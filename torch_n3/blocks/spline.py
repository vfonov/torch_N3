"""The smooth field fit: a cubic tensor B-spline (``spline_smooth -b_spline``).

Each iteration of N3 hands this block a noisy, voxelwise estimate of the log
field and asks for the smooth part of it.  The answer is a tensor product of
uniform cubic B-splines, fitted by least squares with a penalty on the
spline's bending energy:

.. math::  (A^T A + \\lambda\\, N\\, J)\\, c = A^T f

``A`` holds the basis functions at every masked voxel, ``J`` the bending
energy of the basis (the integrated second derivatives), ``N`` the number of
samples, and ``lambda`` is N3's ``-lambda``.  With the default 200 mm knot
spacing there are only a few dozen coefficients, which is what makes the field
smooth: it simply cannot represent anything sharper.

The spline is defined on a box in *world* coordinates and evaluates to exactly
zero outside it, so a field fitted on the coarse estimation grid can be
evaluated at full resolution -- N3 does that through its ``.imp`` file, and
:meth:`BSplineField.evaluate_on` does it directly.

Ported from ``legacy/N3/src/Splines/TBSpline.cc``.
"""

import math

import numpy as np
import torch

#: The legacy's guard against a knot landing exactly on the domain edge
#: (``TBSpline.cc:60``).
EPSILON = 1.0e-14

#: How many dense matrix entries to materialise at once while accumulating the
#: normal equations.  Only affects speed and peak memory, not the result.
_CHUNK_ELEMENTS = 1 << 22


class BSplineField:
    """A cubic tensor B-spline fitted to a masked volume.

    ``distance`` is the knot spacing in world units (N3's ``-distance``) and
    ``lam`` the weight on the bending energy (``-lambda``).  ``domain_world``
    is the box the spline is defined on; by default it is the whole bounding
    box of ``grid``, which is what ``spline_smooth -full_support`` uses.
    """

    def __init__(self, grid, distance=200.0, lam=1e-7, domain_world=None):
        self.grid = grid
        self.distance = float(distance)
        self.lam = float(lam)
        if self.distance <= 0:
            raise ValueError("knot spacing must be positive")

        low, high = _domain_of(grid) if domain_world is None else domain_world
        low, high = np.asarray(low, float), np.asarray(high, float)
        self.domain_world = (np.minimum(low, high), np.maximum(low, high))

        self.device = grid.data.device
        self._coefficients = None
        self._nsamples = 0

        # Number of basis functions per axis, and the knots they sit between.
        # Four extra knots at each end give every voxel four overlapping
        # cubics to be expressed in.
        span = self.domain_world[1] - self.domain_world[0]
        self.n = [int(math.ceil(s / (self.distance * (1.0 + EPSILON)))) + 3
                  for s in span]
        self._knots = [self._axis_knots(axis) for axis in range(3)]
        self._scale = 1.0 / self.distance ** 3

    # ------------------------------------------------------------------ fit

    def fit(self, values, mask=None, subsample=1):
        """Fit to ``values``; only voxels where ``mask`` is true contribute.

        ``subsample`` takes every n-th voxel along each axis (``-subsample``),
        which thins the fit without changing what it can represent.
        """
        values = torch.as_tensor(values, dtype=torch.float64)
        if tuple(values.shape) != tuple(self.grid.shape):
            raise ValueError("values shape %s != grid shape %s"
                             % (tuple(values.shape), tuple(self.grid.shape)))

        step = int(subsample)
        selected = values[::step, ::step, ::step]
        if mask is None:
            inside = torch.ones_like(selected, dtype=torch.bool)
        else:
            inside = torch.as_tensor(mask).to(torch.bool)[::step, ::step, ::step]

        where = torch.nonzero(inside)
        if where.numel() == 0:
            raise RuntimeError("B-spline: no data points inside the mask")

        # Every sample sits in one 4x4x4 block of basis functions.  Voxels
        # outside the domain get an all-zero block -- they contribute nothing
        # but, as in the legacy, still count towards N.
        basis, corner = [], []
        for axis in range(3):
            terms, block = self._axis_basis(
                axis, self._axis_world(self.grid, axis, step))
            basis.append(terms[where[:, axis]])
            corner.append(block[where[:, axis]])

        weights = (basis[0][:, :, None, None] * basis[1][:, None, :, None]
                   * basis[2][:, None, None, :]).reshape(-1, 64)
        columns = self._flat_indices(corner)
        sampled = selected[where[:, 0], where[:, 1], where[:, 2]]
        self._nsamples = int(where.shape[0])

        normal, right = self._normal_equations(columns, weights, sampled)
        penalty = bending_energy_tensor(self.n, self.device)
        system = normal + (self.lam * self._nsamples) * penalty

        self._coefficients = torch.linalg.solve(system, right)
        return self

    def _normal_equations(self, columns, weights, values):
        """Accumulate ``AtA`` and ``AtF`` without ever holding all of ``A``.

        ``A`` has one row per sample and 64 non-zeros in it.  Materialising a
        slab of rows at a time turns the accumulation into two matrix
        products, which is both fast and indifferent to the device.
        """
        size = int(np.prod(self.n))
        normal = torch.zeros((size, size), dtype=torch.float64,
                             device=self.device)
        right = torch.zeros(size, dtype=torch.float64, device=self.device)

        rows = max(1, _CHUNK_ELEMENTS // size)
        for start in range(0, columns.shape[0], rows):
            stop = start + rows
            slab = torch.zeros((min(stop, columns.shape[0]) - start, size),
                               dtype=torch.float64, device=self.device)
            slab.scatter_(1, columns[start:stop], weights[start:stop])
            normal += slab.T @ slab
            right += slab.T @ values[start:stop]
        return normal, right

    # ------------------------------------------------------------- evaluate

    @property
    def coefficients(self):
        """The fitted coefficients, one per basis function."""
        if self._coefficients is None:
            raise RuntimeError("coefficients before fit()")
        return self._coefficients

    def evaluate(self):
        """Evaluate the fitted spline on the grid it was fitted to."""
        return self.evaluate_on(self.grid)

    def evaluate_on(self, grid):
        """Evaluate the fitted spline on any grid in the same world space.

        The basis is a tensor product, so this contracts one axis at a time
        rather than forming the 4x4x4 block per voxel.
        """
        coefficients = self.coefficients.reshape(*self.n)
        neighbours = torch.arange(4, device=self.device)

        terms, block = [], []
        for axis in range(3):
            axis_terms, axis_block = self._axis_basis(
                axis, self._axis_world(grid, axis))
            terms.append(axis_terms)
            block.append(axis_block[:, None] + neighbours)

        # (n0, n1, nz, 4) -> (n0, n1, nz), then the same along y and x.
        field = (coefficients[:, :, block[2]] * terms[2]).sum(-1)
        field = (field[:, block[1], :] * terms[1][..., None]).sum(2)
        field = (field[block[0]] * terms[0][:, :, None, None]).sum(1)
        return field

    # -------------------------------------------------------------- private

    def _axis_knots(self, axis):
        """Knot positions along ``axis``, centred on the domain."""
        low, high = self.domain_world[0][axis], self.domain_world[1][axis]
        count = self.n[axis] + 4
        first = 0.5 * (low + high - self.distance * (self.n[axis] + 3))
        return float(first) + self.distance * torch.arange(
            count, dtype=torch.float64, device=self.device)

    def _axis_world(self, grid, axis, subsample=1):
        """World coordinates of the (sub-sampled) voxel centres along ``axis``."""
        index = torch.arange(0, grid.shape[axis], subsample,
                             dtype=torch.float64, device=self.device)
        return float(grid.start[axis]) + index * float(grid.step[axis])

    def _axis_basis(self, axis, coordinates):
        """The four non-zero cubics at each coordinate, and where they start.

        Returns ``(terms, block)`` with ``terms`` of shape ``(n, 4)`` holding
        the basis values and ``block`` the index of the first of the four
        basis functions.  Coordinates outside the domain get zeros, which is
        how the legacy makes the spline vanish there.
        """
        knots, distance = self._knots[axis], self.distance
        block = torch.ceil((coordinates - knots[3]) / distance).long() - 1
        block = block.clamp(0, self.n[axis] - 4)

        # The four cubic B-spline segments, written as the legacy writes them:
        # differences of cubes of the distance to the surrounding knots.
        rising = self._scale * (coordinates - knots[block + 3]) ** 3
        falling = self._scale * (knots[block + 4] - coordinates) ** 3
        terms = torch.stack([
            falling,
            self._scale * (knots[block + 5] - coordinates) ** 3 - 4.0 * falling,
            self._scale * (coordinates - knots[block + 2]) ** 3 - 4.0 * rising,
            rising,
        ], dim=-1)

        low, high = self.domain_world[0][axis], self.domain_world[1][axis]
        inside = (coordinates >= low) & (coordinates <= high)
        return (torch.where(inside[:, None], terms, torch.zeros_like(terms)),
                torch.where(inside, block, torch.zeros_like(block)))

    def _flat_indices(self, corner):
        """Flat coefficient indices of each sample's 4x4x4 block."""
        base = ((corner[0] * self.n[1] + corner[1]) * self.n[2] + corner[2])
        neighbours = torch.arange(4, device=self.device)
        offsets = ((neighbours[:, None, None] * self.n[1]
                    + neighbours[None, :, None]) * self.n[2]
                   + neighbours[None, None, :]).reshape(-1)
        return base[:, None] + offsets


def _domain_of(grid):
    """The world-space box a ``-full_support`` spline is defined on.

    ``splineSmooth.cc:232`` takes the whole volume, half a voxel beyond the
    outermost voxel centres.
    """
    shape = np.asarray(grid.shape, dtype=np.float64)
    return (grid.start - 0.5 * grid.step,
            grid.start + (shape - 0.5) * grid.step)


# --------------------------------------------------------------- bending energy

def bending_energy_tensor(n, device=None):
    """The 3-D bending energy of a tensor cubic B-spline basis.

    In three dimensions the thin-plate energy expands to
    ``x''yz + xy''z + xyz'' + 2x'y'z + 2x'yz' + 2xy'z'``, and because the
    basis is a tensor product each term is a Kronecker product of the 1-D
    matrices below (``TBSpline.cc:393``).
    """
    axes = [[bending_energy(size, order, device) for order in range(3)]
            for size in n]

    energy = 0.0
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 2
        energy = energy + _kron(axes, order)
    for axis in range(3):
        for other in range(axis + 1, 3):
            order = [0, 0, 0]
            order[axis] = order[other] = 1
            energy = energy + 2.0 * _kron(axes, order)
    return energy


def _kron(axes, order):
    return torch.kron(axes[0][order[0]],
                      torch.kron(axes[1][order[1]], axes[2][order[2]]))


def bending_energy(size, order, device=None):
    """The 1-D matrix of ``integral(b_i^(order) * b_j^(order))``.

    ``size`` basis functions on a uniform knot grid.  Interior functions all
    see the same four-span overlap, so the matrix is banded with a constant
    band; only the first and last two rows differ, because those splines are
    clipped by the end of the domain (``TBSpline.cc:476``).
    """
    if size < 4:
        raise ValueError("bending energy is undefined for fewer than 4 splines")

    products = _segment_products(order)

    # integral[region][offset]: the overlap integral of two splines shifted by
    # `offset` spans, over each of six possible overlap regions.
    integral = [[0.0] * 4 for _ in range(6)]
    for offset in range(4):
        for region in range(4 - offset):
            integral[region][offset] = sum(products[i][i + offset]
                                           for i in range(region + 1))
        for region in range(4 - offset, 4):
            integral[region][offset] = integral[region - 1][offset]
    integral[4] = [products[1][1], products[1][2], products[1][3], 0.0]
    integral[5] = [products[1][1] + products[2][2],
                   products[1][2] + products[2][3], products[1][3], 0.0]

    energy = [[0.0] * size for _ in range(size)]

    def put(i, j, value):
        energy[i][j] = energy[j][i] = value

    for i in range(3):  # the two corners, common to every size
        put(0, i, integral[0][i])
        put(size - 1, size - i - 1, integral[0][i])

    if size == 4:
        put(1, 1, integral[4][0])
        put(2, 2, integral[4][0])
        put(1, 2, integral[4][1])
        put(3, 0, integral[3][3])
    elif size == 5:
        put(1, 1, integral[1][0])
        put(3, 3, integral[1][0])
        put(2, 2, integral[5][0])
        put(1, 2, integral[1][1])
        put(2, 3, integral[1][1])
        put(1, 3, integral[3][2])
        for i in range(size - 3):
            put(i, i + 3, integral[3][3])
    else:
        for band in range(4):  # the repeating interior band
            for i in range(3 - band, size - 3):
                put(i, i + band, integral[3][band])
        put(2, 2, integral[2][0])
        put(size - 3, size - 3, integral[2][0])
        put(1, 2, integral[1][1])
        put(size - 2, size - 3, integral[1][1])
        put(1, 1, integral[1][0])
        put(size - 2, size - 2, integral[1][0])

    return torch.tensor(energy, dtype=torch.float64, device=device)


def _segment_products(order):
    """``D[i][j]``: the integral over one span of segments ``i`` and ``j``.

    A cubic B-spline is four cubic segments laid end to end; here each is
    written on ``[0, 1]`` with coefficients in descending powers, differentiated
    ``order`` times, and multiplied out pairwise.
    """
    segments = [[-1.0, 3.0, -3.0, 1.0], [3.0, -6.0, 0.0, 4.0],
                [-3.0, 3.0, 3.0, 1.0], [1.0, 0.0, 0.0, 0.0]]
    for _ in range(order):
        segments = [[0.0, 3 * c[0], 2 * c[1], c[2]] for c in segments]

    products = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            product = [0.0] * 7
            for a in range(4):
                for b in range(4):
                    product[a + b] += segments[i][a] * segments[j][b]
            # Descending powers, so entry k is the coefficient of x^(6-k) and
            # integrates to itself over 1/(7-k).
            products[i][j] = sum(c / (7 - k) for k, c in enumerate(product))
    return products

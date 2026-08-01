"""The smooth field fit: a cubic tensor B-spline (``spline_smooth -b_spline``).

Each iteration of N3 hands this block a noisy, voxelwise estimate of the log
field and asks for the smooth part of it.  The answer is a tensor product of
uniform cubic B-splines, fitted by least squares with a penalty on the
spline's bending energy:

.. math::  (A^T A + \\lambda\\, N\\, J)\\, c = A^T f

``A`` holds the basis functions at every masked voxel, ``J`` the bending
energy of the basis (the integrated second derivatives), ``N`` the number of
samples, and ``lambda`` is N3's ``-lambda``.  With the default 200 mm knot
spacing a whole head is described by 80 coefficients, which is most of what
makes the field smooth: it simply cannot represent anything sharper.  The
penalty is the rest, and the two have to be set together -- halving the
spacing without raising ``lambda`` gives the fit enough freedom to start
following tissue contrast instead.  See ``python3 -m torch_n3 --help``.

Two solvers answer that least-squares problem, selected by ``solver``:

``"normal"``
    Form ``AtA + lambda*N*J`` and solve it, which is what the legacy does
    (``TBSpline.cc:290``, via LAPACK's ``dsysv``).  Squaring ``A`` squares its
    condition number, and at the default spacing -- where the knots are
    further apart than the head is wide -- that lands around ``1e13``, close
    enough to singular that the last digits of the answer belong to whichever
    BLAS ran it.  This is the reference: bit-for-bit what the repository has
    always computed.

``"qr"``
    Solve the same problem without ever squaring ``A``, by stacking the
    penalty underneath the data and taking a QR factorization of

    .. math::  \\begin{bmatrix} A \\\\ \\sqrt{\\lambda N}\\, D \\end{bmatrix}
               c \\simeq \\begin{bmatrix} f \\\\ 0 \\end{bmatrix},
               \\qquad D^T D = J

    whose normal equations are the ones above, term for term, so it minimises
    the *same* objective -- but its condition number is the square root of the
    other's, and the answer no longer depends on the machine to the degree the
    normal equations do.  On the grid ``brain.mnc`` is estimated on, that is
    ``2.3e6`` against ``5.3e12``, and the fitted field moves ``3.0e-13``
    relative RMS between a CPU and a GPU where the normal equations move it by
    ``2.5e-9``.

``"blocked"``
    The same stacked system and the same answer as ``"qr"`` -- they agree to
    ``4e-15``, which is rounding -- reached without ever holding ``A``.  Its
    rows are sorted, grouped, and folded into a running ``R`` one band at a
    time.  Sorting is what makes the groups cheap: ``A`` is banded once its
    rows are ordered by their first-axis knot, so each group touches one
    window of ``4*n1*n2`` columns instead of all ``n0*n1*n2`` of them.  This
    is the one to reach for at a fine ``-distance``, where the dense stack
    stops fitting in memory: at 12.5 mm on ``chunk.mnc`` (3168 coefficients)
    it peaks at 1.55 GB against ``"qr"``'s 7.53 GB, and runs 3.7x faster.  At
    200 mm there is a single group and it *is* ``"qr"``, plus a sort.

``"dr"``
    The same stacked system, factorised once and then *reparameterized* so
    that the penalty becomes diagonal -- the Demmler-Reinsch basis.  Take the
    triangular factor ``R0`` of ``[A; sqrt(lambda_0 N) D]`` at an anchor weight
    ``lambda_0``, so that ``R0' R0 = A'A + lambda_0 N J`` exactly, and let
    ``D~ = sqrt(N) D R0^-1``.  Then for *every* weight

    .. math::  A^T A + \\lambda N J
               = R_0^T \\bigl(I + (\\lambda - \\lambda_0)\\,
                 \\tilde{D}^T \\tilde{D}\\bigr) R_0,

    and eigendecomposing ``D~'D~ = U diag(gamma) U'`` turns the middle factor
    into a diagonal one.  The fit is then

    .. math::  c = R_0^{-1} U \\,
               \\frac{U^T Q_0^T [f; 0]}{1 + (\\lambda - \\lambda_0)\\gamma},

    a triangular solve around an elementwise division.  At ``lambda ==
    lambda_0`` the divisor is one and this *is* ``"qr"``, back-substitution and
    all; the two agree to rounding, which is what the suite holds it to.

    What it buys is a ``lambda`` sweep.  The QR, the triangular solve and the
    eigendecomposition are all independent of ``lambda``, so a whole grid --
    GCV or REML smoothing-parameter selection, or the ``--lambda`` x
    ``--distance`` tables -- costs one factorization plus one division per
    point instead of one fit per point.  :meth:`BSplineField.refit` is that
    division: on ``brain.mnc``'s estimation grid a further weight costs 0.033
    ms at 200 mm and 0.140 ms at 50 mm, against 5.6 ms and 27.7 ms for a fresh
    ``"qr"`` fit -- 170x to 200x.  ``gamma`` is non-negative and the divisor
    therefore never below one, so no amount of dynamic range in ``gamma``
    (seven decades here) can reach the answer.

    **For a single weight this is the slowest solver, and there is no reason
    to pick it.**  The eigendecomposition of ``D~'D~`` is ``O(k^3)`` on top of
    everything ``"qr"`` does, and it buys nothing until a second weight is
    asked for.  One fit on the same grid: 7.3 ms at 200 mm against ``"qr"``'s
    5.6 ms and ``"normal"``'s 4.9 ms, and 43.6 ms at 50 mm against 27.7 ms and
    8.3 ms -- so a whole 30-iteration pipeline runs 1.8 s at 50 mm where
    ``"qr"`` runs 1.2 s.  It pays from the second weight onwards and is
    dramatic by the fourth; below that, use ``"qr"``.

    **The anchor is what makes this work, and it is not what the textbook
    recipe does.**  Demmler-Reinsch is usually written on the QR of ``A``
    alone, whose ``R`` is then used for ``D~``.  That is unusable here: ``A``
    is the masked design, and at fine knot spacings the mask leaves basis
    functions with no data under them at all.  Measured on ``chunk.mnc``,
    ``cond(A)`` is ``8.4e7`` at 200 mm and ``7.5e12`` at 50 mm, where ``A``
    is *rank deficient* -- 243 of 245 columns -- and ``D R^-1`` overflows into
    a ``gamma`` with 83 non-positive entries reaching ``-5.7e6``.  Clipping
    those at zero does not help; the eigenvectors are as damaged as the
    eigenvalues.  Anchoring on the stacked matrix instead costs nothing and
    fixes it outright, because the penalty rows span precisely the directions
    the data leaves empty: ``cond(R0)`` is the stacked system's ``3.5e6`` and
    ``2.9e5`` at those two spacings.  The price is that the basis is only
    valid at or above its anchor -- below it the divisor can pass through zero
    -- so ``anchor`` belongs at the bottom of the intended grid.

``"sparse"``
    The same stacked system again, held in ``scipy.sparse`` and handed to
    ``lsqr``.  **It does not converge on this problem and should not be used
    for results.**  Rectangular systems rule out every direct sparse solver in
    ``scipy.sparse.linalg`` -- it has no sparse QR -- which leaves iterative
    methods, and LSQR's convergence is governed by the very condition number
    the stacked form was chosen to reduce.  At 200 mm it stops after ~2,950
    iterations reporting ``istop=3`` ("condition number exceeds ``conlim``"),
    18 s later and ``1.8e-3`` away from the direct answer -- worse than the
    gap between this port and the original C++.  Raising the iteration limit
    does not help; it is not stopping early.  Kept because the measurement is
    worth having: see :attr:`BSplineField.solve_info` for what LSQR reports,
    and ``PROBLEMS.md`` for the whole finding.

The three direct solvers agree to far better than the fit is determined to;
where they differ, the stacked pair is the more accurate (they reach a
strictly lower residual than the normal equations).

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

#: The solvers :meth:`BSplineField.fit` can be asked for.  ``"normal"`` is the
#: legacy's own formulation and the reference the others are measured against;
#: see this module's docstring.
SOLVERS = ("normal", "qr", "blocked", "dr", "sparse")

#: The solvers that answer the fit by a direct factorization, and so to the
#: last few bits of float64.  ``"sparse"`` is iterative and is not among them:
#: it stops on a tolerance it cannot reach here.  Anything asserting an exact
#: fit should be parametrised over these, not over :data:`SOLVERS`.
DIRECT_SOLVERS = ("normal", "qr", "blocked", "dr")


class BSplineField:
    """A cubic tensor B-spline fitted to a masked volume.

    ``distance`` is the knot spacing in world units (N3's ``-distance``) and
    ``lam`` the weight on the bending energy (``-lambda``).  ``domain_world``
    is the box the spline is defined on; by default it is the whole bounding
    box of ``grid``, which is what ``spline_smooth -full_support`` uses.
    ``solver`` picks how the least-squares problem is answered -- see the
    module docstring; all of them minimise the same objective.

    ``anchor`` belongs to ``solver="dr"`` alone: it is the weight that solver
    takes its factorization at, and the lowest one :meth:`refit` can then be
    asked for.  It defaults to ``lam``, which makes a single fit behave exactly
    like ``"qr"``; set it to the bottom of the grid you mean to sweep.
    """

    def __init__(self, grid, distance=200.0, lam=1e-7, domain_world=None,
                 solver="normal", anchor=None):
        self.grid = grid
        self.distance = float(distance)
        self.lam = float(lam)
        if self.distance <= 0:
            raise ValueError("knot spacing must be positive")
        if solver not in SOLVERS:
            raise ValueError("unknown solver %r: expected one of %s"
                             % (solver, ", ".join(map(repr, SOLVERS))))
        self.solver = solver

        if anchor is not None and solver != "dr":
            raise ValueError("anchor is meaningful to solver=\"dr\" alone, "
                             "not to %r" % solver)
        self.anchor = self.lam if anchor is None else float(anchor)
        if self.anchor <= 0.0:
            raise ValueError("the anchor weight must be positive")
        if self.anchor > self.lam:
            raise ValueError("anchor %g is above lambda %g: a Demmler-Reinsch "
                             "basis is only valid at or above its anchor"
                             % (self.anchor, self.lam))
        self._dr_basis = None

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

        columns, weights, where = self.design(mask, step)
        sampled = selected[where[:, 0], where[:, 1], where[:, 2]]
        self._nsamples = int(where.shape[0])

        solve = {"qr": self._solve_stacked, "sparse": self._solve_sparse,
                 "blocked": self._solve_blocked, "dr": self._solve_dr,
                 "normal": self._solve_normal}[self.solver]
        self._coefficients = solve(columns, weights, sampled)
        return self

    def design(self, mask=None, subsample=1):
        """Where each sample sits in the basis: ``(columns, weights, where)``.

        One row per masked voxel, holding the 64 basis functions that voxel
        falls under (``weights``) and their flat coefficient indices
        (``columns``), plus the voxel indices themselves (``where``).  This is
        the design matrix ``A`` in the sparse form it is actually built in --
        every row has 64 non-zeros and no other -- which is why the solvers
        can scatter it into slabs instead of holding it whole.

        :meth:`fit` uses it to build a system.  :mod:`torch_n3.optimize` uses
        it to *evaluate*: ``(weights * c[columns]).sum(1)`` is the field at the
        samples, differentiable in ``c``, with no dense matrix anywhere.  Both
        want exactly these two tensors, which is why this is public.
        """
        step = int(subsample)
        if mask is None:
            shape = tuple(len(range(0, length, step))
                          for length in self.grid.shape)
            inside = torch.ones(shape, dtype=torch.bool, device=self.device)
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
        return self._flat_indices(corner), weights, where

    def _solve_normal(self, columns, weights, values):
        """Form the penalised normal equations and solve them (the legacy's way)."""
        normal, right = self._normal_equations(columns, weights, values)
        penalty = bending_energy_tensor(self.n, self.device)
        system = normal + (self.lam * self._nsamples) * penalty
        return torch.linalg.solve(system, right)

    def _solve_stacked(self, columns, weights, values):
        """Solve the same fit as ``[A; sqrt(lam*N) D] c ~ [f; 0]``, in one go.

        The design matrix is built whole -- one row per sample, 64 non-zeros
        in each -- with the penalty rows written underneath it, and handed to
        a least-squares driver.

        That is the one real cost of this solver: ``A`` is held dense, where
        the normal equations only ever held ``AtA``, which is ``N x size``
        doubles against ``size x size``.  N3 estimates on the grid ``-shrink``
        leaves, so on ``brain.mnc`` it is 25 MB at the default shrink of 4 and
        600 MB at ``-shrink 1``; peak RSS for the whole pipeline measured 1.0
        GB against 1.3 GB there.  Time is not the issue -- 10 iterations of
        the shipped protocol run 0.38 s against 0.41 s on a CPU.
        """
        size = int(np.prod(self.n))
        factor = bending_energy_factor(self.n, self.device)
        scale = math.sqrt(self.lam * self._nsamples)

        stacked = torch.zeros((columns.shape[0] + factor.shape[0], size),
                              dtype=torch.float64, device=self.device)
        stacked[:columns.shape[0]].scatter_(1, columns, weights)
        stacked[columns.shape[0]:] = scale * factor

        right = torch.zeros(stacked.shape[0], dtype=torch.float64,
                            device=self.device)
        right[:values.shape[0]] = values

        # "gels" is Householder QR, and the one driver both CPU and CUDA
        # implement; naming it keeps the same factorization on either device,
        # where the default would pick a different one per device and put the
        # machine back into the answer.  It reports rank deficiency as an
        # error rather than returning a minimum-norm answer for it, which is
        # what we want -- with lambda above zero it takes a nearly empty mask.
        return torch.linalg.lstsq(stacked, right, driver="gels").solution

    def _solve_blocked(self, columns, weights, values):
        """``"qr"``'s answer, accumulated a band at a time instead of at once.

        The design matrix is never held.  Its rows arrive in groups and each
        group is folded into a running ``R`` by re-triangularising ``[R;
        group]`` -- a tall-skinny QR, which gives the same ``R`` as one
        factorization of the whole stack.  The right-hand side rides along as
        an extra *column*, so the rotations that triangularise ``A`` deliver
        ``Q^T f`` in it and the fit is one back-substitution at the end.

        What makes the groups cheap is that ``A`` is banded once its rows are
        sorted.  A sample whose first-axis corner is ``k`` has its 64
        non-zeros inside the flat columns ``[k*n1*n2, (k+4)*n1*n2)`` -- one
        contiguous window of width ``4*n1*n2``, because the flat index runs
        first-axis-slowest.  Take the groups in increasing ``k`` and two
        things follow: a group can only touch the window, and every row of
        ``R`` above the window is already final, since the incoming rows are
        zero in the columns those rows pivot on.  So each step factorises a
        ``(window + group)`` by ``window`` block rather than anything the size
        of the whole system.

        The bending-energy rows are the exception -- ``D`` comes from an
        eigendecomposition and is dense across every column, so it cannot join
        the band.  It goes in at the end, as one ``2*size`` by ``size`` QR,
        which is small: the sweep above has already absorbed the ``N`` rows
        that made the problem big.

        At the shipped 200 mm spacing this degenerates to :meth:`_solve_stacked`
        -- ``n0`` is 4, so there is a single group and the window is the whole
        system.  It pays at fine spacings, where ``size`` grows as the cube of
        the knot count but the window only as the square.
        """
        size = int(np.prod(self.n))
        stride = self.n[1] * self.n[2]
        width = 4 * stride

        # Sort the samples by the first-axis corner their block starts at.
        # `columns[:, 0]` is that block's flat index, and the two trailing
        # axes contribute less than one stride, so integer division recovers
        # the corner exactly.
        corner = torch.div(columns[:, 0], stride, rounding_mode="floor")
        order = torch.argsort(corner, stable=True)
        corner, columns = corner[order], columns[order]
        weights, values = weights[order], values[order]

        groups = self.n[0] - 3
        edges = torch.searchsorted(
            corner.contiguous(),
            torch.arange(groups + 1, device=self.device, dtype=corner.dtype))

        # `state` holds R alongside the transformed right-hand side, so the
        # last column is not part of the triangle.
        state = torch.zeros((size, size + 1), dtype=torch.float64,
                            device=self.device)

        for k in range(groups):
            low, high = int(edges[k]), int(edges[k + 1])
            if high <= low:
                continue
            start = k * stride
            stop = min(size, start + width)
            span = stop - start

            block = torch.zeros((high - low, span + 1), dtype=torch.float64,
                                device=self.device)
            block[:, :span].scatter_(1, columns[low:high] - start,
                                     weights[low:high])
            block[:, span] = values[low:high]

            carried = torch.cat([state[start:stop, start:stop],
                                 state[start:stop, size:size + 1]], dim=1)
            triangle = torch.linalg.qr(torch.cat([carried, block], dim=0),
                                       mode="r")[1]

            # Keep the rows that still span the window's columns.  The row
            # below them, if the block was tall enough to produce one, is zero
            # in every column of `A` -- it carries only the residual -- and
            # can never affect the fit again.
            kept = min(span, triangle.shape[0])
            state[start:stop, start:stop] = 0.0
            state[start:stop, size] = 0.0
            state[start:start + kept, start:stop] = triangle[:kept, :span]
            state[start:start + kept, size] = triangle[:kept, span]

        factor = bending_energy_factor(self.n, self.device)
        scale = math.sqrt(self.lam * self._nsamples)
        penalty = torch.cat([scale * factor,
                             torch.zeros((factor.shape[0], 1),
                                         dtype=torch.float64,
                                         device=self.device)], dim=1)
        final = torch.linalg.qr(torch.cat([state, penalty], dim=0),
                                mode="r")[1]

        return torch.linalg.solve_triangular(
            final[:size, :size], final[:size, size:], upper=True)[:, 0]

    def _solve_dr(self, columns, weights, values):
        """Factorise once into the Demmler-Reinsch basis, then solve by division.

        The stacked matrix is built exactly as :meth:`_solve_stacked` builds
        it, but at the *anchor* weight and with the right-hand side carried
        along as an extra column, so that triangularising it delivers ``R0``
        and ``Q0'[f; 0]`` together and ``Q0`` itself is never formed -- the
        same trick :meth:`_solve_blocked` uses to get its back-substitution
        for free.

        Everything expensive is in that factorization and in the
        eigendecomposition :class:`DemmlerReinschBasis` does on top of it, and
        neither depends on ``lambda``.  The basis is kept so that
        :meth:`refit` can sweep one.
        """
        size = int(np.prod(self.n))
        factor = bending_energy_factor(self.n, self.device)
        scale = math.sqrt(self.anchor * self._nsamples)
        rows = columns.shape[0]

        stacked = torch.zeros((rows + factor.shape[0], size + 1),
                              dtype=torch.float64, device=self.device)
        stacked[:rows, :size].scatter_(1, columns, weights)
        stacked[:rows, size] = values
        stacked[rows:, :size] = scale * factor

        triangle = torch.linalg.qr(stacked, mode="r")[1]
        self._dr_basis = DemmlerReinschBasis(
            triangle[:size, :size], triangle[:size, size], factor,
            self._nsamples, self.anchor)
        return self._dr_basis.coefficients(self.lam)

    def _solve_sparse(self, columns, weights, values):
        """``"qr"``'s stacked system, held sparse and solved iteratively.

        Exactly the matrix :meth:`_solve_stacked` builds -- ``[A; sqrt(lam*N)
        D]`` against ``[f; 0]`` -- but ``A`` is stored as the sparse thing it
        is: 64 non-zeros in a row of ``size``, which at the shipped 200 mm
        spacing is 80% full and at 25 mm is 2%.  Nothing here forms ``AtA`` or
        ``At f``, so the conditioning is the stacked system's, as in ``"qr"``.

        The solver has to be ``lsqr`` rather than ``spsolve`` because the
        system is rectangular -- ``spsolve`` takes square systems only, and a
        square one here would mean the normal equations again.  ``lsqr``
        applies its own bidiagonalisation to ``A`` directly and never forms
        them, which is the whole point; being iterative, it stops on a
        tolerance rather than at a fixed cost, and that tolerance is what
        ``atol``/``btol`` set below.

        ``scipy`` is imported here rather than at the top of the module so
        that it is needed only by whoever asks for this solver, and the solve
        runs on the CPU whatever device the fit was assembled on.
        """
        from scipy.sparse import coo_matrix, csr_matrix, vstack
        from scipy.sparse.linalg import lsqr

        size = int(np.prod(self.n))
        samples, block = columns.shape
        scale = math.sqrt(self.lam * self._nsamples)

        sparse_design = coo_matrix(
            (weights.reshape(-1).cpu().numpy(),
             (np.repeat(np.arange(samples), block),
              columns.reshape(-1).cpu().numpy())),
            shape=(samples, size))
        factor = csr_matrix(
            bending_energy_factor(self.n).cpu().numpy() * scale)

        stacked = vstack([sparse_design, factor]).tocsr()
        right = np.concatenate([values.cpu().numpy(),
                                np.zeros(factor.shape[0])])

        # Tolerances at the float64 floor: this is asked to reproduce a direct
        # solve, not to stop early.  `lsqr` measures both against norms of its
        # own residual, so they are relative and need no scaling by the data.
        # It will not reach them -- see :attr:`solve_info`.
        answer = lsqr(stacked, right, atol=1e-14, btol=1e-14,
                      iter_lim=20 * size)
        self.solve_info = {"istop": int(answer[1]), "iterations": int(answer[2]),
                           "residual": float(answer[3])}
        return torch.as_tensor(answer[0], dtype=torch.float64,
                               device=self.device)

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

    @coefficients.setter
    def coefficients(self, values):
        """Set the coefficients without fitting anything.

        For a caller that chose them some other way -- :mod:`torch_n3.optimize`
        arrives at them by gradient descent -- and then wants the evaluation
        machinery below.  Nothing else about the object changes, so
        :attr:`solve_info` and :attr:`dr_basis` stay as they were, which for an
        unfitted spline means absent.
        """
        values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        size = int(np.prod(self.n))
        if values.numel() != size:
            raise ValueError("expected %d coefficients, got %d"
                             % (size, values.numel()))
        self._coefficients = values

    @property
    def dr_basis(self):
        """The Demmler-Reinsch decomposition, once ``solver="dr"`` has fitted."""
        if self._dr_basis is None:
            raise RuntimeError("no Demmler-Reinsch basis: that needs "
                               "solver=\"dr\" and a completed fit()")
        return self._dr_basis

    def refit(self, lam):
        """Move to another ``lambda`` without factorising again.

        Only ``solver="dr"`` can do this, and only at or above its ``anchor``.
        The whole cost is one elementwise division and a triangular solve, so
        a ``lambda`` grid is a sweep over this rather than a sequence of fits;
        the answer is the same one a fresh fit at ``lam`` would reach.
        """
        basis = self.dr_basis
        self._coefficients = basis.coefficients(lam)
        self.lam = float(lam)
        return self

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


# ------------------------------------------------------- Demmler-Reinsch basis

class DemmlerReinschBasis:
    """A penalised least-squares fit reparameterized so the penalty is diagonal.

    Built from the triangular factor of the *stacked* system at an anchor
    weight -- ``R0' R0 = A'A + lambda_0 N J`` -- together with the transformed
    right-hand side ``rhs = Q0'[f; 0]``, which the same factorization produces
    when the data rides along as an extra column.  See this module's docstring
    for the algebra and for why the anchor is on the stacked matrix rather than
    on ``A`` alone.

    Everything here is independent of ``lambda``: the triangular solve that
    forms ``D~ = sqrt(N) D R0^-1`` (Step 2), the eigendecomposition ``D~'D~ =
    U diag(gamma) U'`` (Step 3), and the projection of the right-hand side into
    that basis (Step 4).  :meth:`coefficients` is Steps 5 and 6, and is the
    only part a ``lambda`` sweep repeats.

    ``gamma`` is the penalty spectrum in the transformed basis, sorted
    ascending by ``eigh``.  Its four smallest entries are zero to working
    precision in three dimensions -- ``J`` cannot see an affine field -- and
    those are the unpenalised trend components of the classical
    Demmler-Reinsch construction.

    References
    ----------
    Demmler, A. & Reinsch, C. (1975). "Oscillation matrices with spline
    smoothing." *Numerische Mathematik* 24, 375-382.
    Eilers, P. H. C. & Marx, B. D. (1996). "Flexible smoothing with B-splines
    and penalties." *Statistical Science* 11, 89-121.
    Ruppert, D., Wand, M. P. & Carroll, R. J. (2003). *Semiparametric
    Regression*, ch. 3.
    """

    def __init__(self, triangle, rhs, factor, nsamples, anchor):
        self.triangle = triangle
        self.rhs = rhs
        self.nsamples = int(nsamples)
        self.anchor = float(anchor)

        # Step 2.  `left=False` solves `X @ triangle = factor`, which is
        # `D R0^-1` -- a triangular solve, never an explicit inverse.
        self.d_tilde = math.sqrt(self.nsamples) * torch.linalg.solve_triangular(
            triangle, factor, upper=True, left=False)

        # Step 3.  `D~'D~` is a Gram matrix, so its spectrum is non-negative;
        # the clamp is for the handful of entries that come back a rounding
        # error below zero, as in `bending_energy_factor`.
        gamma, vectors = torch.linalg.eigh(self.d_tilde.T @ self.d_tilde)
        self.gamma = gamma.clamp(min=0.0)
        self.vectors = vectors

        # Step 4.
        self.projected = vectors.T @ rhs

    def divisor(self, lam):
        """``1 + (lambda - lambda_0) * gamma`` -- what Step 5 divides by.

        At or above the anchor every entry is at least one, which is the whole
        point of the reparameterization: there is nothing here to cancel, and
        ``gamma``'s dynamic range cannot reach the answer.
        """
        return 1.0 + (float(lam) - self.anchor) * self.gamma

    def coefficients(self, lam):
        """Steps 5 and 6: the elementwise division, then back to coefficients."""
        lam = float(lam)
        if lam < self.anchor:
            raise ValueError(
                "lambda %g is below this basis's anchor %g; the divisor is "
                "only bounded away from zero at or above it, so anchor the "
                "basis at the bottom of the grid you mean to sweep"
                % (lam, self.anchor))

        alpha = self.projected / self.divisor(lam)
        return torch.linalg.solve_triangular(
            self.triangle, (self.vectors @ alpha)[:, None], upper=True)[:, 0]

    @property
    def cond(self):
        """Condition number of ``R0``, i.e. of the stacked system it factorised."""
        return float(torch.linalg.cond(self.triangle))


def fit_penalized_spline_dr(design, factor, values, lam, anchor=None,
                            return_diagnostics=False):
    """Fit ``(B'B + lam D'D) c = B'y`` through the Demmler-Reinsch basis.

    The standalone form of what ``BSplineField(..., solver="dr")`` does, on
    matrices handed in directly: ``design`` is ``B`` (``n x k``), ``factor`` is
    ``D`` (``p x k``, the penalty's square root, ``D'D = J``), ``values`` is
    ``y``.  Note the convention -- ``lam`` multiplies ``D'D`` with no sample
    count in it, so a caller using N3's ``lambda * N * J`` folds the ``N`` in
    itself.

    ``anchor`` is the weight the factorization is taken at, defaulting to
    ``lam``; the returned basis is valid at that weight and above.  Fitting a
    grid means calling this once at the bottom of it and then
    :meth:`DemmlerReinschBasis.coefficients` per point -- the QR and the
    eigendecomposition do not depend on ``lam`` and must not be repeated.

    Returns ``c``, or ``(c, basis)`` when ``return_diagnostics`` is set, the
    basis carrying ``gamma``, ``vectors``, ``triangle``, ``d_tilde`` and
    ``cond``.
    """
    design = torch.as_tensor(design, dtype=torch.float64)
    factor = torch.as_tensor(factor, dtype=torch.float64)
    values = torch.as_tensor(values, dtype=torch.float64)
    lam = float(lam)
    anchor = lam if anchor is None else float(anchor)

    size = design.shape[1]
    stacked = torch.zeros((design.shape[0] + factor.shape[0], size + 1),
                          dtype=torch.float64, device=design.device)
    stacked[:design.shape[0], :size] = design
    stacked[:design.shape[0], size] = values
    stacked[design.shape[0]:, :size] = math.sqrt(anchor) * factor

    triangle = torch.linalg.qr(stacked, mode="r")[1]
    # `nsamples` is 1 here: this signature carries the sample count inside
    # `lam` already, so `D~` is `D R0^-1` unscaled.
    basis = DemmlerReinschBasis(triangle[:size, :size], triangle[:size, size],
                                factor, 1, anchor)

    coefficients = basis.coefficients(lam)
    return (coefficients, basis) if return_diagnostics else coefficients


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


def bending_energy_factor(n, device=None):
    """``D`` with ``D^T D = J``, the stacked solver's half of the penalty.

    ``J`` is a Gram matrix of integrals, so it is symmetric positive
    *semi*-definite -- singular, in fact, by exactly the four dimensions an
    affine field spans, which bends not at all and is what the penalty is
    meant to leave alone.  That rules out a Cholesky factor and leaves the
    symmetric eigendecomposition, which does not mind: ``J = V diag(w) V^T``
    gives ``D = diag(sqrt(w)) V^T``, with the handful of eigenvalues that come
    back a rounding error below zero clamped away.  ``J`` is one coefficient
    square (80 of them at the default spacing), so this costs nothing next to
    the fit.
    """
    energy = bending_energy_tensor(n, device)
    values, vectors = torch.linalg.eigh(energy)
    return torch.sqrt(values.clamp(min=0.0))[:, None] * vectors.T


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

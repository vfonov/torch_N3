"""The *legacy* backend: N3's blocks as implemented by the original C++.

Every function here is a thin wrapper around the CFFI shim in
``torch_n3._legacy``.  No mathematics happens in this file -- that is the whole
point: it is the oracle each block in :mod:`torch_n3.blocks` is tested
against, and it exports the same names so that either can be dropped into the
pipeline (see :func:`torch_n3.backends.resolve`).

Tensors in, tensors out; the conversion to and from ``numpy`` -- and to and
from the CPU -- happens here, because the C code knows nothing about either.

**No MINC file is involved.**  That is worth saying because the original N3 is
a Perl script driving a dozen separate executables, which can only talk to each
other through files, so every intermediate volume it computes is rounded to a
12- or 16-bit MINC image on the way out and rescaled on the way back in.  None
of that applies here: these functions are the same C++ *routines* called
directly, on ``float64`` buffers, in one process.  So ``backend="legacy"``
gives the original arithmetic without the original's quantisation, and does
*not* reproduce the installed programs bit for bit -- on ``brain.mnc`` it lands
3.7e-3 from ``brain_nu_ref.mnc``, slightly further out than the PyTorch blocks
do.  What it is for is comparing block against block with nothing rounded in
between.

Build the extension first::

    python3 torch_n3/_legacy/build_legacy.py
"""

import numpy as np
import torch

from torch_n3._legacy._n3legacy import ffi, lib


def _as_double_array(values):
    """Return a contiguous float64 copy suitable for passing to the shim."""
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_tensor(values):
    """Hand a result back in the form the rest of the package expects."""
    return torch.from_numpy(np.ascontiguousarray(values))


def _as_mask_array(mask):
    """A contiguous ``unsigned char`` mask for the shim."""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(mask, dtype=bool), dtype=np.uint8)


def _in(array):
    """A ``const double *`` view of ``array``.

    The caller must keep ``array`` alive for the duration of the call: cffi
    does not take a reference to the buffer it points into.
    """
    return ffi.cast("const double *", ffi.from_buffer(array))


def _out(array):
    return ffi.cast("double *", ffi.from_buffer(array))


def _mask_in(array):
    return ffi.cast("const unsigned char *", ffi.from_buffer(array))


def histogram_range(values, initial=None):
    """Intensity range N3 would choose for ``volume_hist -auto_range``.

    Returns ``(min, max)``.  Note this is *not* simply ``(values.min(),
    values.max())`` -- the legacy scan uses an ``else if`` that lets a sample
    update only one bound per visit.  The difference is invisible except on
    degenerate inputs, but the port reproduces it so the two agree exactly.

    ``initial`` is the ``(min, max)`` the scan starts from.  ``volume_hist``
    seeds it with the range of the *whole* volume and then scans only the
    voxels selected by the mask, so masked calls must pass that in; the
    default reproduces the unmasked case.
    """
    values = _as_double_array(values).ravel()
    if initial is None:
        initial = (values.max(), values.min())
    out = np.zeros(2, dtype=np.float64)
    if lib.n3_histogram_range(_in(values), values.size,
                              float(initial[0]), float(initial[1]),
                              _out(out)) != 0:
        raise ValueError("histogram_range: empty input")
    return float(out[0]), float(out[1])


def histogram(values, bins, value_range, parzen=True):
    """Histogram of ``values`` with ``bins`` bins spanning ``value_range``.

    Bin *centres* are evenly spaced from ``value_range[0]`` to
    ``value_range[1]``; the outermost half-bins are open, so samples beyond
    them are dropped.  With ``parzen=True`` each sample is split linearly
    between its two neighbouring centres (N3's ``-parzen`` / ``-window``),
    which is why the counts are floats rather than integers.
    """
    values = _as_double_array(values).ravel()
    counts = np.zeros(int(bins), dtype=np.float64)
    lo, hi = float(value_range[0]), float(value_range[1])
    if lib.n3_histogram(_in(values), values.size, int(bins), lo, hi,
                        1 if parzen else 0, _out(counts)) != 0:
        raise ValueError("histogram: bad arguments")
    return _as_tensor(counts)


def bin_centers(bins, value_range):
    """The intensity each histogram bin is centred on."""
    return _as_tensor(np.linspace(float(value_range[0]), float(value_range[1]),
                                  int(bins)))


def sharpen_lut(counts, value_range, fwhm, noise, deblur=False):
    """N3's sharpened intensity mapping, one value per histogram bin.

    Deconvolves ``counts`` with a Gaussian of width ``fwhm`` using a Wiener
    filter with constant ``noise``, then returns the conditional expectation
    ``E[u | v]`` of the true intensity given the measured one.  Applying this
    as a lookup table is what "sharpens" the histogram.

    ``deblur=True`` skips the deconvolution (N3's ``-blur`` flag).
    """
    counts = _as_double_array(counts).ravel()
    lut = np.zeros(counts.size, dtype=np.float64)
    lo, hi = float(value_range[0]), float(value_range[1])
    if lib.n3_sharpen_lut(_in(counts), counts.size, lo, hi, float(fwhm),
                          float(noise), 1 if deblur else 0, _out(lut)) != 0:
        raise ValueError("sharpen_lut: degenerate histogram")
    return _as_tensor(lut)


def correct_field(field, mask, step):
    """Extend ``field`` from the mask into the rest of the volume.

    The spline that N3 fits is exactly zero outside its domain, so before
    dividing, ``nu_evaluate`` runs ``correct_field``: a multigrid Gauss-Seidel
    solve of Laplace's equation that grows the masked field outward smoothly
    (``torch_n3/_legacy/n3/CorrectField/correctField.cc``).  Returns a new array.
    """
    out = _as_double_array(field)
    flags = _as_mask_array(mask)
    if out.shape != flags.shape:
        raise ValueError("field and mask must have the same shape")

    count = ffi.new("int[3]", [int(n) for n in out.shape])
    steps = ffi.new("double[3]", [float(s) for s in step])
    lib.n3_correct_field(_out(out), _mask_in(flags), count, steps)
    return _as_tensor(out)


class BSplineField:
    """A cubic tensor B-spline fitted to a masked volume.

    Wraps the legacy ``TBSplineVolume``.  ``distance`` is the knot spacing in
    world units (N3's ``-distance``) and ``lam`` the weight on the bending
    energy penalty (``-lambda``); the fit solves

        (AtA + lam * n_samples * J) c = At f

    where ``J`` is the bending energy tensor.  The spline evaluates to exactly
    zero outside its domain.

    The domain is the whole bounding box of the fitting grid, which is what
    ``spline_smooth -full_support`` uses and what ``nu_estimate`` asks for.
    Because it is remembered in *world* coordinates, a spline fitted on the
    coarse estimation grid can be evaluated at full resolution -- exactly the
    round trip the ``.imp`` mapping file performs between ``nu_estimate`` and
    ``nu_evaluate``.
    """

    def __init__(self, grid, distance=200.0, lam=1e-7, domain_world=None,
                 solver="normal"):
        # The oracle has one solver: the normal equations, as TBSpline.cc
        # forms them.  Asking it for the stacked QR would silently get the
        # other answer, so say so instead.
        if solver != "normal":
            raise ValueError(
                "the legacy backend only solves the normal equations; "
                "solver=%r is implemented by the torch backend alone" % solver)
        self.grid = grid
        self.solver = solver
        self.distance = float(distance)
        self.lam = float(lam)
        self.domain_world = (_domain_of(grid) if domain_world is None
                             else tuple(np.asarray(d, float) for d in domain_world))
        self._handle = self._make_handle(grid, allocate=True)
        self._fitted = False

    def _make_handle(self, grid, allocate):
        """Create a ``TBSplineVolume`` over ``grid`` sharing this domain.

        The legacy splines measure position as ``voxel index * step`` with the
        grid's own origin at zero, so the world-space domain has to be
        re-expressed relative to whichever grid is being used.
        """
        lo, hi = (np.asarray(d) - grid.start for d in self.domain_world)
        domain = ffi.new("double[6]", [float(v) for pair in zip(lo, hi)
                                       for v in pair])
        return lib.n3_spline_create_on_domain(
            domain,
            ffi.new("double[3]", [0.0, 0.0, 0.0]),
            ffi.new("double[3]", [float(s) for s in grid.step]),
            ffi.new("int[3]", [int(n) for n in grid.shape]),
            self.distance, self.lam, 1 if allocate else 0)

    def fit(self, values, mask=None, subsample=1):
        """Fit to ``values``; only voxels where ``mask`` is true contribute."""
        values = _as_double_array(values)
        if values.shape != tuple(self.grid.shape):
            raise ValueError("values shape %s != grid shape %s"
                             % (values.shape, tuple(self.grid.shape)))

        if mask is None:
            flags, mask_ptr = None, ffi.NULL
        else:
            flags = _as_mask_array(mask)
            mask_ptr = _mask_in(flags)

        if lib.n3_spline_add_volume(self._handle, _in(values), mask_ptr,
                                    int(subsample)) != 0:
            raise RuntimeError("B-spline: could not add data points")
        if lib.n3_spline_fit(self._handle) != 0:
            raise RuntimeError("B-spline fit failed (singular normal equations)")
        self._fitted = True
        return self

    @property
    def coefficients(self):
        n = lib.n3_spline_n_coefficients(self._handle)
        out = np.zeros(n, dtype=np.float64)
        lib.n3_spline_coefficients(self._handle, _out(out))
        return _as_tensor(out)

    def evaluate(self):
        """Evaluate the fitted spline on the grid it was fitted to."""
        return self.evaluate_on(self.grid)

    def evaluate_on(self, grid):
        """Evaluate the fitted spline on any grid in the same world space."""
        if not self._fitted:
            raise RuntimeError("evaluate() before fit()")

        out = np.zeros(int(np.prod(grid.shape)), dtype=np.float64)
        if grid is self.grid:
            lib.n3_spline_evaluate_grid(self._handle, _out(out))
        else:
            coef = _as_double_array(self.coefficients)
            handle = self._make_handle(grid, allocate=False)
            try:
                lib.n3_spline_set_coefficients(handle, _in(coef), coef.size)
                lib.n3_spline_evaluate_grid(handle, _out(out))
            finally:
                lib.n3_spline_free(handle)
        return _as_tensor(out.reshape(tuple(grid.shape)))

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            lib.n3_spline_free(handle)
            self._handle = None


def _domain_of(grid):
    """The world-space box a ``-full_support`` spline is defined on.

    ``splineSmooth.cc:232`` takes the whole volume, half a voxel beyond the
    outermost voxel centres.
    """
    shape = np.asarray(grid.shape, dtype=np.float64)
    return (grid.start - 0.5 * grid.step,
            grid.start + (shape - 0.5) * grid.step)

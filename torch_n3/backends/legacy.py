"""The *legacy* backend: N3's blocks as implemented by the original C++.

Every function here is a thin, numpy-friendly wrapper around the CFFI shim in
``torch_n3._legacy``.  No mathematics happens in this file -- that is the whole
point.  During Stage 2 these functions are the oracle each PyTorch block is
tested against.

Build the extension first::

    python3 torch_n3/_legacy/build_legacy.py
"""

import numpy as np

from torch_n3._legacy._n3legacy import ffi, lib


def _as_double_array(values):
    """Return a contiguous float64 copy suitable for passing to the shim."""
    return np.ascontiguousarray(values, dtype=np.float64)


def _in(array):
    return ffi.cast("const double *", ffi.from_buffer(array))


def _out(array):
    return ffi.cast("double *", ffi.from_buffer(array))


def histogram_range(values):
    """Intensity range N3 would choose for ``volume_hist -auto_range``.

    Returns ``(min, max)``.  Note this is *not* simply ``(values.min(),
    values.max())`` -- the legacy scan uses an ``else if`` that lets a sample
    update only one bound per visit.  The difference is invisible except on
    degenerate inputs, but the port reproduces it so the two agree exactly.
    """
    values = _as_double_array(values).ravel()
    out = np.zeros(2, dtype=np.float64)
    if lib.n3_histogram_range(_in(values), values.size, _out(out)) != 0:
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
    return counts


def bin_centers(bins, value_range):
    """The intensity each histogram bin is centred on."""
    return np.linspace(float(value_range[0]), float(value_range[1]), int(bins))


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
    return lut


class BSplineField:
    """A cubic tensor B-spline fitted to a masked volume.

    Wraps the legacy ``TBSplineVolume``.  ``distance`` is the knot spacing in
    world units (N3's ``-distance``) and ``lam`` the weight on the bending
    energy penalty (``-lambda``); the fit solves

        (AtA + lam * n_samples * J) c = At f

    where ``J`` is the bending energy tensor.  The spline evaluates to exactly
    zero outside its domain.
    """

    def __init__(self, shape, start=(0.0, 0.0, 0.0), step=(1.0, 1.0, 1.0),
                 distance=200.0, lam=1e-7):
        self.shape = tuple(int(s) for s in shape)
        self._handle = lib.n3_spline_create(
            ffi.new("double[3]", [float(v) for v in start]),
            ffi.new("double[3]", [float(v) for v in step]),
            ffi.new("int[3]", [int(v) for v in self.shape]),
            float(distance), float(lam))
        self._fitted = False

    def fit(self, volume, mask=None):
        """Fit to ``volume``; only voxels where ``mask`` is true contribute."""
        volume = np.ascontiguousarray(volume, dtype=np.float64)
        if volume.shape != self.shape:
            raise ValueError(
                "volume shape %s != spline shape %s" % (volume.shape, self.shape))
        if mask is None:
            mask = np.ones(self.shape, dtype=bool)
        mask = np.ascontiguousarray(mask).astype(bool)

        # Kept as an explicit loop: it mirrors the legacy fitting loop exactly
        # and is only used as the reference implementation.
        add = lib.n3_spline_add
        handle = self._handle
        for x, y, z in zip(*np.nonzero(mask)):
            add(handle, int(x), int(y), int(z), float(volume[x, y, z]))

        if lib.n3_spline_fit(handle) != 0:
            raise RuntimeError("B-spline fit failed (singular normal equations)")
        self._fitted = True
        return self

    @property
    def coefficients(self):
        n = lib.n3_spline_n_coefficients(self._handle)
        out = np.zeros(n, dtype=np.float64)
        lib.n3_spline_coefficients(self._handle, _out(out))
        return out

    def evaluate(self):
        """Evaluate the fitted spline on the whole voxel grid."""
        if not self._fitted:
            raise RuntimeError("evaluate() before fit()")
        out = np.zeros(int(np.prod(self.shape)), dtype=np.float64)
        lib.n3_spline_evaluate_grid(self._handle, _out(out))
        return out.reshape(self.shape)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            lib.n3_spline_free(handle)
            self._handle = None

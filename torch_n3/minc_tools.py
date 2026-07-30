"""The two MINC command-line tools the N3 drivers lean on, as array code.

Everything else N3 does lives in its own C++ (and is wrapped in
:mod:`torch_n3.backends.legacy`); these two are general MINC utilities that
happen to sit on the pipeline's critical path, so they are written out here.
Both are pinned against the installed binaries in ``tests/test_minc_tools.py``.
"""

import numpy as np


def apply_lut(values, lut, value_range):
    """Look ``values`` up in ``lut``, as ``minclookup -continuous`` does.

    ``lut`` holds one output value per entry, evenly spaced across
    ``value_range``; ``minclookup`` normalises each voxel to ``[0, 1]`` over
    the range given by ``-range``, interpolates linearly between the two
    neighbouring table entries, and clamps anything outside.

    This is how the sharpened histogram becomes a sharpened volume: the table
    produced by ``sharpen_hist`` *is* the mapping ``E[u | v]``.
    """
    lut = np.asarray(lut, dtype=np.float64)
    domain = np.linspace(float(value_range[0]), float(value_range[1]), lut.size)
    return np.interp(values, domain, lut)


def bimodal_threshold(values, bins=2000):
    """Split ``values`` into background and foreground, as ``mincstats -biModalT``.

    This is Otsu's method: histogram the data, then choose the boundary that
    maximises the variance *between* the two resulting groups.  The returned
    threshold is the centre of the winning bin, which is what ``mincstats``
    reports and what ``nu_evaluate`` uses when no mask is supplied.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return float(lo)

    width = (hi - lo) / bins
    index = np.clip(((values - lo) / width).astype(int), 0, bins - 1)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    centres = lo + (np.arange(bins) + 0.5) * width

    # Cumulative weight and mean of the group at or below each candidate bin,
    # and of the group above it.
    weight_lo = np.cumsum(counts)
    weight_hi = counts.sum() - weight_lo
    sum_lo = np.cumsum(counts * centres)
    sum_hi = (counts * centres).sum() - sum_lo

    with np.errstate(invalid="ignore", divide="ignore"):
        between = weight_lo * weight_hi * (sum_lo / weight_lo - sum_hi / weight_hi) ** 2

    return float(centres[np.nanargmax(between)])

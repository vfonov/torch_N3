"""The two MINC command-line tools the N3 drivers depend on, as array code.

Everything else N3 does is in its own C++, ported block by block in
:mod:`torch_n3.blocks`.  These two are general MINC utilities that sit on the
pipeline's critical path.  Both are compared against the installed binaries in
``tests/test_minc_tools.py``.
"""

import torch


def apply_lut(values, lut, value_range):
    """Look ``values`` up in ``lut``, as ``minclookup -continuous`` does.

    ``lut`` holds one output value per entry, evenly spaced across
    ``value_range``; ``minclookup`` normalises each voxel to ``[0, 1]`` over
    the range given by ``-range``, interpolates linearly between the two
    neighbouring table entries, and clamps anything outside.

    This is how the sharpened histogram becomes a sharpened volume: the table
    produced by ``sharpen_hist`` is the mapping ``E[u | v]``.
    """
    values = torch.as_tensor(values, dtype=torch.float64)
    lut = torch.as_tensor(lut, dtype=torch.float64).reshape(-1)
    entries = lut.numel()
    if entries == 1:
        return torch.full_like(values, float(lut[0]))

    low, high = float(value_range[0]), float(value_range[1])
    if high == low:
        raise ValueError("apply_lut: empty range")

    # Interpolate between the two table entries that bracket each value, using
    # the entries' own positions -- which is the search minclookup performs,
    # and is not quite the same in floating point as scaling the value into
    # units of table entries.
    domain = torch.linspace(low, high, entries, dtype=torch.float64,
                            device=values.device)
    left = (torch.searchsorted(domain, values.contiguous(), right=True) - 1)
    left = left.clamp(0, entries - 2)
    slope = (lut[left + 1] - lut[left]) / (domain[left + 1] - domain[left])

    mapped = lut[left] + slope * (values - domain[left])
    mapped = torch.where(values <= low, lut[0], mapped)
    return torch.where(values >= high, lut[-1], mapped)


def bimodal_threshold(values, bins=2000):
    """Split ``values`` into background and foreground, as ``mincstats -biModalT``.

    Otsu's method: histogram the data, then choose the boundary that maximises
    the variance *between* the two resulting groups.  The returned threshold is
    the centre of the winning bin, which is what ``mincstats`` reports and what
    ``nu_evaluate`` uses when no mask is supplied.
    """
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return low

    width = (high - low) / bins
    index = ((values - low) / width).long().clamp(0, bins - 1)
    counts = torch.bincount(index, minlength=bins).to(torch.float64)
    centres = low + (torch.arange(bins, dtype=torch.float64,
                                  device=values.device) + 0.5) * width

    # Cumulative weight and mean of the group at or below each candidate bin,
    # and of the group above it.
    weight_low = counts.cumsum(0)
    weight_high = counts.sum() - weight_low
    sum_low = (counts * centres).cumsum(0)
    sum_high = (counts * centres).sum() - sum_low

    between = (weight_low * weight_high
               * (sum_low / weight_low - sum_high / weight_high) ** 2)
    between = torch.where(torch.isnan(between),
                          torch.full_like(between, -float("inf")), between)
    return float(centres[int(between.argmax())])

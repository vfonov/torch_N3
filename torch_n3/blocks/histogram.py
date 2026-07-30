"""The masked intensity histogram (``volume_hist``).

N3 measures the intensity distribution of the volume inside the mask and works
on that; ``sharpen_hist`` never sees the volume itself.  Two details of the
legacy histogram matter enough to be reproduced exactly:

* the bin *centres* -- not the bin edges -- span the requested range, so a
  histogram of ``n`` bins over ``[lo, hi]`` has bin width ``(hi - lo)/(n - 1)``
  and reaches half a bin beyond either end;
* ``-parzen`` splits each sample linearly between its two neighbouring centres
  rather than dropping it in one bin, which is why the counts are floats.

Ported from ``legacy/N3/src/VolumeHist/{DHistogram.cc,WHistogram.h}``.
"""

import torch


def histogram_range(values, initial=None):
    """The range ``volume_hist -auto_range`` would choose for ``values``.

    Not simply ``(min, max)``.  The legacy scan is

    .. code-block:: c

        if (v < lo) lo = v; else if (v > hi) hi = v;

    started from the range of the *whole* volume with the bounds **crossed**
    (``lo`` = the volume maximum, ``hi`` = its minimum) and then run over the
    masked voxels only, so the ``else`` genuinely bites: until some sample
    fails to be a new minimum, no sample can raise ``hi``.

    That makes the scan order-dependent, so it is reproduced here rather than
    replaced by ``min``/``max``.  Writing ``j`` for the first sample that is
    not a new running minimum, the scan is equivalent to taking the minimum
    over everything and the maximum over the tail from ``j`` -- which
    vectorises as a cumulative minimum.

    ``initial`` is the ``(lo, hi)`` the scan starts from; the default
    reproduces an unmasked call.
    """
    values = torch.as_tensor(values).reshape(-1)
    if values.numel() == 0:
        raise ValueError("histogram_range: empty input")

    if initial is None:
        low, high = values.max(), values.min()
    else:
        low = values.new_tensor(float(initial[0]))
        high = values.new_tensor(float(initial[1]))

    # The running minimum as it stands *before* each sample is visited.
    before = torch.cummin(torch.cat([low.reshape(1), values[:-1]]), 0).values
    raises_high = (values >= before) & (values > high)

    low = torch.minimum(low, values.min())
    first = torch.nonzero(raises_high)
    if first.numel():
        high = values[int(first[0, 0]):].max()

    low, high = float(low), float(high)
    if high <= low:
        high = low + 1.0
    return low, high


def histogram(values, bins, value_range, parzen=True):
    """Histogram of ``values`` with ``bins`` bins centred across ``value_range``.

    Samples beyond the outermost bin *centres* are discarded.  With
    ``parzen=True`` -- what ``nu_correct`` uses -- each sample contributes
    ``1 - offset`` to the nearer centre and ``offset`` to the next one, so a
    voxel sitting between two bins is shared between them instead of being
    rounded into one.
    """
    values = torch.as_tensor(values).reshape(-1)
    bins = int(bins)
    if bins <= 0:
        raise ValueError("histogram: need at least one bin")

    low, high = sorted((float(value_range[0]), float(value_range[1])))
    width = _bin_width(low, high, bins)
    edge = low - 0.5 * width  # the outer edge of the first bin

    counts = torch.zeros(bins, dtype=values.dtype, device=values.device)
    if parzen:
        _add_split(counts, values, low, high, edge, width, bins)
    else:
        _add_whole(counts, values, edge, width, bins)
    return counts


def bin_centers(bins, value_range):
    """The intensity each histogram bin is centred on."""
    return torch.linspace(float(value_range[0]), float(value_range[1]),
                          int(bins), dtype=torch.float64)


def _add_split(counts, values, low, high, edge, width, bins):
    """``WHistogram::add``: share each sample between two bin centres."""
    values = values[(values >= low) & (values <= high)]

    location = (values - edge) / width
    index = location.floor().long().clamp(0, bins - 1)
    offset = location - index - 0.5

    # The legacy tries three cases in this order; they are disjoint.
    exact = offset == 0
    upper = (offset > 0) & (index <= bins - 2)
    lower = ~exact & ~upper & (index >= 1)

    ones = torch.ones_like(offset)
    counts.index_add_(0, index[exact], ones[exact])
    counts.index_add_(0, index[upper], 1.0 - offset[upper])
    counts.index_add_(0, index[upper] + 1, offset[upper])
    counts.index_add_(0, index[lower], 1.0 + offset[lower])
    counts.index_add_(0, index[lower] - 1, -offset[lower])


def _add_whole(counts, values, edge, width, bins):
    """``DHistogram::add``: one sample, one bin.

    This variant accepts samples out to the bin *edges*, half a bin further
    than the Parzen one does.
    """
    values = values[(values >= edge) & (values <= edge + width * bins)]
    index = ((values - edge) / width).long().clamp(0, bins - 1)
    counts.index_add_(0, index, torch.ones_like(values))


def _bin_width(low, high, bins):
    """``DHistogram``'s width, including its two degenerate cases."""
    if high == low:
        return 1.0 / bins
    if bins <= 1:
        return high - low
    return (high - low) / (bins - 1)

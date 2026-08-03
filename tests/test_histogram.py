"""Stage 2: :mod:`torch_n3.blocks.histogram` against ``volume_hist``.

Two oracles, in order of strictness.  The CFFI shim runs the legacy
``WHistogram`` class itself, so the port must match it to the last bit
summation order allows; ``volume_hist``'s recorded output is the end-to-end
check that the shim is being asked the right question.
"""

import math

import pytest
import torch

from tests.conftest import assert_close
from tests.inputs import masked_log
from torch_n3 import blocks
from torch_n3.backends import legacy


def test_range_matches_the_legacy_scan(chunk, chunk_mask):
    """The ``-auto_range`` scan, crossed initial bounds and all."""
    values, inside = masked_log(chunk, chunk_mask)
    initial = (values.max(), values.min())

    assert (blocks.histogram_range(values[inside], initial=initial)
            == legacy.histogram_range(values[inside], initial=initial))


def test_range_reproduces_the_else_if_on_a_decreasing_run():
    """A strictly decreasing sample never gets to raise the upper bound.

    The legacy's ``else if`` means a value that is a new *minimum* cannot also
    be a new maximum, so a monotonically falling sequence leaves the upper bound
    where it started: the one case where this is not simply ``(min, max)``.  The
    two implementations must agree on it.
    """
    falling = torch.arange(10.0, 0.0, -1.0, dtype=torch.float64)
    initial = (100.0, -100.0)

    assert (blocks.histogram_range(falling, initial=initial)
            == legacy.histogram_range(falling, initial=initial))


@pytest.mark.parametrize("parzen", [True, False])
def test_counts_match_the_legacy_histogram(chunk, chunk_mask, parzen):
    values, inside = masked_log(chunk, chunk_mask)
    value_range = legacy.histogram_range(values[inside],
                                         initial=(values.max(), values.min()))

    port = blocks.histogram(values[inside], 200, value_range, parzen)
    oracle = legacy.histogram(values[inside], 200, value_range, parzen)

    # Both add one sample at a time into the same bins; only the order of the
    # additions differs, and 132k of them accumulate about that much rounding.
    assert_close(port, oracle, atol=1e-9)


def test_counts_match_the_volume_hist_binary(legacy_output, chunk, chunk_mask):
    """`nu_volume_hist_1`, with the ``-window`` (Parzen) variant N3 uses."""
    recorded = legacy_output["volume_hist.chunk"]

    inside = chunk_mask.data != 0
    selected = chunk.data[inside]
    value_range = blocks.histogram_range(
        selected, initial=(chunk.data.max(), chunk.data.min()))
    counts = blocks.histogram(selected, 200, value_range)

    # volume_hist writes both columns with "%lf", so six decimals is the last
    # digit it reports and there is nothing finer to agree to.
    assert_close(blocks.bin_centers(200, value_range), recorded[:, 0], atol=1e-6)
    assert_close(counts, recorded[:, 1], atol=1e-6)
    assert float(counts.sum()) == pytest.approx(float(recorded[:, 1].sum()),
                                                rel=1e-4)


def test_parzen_splits_a_sample_between_two_bins():
    """One sample halfway between two centres puts half its weight in each."""
    counts = blocks.histogram(torch.tensor([0.5]), 3, (0.0, 2.0), parzen=True)

    assert_close(counts, [0.5, 0.5, 0.0], atol=1e-12)


def test_samples_beyond_the_outer_centres_are_dropped():
    """The first and last bins are half-open, as ``WHistogram::add`` has them."""
    values = torch.tensor([-0.5, 0.0, 1.0, 2.0, 2.5])

    counts = blocks.histogram(values, 3, (0.0, 2.0), parzen=True)

    assert float(counts.sum()) == 3.0


# The Gaussian Parzen window is a modification, not a port, so it has no
# oracle: what follows states the properties it is supposed to have.  What it
# does to the pipeline is measured by ``python3 -m tests.parzen``.


def test_gaussian_window_weights_the_bins_by_the_kernel():
    """A sample on a centre spreads symmetrically, in the kernel's own ratios."""
    counts = blocks.histogram(torch.tensor([5.0], dtype=torch.float64), 11,
                              (0.0, 10.0), sigma=1.0)

    assert_close(counts[4], counts[6], atol=1e-15)
    # exp(-1/2) between neighbours one bin apart, exp(-2) two bins out.
    assert float(counts[4] / counts[5]) == pytest.approx(math.exp(-0.5))
    assert float(counts[3] / counts[5]) == pytest.approx(math.exp(-2.0))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 3.0])
def test_gaussian_window_conserves_the_sample_count(chunk, chunk_mask, sigma):
    """Every retained sample contributes exactly 1, as the linear split does.

    The kernel is renormalised per sample rather than truncated, so a voxel
    near the end of the range does not count for less than one.
    """
    values, inside = masked_log(chunk, chunk_mask)
    value_range = blocks.histogram_range(values[inside],
                                         initial=(values.max(), values.min()))

    linear = blocks.histogram(values[inside], 200, value_range)
    gaussian = blocks.histogram(values[inside], 200, value_range, sigma=sigma)

    assert float(gaussian.sum()) == pytest.approx(float(linear.sum()), rel=1e-12)


def test_gaussian_window_smooths_more_than_the_linear_split(chunk, chunk_mask):
    """The window's purpose: a wider kernel, so a smoother histogram."""
    values, inside = masked_log(chunk, chunk_mask)
    value_range = blocks.histogram_range(values[inside],
                                         initial=(values.max(), values.min()))

    def roughness(counts):
        return float(counts.diff().abs().sum())

    linear = blocks.histogram(values[inside], 200, value_range)
    narrow = blocks.histogram(values[inside], 200, value_range, sigma=0.5)
    wide = blocks.histogram(values[inside], 200, value_range, sigma=2.0)

    assert roughness(wide) < roughness(narrow) < roughness(linear)


def test_a_narrow_gaussian_window_becomes_plain_binning():
    """As the kernel shrinks below a bin it degenerates to nearest-centre.

    Which is ``parzen=False``: a check that the kernel is centred on the sample
    and evaluated at the bin centres rather than offset by half a bin.
    """
    values = torch.tensor([1.2, 3.7, 5.4, 8.9], dtype=torch.float64)

    counts = blocks.histogram(values, 11, (0.0, 10.0), sigma=1e-3)

    assert_close(counts, blocks.histogram(values, 11, (0.0, 10.0),
                                          parzen=False), atol=1e-12)


def test_a_sample_exactly_between_two_centres_is_halved():
    """The one place the narrow limit is not ``parzen=False``.

    Both bins are the same distance away, so the kernel gives them equal weight
    however narrow it is, where plain binning rounds the tie one way.  Half each
    is also the linear split's answer.
    """
    counts = blocks.histogram(torch.tensor([5.5], dtype=torch.float64), 11,
                              (0.0, 10.0), sigma=1e-3)

    assert_close(counts[5:7], [0.5, 0.5], atol=1e-12)


def test_the_window_needs_the_window_flag():
    with pytest.raises(ValueError, match="no window without"):
        blocks.histogram(torch.tensor([0.5]), 3, (0.0, 2.0), parzen=False,
                         sigma=1.0)


def test_the_legacy_backend_has_no_gaussian_window():
    """The oracle is N3, and N3 does not have this."""
    with pytest.raises(ValueError, match="torch backend"):
        legacy.histogram(torch.tensor([0.5]), 3, (0.0, 2.0), sigma=1.0)

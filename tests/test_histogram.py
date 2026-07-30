"""Stage 2: :mod:`torch_n3.blocks.histogram` against ``volume_hist``.

Two oracles, in order of strictness.  The CFFI shim runs the legacy
``WHistogram`` class itself, so the port has to match it to the last bit that
summation order allows; the installed ``volume_hist`` binary is the end-to-end
check that the shim is being asked the right question in the first place.
"""

import numpy as np
import pytest
import torch

from tests.conftest import assert_close
from torch_n3 import blocks
from torch_n3.backends import legacy


def masked_log(volume, mask):
    """What the pipeline actually histograms: the masked log volume."""
    inside = mask.data != 0
    values = torch.log(volume.data.clamp(min=1.0))
    return torch.where(inside, values, torch.zeros_like(values)), inside


def test_range_matches_the_legacy_scan(chunk, chunk_mask):
    """The ``-auto_range`` scan, crossed initial bounds and all."""
    values, inside = masked_log(chunk, chunk_mask)
    initial = (values.max(), values.min())

    assert (blocks.histogram_range(values[inside], initial=initial)
            == legacy.histogram_range(values[inside], initial=initial))


def test_range_reproduces_the_else_if_on_a_decreasing_run():
    """A strictly decreasing sample never gets to raise the upper bound.

    The legacy's ``else if`` means a value that is a new *minimum* cannot also
    be a new maximum, so a monotonically falling sequence leaves the upper
    bound at where it started -- the one case where this is not just
    ``(min, max)``.  It is a quirk, not a feature, but the two implementations
    have to agree on it.
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

    ours = blocks.histogram(values[inside], 200, value_range, parzen)
    theirs = legacy.histogram(values[inside], 200, value_range, parzen)

    # Both add one sample at a time into the same bins; only the order of the
    # additions differs, and 130k of them accumulate about that much rounding.
    assert_close(ours, theirs, atol=1e-9)


def test_counts_match_the_volume_hist_binary(workspace, chunk, chunk_mask):
    """`nu_volume_hist_1`, with the ``-window`` (Parzen) variant N3 uses."""
    source = workspace.write("chunk.mnc", chunk)
    mask = workspace.write("mask.mnc", chunk_mask, store_dtype="int16")
    workspace.run("volume_hist", "-bins", 200, "-auto_range", "-mask", mask,
                  "-clobber", "-text", "-select", 1, "-quiet", "-window",
                  source, workspace.at("hist.txt"))
    reference = np.loadtxt(workspace.at("hist.txt"))

    inside = chunk_mask.data != 0
    selected = chunk.data[inside]
    value_range = blocks.histogram_range(
        selected, initial=(chunk.data.max(), chunk.data.min()))
    counts = blocks.histogram(selected, 200, value_range)

    assert_close(blocks.bin_centers(200, value_range), reference[:, 0],
                 atol=1e-6)
    # volume_hist reads the volume through volume_io, which rescales it onto a
    # single grid for the whole file, so a voxel near a bin edge can land on
    # the other side of it; the counts either side then differ by that voxel.
    assert_close(counts, reference[:, 1], atol=2.0)
    assert float(counts.sum()) == pytest.approx(reference[:, 1].sum(), rel=1e-4)


def test_parzen_splits_a_sample_between_two_bins():
    """One sample halfway between two centres puts half its weight in each."""
    counts = blocks.histogram(torch.tensor([0.5]), 3, (0.0, 2.0), parzen=True)

    assert_close(counts, [0.5, 0.5, 0.0], atol=1e-12)


def test_samples_beyond_the_outer_centres_are_dropped():
    """The first and last bins are half-open, as ``WHistogram::add`` has them."""
    values = torch.tensor([-0.5, 0.0, 1.0, 2.0, 2.5])

    counts = blocks.histogram(values, 3, (0.0, 2.0), parzen=True)

    assert float(counts.sum()) == 3.0

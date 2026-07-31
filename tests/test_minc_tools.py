"""`torch_n3.minc_tools` against the MINC programs it stands in for.

Both comparisons use answers recorded from the real programs; see
:mod:`tests.reference`.
"""

import torch

from tests.conftest import assert_close
from torch_n3.minc_tools import apply_lut, bimodal_threshold


def test_apply_lut_matches_minclookup(legacy_output):
    """Our continuous lookup is `minclookup -continuous`.

    The table and the intensities are the ones the program was given -- a few
    thousand real voxels out of the masked log volume, spread across the
    domain -- so this checks the interpolation rule and nothing else.
    """
    values = legacy_output["minclookup.values"]
    table = legacy_output["minclookup.lut"]
    value_range = tuple(float(v) for v in legacy_output["minclookup.range"])

    mapped = apply_lut(values, table, value_range)

    assert_close(mapped, legacy_output["minclookup.mapped"], atol=1e-9)


def test_apply_lut_clamps_outside_the_domain():
    """Values past either end map to the end of the table, as minclookup does."""
    lut = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    values = torch.tensor([-5.0, 0.0, 0.5, 1.0, 7.0], dtype=torch.float64)

    mapped = apply_lut(values, lut, (0.0, 1.0))

    assert_close(mapped, [10.0, 10.0, 20.0, 30.0, 30.0], atol=1e-12)


def test_bimodal_threshold_matches_mincstats(legacy_output, chunk):
    """`nu_evaluate`'s automatic mask comes from `mincstats -biModalT`.

    The bound is the last digit ``mincstats`` printed: it reports four
    decimals, so ``1e-4`` is the finest agreement it can be asked for.  (This
    was a *relative* ``1e-3`` until 2026-07-31, which on a value in the
    hundreds of thousands allowed a difference of 238 -- some two million
    times looser than the number it was comparing against, and enough for the
    threshold to land in a different tissue.)
    """
    recorded = legacy_output.scalar("mincstats.bimodal_threshold_chunk")

    threshold = bimodal_threshold(chunk.data)

    assert abs(threshold - recorded) < 1e-4


def test_bimodal_threshold_separates_two_clusters():
    """The threshold falls in the gap, so it classifies both groups correctly."""
    generator = torch.Generator().manual_seed(7)
    background = 10.0 + torch.randn(5000, dtype=torch.float64,
                                    generator=generator)
    foreground = 50.0 + 3.0 * torch.randn(5000, dtype=torch.float64,
                                          generator=generator)

    threshold = bimodal_threshold(torch.cat([background, foreground]))

    assert float((background < threshold).to(torch.float64).mean()) > 0.99
    assert float((foreground > threshold).to(torch.float64).mean()) > 0.99

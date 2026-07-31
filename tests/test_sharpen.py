"""Stage 2: :mod:`torch_n3.blocks.sharpen` against ``sharpen_hist``.

The deconvolution is the one place where N3 does something a reader cannot
check by eye, so it gets both a parity test against the original code and a
couple of tests that say what the mapping is *for*.
"""

import pytest
import torch

from tests.conftest import assert_close
from tests.inputs import two_tissue_histogram
from torch_n3 import blocks
from torch_n3.backends import legacy


#: The domain the synthetic histogram is defined over.
VALUE_RANGE = (4.0, 6.0)


@pytest.fixture(scope="module")
def two_tissues():
    """A histogram of two overlapping tissue peaks, in log intensity."""
    return two_tissue_histogram(200, VALUE_RANGE)


@pytest.mark.parametrize("deblur", [False, True])
def test_matches_the_legacy_deconvolution(two_tissues, deblur):
    value_range = VALUE_RANGE

    ours = blocks.sharpen_lut(two_tissues, value_range, 0.15, 0.01, deblur)
    theirs = legacy.sharpen_lut(two_tissues, value_range, 0.15, 0.01, deblur)

    assert_close(ours, theirs, atol=1e-11)


def test_matches_the_legacy_on_a_real_histogram(chunk, chunk_mask):
    """The same, on the histogram the pipeline actually produces."""
    inside = chunk_mask.data != 0
    values = torch.log(chunk.data.clamp(min=1.0))
    values = torch.where(inside, values, torch.zeros_like(values))

    value_range = blocks.histogram_range(values[inside],
                                         initial=(values.max(), values.min()))
    counts = blocks.histogram(values[inside], 200, value_range)

    assert_close(blocks.sharpen_lut(counts, value_range, 0.15, 0.01),
                 legacy.sharpen_lut(counts, value_range, 0.15, 0.01),
                 atol=1e-9)


def test_matches_the_sharpen_hist_binary(legacy_output, two_tissues):
    """What the installed program answered on this histogram."""
    lut = blocks.sharpen_lut(two_tissues, VALUE_RANGE, 0.15, 0.01)

    # sharpen_hist writes "%lf", so six decimals is all it reports.
    assert_close(lut, legacy_output["sharpen_hist.two_tissues_lut"], atol=1e-6)


def test_the_mapping_pulls_intensities_towards_the_tissue_peaks(two_tissues):
    """What "sharpening" means: the mapped range is narrower than the input.

    Each intensity is replaced by the mean of the deconvolved distribution
    near it, so voxels between the two peaks are pulled onto one of them and
    the histogram of the result is more sharply peaked than the one measured.
    """
    value_range = VALUE_RANGE
    lut = blocks.sharpen_lut(two_tissues, value_range, 0.15, 0.01)
    centres = blocks.bin_centers(two_tissues.numel(), value_range)

    populated = two_tissues > 0.01 * two_tissues.max()
    assert torch.all(torch.isfinite(lut))
    assert (lut[populated].max() - lut[populated].min()) < (
        centres[populated].max() - centres[populated].min())


def test_deblur_leaves_the_histogram_alone(two_tissues):
    """``-blur`` skips the deconvolution, so the mapping only smooths."""
    value_range = VALUE_RANGE
    sharpened = blocks.sharpen_lut(two_tissues, value_range, 0.15, 0.01)
    blurred = blocks.sharpen_lut(two_tissues, value_range, 0.15, 0.01,
                                 deblur=True)
    centres = blocks.bin_centers(two_tissues.numel(), value_range)

    populated = two_tissues > 0.01 * two_tissues.max()
    def moved(lut):
        return float((lut - centres)[populated].abs().mean())

    assert moved(sharpened) > moved(blurred)

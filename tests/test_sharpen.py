"""Stage 2: :mod:`torch_n3.blocks.sharpen` against ``sharpen_hist``.

The deconvolution is the one place where N3 does something a reader cannot
check by eye, so it gets both a parity test against the original code and a
couple of tests that say what the mapping is *for*.
"""

import numpy as np
import pytest
import torch

from tests.conftest import assert_close
from torch_n3 import blocks
from torch_n3.backends import legacy


@pytest.fixture(scope="module")
def two_tissues():
    """A histogram of two overlapping tissue peaks, in log intensity."""
    centres = torch.linspace(4.0, 6.0, 200, dtype=torch.float64)
    return sum(torch.exp(-0.5 * ((centres - peak) / 0.25) ** 2) * height
               for peak, height in [(4.6, 3000.0), (5.4, 5000.0)])


@pytest.mark.parametrize("deblur", [False, True])
def test_matches_the_legacy_deconvolution(two_tissues, deblur):
    value_range = (4.0, 6.0)

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


def test_matches_the_sharpen_hist_binary(workspace, two_tissues):
    """The installed program, driven through its text-file interface."""
    bins, value_range = two_tissues.numel(), (4.0, 6.0)
    centres = blocks.bin_centers(bins, value_range)

    with open(workspace.at("hist.txt"), "w") as fp:
        fp.write("#  domain: %.15g  %.15g\n" % value_range)
        for centre, count in zip(centres, two_tissues):
            fp.write("  %.15g       %.15g\n" % (centre, count))
    workspace.run("sharpen_hist", "-clobber", "-fwhm", 0.15, "-noise", 0.01,
                  "-quiet", workspace.at("hist.txt"), workspace.at("hist.sharp"))
    reference = np.loadtxt(workspace.at("hist.sharp"))[:, 1]

    lut = blocks.sharpen_lut(two_tissues, value_range, 0.15, 0.01)

    # sharpen_hist writes "%lf", so six decimals is all it reports.
    assert_close(lut, reference, atol=1e-6)


def test_the_mapping_pulls_intensities_towards_the_tissue_peaks(two_tissues):
    """What "sharpening" means: the mapped range is narrower than the input.

    Each intensity is replaced by the mean of the deconvolved distribution
    near it, so voxels between the two peaks are pulled onto one of them and
    the histogram of the result is more sharply peaked than the one measured.
    """
    value_range = (4.0, 6.0)
    lut = blocks.sharpen_lut(two_tissues, value_range, 0.15, 0.01)
    centres = blocks.bin_centers(two_tissues.numel(), value_range)

    populated = two_tissues > 0.01 * two_tissues.max()
    assert torch.all(torch.isfinite(lut))
    assert (lut[populated].max() - lut[populated].min()) < (
        centres[populated].max() - centres[populated].min())


def test_deblur_leaves_the_histogram_alone(two_tissues):
    """``-blur`` skips the deconvolution, so the mapping only smooths."""
    value_range = (4.0, 6.0)
    sharpened = blocks.sharpen_lut(two_tissues, value_range, 0.15, 0.01)
    blurred = blocks.sharpen_lut(two_tissues, value_range, 0.15, 0.01,
                                 deblur=True)
    centres = blocks.bin_centers(two_tissues.numel(), value_range)

    populated = two_tissues > 0.01 * two_tissues.max()
    def moved(lut):
        return float((lut - centres)[populated].abs().mean())

    assert moved(sharpened) > moved(blurred)

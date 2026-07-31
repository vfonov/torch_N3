"""Stage 1: the legacy backend really is the legacy algorithm.

These tests pin the CFFI shim against the *installed* N3 programs, so that the
Stage 2 tests can trust it as an oracle.  If the shim ever drifts from what
`sharpen_hist` and friends actually do, these fail rather than the PyTorch
tests, which keeps the blame in the right place.
"""

import pytest
import torch

from tests.conftest import assert_close
from torch_n3.backends import legacy
from torch_n3.volume import Volume


def unit_grid(shape):
    """A 1 mm isotropic grid at the origin -- geometry only, no data."""
    return Volume(torch.zeros(shape), start=(0.0, 0.0, 0.0), step=(1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def gaussian_mixture():
    """A bimodal 1-D sample, standing in for the intensities of a masked brain."""
    generator = torch.Generator().manual_seed(20240730)
    return torch.cat([
        2.0 + 0.35 * torch.randn(4000, dtype=torch.float64, generator=generator),
        4.0 + 0.50 * torch.randn(6000, dtype=torch.float64, generator=generator)])


def test_histogram_counts_are_conserved(gaussian_mixture):
    value_range = legacy.histogram_range(gaussian_mixture)
    counts = legacy.histogram(gaussian_mixture, 200, value_range, parzen=True)

    # Every sample lands in the histogram except those in the outer half-bins.
    assert float(counts.sum()) == pytest.approx(gaussian_mixture.numel(),
                                                rel=1e-3)
    assert float(counts.min()) >= 0.0


def test_parzen_smooths_relative_to_plain_binning(gaussian_mixture):
    value_range = legacy.histogram_range(gaussian_mixture)
    plain = legacy.histogram(gaussian_mixture, 200, value_range, parzen=False)
    parzen = legacy.histogram(gaussian_mixture, 200, value_range, parzen=True)

    assert float(plain.sum()) == pytest.approx(float(parzen.sum()), rel=1e-3)
    # Splitting each sample between two bins cannot increase roughness.
    assert float(parzen.diff().abs().sum()) < float(plain.diff().abs().sum())


def test_sharpen_lut_matches_the_installed_sharpen_hist(legacy_output):
    """The shim and the real `sharpen_hist` binary must agree.

    On the histogram the binary was handed, which is recorded next to its
    answer -- so this compares the deconvolution and nothing else.
    """
    counts = legacy_output["sharpen_hist.chunk_counts"]
    value_range = tuple(float(v) for v in legacy_output["sharpen_hist.chunk_range"])

    from_shim = legacy.sharpen_lut(counts, value_range, fwhm=0.15, noise=0.01)

    # sharpen_hist writes its lookup table with "%lf", i.e. six decimals, so
    # agreement to ~1e-6 is agreement to the last digit the binary reports.
    assert_close(from_shim, legacy_output["sharpen_hist.chunk_lut"], atol=1e-6)


def test_bspline_reproduces_a_linear_ramp():
    """A plane lies in the span of the B-spline basis, so it must fit exactly."""
    shape = (12, 10, 8)
    x, y, z = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    ramp = 1.0 + 0.5 * x - 0.25 * y + 0.125 * z

    spline = legacy.BSplineField(unit_grid(shape), distance=4.0, lam=1e-7)

    assert_close(spline.fit(ramp).evaluate(), ramp, atol=1e-6)


def test_bspline_smooths_away_noise():
    """With knots far apart, the fit keeps the trend and drops the noise."""
    shape = (16, 16, 16)
    x, _, _ = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    trend = 0.1 * x
    torch.manual_seed(0)
    noisy = trend + 0.5 * torch.randn(shape, dtype=torch.float64)

    fitted = legacy.BSplineField(unit_grid(shape), distance=12.0,
                                 lam=1e-7).fit(noisy).evaluate()

    assert float((fitted - trend).abs().max()) < float(
        (noisy - trend).abs().max()) / 3


def test_legacy_test_volumes_are_available(chunk, chunk_mask, brain,
                                           brain_reference):
    """The end-to-end comparisons need these fixtures."""
    assert chunk.shape == chunk_mask.shape
    assert brain.shape == brain_reference.shape

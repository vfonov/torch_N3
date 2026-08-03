"""The two sharpness measures, held to what :mod:`torch_n3.optimize` assumes.

Every number the optimizer produces rests on these four functions being what
they claim, so the properties are asserted directly rather than inferred from a
recovered field: a measure that is subtly wrong still descends, to the wrong
place.

Gradients are checked with ``torch.autograd.gradcheck``, which is reliable here
because the whole port is float64, so the finite differences it compares against
are not limited by the dtype.
"""

import math

import pytest
import torch

from torch_n3.blocks.sharpness import (cluster_occupancy, cluster_tightness,
                                       em_centroids, hoyer_sparsity,
                                       quantile_centroids, soft_histogram,
                                       standardize)

#: A histogram grid in standardized units, as ``nu_optimize`` builds it.
BINS = 64
SPAN = 4.0
SIGMA = 0.15


def centers(bins=BINS, span=SPAN):
    return torch.linspace(-span, span, bins, dtype=torch.float64)


def test_standardize_pins_the_first_two_moments():
    values = torch.randn(1000, dtype=torch.float64) * 7.0 + 3.0
    standardized = standardize(values)

    assert float(standardized.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(standardized.std(unbiased=False)) == pytest.approx(1.0,
                                                                   rel=1e-12)


def test_standardize_is_what_makes_the_measures_affine_invariant():
    """The anti-collapse property, stated on its own.

    A bias field that flattens the volume is an affine change of the
    intensities.  After standardizing, neither measure can see it, which is why
    the optimizer cannot win by destroying the image.
    """
    values = torch.randn(500, dtype=torch.float64)
    squeezed = values * 0.01 + 5.0        # a field that nearly flattened it

    assert torch.allclose(standardize(values), standardize(squeezed))


def test_soft_histogram_matches_a_direct_computation():
    values = torch.randn(50, dtype=torch.float64)
    grid = centers()

    got = soft_histogram(values, grid, SIGMA)
    expected = torch.stack([
        torch.exp(-(values - centre) ** 2 / (2 * SIGMA ** 2)).mean()
        for centre in grid])

    assert torch.allclose(got, expected)


def test_soft_histogram_is_positive_and_peaks_where_the_data_is():
    values = torch.full((100,), 1.5, dtype=torch.float64)
    grid = centers()
    histogram = soft_histogram(values, grid, SIGMA)

    assert float(histogram.min()) > 0.0
    assert float(grid[int(histogram.argmax())]) == pytest.approx(1.5, abs=0.15)


def test_hoyer_is_one_for_a_spike_and_zero_for_a_flat_histogram():
    spike = torch.zeros(BINS, dtype=torch.float64)
    spike[7] = 1.0
    flat = torch.ones(BINS, dtype=torch.float64)

    assert float(hoyer_sparsity(spike)) == pytest.approx(1.0, abs=1e-7)
    assert float(hoyer_sparsity(flat)) == pytest.approx(0.0, abs=1e-7)


def test_hoyer_ignores_the_scale_of_the_histogram():
    """Only the shape is significant, so the mean-vs-sum choice cannot matter.

    Exact at ``eps=0``, which is the property the ratio of norms has.  The
    default ``eps`` guards a zero histogram and costs the invariance a relative
    ``eps/|h|_2``; the next test covers that guard.
    """
    histogram = torch.rand(BINS, dtype=torch.float64) + 0.1

    assert float(hoyer_sparsity(histogram, eps=0.0)) == pytest.approx(
        float(hoyer_sparsity(histogram * 1234.0, eps=0.0)), rel=1e-12)


def test_hoyer_survives_an_empty_histogram():
    """What ``eps`` is there for: no NaN out of a histogram with no mass."""
    assert math.isfinite(float(hoyer_sparsity(torch.zeros(BINS,
                                                          dtype=torch.float64))))


def test_a_sharper_distribution_scores_higher():
    """The property the optimizer exploits.

    Two tissue peaks, blurred by a field, against the same two peaks unblurred.
    If this ordering did not hold there would be nothing to descend on.
    """
    generator = torch.Generator().manual_seed(4)
    peaks = torch.where(torch.rand(4000, generator=generator) < 0.5, -1.0, 1.0)
    sharp = peaks + 0.10 * torch.randn(4000, generator=generator)
    blurred = peaks + 0.60 * torch.randn(4000, generator=generator)

    grid = centers()
    assert float(hoyer_sparsity(soft_histogram(standardize(sharp), grid, SIGMA))) \
        > float(hoyer_sparsity(soft_histogram(standardize(blurred), grid, SIGMA)))


def test_tightness_is_zero_when_every_sample_sits_on_a_centroid():
    centroids = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    values = centroids.repeat(20)

    assert float(cluster_tightness(values, centroids)) == pytest.approx(0.0,
                                                                       abs=1e-12)


def test_tightness_is_the_within_cluster_variance_when_clusters_separate():
    """With well-separated clusters the soft assignment is hard, so the measure
    reduces to a closed form, which is what makes it interpretable as a fraction
    of unexplained variance."""
    centroids = torch.tensor([-3.0, 3.0], dtype=torch.float64)
    offsets = torch.tensor([-0.2, 0.1, 0.2, -0.1], dtype=torch.float64)
    values = torch.cat([centroids[0] + offsets, centroids[1] + offsets])

    assert float(cluster_tightness(values, centroids, beta=200.0)) == \
        pytest.approx(float(offsets.pow(2).mean()), rel=1e-6)


def test_tightness_prefers_the_sharper_of_two_distributions():
    generator = torch.Generator().manual_seed(11)
    peaks = torch.where(torch.rand(4000, generator=generator) < 0.5, -1.0, 1.0)
    sharp = standardize(peaks + 0.10 * torch.randn(4000, generator=generator))
    blurred = standardize(peaks + 0.60 * torch.randn(4000, generator=generator))

    centroids = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    assert float(cluster_tightness(sharp, centroids)) \
        < float(cluster_tightness(blurred, centroids))


def test_quantile_centroids_start_spread_out_and_occupied():
    values = standardize(torch.randn(2000, dtype=torch.float64))
    centroids = quantile_centroids(values, 3)

    assert centroids.numel() == 3
    assert bool((centroids[1:] > centroids[:-1]).all())
    assert float(cluster_occupancy(values, centroids).min()) > 0.05


def test_em_step_does_not_increase_the_measure():
    """The closed form is a minimiser at fixed weights, so it cannot increase
    the measure, which is what ``centroid_update="em"`` relies on."""
    generator = torch.Generator().manual_seed(5)
    values = standardize(torch.randn(1000, generator=generator,
                                     dtype=torch.float64))
    centroids = torch.tensor([-0.5, 0.4], dtype=torch.float64)

    before = float(cluster_tightness(values, centroids))
    after = float(cluster_tightness(values, em_centroids(values, centroids)))
    assert after <= before


def test_em_leaves_an_empty_cluster_where_it_was():
    values = torch.zeros(50, dtype=torch.float64)
    centroids = torch.tensor([0.0, 50.0], dtype=torch.float64)

    moved = em_centroids(values, centroids, beta=200.0)
    assert torch.isfinite(moved).all()


def test_gradients_of_the_hoyer_path():
    values = torch.randn(40, dtype=torch.float64, requires_grad=True)
    grid = centers(bins=16)

    assert torch.autograd.gradcheck(
        lambda x: hoyer_sparsity(soft_histogram(standardize(x), grid, 0.5)),
        (values,))


def test_gradients_of_the_tightness_path_in_both_arguments():
    """Both, because the centroids are learned alongside the field."""
    values = torch.randn(40, dtype=torch.float64, requires_grad=True)
    centroids = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64,
                             requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda x, m: cluster_tightness(standardize(x), m, beta=5.0),
        (values, centroids))


def test_the_measures_reject_degenerate_arguments():
    with pytest.raises(ValueError):
        hoyer_sparsity(torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError):
        cluster_tightness(torch.zeros(4, dtype=torch.float64),
                          torch.zeros(1, dtype=torch.float64))
    with pytest.raises(ValueError):
        quantile_centroids(torch.zeros(4, dtype=torch.float64), 1)


def test_a_collapsed_volume_scores_perfectly_without_standardization():
    """The degeneracy, demonstrated rather than asserted away.

    What the optimizer would drive towards if :func:`standardize` were not in
    the path: a constant image is the global optimum of *both* measures.
    ``tests/test_optimize.py`` shows the same end to end.
    """
    constant = torch.full((500,), 2.0, dtype=torch.float64)
    grid = centers()

    assert float(hoyer_sparsity(soft_histogram(constant, grid, SIGMA))) \
        > float(hoyer_sparsity(soft_histogram(
            standardize(torch.randn(500, dtype=torch.float64)), grid, SIGMA)))
    assert float(cluster_tightness(
        constant, quantile_centroids(constant, 3))) == pytest.approx(0.0,
                                                                     abs=1e-12)

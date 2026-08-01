"""``nu_optimize``: the descent, the degeneracy it avoids, and its randomness.

The measures themselves are pinned in ``tests/test_sharpness.py``; what is
asserted here is that driving them with a spline field actually estimates a
bias field, that the thing standing between this method and a destroyed image
really is the standardization, and that a stochastic estimator inside this
repo is still reproducible.

``chunk.mnc`` throughout, and small iteration budgets: these are properties of
the method, and the numbers that decide whether it is *better than N3* come
from ``experiments/``, not from here.
"""

import math

import pytest
import torch

from experiments.simulation import random_bias_field
from torch_n3.blocks.sharpness import (hoyer_sparsity, soft_histogram,
                                       standardize)
from torch_n3.blocks.spline import BSplineField
from torch_n3.optimize import DEFAULTS, nu_optimize
from torch_n3.pipeline import nu_evaluate

#: Enough iterations to move, few enough to keep the suite quick.
BUDGET = dict(max_iterations=40, sample_size=8192, distance=75.0)

#: The planted field these tests recover, as a log peak-to-peak amplitude.
AMPLITUDE = 0.2


@pytest.fixture(scope="module")
def inside(chunk, chunk_mask):
    return chunk_mask.resample_like(chunk).data != 0


@pytest.fixture(scope="module")
def planted(chunk, inside):
    return random_bias_field(chunk, inside, AMPLITUDE, seed=1)


@pytest.fixture(scope="module")
def trial(chunk, planted):
    return chunk.like(chunk.data * planted)


def non_uniformity(field):
    return float(field.std(unbiased=False) / field.mean())


def unexplained(estimated, baseline, planted, inside):
    """The score ``experiments/`` uses, so the two are comparable."""
    ratio = estimated / baseline / planted[inside]
    ratio = ratio / ratio.mean()
    return float(ratio.std(unbiased=False))


@pytest.mark.parametrize("objective", ["hoyer", "tightness"])
def test_the_loss_goes_down(trial, chunk_mask, objective):
    """The whole point of the reformulation: one number, and it descends."""
    field = nu_optimize(trial, mask=chunk_mask, objective=objective, **BUDGET)
    history = field.optimize_info["loss"]

    assert len(history) >= 2
    assert history[-1] < history[0]


def test_it_removes_some_of_the_planted_field(trial, chunk, chunk_mask,
                                              planted, inside):
    """Recovery, against the only bound that needs no calibration.

    Doing nothing scores exactly the non-uniformity that was planted.  An
    estimator has to beat that or it is not estimating anything -- and on this
    small crop even N3 barely does (3.25% of a 3.29% field at this spacing),
    so the bar is deliberately the honest one rather than a tight one.

    ``hoyer`` only.  ``tightness`` does not meet this bar with a budget long
    enough to converge -- see the test below, which is where that is recorded.
    """
    settings = dict(objective="hoyer", **BUDGET)
    baseline = nu_optimize(chunk, mask=chunk_mask,
                           **settings).evaluate_on(chunk)[inside]
    estimated = nu_optimize(trial, mask=chunk_mask,
                            **settings).evaluate_on(trial)[inside]

    assert unexplained(estimated, baseline, planted, inside) \
        < non_uniformity(planted[inside])


def test_the_standardization_is_what_prevents_the_collapse(trial, chunk_mask,
                                                           inside):
    """The degeneracy is reachable *inside the model class*, and standardizing
    is what removes it.

    The worry is not abstract: a smooth spline field with enough freedom can
    approximate the log volume itself, and subtracting it leaves something
    close to a constant.  So the test builds exactly that field -- a
    least-squares spline fit to the log volume, which is the flattest thing
    the model can produce -- and compares it against doing nothing.

    Unstandardized, the flattening field wins: the objective genuinely prefers
    a destroyed image.  Standardized, it loses, because an affine change in
    the intensities is now invisible.  That reversal is the whole argument for
    :func:`~torch_n3.blocks.sharpness.standardize`.
    """
    grid = trial.shrink(DEFAULTS["shrink"])
    observed = torch.log(grid.data.clamp(min=1.0))
    masked = (grid.data > 1.0) & (chunk_mask.resample_like(grid).data != 0)

    # Fine knots, so the field has the freedom to chase the image; that
    # freedom is precisely what the degeneracy needs.
    spline = BSplineField(grid, 20.0, 1e-7).fit(observed, masked)
    columns, weights, where = spline.design(masked, 1)
    values = observed[where[:, 0], where[:, 1], where[:, 2]]
    flattened = values - (weights * spline.coefficients[columns]).sum(1)

    # Centres over each candidate's own range: without standardization there
    # is no common frame, and a histogram whose centres miss the data is not
    # a histogram (see hoyer_sparsity on what it returns then).
    def raw_sparsity(sample):
        centers = torch.linspace(float(sample.min()), float(sample.max()), 128,
                                 dtype=torch.float64)
        width = float(sample.std(unbiased=False)) * 0.15
        return float(hoyer_sparsity(soft_histogram(sample, centers, width)))

    def standardized_sparsity(sample):
        centers = torch.linspace(-4.0, 4.0, 128, dtype=torch.float64)
        return float(hoyer_sparsity(
            soft_histogram(standardize(sample), centers, 0.15)))

    assert non_uniformity(flattened) < 0.2 * non_uniformity(values)
    assert raw_sparsity(flattened) > raw_sparsity(values)
    assert standardized_sparsity(flattened) < standardized_sparsity(values)

    # And the shipped path, which standardizes, leaves a volume with contrast.
    field = nu_optimize(trial, mask=chunk_mask, **BUDGET)
    corrected = nu_evaluate(trial, field, mask=chunk_mask).data[inside]
    assert non_uniformity(corrected) > 0.1


def test_a_seed_names_one_run(trial, chunk_mask):
    """Subsampling makes this stochastic; the seed is what makes it a method."""
    first = nu_optimize(trial, mask=chunk_mask, seed=3,
                        **BUDGET).evaluate_on(trial)
    again = nu_optimize(trial, mask=chunk_mask, seed=3,
                        **BUDGET).evaluate_on(trial)

    assert torch.equal(first, again)


def test_two_seeds_agree_on_the_field(trial, chunk_mask, inside):
    """What makes subsampling admissible at all.

    Two draws of the same size must estimate the same field to much better
    than the field itself varies, or the answer is a property of which voxels
    were drawn.  The bound is a tenth of the planted non-uniformity: loose,
    but stated in units of the thing being measured rather than fitted to
    what came out.
    """
    settings = dict(sample_size=4096, max_iterations=BUDGET["max_iterations"],
                    distance=BUDGET["distance"])
    one = nu_optimize(trial, mask=chunk_mask, seed=1,
                      **settings).evaluate_on(trial)[inside]
    other = nu_optimize(trial, mask=chunk_mask, seed=2,
                        **settings).evaluate_on(trial)[inside]

    difference = ((one - other) ** 2).mean().sqrt() / one.mean()
    assert float(difference) < 0.1 * AMPLITUDE


def test_the_result_is_a_field_the_rest_of_the_pipeline_accepts(trial,
                                                                chunk_mask):
    """Drop-in, which is what makes the comparison with N3 possible at all."""
    field = nu_optimize(trial, mask=chunk_mask, **BUDGET)

    assert isinstance(field, BSplineField)
    corrected = nu_evaluate(trial, field, mask=chunk_mask)
    assert corrected.data.shape == trial.data.shape
    assert torch.isfinite(corrected.data).all()


def test_the_gauge_is_fixed(trial, chunk_mask, inside):
    """The log field is mean zero over the mask, so the field is mean one.

    Neither measure can see a constant, so without a gauge the returned field
    would be arbitrary up to a factor and two runs would not be comparable.
    """
    field = nu_optimize(trial, mask=chunk_mask, **BUDGET)
    values = field.evaluate_on(trial)[inside]

    assert float(torch.log(values).mean()) == pytest.approx(0.0, abs=0.02)


def test_tightness_reports_its_tissue_model(trial, chunk_mask):
    """The centroids are learned, so they are diagnostics worth having out."""
    field = nu_optimize(trial, mask=chunk_mask, objective="tightness",
                        classes=3, **BUDGET)
    info = field.optimize_info

    assert len(info["centroids"]) == 3
    assert len(info["occupancy"]) == 3
    assert math.isclose(sum(info["occupancy"]), 1.0, rel_tol=1e-9)
    # No cluster may quietly die: that is a fit with fewer classes than asked.
    assert min(info["occupancy"]) > 0.01


def test_tightness_makes_its_own_estimate_worse_as_it_converges(trial, chunk,
                                                                chunk_mask,
                                                                planted,
                                                                inside):
    """The finding that decides ``tightness`` is not usable, kept as a test.

    Its loss falls monotonically while the field it produces grows more and
    more non-uniform and the recovery gets worse.  That is not a tuning
    problem: within-cluster variance is minimised by a few widely separated
    spikes, and a smooth field can move towards that by *amplifying* contrast.
    :func:`~torch_n3.blocks.sharpness.standardize` pins the first two moments
    of the intensities; nothing pins the third.

    Asserted as a direction rather than a number, because the number is
    enormous and volume-dependent -- on ``brain_nu_ref.mnc`` at 200 mm the
    field reached 257% non-uniformity while the loss kept falling.  If this
    test ever fails, ``tightness`` has been fixed, and the module docstring
    and ``optimize.PENALTY`` need rewriting rather than the test.
    """
    def run(iterations):
        settings = dict(objective="tightness", penalty=1.0,
                        sample_size=BUDGET["sample_size"],
                        distance=BUDGET["distance"],
                        max_iterations=iterations)
        field = nu_optimize(trial, mask=chunk_mask, **settings)
        values = field.evaluate_on(trial)[inside]
        return field.optimize_info["loss"][-1], non_uniformity(values)

    short_loss, short_field = run(20)
    long_loss, long_field = run(200)

    assert long_loss < short_loss           # the objective is being minimised
    assert long_field > short_field         # and the field is running away
    assert long_field > non_uniformity(planted[inside])


def test_em_centroids_reach_the_same_kind_of_answer(trial, chunk_mask):
    field = nu_optimize(trial, mask=chunk_mask, objective="tightness",
                        centroid_update="em", **BUDGET)
    history = field.optimize_info["loss"]

    assert history[-1] < history[0]
    assert min(field.optimize_info["occupancy"]) > 0.01


def test_a_stochastic_objective_refuses_the_line_search(trial, chunk_mask):
    """``resample="always"`` with L-BFGS is wrong, and says so.

    Not a style preference: the strong-Wolfe line search and the curvature
    pairs both assume the objective is the same function between evaluations.
    """
    with pytest.raises(ValueError, match="stochastic"):
        nu_optimize(trial, mask=chunk_mask, resample="always",
                    optimizer="lbfgs", **BUDGET)


def test_resampling_runs_under_adam(trial, chunk_mask):
    field = nu_optimize(trial, mask=chunk_mask, resample="always",
                        optimizer="adam", learning_rate=0.02, **BUDGET)

    assert field.optimize_info["iterations"] == BUDGET["max_iterations"]


def test_too_few_bins_for_the_kernel_is_an_error(trial, chunk_mask):
    """A comb of separate spikes has a sparsity, and it means nothing."""
    with pytest.raises(ValueError, match="bin spacing"):
        nu_optimize(trial, mask=chunk_mask, bins=8, sigma=0.01, **BUDGET)


def test_unknown_settings_are_rejected(trial, chunk_mask):
    for bad in (dict(objective="entropy"), dict(optimizer="newton"),
                dict(precondition="magic"), dict(init="warm"),
                dict(centroid_update="kmeans"), dict(resample="sometimes")):
        with pytest.raises(ValueError):
            nu_optimize(trial, mask=chunk_mask, **dict(BUDGET, **bad))

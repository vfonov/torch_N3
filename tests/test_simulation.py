"""The generators behind ``experiments/``, held to what they promise.

The experiment itself is hours long and lives outside the suite (nothing in
``experiments/`` is collected: ``pytest.ini`` sets ``testpaths = tests``).  What
a fast test covers is the handful of functions every one of its rows depends on:
a field of the stated amplitude, noise of the stated sigma, and a score that is
zero when nothing was left behind.  If any of those is wrong, every number in
the sweep is wrong in a way no amount of averaging reveals.

Randomness is testable as long as it is seeded: each case fixes a seed and
asserts a property that must hold at that seed, and one asserts that a different
seed gives a different field.

``chunk.mnc`` throughout: these are properties of the arithmetic, and the small
volume makes them fast.
"""

import math

import pytest
import torch

from experiments import simulation

#: Amplitudes to check the normalisation at: the shipped pair plus the severe
#: case the sweep runs.
AMPLITUDES = [0.2, 0.4, 0.8]

#: Enough voxels that a sample standard deviation lands close to the one that
#: was asked for; the bound below is loose enough for this many.
SAMPLING = 0.02


@pytest.fixture(scope="module")
def inside(chunk, chunk_mask):
    return chunk_mask.resample_like(chunk).data != 0


@pytest.mark.parametrize("log_range", AMPLITUDES)
def test_planted_field_has_the_amplitude_it_was_asked_for(chunk, inside,
                                                          log_range):
    """Log peak-to-peak inside the mask is the amplitude, exactly.

    Exactly, because the field is rescaled to it by construction.  This is the
    definition the sweep's ``--amplitude`` axis rests on.
    """
    field = simulation.random_bias_field(chunk, inside, log_range, seed=1)
    values = field[inside]
    span = float(torch.log(values.max()) - torch.log(values.min()))
    assert span == pytest.approx(log_range, rel=1e-12)


def test_planted_field_is_normalised_inside_the_mask(chunk, inside):
    """Mean 1 over the mask: a bias field is defined up to a global scale."""
    field = simulation.random_bias_field(chunk, inside, 0.2, seed=2)
    assert float(field[inside].mean()) == pytest.approx(1.0, rel=1e-12)


def test_planted_field_is_positive_everywhere(chunk, inside):
    """Including outside the mask, where it still multiplies the volume."""
    field = simulation.random_bias_field(chunk, inside, 0.8, seed=3)
    assert float(field.min()) > 0.0


def test_a_seed_names_one_field(chunk, inside):
    """Same seed, same field, bit for bit -- a trial can be re-run."""
    first = simulation.random_bias_field(chunk, inside, 0.4, seed=7)
    again = simulation.random_bias_field(chunk, inside, 0.4, seed=7)
    assert torch.equal(first, again)


def test_different_seeds_name_different_fields(chunk, inside):
    """Fifty seeds are fifty experiments."""
    first = simulation.random_bias_field(chunk, inside, 0.4, seed=7)
    other = simulation.random_bias_field(chunk, inside, 0.4, seed=8)
    assert not torch.equal(first, other)


def test_a_smoother_field_is_easier_for_the_basis(chunk, chunk_mask, inside):
    """``--field-scale`` does what it says: longer waves, smaller floor.

    The floor is the residual of fitting the planted field with the spline
    itself, so it measures nothing but whether the basis can express the
    field.  A field built from longer wavelengths must be expressible at least
    as well as one built from shorter ones at the same knot spacing.
    """
    floors = [
        simulation.basis_floor(
            chunk, chunk_mask,
            simulation.random_bias_field(chunk, inside, 0.4, seed=11,
                                         scale=scale),
            inside, distance=75.0, lam=1e-7, solver="normal")
        for scale in (200.0, 800.0)]
    assert floors[1] < floors[0]


def test_the_basis_reproduces_a_unit_field_exactly(chunk, chunk_mask, inside):
    """What ``recovery``'s oracle baseline rests on.

    The oracle method is given the planted field and fits it; on the untouched
    volume there is no planted field, so it is given unity, and every oracle
    trial is divided by whatever that fit produced.  For that division to be the
    no-op the ceiling requires, the fit must return unity.

    It does, by construction: cubic B-splines are a partition of unity, so a
    constant is in the span, and the bending energy of a constant is zero, so
    the penalty cannot pull the fit off it at any ``lam``.  The bound is float64
    rounding through a solve of a matrix whose condition number CLAUDE.md puts
    at ~5e12, not a measured margin.
    """
    unit = torch.ones_like(chunk.data)
    fitted = simulation.basis_field(chunk, chunk_mask, unit, distance=75.0,
                                    lam=1e-7, solver="normal")

    assert float((fitted[inside] - 1.0).abs().max()) < 1e-9


def test_noise_has_the_sigma_the_snr_asked_for(chunk, inside):
    """SNR is the masked mean over the noise standard deviation."""
    sigma = simulation.noise_sigma(chunk, inside, 20.0)
    assert sigma == pytest.approx(float(chunk.data[inside].mean()) / 20.0)

    noisy = simulation.add_noise(chunk.data, sigma, seed=5)
    drawn = (noisy - chunk.data)[inside]
    assert float(drawn.std(unbiased=False)) == pytest.approx(sigma,
                                                             rel=SAMPLING)
    assert abs(float(drawn.mean())) < SAMPLING * sigma


def test_infinite_snr_is_the_noiseless_control(chunk, inside):
    """``--snr inf`` must leave the volume alone, not merely nearly so."""
    assert simulation.noise_sigma(chunk, inside, math.inf) == 0.0
    assert torch.equal(simulation.add_noise(chunk.data, 0.0, seed=5),
                       chunk.data)


def test_a_seed_names_one_noise_draw(chunk):
    """Same seed, same noise: the noise of a trial is reproducible alone."""
    first = simulation.add_noise(chunk.data, 10.0, seed=13)
    again = simulation.add_noise(chunk.data, 10.0, seed=13)
    assert torch.equal(first, again)


def test_a_perfect_recovery_scores_zero(chunk, inside):
    """The score is zero when the recovered field is the planted one.

    Zero to rounding, necessarily: every cell of the sweep is read as a distance
    from this.
    """
    planted = simulation.random_bias_field(chunk, inside, 0.4, seed=17)[inside]
    baseline = torch.ones_like(planted)

    ratio = simulation.unexplained(planted, baseline, planted)
    assert simulation.residual_percent(ratio) < 1e-10
    assert simulation.log_rms(ratio) < 1e-12


def test_the_score_ignores_a_global_scale(chunk, inside):
    """A field recovered up to a constant factor is a perfect recovery.

    N3 does not fix the overall scale -- ``-normalize_field`` is off by
    default -- so a score that charged for one would be measuring the wrong
    thing.
    """
    planted = simulation.random_bias_field(chunk, inside, 0.2, seed=19)[inside]
    baseline = torch.ones_like(planted)

    ratio = simulation.unexplained(planted * 3.7, baseline, planted)
    assert simulation.residual_percent(ratio) < 1e-10


def test_the_score_is_the_non_uniformity_left_behind(chunk, inside):
    """A field missed entirely scores the non-uniformity that was planted.

    The other anchor: recover nothing, and the score is what was there to
    remove.  Between the two, a cell of the sweep reads as a fraction.
    """
    planted = simulation.random_bias_field(chunk, inside, 0.2, seed=23)[inside]
    baseline = torch.ones_like(planted)

    ratio = simulation.unexplained(baseline, baseline, planted)
    assert simulation.residual_percent(ratio) == pytest.approx(
        simulation.non_uniformity_percent(1.0 / planted), rel=1e-9)

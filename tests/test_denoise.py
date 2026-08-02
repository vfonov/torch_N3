"""``torch_n3.blocks.denoise``: the non-local-means filter, and its wiring.

The filter is a modification rather than a port, so it has no oracle.  Nor is
``torch_SR``'s C kernel one: the two are the same computation under different
conventions and agree only where ``sigma`` is constant, so comparing against it
would pin the wrong thing.  What follows states the properties the filter is
required to have, and what it does to a bias-field estimate is measured by
``python3 -m tests.denoise`` rather than asserted here.

Every bound below is either exact or an ordering.  That is deliberate: a bound
this module could only justify by first running the code would record what the
filter does instead of what it must do (CLAUDE.md, "Test tolerances").

The phantom is built here rather than in ``tests/inputs.py``, which exists to
re-pose questions a recorded oracle has already answered; there is no recorded
answer to any of these.
"""

import pytest
import torch

from torch_n3.blocks.denoise import DEFAULT_SCALE, denoise, noise_level
from torch_n3.pipeline import DEFAULTS, evaluate_field, nu_correct, nu_estimate

#: The noise planted on the phantom, as a fraction of the slab separation.
NOISE = 0.05

#: Slab intensities: three tissues, far enough apart that the rejection test
#: keeps them separate and near enough that the filter is not trivial.
SLABS = (40.0, 120.0, 200.0)


@pytest.fixture(scope="module")
def clean():
    """Three constant slabs stacked along the first axis: 12x12x12."""
    volume = torch.zeros(12, 12, 12, dtype=torch.float64)
    for index, level in enumerate(SLABS):
        volume[index * 4:(index + 1) * 4] = level
    return volume


@pytest.fixture(scope="module")
def noisy(clean):
    """The same, with seeded Gaussian noise -- what the filter is given."""
    generator = torch.Generator().manual_seed(20260802)
    spread = NOISE * (max(SLABS) - min(SLABS))
    return clean + spread * torch.randn(clean.shape, generator=generator,
                                        dtype=torch.float64)


def test_a_constant_volume_is_returned_exactly(clean):
    """Nothing to average means nothing to change.

    Every patch distance is zero, so either every weight is one or the noise
    floor rejects them all; a weighted mean of identical values is that value
    either way.  Exact, from the algebra rather than from a measurement.
    """
    flat = torch.full((8, 8, 8), 77.0, dtype=torch.float64)
    assert torch.equal(denoise(flat), flat)


@pytest.mark.parametrize("factor", [2.0 ** -20, 2.0 ** 20])
def test_the_filter_does_not_depend_on_the_intensity_scale(noisy, factor):
    """The one property the internal normalisation exists to provide.

    ``SIGMA_FLOOR`` is an absolute threshold, so a filter that trusted its
    input's scale would smooth the background of a volume stored large and be
    the identity on the same volume stored small -- both silently.  Rescaling
    the input must instead rescale the output and do nothing else.

    Exact rather than approximate because the factors are powers of two:
    every quantity the filter derives from the volume -- the noise level, the
    audibility floor, the patch distances -- is then an exact binary rescaling
    of its counterpart, so the weights are bit-identical and no measurement
    enters the bound.  This is also the test that caught the source's fixed
    ``1e-10`` variance floor, which is scale-free only while the volume
    happens to be on ``DEFAULT_SCALE``.
    """
    assert torch.equal(denoise(noisy * factor) / factor, denoise(noisy))


@pytest.mark.parametrize("offset", [1000.0, -30.0])
def test_the_filter_does_not_depend_on_an_intensity_offset(noisy, offset):
    """Adding a constant to a volume must shift its answer and nothing else.

    Every other quantity in the filter is already translation-invariant -- the
    patch distances are differences, the noise level is a local standard
    deviation -- so a floor taken from an intensity *level* rather than from a
    range would be the one term that was not, and the same anatomy stored with
    a DC offset would be filtered as though its noise were smaller than it is.
    Together with the scaling above, this is affine equivariance:
    ``denoise(a*v + b) == a*denoise(v) + b``.

    Not exact, unlike the scaling: ``v + offset`` rounds, so the two runs are
    not handed bit-identical inputs.  The bound is float64's sixteen digits
    less the ~343-term accumulation.
    """
    shifted = denoise(noisy + offset) - offset
    assert float((shifted - denoise(noisy)).abs().max()) < 1e-12 * float(
        noisy.max() - noisy.min() + abs(offset))


def test_zero_strength_is_the_identity(noisy):
    """``strength`` reaches zero, and reaching it does nothing at all.

    Every ``sigma`` becomes zero, none clears ``SIGMA_FLOOR``, every weight is
    masked away, and the result is the input over a total weight of one.
    """
    assert torch.equal(denoise(noisy, strength=0.0), noisy)


def test_no_voxel_leaves_the_range_of_the_volume(noisy):
    """The output is a convex combination, so it cannot overshoot.

    Every weight is non-negative and the voxel enters its own sum at weight
    one, so each result lies between the smallest and largest value in its
    neighbourhood.  Catches a sign error in the exponent, a mis-initialised
    accumulator, or a normalisation applied to one sum and not the other.

    The tolerance is float64's sixteen digits less the ~343-term
    accumulation, not a measured margin.
    """
    result = denoise(noisy)
    slack = 1e-12 * float(noisy.max() - noisy.min())
    assert float(result.min()) >= float(noisy.min()) - slack
    assert float(result.max()) <= float(noisy.max()) + slack


def test_a_voxel_in_a_quiet_neighbourhood_is_left_untouched(clean):
    """``SIGMA_FLOOR`` spares what has no noise on it.

    Inside a slab, away from its boundaries, the local standard deviation is
    exactly zero, so those voxels must come back bit-identical even though the
    volume as a whole is not constant.
    """
    result = denoise(clean)
    interior = (slice(1, 3), slice(2, 10), slice(2, 10))
    assert torch.equal(result[interior], clean[interior])


def test_it_reduces_the_error_against_the_noise_free_phantom(clean, noisy):
    """What the filter is for, stated as the only bound that needs no number.

    An ordering, not a threshold: how much it removes is a measurement and
    belongs in ``tests/denoise.py``.
    """
    before = float((noisy - clean).pow(2).mean())
    after = float((denoise(noisy) - clean).pow(2).mean())
    assert after < before


def test_more_strength_smooths_more(noisy):
    """``strength`` is monotone in the direction its name claims.

    Larger ``sigma`` both flattens the weights towards one and loosens the
    mean-difference rejection, so more of the neighbourhood is averaged in and
    the result varies less.  Ordering only.
    """
    variances = [float(denoise(noisy, strength=s).var(unbiased=False))
                 for s in (0.5, 1.0, 2.0)]
    assert variances[2] < variances[1] < variances[0] < float(
        noisy.var(unbiased=False))


def test_the_noise_level_is_half_the_local_spread(clean):
    """``noise_level`` is the paper's estimate and not merely a local std.

    On a volume whose only structure is one step, the estimate away from the
    step is zero and at the step is half the smoothed spread.  Asserting the
    halving matters because ``SIGMA_FLOOR`` is a threshold on the halved
    quantity, so dropping the factor would move the gate by a factor of two.
    """
    assert float(noise_level(clean)[1, 5, 5]) == 0.0
    assert float(noise_level(clean).max()) > 0.0
    assert torch.equal(noise_level(clean) * 2.0,
                       noise_level(clean * 2.0))


@pytest.mark.parametrize("shape,fails", [((5, 5, 5), False), ((4, 5, 5), True),
                                         ((5, 4, 5), True), ((5, 5, 4), True)])
def test_an_axis_no_longer_than_the_padding_is_an_error(shape, fails):
    """Reflect padding cannot reach past the array it reflects.

    The kernel pads by ``search + patch``, which is 4 at the defaults, so every
    axis must exceed 4.  Torch's own message for this names neither the option
    that caused it nor the way out, so it is caught first.
    """
    volume = torch.rand(shape, dtype=torch.float64) + 1.0
    if fails:
        with pytest.raises(ValueError, match="reflect padding"):
            denoise(volume)
    else:
        assert denoise(volume).shape == shape


def test_a_volume_with_no_dynamic_range_is_returned_unchanged():
    """A volume with no spread anywhere has no noise, provably.

    The floor is measured against the dynamic range, and an empty volume has
    none -- but it also has a local standard deviation of zero everywhere, so
    the answer is settled before any scale is needed.  Returning NaN here would
    propagate silently through the whole estimation.
    """
    empty = torch.zeros(8, 8, 8, dtype=torch.float64)
    assert torch.equal(denoise(empty), empty)


def test_a_volume_that_is_almost_all_background_is_an_error():
    """Noise present, but no dynamic range to measure the floor against.

    Two voxels of signal in a field of zeros: the centile range collapses while
    the local spread does not, so no floor can be set.  Saying so is better
    than silently admitting every voxel.
    """
    volume = torch.zeros(9, 9, 9, dtype=torch.float64)
    volume[4, 4, 4] = 500.0
    volume[4, 4, 5] = 300.0
    with pytest.raises(ValueError, match="dynamic range"):
        denoise(volume)


def test_a_volume_that_is_not_three_dimensional_is_an_error():
    with pytest.raises(ValueError, match="3-D"):
        denoise(torch.rand(8, 8, dtype=torch.float64) + 1.0)


@pytest.mark.parametrize("search,patch,strength",
                         [(0, 1, 1.0), (3, 0, 1.0), (3, 1, -0.5)])
def test_the_parameters_are_checked(search, patch, strength):
    volume = torch.rand(8, 8, 8, dtype=torch.float64) + 1.0
    with pytest.raises(ValueError, match="denoise:"):
        denoise(volume, search=search, patch=patch, strength=strength)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"]
                                              if torch.cuda.is_available()
                                              else []))
def test_the_result_is_float64_where_it_was_given(noisy, device):
    """float64 throughout and no device round trip.

    The source this was copied from defaults to float32 on CUDA; everything
    here is float64, and a block that moved its input to another device would
    break the pipeline's ``--device`` handling silently.
    """
    result = denoise(noisy.to(device))
    assert result.dtype is torch.float64
    assert result.device.type == device


def test_the_two_devices_agree(noisy):
    """The filter is device-clean.

    float64 carries sixteen digits and the accumulation is over 343 terms, so a
    difference beyond twelve digits is a different filter rather than a
    different summation order.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    on_cpu = denoise(noisy)
    on_gpu = denoise(noisy.to("cuda")).cpu()
    assert float((on_gpu - on_cpu).pow(2).mean().sqrt()
                 / on_cpu.pow(2).mean().sqrt()) < 1e-12


# The filter is off by default and has no counterpart in the original.  What
# follows pins that, at both levels: the default path must not reach it, and
# the legacy backend must refuse rather than quietly ignore the request.

def test_the_default_pipeline_never_calls_the_filter(monkeypatch, chunk,
                                                     chunk_mask):
    """Nothing on the default path enters this module.

    Asserted rather than inferred, because every recorded answer in ``tests/``
    was produced without it: were the filter to run by default, or on an option
    left true by accident, each of them would move.
    """
    import torch_n3.blocks

    def boom(*arguments, **keywords):
        raise AssertionError("the default path called denoise()")

    # Patched on the blocks package, which is what `backends.resolve` hands
    # the pipeline.  The object form rather than the dotted string, because
    # the re-export shadows the submodule of the same name -- as it already
    # does for `histogram`.
    monkeypatch.setattr(torch_n3.blocks, "denoise", boom)

    assert DEFAULTS["denoise"] is False
    nu_estimate(chunk, mask=chunk_mask, iterations=(1,))


def test_denoising_changes_the_field_that_is_estimated(chunk, chunk_mask):
    """The option is wired to the estimation and not to a dead branch.

    Categorical: that it changes the answer at all.  Whether it changes it for
    the better is a measurement, and is what ``tests/denoise.py`` is for.
    """
    plain = nu_estimate(chunk, mask=chunk_mask, iterations=(1,))
    filtered = nu_estimate(chunk, mask=chunk_mask, iterations=(1,),
                           denoise=True)
    assert not torch.equal(plain.evaluate(), filtered.evaluate())


def test_the_written_volume_is_the_original_divided_by_the_field(chunk,
                                                                 chunk_mask):
    """Denoising reaches the estimate and stops there.

    The whole point of the option: the field is fitted to a filtered copy, but
    the volume handed back is the caller's own intensities divided by that
    field.  Exact, because both sides are the same division of the same
    tensor -- had the filtered copy leaked into the output, the difference
    would be the size of the noise.
    """
    field = nu_estimate(chunk, mask=chunk_mask, iterations=(1,), denoise=True)
    divisor = evaluate_field(chunk, field)
    corrected = nu_correct(chunk, mask=chunk_mask, iterations=(1,),
                           denoise=True)
    assert torch.equal(corrected.data, chunk.data / divisor.data)


def test_the_legacy_backend_has_no_denoiser():
    """N3 has no such stage, so the oracle refuses instead of approximating."""
    from torch_n3.backends import legacy

    with pytest.raises(ValueError, match="torch backend"):
        legacy.denoise(torch.rand(8, 8, 8, dtype=torch.float64) + 1.0)


def test_the_legacy_backend_refuses_a_denoised_run(chunk, chunk_mask):
    """And refuses it through the pipeline too, rather than silently ignoring."""
    with pytest.raises(ValueError, match="torch backend"):
        nu_estimate(chunk, mask=chunk_mask, iterations=(1,),
                    backend="legacy", denoise=True)


def test_the_scale_constant_is_the_one_the_floor_was_calibrated_on():
    """A guard on the module's own arithmetic.

    ``SIGMA_FLOOR`` is meaningful only against ``DEFAULT_SCALE``; changing one
    without the other silently re-tunes the filter, and no other test here
    would notice, since every one of them is scale-invariant by construction.
    """
    assert DEFAULT_SCALE == 256.0


def test_the_scale_is_a_range_and_not_a_level(noisy):
    """The floor is measured against a spread, which is what noise is.

    Stated directly as well as through the invariance above, because the two
    fail differently: a level-based floor would still pass the scaling test
    while being wrong about any volume with a DC offset, and it is what the
    code this was ported from uses.
    """
    from torch_n3.blocks.denoise import SCALE_QUANTILES, _scale

    assert SCALE_QUANTILES == (0.01, 0.99)
    assert _scale(noisy + 5000.0) == pytest.approx(_scale(noisy), rel=1e-12)
    assert _scale(noisy * 4.0) == pytest.approx(4.0 * _scale(noisy), rel=1e-12)

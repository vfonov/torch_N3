"""The tests ``legacy/N3/testing/CMakeLists.txt`` defines, as comparisons.

The legacy suite mostly checks that its programs *run*; the one test with a
numerical target is ``nu_reference_1``, which re-runs ``nu_estimate`` +
``nu_evaluate`` on ``brain.mnc.gz`` and requires the result to stay within
``1e-4`` relative RMS of ``brain_nu_ref.mnc.gz``.  Here every case is turned
into a comparison against the installed programs instead, so that the Python
pipeline has to reproduce them rather than merely not crash.

Each stage runs on both backends, so a failure says whether the port or the
plumbing around it is at fault.

On tolerances: the individual blocks agree with the legacy to the precision
their tests record.  The *pipeline* cannot, for two reasons.  The legacy
passes every intermediate volume between programs as a MINC file -- 12-bit
here, scaled slice by slice -- so each stage rounds its result before the next
one reads it.  And the iteration amplifies: see
``test_the_iteration_amplifies_small_differences``.
"""

import pytest
import torch

from tests.conftest import assert_close, span
from torch_n3.pipeline import (DEFAULTS, _sharpen, _smooth, evaluate_field,
                               nu_correct, nu_estimate)
from torch_n3.volume import Volume

BACKENDS = ["torch", "legacy"]


def relative_rms(result, reference):
    """The measure ``compare_nu_result.pl`` uses: RMS error over mean signal."""
    difference = torch.as_tensor(result) - torch.as_tensor(reference)
    return float((difference ** 2).mean().sqrt() / torch.as_tensor(reference).mean())


# --------------------------------------------------------------- single stages

@pytest.mark.parametrize("backend", BACKENDS)
def test_sharpen_matches_sharpen_volume(workspace, chunk, chunk_mask, backend):
    """`nu_sharpen_volume_1`: one pass of histogram sharpening."""
    inside = chunk_mask.data != 0
    log_volume = torch.log(chunk.data.clamp(min=1.0))
    log_volume = torch.where(inside, log_volume, torch.zeros_like(log_volume))

    source = workspace.write("log.mnc", chunk.like(log_volume))
    mask = workspace.write("mask.mnc", chunk.like(inside.to(torch.float64)),
                           store_dtype="int16")
    workspace.run("sharpen_volume", "-parzen", "-bins", 200,
                  "-fwhm", 0.15, "-noise", 0.01, "-clobber", "-quiet",
                  mask, source, workspace.at("sharp.mnc"))
    reference = workspace.read("sharp.mnc").data
    reference = torch.where(inside, reference, torch.zeros_like(reference))

    sharpened = _sharpen(log_volume, inside,
                         dict(DEFAULTS, bins=200, backend=backend))

    # sharpen_volume's output is a 16-bit MINC file spanning the mapped range.
    assert_close(sharpened, reference, atol=span(reference[inside]) / 65535)


@pytest.mark.parametrize("backend", BACKENDS)
def test_smooth_matches_spline_smooth(workspace, chunk, chunk_mask, backend):
    """`spline_smooth -full_support -b_spline`, the field-smoothing stage."""
    inside = chunk_mask.data != 0
    torch.manual_seed(3)
    bumpy = (0.05 * torch.cos(torch.linspace(0, 6, chunk.data.numel(),
                                             dtype=torch.float64))
             ).reshape(chunk.shape) + 0.01 * torch.randn(chunk.shape,
                                                         dtype=torch.float64)
    bumpy = torch.where(inside, bumpy, torch.zeros_like(bumpy))

    source = workspace.write("working.mnc", chunk.like(bumpy))
    mask = workspace.write("mask.mnc", chunk.like(inside.to(torch.float64)),
                           store_dtype="int16")
    workspace.run("spline_smooth", "-full_support", "-clobber", "-quiet",
                  "-distance", 200, "-b_spline", "-lambda", 1e-7,
                  "-subsample", 1, "-mask", mask, source,
                  workspace.at("residue.mnc"))
    reference = workspace.read("residue.mnc").data

    smoothed = _smooth(bumpy, inside, chunk, dict(DEFAULTS, backend=backend))

    assert_close(smoothed, reference, atol=span(reference) / 65535)


@pytest.mark.parametrize("backend", BACKENDS)
def test_spline_evaluates_on_a_finer_grid_like_evaluate_field(
        workspace, chunk, chunk_mask, backend):
    """`nu_imp2field`: a spline fitted coarse, evaluated at full resolution.

    This is the round trip N3 makes through the ``.imp`` mapping file between
    ``nu_estimate`` and ``nu_evaluate``, and the reason the estimation can
    afford to run on a coarse grid at all.
    """
    from torch_n3 import backends

    grid = chunk.shrink(4)
    inside = chunk_mask.data != 0
    coarse_inside = chunk_mask.resample_like(grid).data != 0

    zz, yy, xx = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)
                                  for n in grid.shape], indexing="ij")
    field = 1.0 + 0.01 * zz - 0.005 * yy + 0.002 * xx

    spline = backends.resolve(backend).BSplineField(grid, distance=200.0,
                                                    lam=1e-7)
    spline.fit(field, coarse_inside)

    # The legacy route: write the field, fit a compact spline to it, evaluate.
    source = workspace.write("field.mnc", grid.like(field))
    coarse_mask = workspace.write("coarse_mask.mnc",
                                  grid.like(coarse_inside.to(torch.float64)),
                                  store_dtype="int16")
    fine = workspace.write("fine.mnc", chunk)
    fine_mask = workspace.write("fine_mask.mnc",
                                chunk.like(inside.to(torch.float64)),
                                store_dtype="int16")
    workspace.run("spline_smooth", "-full_support", "-clobber", "-quiet",
                  "-distance", 200, "-b_spline", "-lambda", 1e-7,
                  "-subsample", 1, "-novolume", "-mask", coarse_mask,
                  source, "-compact", workspace.at("field.imp"))
    workspace.run("evaluate_field", "-clobber", "-mask", fine_mask,
                  "-like", fine, workspace.at("field.imp"),
                  workspace.at("evaluated.mnc"))
    reference = workspace.read("evaluated.mnc").data

    evaluated = spline.evaluate_on(chunk)
    evaluated = torch.where(inside, evaluated, torch.zeros_like(evaluated))

    assert_close(evaluated, reference, atol=span(reference) / 65535)


# ------------------------------------------------------------------ end to end

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("iterations,shrink,fwhm", [(1, 3, 0.2), (3, 4, 0.15)])
def test_nu_correct_tracks_the_legacy_pipeline(workspace, chunk, chunk_mask,
                                               iterations, shrink, fwhm,
                                               backend):
    """`nu_correct_8`: the whole thing, against the installed `nu_correct`."""
    source = workspace.write("chunk.mnc", chunk, store_dtype="int16")
    mask = workspace.write("mask.mnc",
                           chunk.like((chunk_mask.data != 0).to(torch.float64)),
                           store_dtype="int16")
    workspace.run("nu_correct", "-clobber", "-quiet", "-mapping_dir",
                  workspace.at(""), "-fwhm", fwhm, "-shrink", shrink,
                  "-stop", 0.001, "-iterations", iterations, "-mask", mask,
                  source, workspace.at("nu.mnc"))
    reference = workspace.read("nu.mnc").data

    corrected = nu_correct(chunk, mask=chunk_mask, evaluation_mask=chunk_mask,
                           fwhm=fwhm, shrink=shrink, backend=backend,
                           iterations=(iterations,), stop=(0.001,))

    assert relative_rms(corrected.data, reference) < 1e-3


def test_nu_estimate_recovers_a_planted_field():
    """The point of the exercise, on a phantom where the answer is known.

    Two tissues, a little noise, and a smooth multiplicative field: N3 should
    hand the field back.  Note it never sees the tissue values -- all it has
    to go on is that the intensity histogram of the *corrected* volume should
    be sharper than the one it measures.
    """
    shape, step = (40, 40, 40), (2.0, 2.0, 2.0)
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)
                               for n in shape], indexing="ij")
    radius = (z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2

    torch.manual_seed(0)
    tissue = torch.where(radius < 12 ** 2, 200.0, 100.0)
    noisy = tissue * (1 + 0.02 * torch.randn(shape, dtype=torch.float64))
    inside = radius < 17 ** 2
    planted = torch.exp(0.20 * (x / shape[2] - 0.5) + 0.15 * (z / shape[0] - 0.5))

    volume = Volume(torch.where(inside, noisy * planted,
                                torch.zeros_like(noisy)), (0, 0, 0), step)
    mask = Volume(inside.to(torch.float64), (0, 0, 0), step)

    field = nu_estimate(volume, mask=mask, distance=80.0, shrink=1,
                        iterations=(50,), stop=(5e-4,))

    # A bias field is only defined up to a global scale, so what has to be
    # flat is the ratio, not the difference.
    ratio = (field.evaluate_on(volume) / planted)[inside]
    planted_variation = float(planted[inside].std() / planted[inside].mean())
    assert float(ratio.std() / ratio.mean()) < planted_variation / 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_the_legacy_reference_volume(brain, model_mask, brain_reference,
                                             backend):
    """`nu_reference_1`, the legacy suite's one numerical regression test.

    The legacy asks for ``1e-4`` relative RMS, which is a comparison of the
    legacy against itself and holds exactly.  We cannot reach it, and the
    reason is not the algorithm: legacy N3 hands every intermediate volume to
    the next program as a 12-bit, slice-scaled MINC file, so every stage
    rounds before the next one reads, thirty times over.  Working in float64
    instead costs about 3e-3 here -- a third of a percent, well below the
    noise of the images N3 is used on.
    """
    corrected = nu_correct(brain, mask=model_mask, backend=backend)

    assert relative_rms(corrected.data, brain_reference.data) < 5e-3


def test_the_iteration_amplifies_small_differences(brain, model_mask):
    """Why the two backends part company end to end, though the blocks agree.

    N3's loop feeds its own output back in, so a difference of one part in
    ``1e7`` after a single iteration is a difference of one part in ``1e3``
    after thirty.  This is a property of the algorithm, not of either
    implementation, and it is the reason the end-to-end tolerances above are
    so much looser than the per-block ones.
    """
    def divisor(backend, iterations):
        field = nu_estimate(brain, mask=model_mask, backend=backend,
                            iterations=(iterations,), stop=(0.0,))
        return evaluate_field(brain, field, backend=backend).data

    early = relative_rms(divisor("torch", 1), divisor("legacy", 1))
    late = relative_rms(divisor("torch", 10), divisor("legacy", 10))

    assert early < 1e-6
    assert late > 100 * early

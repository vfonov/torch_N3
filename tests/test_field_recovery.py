"""Can it recover a bias field we planted ourselves?

Every other test here asks whether the port reproduces the original N3.  This
one asks the question a user actually has: given a volume with a known
non-uniformity, does N3 hand the non-uniformity back?

The substrate is ``brain_nu_ref.mnc.gz`` -- real anatomy that has already been
through ``nu_correct``, so what little non-uniformity remains in it is small
compared with what we are about to add.  A smooth multiplicative field of a
set amplitude goes on top, the result is written out as
``brain_nu_artificial.mnc``, and every implementation available is asked to
correct it: the PyTorch blocks, the same pipeline running the original C++
blocks, and -- if it is on ``PATH`` -- the installed ``nu_correct`` itself.

Comparing the two backends is the test that always runs and is always fair:
same pipeline, same input file, only the blocks differ.  The binary is the
stronger check when it is there, because it shares no code with either.

What none of them can be held to is an absolute residual.  N3 does not recover
a field perfectly -- it leaves about 0.9% here whoever runs it -- so a
tolerance on that would be a statement about N3, not about this code.
"""

import math
from dataclasses import dataclass

import pytest
import torch

from tests.conftest import (Workspace, legacy_data, program_available,
                            requires_program)
from torch_n3.pipeline import nu_estimate
from torch_n3.volume import Volume, load_volume

#: Log peak-to-peak amplitude of the planted fields.  A field spanning ``0.2``
#: in the log domain ranges over a factor of ``exp(0.2) = 1.22``, i.e. roughly
#: +-10% about its mean -- what the BrainWeb phantoms call "20% RF", and about
#: what a 1.5 T head coil produces.  ``0.4`` is their harder case.
AMPLITUDES = [0.2, 0.4]

#: The two block implementations, both driving the same pipeline.
BACKENDS = ["torch", "legacy"]

#: The installed program, which shares no code with either of them.
BINARY = "nu_correct"

#: How closely two implementations must agree on the field they recovered,
#: as a relative RMS difference over the mask.  One number for every pairing,
#: fixed in advance: it is what is being *required*, not what was measured.
#: If it fails, see the note on tolerances in CLAUDE.md -- the first thing to
#: check is whether the two ran the same number of iterations.
AGREEMENT = 1e-3

HAVE_BINARY = program_available(BINARY)
needs_the_binary = requires_program(BINARY)


def synthetic_bias_field(volume, inside, log_range):
    """A smooth multiplicative field of exactly ``log_range`` log peak-to-peak.

    Deliberately not a B-spline: a few low-order harmonics across the volume,
    which is the shape coil sensitivity actually takes and which N3's basis
    can only approximate.  Normalised to mean 1 inside ``inside``, since a
    bias field is only ever defined up to a global scale.
    """
    axes = [torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
            for n in volume.shape]
    u, v, w = torch.meshgrid(*axes, indexing="ij")
    shape = (0.55 * torch.cos(0.9 * u + 0.3) + 0.40 * torch.sin(0.8 * v - 0.5)
             + 0.30 * w + 0.25 * u * v)

    spread = float(shape[inside].max() - shape[inside].min())
    field = torch.exp(shape * (log_range / spread))
    return field / field[inside].mean()


def non_uniformity(field, inside):
    """How much a field varies across the mask, as a coefficient of variation.

    This is the number N3 exists to reduce, and comparing it before and after
    is how the paper reports its own results.
    """
    values = field[inside]
    return float(values.std(unbiased=False) / values.mean())


@dataclass
class Recovery:
    """One planted-field experiment, corrected by every implementation."""

    log_range: float
    inside: torch.Tensor
    planted: torch.Tensor                      # the field we put on
    baseline: dict                             # what each finds in the reference
    recovered: dict                            # what each finds after planting
    reference: Volume                          # brain_nu_ref, our starting point
    corrected: Volume                          # our correction of the artificial

    def unexplained(self, source):
        """The part of ``source``'s answer that is not the planted field.

        Divided by that implementation's own baseline as well, because the
        reference is not perfectly uniform to begin with and N3 removes that
        too; without dividing it out the comparison would charge this code for
        non-uniformity it corrected successfully.  Renormalised, since only
        the shape of a field means anything.
        """
        ratio = (self.recovered[source] / self.baseline[source]
                 / self.planted)[self.inside]
        return ratio / ratio.mean()

    def residual(self, source):
        """How much non-uniformity ``source`` failed to take out."""
        return float(self.unexplained(source).std(unbiased=False))

    def disagreement(self, one, other):
        """How far apart two implementations' recovered fields are.

        Relative RMS over the mask.  Deliberately not the largest single
        difference: that is an extreme-value statistic over a quarter of a
        million voxels, dominated by a few on the edge of the mask, and no
        fixed bound on it would mean much.
        """
        difference = self.unexplained(one) - self.unexplained(other)
        return float((difference ** 2).mean().sqrt())


def _run_the_binary(workspace, volume_path, mask_path, name):
    """``nu_correct`` on a file, returning the field it divided out."""
    workspace.run(BINARY, "-clobber", "-quiet", "-mapping_dir",
                  workspace.at(""), "-mask", mask_path, volume_path,
                  workspace.at(name))
    return load_volume(volume_path).data / workspace.read(name).data


@pytest.fixture(scope="module")
def baseline(tmp_path_factory, brain_reference, model_mask):
    """What each implementation finds in the *untouched* reference volume.

    Not zero: `brain_nu_ref.mnc.gz` still carries about 0.5% of
    non-uniformity, and each implementation has its own opinion of what it is.
    Measuring it once here keeps that out of the comparisons below.
    """
    workspace = Workspace(tmp_path_factory.mktemp("baseline"))
    inside = model_mask.resample_like(brain_reference).data != 0

    fields = {backend: nu_estimate(brain_reference, mask=model_mask,
                                   backend=backend).evaluate_on(brain_reference)
              for backend in BACKENDS}

    if not HAVE_BINARY:
        return fields

    reference = workspace.write("brain_nu_ref.mnc", brain_reference,
                                like=legacy_data("brain_nu_ref.mnc.gz"),
                                store_dtype="int16")
    mask = workspace.write("mask.mnc",
                           brain_reference.like(inside.to(torch.float64)),
                           store_dtype="int16")
    fields[BINARY] = _run_the_binary(workspace, reference, mask, "nu_ref.mnc")
    return fields


@pytest.fixture(scope="module", params=AMPLITUDES,
                ids=["%d%%" % (a * 100) for a in AMPLITUDES])
def recovery(request, tmp_path_factory, brain_reference, model_mask, baseline):
    log_range = request.param
    workspace = Workspace(tmp_path_factory.mktemp("recovery"))
    reference = brain_reference
    inside = model_mask.resample_like(reference).data != 0

    planted = synthetic_bias_field(reference, inside, log_range)

    # Write the artificial volume as a real MINC file, in the reference's own
    # 16-bit storage, so that every implementation reads exactly the same
    # numbers -- otherwise the comparison would be partly about quantisation.
    artificial = workspace.write("brain_nu_artificial.mnc",
                                 reference.like(reference.data * planted),
                                 like=legacy_data("brain_nu_ref.mnc.gz"),
                                 store_dtype="int16")
    volume = load_volume(artificial)
    mask = workspace.write("mask.mnc", reference.like(inside.to(torch.float64)),
                           store_dtype="int16")

    recovered = {backend: nu_estimate(volume, mask=model_mask, backend=backend
                                      ).evaluate_on(volume)
                 for backend in BACKENDS}
    if HAVE_BINARY:
        recovered[BINARY] = _run_the_binary(workspace, artificial, mask,
                                            "nu.mnc")

    return Recovery(log_range=log_range, inside=inside, planted=planted,
                    baseline=baseline, recovered=recovered, reference=reference,
                    corrected=volume.like(volume.data / recovered["torch"]))


def test_the_planted_field_has_the_amplitude_asked_for(recovery):
    """The experiment is only as good as its setup, so check the setup."""
    field = recovery.planted[recovery.inside]

    assert math.log(float(field.max() / field.min())) == pytest.approx(
        recovery.log_range, rel=1e-9)
    assert float(field.mean()) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_planted_field_is_recovered(recovery, backend):
    """Most of the non-uniformity we put in comes back out."""
    planted = non_uniformity(recovery.planted, recovery.inside)

    assert recovery.residual(backend) < planted / 4


def test_the_two_backends_recover_the_same_field(recovery):
    """The comparison that is always available: same pipeline, other blocks.

    The blocks themselves agree to between 1e-13 and 1e-6 (``test_histogram.py``
    and friends); this asks whether that survives the loop, which feeds its own
    output back in.

    Mostly it does, but the margin here is thin and the reason is worth
    knowing: at 40% the two stop at different iterations, because the rule is
    ``change < 0.001`` and one reaches 0.000975 where the other is still at
    0.001011.  A whole extra field update moves the answer far more than any
    block difference does.  If this fails, check the iteration counts before
    anything else.
    """
    assert recovery.disagreement("torch", "legacy") < AGREEMENT


@needs_the_binary
@pytest.mark.parametrize("backend", BACKENDS)
def test_it_recovers_as_much_as_the_installed_nu_correct(recovery, backend):
    """The stronger check, when the original is installed: no shared code.

    N3 leaves about 0.9% of the field behind on this data no matter who runs
    it, so what an implementation can be held to is not an absolute residual
    but whether it agrees with the original about which field is there.
    """
    assert recovery.disagreement(backend, BINARY) < AGREEMENT


def test_the_correction_restores_the_reference_volume(recovery):
    """End to end, in the terms a user would put it in.

    Correcting the artificial volume has to bring it back towards the volume
    it was made from -- and by much more than it moves the intensities around
    for other reasons.
    """
    inside = recovery.inside
    reference = recovery.reference.data[inside]

    def distance(volume):
        values = volume.data[inside]
        values = values * (reference.mean() / values.mean())  # scale is free
        return float(((values - reference) ** 2).mean().sqrt() / reference.mean())

    before = distance(recovery.reference.like(recovery.reference.data
                                              * recovery.planted))
    after = distance(recovery.corrected)

    assert after < before / 3

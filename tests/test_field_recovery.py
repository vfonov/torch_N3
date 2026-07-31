"""Can it recover a bias field we planted ourselves?

Every other test here asks whether the port reproduces the original N3.  This
one asks the question a user actually has: given a volume with a known
non-uniformity, does N3 hand the non-uniformity back?

The substrate is ``brain_nu_ref.mnc.gz`` -- real anatomy that has already been
through ``nu_correct``, so what little non-uniformity remains in it is small
compared with what we are about to add.  A smooth multiplicative field of a
set amplitude goes on top, and three implementations are asked to correct the
result: the PyTorch blocks, the same pipeline running the original C++ blocks,
and the installed ``nu_correct`` -- whose answer was recorded once, from a
volume written out as ``brain_nu_artificial.mnc``, and is read back from
``tests/reference/`` rather than recomputed.

Comparing the two backends is the fairest comparison: same pipeline, same
input, only the blocks differ.  ``nu_correct`` is the stronger one, because it
shares no code with either.

What none of them can be held to is an absolute residual.  N3 does not recover
a field perfectly -- it leaves about 0.9% here whoever runs it -- so a
tolerance on that would be a statement about N3, not about this code.

Everything below is computed inside the model mask and nowhere else, which is
also all that is recorded.
"""

import math
from dataclasses import dataclass

import pytest
import torch

from tests.conftest import legacy_data
from tests.inputs import as_stored, synthetic_bias_field
from torch_n3.pipeline import nu_estimate

#: Log peak-to-peak amplitude of the planted fields.  A field spanning ``0.2``
#: in the log domain ranges over a factor of ``exp(0.2) = 1.22``, i.e. roughly
#: +-10% about its mean -- what the BrainWeb phantoms call "20% RF", and about
#: what a 1.5 T head coil produces.  ``0.4`` is their harder case.
AMPLITUDES = [0.2, 0.4]

#: The two block implementations, both driving the same pipeline.
BACKENDS = ["torch", "legacy"]

#: The installed program.  Its answers are recorded, so nothing is run here.
BINARY = "nu_correct"

#: Which recorded ``nu_correct`` run goes with which planted amplitude.
RECORDED = {0.2: "nu_correct.planted_20_field",
            0.4: "nu_correct.planted_40_field"}

#: How closely two implementations must agree on the field they recovered,
#: as a relative RMS difference over the mask.  One number for every pairing,
#: fixed in advance: it is what is being *required*, not what was measured.
#: If it fails, see the note on tolerances in CLAUDE.md -- the first thing to
#: check is whether the two ran the same number of iterations.
AGREEMENT = 1e-3


def non_uniformity(field):
    """How much a field varies, as a coefficient of variation.

    This is the number N3 exists to reduce, and comparing it before and after
    is how the paper reports its own results.
    """
    return float(field.std(unbiased=False) / field.mean())


@dataclass
class Recovery:
    """One planted-field experiment, corrected by every implementation.

    Every tensor here is already restricted to the mask and flattened.
    """

    log_range: float
    planted: torch.Tensor      # the field we put on
    baseline: dict             # what each implementation finds in the reference
    recovered: dict            # what each finds after the field was planted
    reference: torch.Tensor    # brain_nu_ref, the volume we started from
    corrected: torch.Tensor    # our correction of the artificial volume

    def unexplained(self, source):
        """The part of ``source``'s answer that is not the planted field.

        Divided by that implementation's own baseline as well, because the
        reference is not perfectly uniform to begin with and N3 removes that
        too; without dividing it out the comparison would charge this code for
        non-uniformity it corrected successfully.  Renormalised, since only
        the shape of a field means anything.
        """
        ratio = self.recovered[source] / self.baseline[source] / self.planted
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


@pytest.fixture(scope="module")
def baseline(legacy_output, tmp_path_factory, brain_reference, model_mask):
    """What each implementation finds in the *untouched* reference volume.

    Not zero: ``brain_nu_ref.mnc.gz`` still carries about 0.5% of
    non-uniformity, and each implementation has its own opinion of what it is.
    Measuring it once here keeps that out of the comparisons below.
    """
    inside = model_mask.resample_like(brain_reference).data != 0
    volume = as_stored(tmp_path_factory.mktemp("baseline"), "reference.mnc",
                       brain_reference, legacy_data("brain_nu_ref.mnc.gz"))

    fields = {backend: _estimate(volume, model_mask, backend, inside)
              for backend in BACKENDS}
    fields[BINARY] = legacy_output["nu_correct.reference_field"]
    return fields


@pytest.fixture(scope="module", params=AMPLITUDES,
                ids=["%d%%" % (a * 100) for a in AMPLITUDES])
def recovery(request, legacy_output, tmp_path_factory, brain_reference,
             model_mask, baseline):
    log_range = request.param
    inside = model_mask.resample_like(brain_reference).data != 0
    planted = synthetic_bias_field(brain_reference, inside, log_range)

    # ``brain_nu_artificial.mnc``: the same file the recorded run was handed,
    # written the same way, so both see the same quantised intensities.  No
    # legacy program runs -- this is our own MINC I/O.
    volume = as_stored(tmp_path_factory.mktemp("recovery"),
                       "brain_nu_artificial.mnc",
                       brain_reference.like(brain_reference.data * planted),
                       legacy_data("brain_nu_ref.mnc.gz"))

    recovered = {backend: _estimate(volume, model_mask, backend, inside)
                 for backend in BACKENDS}
    recovered[BINARY] = legacy_output[RECORDED[log_range]]

    return Recovery(log_range=log_range, planted=planted[inside],
                    baseline=baseline, recovered=recovered,
                    reference=brain_reference.data[inside],
                    corrected=volume.data[inside] / recovered["torch"])


def _estimate(volume, mask, backend, inside):
    """The field ``backend`` finds in ``volume``, inside the mask."""
    field = nu_estimate(volume, mask=mask, backend=backend)
    return field.evaluate_on(volume)[inside]


def test_the_planted_field_has_the_amplitude_asked_for(recovery):
    """The experiment is only as good as its setup, so check the setup."""
    field = recovery.planted

    assert math.log(float(field.max() / field.min())) == pytest.approx(
        recovery.log_range, rel=1e-9)
    assert float(field.mean()) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_planted_field_is_recovered(recovery, backend):
    """Most of the non-uniformity we put in comes back out."""
    assert recovery.residual(backend) < non_uniformity(recovery.planted) / 4


def test_the_two_backends_recover_the_same_field(recovery):
    """The fairest comparison: same pipeline, same input, other blocks.

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


@pytest.mark.parametrize("backend", BACKENDS)
def test_it_recovers_as_much_as_nu_correct_did(recovery, backend):
    """The stronger check: no shared code with either implementation.

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
    reference = recovery.reference

    def distance(values):
        values = values * (reference.mean() / values.mean())  # scale is free
        return float(((values - reference) ** 2).mean().sqrt() / reference.mean())

    before = distance(reference * recovery.planted)
    after = distance(recovery.corrected)

    assert after < before / 3

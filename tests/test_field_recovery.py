"""Can it recover a bias field we planted ourselves?

Every other test here asks whether the port reproduces the original N3.  This
one asks the question a user actually has: given a volume with a known
non-uniformity, does N3 hand the non-uniformity back?

The substrate is ``brain_nu_ref.mnc`` -- real anatomy that has already been
through ``nu_correct``, so what little non-uniformity remains in it is small
compared with what we are about to add.  A smooth multiplicative field of a
set amplitude goes on top, and three implementations are asked to correct the
result at each of several knot spacings: the PyTorch blocks, the same pipeline
running the original C++ blocks, and the installed ``nu_correct`` -- whose
answers were recorded once, from volumes written out as
``brain_nu_artificial.mnc``, and are read back from ``tests/reference/``
rather than recomputed.

Comparing the two backends is the fairest comparison: same pipeline, same
input, only the blocks differ.  ``nu_correct`` is the stronger one, because it
shares no code with either.  Sweeping ``-distance`` is what makes either worth
much: it is the B-spline block's main parameter, it changes the number of
coefficients five-fold over the range used here, and a fault in knot placement
would most likely show at some spacings and not others.

**Every run is given a fixed iteration count with the early stop disabled.**
That is not a detail.  Left alone, N3 stops at ``change < 0.001``, and in four
of the six cells here the two backends land on opposite sides of it and run a
different number of iterations -- an entire extra field update, which moves
the answer several times more than any block-level difference.  Holding the
count fixed makes this a comparison of implementations rather than of where a
threshold happened to fall; it improves agreement by up to 8x.

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

#: Knot spacings to run each amplitude at.  200 mm is what ``nu_correct``
#: ships with; the other two give the spline three and five times as many
#: coefficients (80, 150, 392 on this volume).
DISTANCES = [200.0, 100.0, 50.0]
DEFAULT_DISTANCE = 200.0

#: The two block implementations, both driving the same pipeline.
BACKENDS = ["torch", "legacy"]

#: The installed program.  Its answers are recorded, so nothing is run here.
BINARY = "nu_correct"

#: Iterations every implementation runs, with the early stop disabled, so
#: that they all do the same work.  See the module docstring.
PROTOCOL = dict(iterations=(30,), stop=(0.0,))

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
    distance: float            # the knot spacing every implementation used
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

    def distance_from_reference(self, values):
        """How far ``values`` sit from the volume the experiment started from.

        Relative RMS, after matching the means -- N3 does not fix the overall
        scale, so only the shape of the difference is meaningful.
        """
        reference = self.reference
        values = values * (reference.mean() / values.mean())
        return float(((values - reference) ** 2).mean().sqrt() / reference.mean())

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

    Not zero: ``brain_nu_ref.mnc`` still carries about 0.5% of non-uniformity,
    each implementation has its own opinion of what it is, and that opinion
    depends on the knot spacing like everything else.  Measuring it once per
    spacing keeps it out of the comparisons below.
    """
    inside = model_mask.resample_like(brain_reference).data != 0
    volume = as_stored(tmp_path_factory.mktemp("baseline"), "reference.mnc",
                       brain_reference, legacy_data("brain_nu_ref.mnc"))

    fields = {}
    for distance in DISTANCES:
        fields[distance] = {
            backend: _estimate(volume, model_mask, backend, distance, inside)
            for backend in BACKENDS}
        fields[distance][BINARY] = legacy_output[
            "nu_correct.reference_field_d%d" % distance]
    return fields


#: Every cell of the sweep, in the order the ids below name them.
CELLS = [(amplitude, distance)
         for amplitude in AMPLITUDES for distance in DISTANCES]


@pytest.fixture(scope="module")
def recoveries(legacy_output, tmp_path_factory, brain_reference, model_mask,
               baseline):
    """Every (amplitude, spacing) experiment, built once.

    One dictionary rather than a parametrised fixture, because
    :func:`test_a_stiffer_spline_recovers_a_smooth_field_better` compares
    across spacings and would otherwise have to redo them all.
    """
    inside = model_mask.resample_like(brain_reference).data != 0
    directory = tmp_path_factory.mktemp("recovery")
    experiments = {}

    for log_range in AMPLITUDES:
        planted = synthetic_bias_field(brain_reference, inside, log_range)

        # ``brain_nu_artificial.mnc``: the same file the recorded run was
        # handed, written the same way, so both see the same quantised
        # intensities.  No legacy program runs -- this is our own MINC I/O.
        volume = as_stored(directory, "brain_nu_artificial_%d.mnc"
                           % (log_range * 100),
                           brain_reference.like(brain_reference.data * planted),
                           legacy_data("brain_nu_ref.mnc"))

        for distance in DISTANCES:
            recovered = {backend: _estimate(volume, model_mask, backend,
                                            distance, inside)
                         for backend in BACKENDS}
            recovered[BINARY] = legacy_output[
                "nu_correct.planted_%d_field_d%d" % (log_range * 100, distance)]

            experiments[(log_range, distance)] = Recovery(
                log_range=log_range, distance=distance,
                planted=planted[inside], baseline=baseline[distance],
                recovered=recovered, reference=brain_reference.data[inside],
                corrected=volume.data[inside] / recovered["torch"])
    return experiments


@pytest.fixture(params=CELLS,
                ids=["%d%%-%dmm" % (a * 100, d) for a, d in CELLS])
def recovery(request, recoveries):
    return recoveries[request.param]


def _estimate(volume, mask, backend, distance, inside):
    """The field ``backend`` finds in ``volume``, inside the mask."""
    field = nu_estimate(volume, mask=mask, backend=backend, distance=distance,
                        **PROTOCOL)
    return field.evaluate_on(volume)[inside]


def test_the_planted_field_has_the_amplitude_asked_for(recovery):
    """The experiment is only as good as its setup, so check the setup."""
    field = recovery.planted

    assert math.log(float(field.max() / field.min())) == pytest.approx(
        recovery.log_range, rel=1e-9)
    assert float(field.mean()) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_planted_field_is_recovered(recovery, backend):
    """At the very least, the non-uniformity has to be halved.

    A floor on being useful rather than a description of performance: a bias
    corrector that leaves more than half of a known field behind is not doing
    its job, whatever the knot spacing.  What N3 actually manages is a good
    deal better at 200 mm -- it leaves about 7% -- and degrades to 24-37% at
    50 mm, as the spline is given freedom it does not need.
    """
    assert recovery.residual(backend) < non_uniformity(recovery.planted) / 2


def test_the_two_backends_recover_the_same_field(recovery):
    """The fairest comparison: same pipeline, same input, other blocks.

    The blocks themselves agree to between 1e-13 and 1e-6 (``test_histogram.py``
    and friends); this asks whether that survives thirty rounds of a loop that
    feeds its own output back in.  What is left is genuine amplification, now
    that the iteration count is held fixed -- see the module docstring for why
    that mattered.
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


@pytest.mark.parametrize("backend", BACKENDS + [BINARY])
def test_a_stiffer_spline_recovers_a_smooth_field_better(recoveries, backend):
    """Freedom the field does not need is spent on anatomy instead.

    The planted field is smooth by construction, as a real one is, so the
    stiffest spline in the sweep is already able to represent it.  Giving the
    fit more coefficients cannot help it and does measurably hurt: the extra
    degrees of freedom go into following tissue contrast, which comes back as
    field that was never planted.

    Stated as a bare ordering between the ends of the sweep -- no factor to
    choose.  It is not monotone at every step and is not asserted to be.
    """
    for log_range in AMPLITUDES:
        stiffest = recoveries[(log_range, max(DISTANCES))].residual(backend)
        floppiest = recoveries[(log_range, min(DISTANCES))].residual(backend)
        assert stiffest < floppiest


@pytest.mark.parametrize("log_range", AMPLITUDES)
def test_the_correction_restores_the_reference_volume(recoveries, log_range):
    """End to end at the shipped protocol, in the terms a user would put it in.

    Correcting the artificial volume has to bring it back towards the volume
    it was made from.  Asserted as a bare ordering, with no factor: at 200 mm
    it in fact removes three quarters of the deviation at 20% and seven
    eighths at 40%.

    Only at 200 mm, because it is not true at every spacing -- see the next
    test, which is what that costs.
    """
    recovery = recoveries[(log_range, DEFAULT_DISTANCE)]

    before = recovery.distance_from_reference(recovery.reference
                                              * recovery.planted)
    after = recovery.distance_from_reference(recovery.corrected)

    assert after < before


@pytest.mark.parametrize("log_range", AMPLITUDES)
def test_too_fine_a_spline_undoes_the_correction(recoveries, log_range):
    """The same thing the residual says, in voxels rather than field.

    At the finest spacing in the sweep the correction leaves the volume
    *further* from the truth than the planted field left it -- 1.10x the
    original deviation at 20%.  N3 removes the field and puts back anatomy it
    mistook for one.

    That is not a defect in this port; it is why ``-distance`` is the knob
    N3 documents most carefully, and why 200 mm is the shipped default.  It is
    asserted here as a bare ordering so that the sweep records the cost rather
    than averaging over it.
    """
    at_default = recoveries[(log_range, DEFAULT_DISTANCE)]
    at_finest = recoveries[(log_range, min(DISTANCES))]

    assert (at_default.distance_from_reference(at_default.corrected)
            < at_finest.distance_from_reference(at_finest.corrected))

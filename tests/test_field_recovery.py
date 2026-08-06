"""Recovery of a planted bias field.

Every other test here asks whether the port reproduces the original N3.  This
one asks the question an application has: given a volume with a known
non-uniformity, does N3 return that non-uniformity?

The substrate is ``brain_nu_ref.mnc``, real anatomy already processed by
``nu_correct``, so the non-uniformity remaining in it is small compared with the
field added here.  A smooth multiplicative field of a set amplitude is applied,
and three implementations correct the result at each of several knot spacings:
the PyTorch blocks, the same pipeline running the original C++ blocks, and the
installed ``nu_correct``, whose answers were recorded once from volumes written
out as ``brain_nu_artificial.mnc`` and are read back from ``tests/reference/``
rather than recomputed.

Comparing the two backends is the most controlled comparison: same pipeline,
same input, only the blocks differ.  ``nu_correct`` is the stronger one, sharing
no code with either.  Sweeping ``-distance`` is what makes either informative:
it is the B-spline block's main parameter, it changes the number of coefficients
five-fold over the range used here, and a fault in knot placement would most
likely appear at some spacings and not others.

**Every run is given a fixed iteration count with the early stop disabled.**
Left alone, N3 stops at ``change < 0.001``, and in four of the six cells here
the two backends land on opposite sides of it and run a different number of
iterations: an entire extra field update, which moves the answer several times
more than any block-level difference.  Holding the count fixed makes this a
comparison of implementations rather than of where a threshold happened to fall,
and improves agreement by up to 8x.

Everything below is computed inside the model mask and nowhere else, which is
also all that is recorded.
"""

import math
from dataclasses import dataclass

import pytest
import torch

from tests.conftest import legacy_data
from tests.inputs import as_stored, synthetic_bias_field
from torch_n3.pipeline import V1_0, nu_estimate

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
DEFAULT_LAMBDA = 1e-7

#: The two block implementations, both driving the same pipeline.
BACKENDS = ["torch", "legacy"]

#: The installed program.  Its answers are recorded, so nothing is run here.
BINARY = "nu_correct"

#: Iterations every implementation runs, with the early stop disabled, so
#: that they all do the same work.  See the module docstring.  Pinned to
#: ``V1_0``'s fwhm/histogram/rounding: ``brain_nu_ref.mnc`` and every
#: recorded answer this file compares against were produced under N3's own
#: protocol, not whatever ``torch_n3.pipeline.DEFAULTS`` currently defaults to.
PROTOCOL = dict(V1_0, iterations=(30,), stop=(0.0,))

#: Bending-energy weights for the trade-off test below.  ``1e-7`` is the
#: shipped default; the others are what a 50 mm spline needs to behave.
LAMBDAS = [1e-5, 1e-4]

#: How closely two implementations must agree on the field they recovered,
#: as a relative RMS difference over the mask.  One number for every pairing,
#: fixed in advance: it is what is being *required*, not what was measured.
#: If it fails, see the note on tolerances in CLAUDE.md -- the first thing to
#: check is whether the two ran the same number of iterations.
AGREEMENT = 1e-3


def non_uniformity(field):
    """How much a field varies, as a coefficient of variation.

    The number N3 exists to reduce; comparing it before and after is how the
    paper reports its own results.
    """
    return float(field.std(unbiased=False) / field.mean())


@dataclass
class Recovery:
    """One planted-field experiment, corrected by every implementation.

    Every tensor here is already restricted to the mask and flattened.
    """

    log_range: float
    distance: float            # the knot spacing every implementation used
    planted: torch.Tensor      # the field that was applied
    baseline: dict             # what each implementation finds in the reference
    recovered: dict            # what each finds after the field was planted
    reference: torch.Tensor    # brain_nu_ref, the starting volume
    corrected: torch.Tensor    # this port's correction of the artificial volume

    def unexplained(self, source):
        """The part of ``source``'s answer that is not the planted field.

        Divided by that implementation's own baseline as well, because the
        reference is not perfectly uniform and N3 removes that too; without
        dividing it out, the comparison would charge this code for
        non-uniformity it corrected successfully.  Renormalised, since only a
        field's shape is significant.
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
        difference, which is an extreme-value statistic over a quarter of a
        million voxels, dominated by a few at the mask edge, and which supports
        no fixed bound.
        """
        difference = self.unexplained(one) - self.unexplained(other)
        return float((difference ** 2).mean().sqrt())


@pytest.fixture(scope="module")
def baseline(legacy_output, tmp_path_factory, brain_reference, model_mask):
    """What each implementation finds in the *untouched* reference volume.

    Not zero: ``brain_nu_ref.mnc`` still carries about 0.5% non-uniformity, each
    implementation estimates it differently, and that estimate depends on the
    knot spacing.  Measuring it once per spacing keeps it out of the comparisons
    below.
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
    :func:`test_a_stiffer_spline_recovers_a_smooth_field_better` compares across
    spacings and would otherwise recompute them all.
    """
    inside = model_mask.resample_like(brain_reference).data != 0
    directory = tmp_path_factory.mktemp("recovery")
    experiments = {}

    for log_range in AMPLITUDES:
        planted = synthetic_bias_field(brain_reference, inside, log_range)

        # ``brain_nu_artificial.mnc``: the same file the recorded run was
        # handed, written the same way, so both see the same quantised
        # intensities.  No legacy program runs; this is the port's own MINC I/O.
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


@pytest.fixture(scope="module")
def regularized(tmp_path_factory, brain_reference, model_mask):
    """Residual left at the *finest* spacing, as the penalty is raised.

    Torch backend only: this is a claim about what N3 does rather than about who
    implements it, and the parity of the two backends is established by the
    sweep above.
    """
    inside = model_mask.resample_like(brain_reference).data != 0
    directory = tmp_path_factory.mktemp("regularized")
    reference = as_stored(directory, "reference.mnc", brain_reference,
                          legacy_data("brain_nu_ref.mnc"))
    distance = min(DISTANCES)
    left = {}

    for log_range in AMPLITUDES:
        planted = synthetic_bias_field(brain_reference, inside, log_range)
        volume = as_stored(directory, "artificial_%d.mnc" % (log_range * 100),
                           brain_reference.like(brain_reference.data * planted),
                           legacy_data("brain_nu_ref.mnc"))
        for lam in [DEFAULT_LAMBDA] + LAMBDAS:
            settings = dict(distance=distance, lam=lam, **PROTOCOL)
            base = nu_estimate(reference, mask=model_mask,
                               **settings).evaluate_on(reference)[inside]
            got = nu_estimate(volume, mask=model_mask,
                              **settings).evaluate_on(volume)[inside]
            ratio = got / base / planted[inside]
            ratio = ratio / ratio.mean()
            left[(log_range, lam)] = float(ratio.std(unbiased=False))
    return left


def _estimate(volume, mask, backend, distance, inside):
    """The field ``backend`` finds in ``volume``, inside the mask."""
    field = nu_estimate(volume, mask=mask, backend=backend, distance=distance,
                        **PROTOCOL)
    return field.evaluate_on(volume)[inside]


def test_the_planted_field_has_the_amplitude_asked_for(recovery):
    """A check on the experiment's setup."""
    field = recovery.planted

    assert math.log(float(field.max() / field.min())) == pytest.approx(
        recovery.log_range, rel=1e-9)
    assert float(field.mean()) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_planted_field_is_recovered(recovery, backend):
    """At the very least, the non-uniformity has to be halved.

    A floor on usefulness rather than a description of performance: a bias
    corrector that leaves more than half of a known field behind is not working,
    whatever the knot spacing.  N3 does considerably better at 200 mm, leaving
    about 7%, and degrades to 24-37% at 50 mm as the spline is given freedom it
    does not need.
    """
    assert recovery.residual(backend) < non_uniformity(recovery.planted) / 2


def test_the_two_backends_recover_the_same_field(recovery):
    """The controlled comparison: same pipeline, same input, other blocks.

    The blocks themselves agree to between 1e-13 and 1e-6 (``test_histogram.py``
    and the other block tests); this asks whether that survives thirty rounds of
    a loop that feeds its own output back in.  What remains is amplification,
    the iteration count being held fixed; see the module docstring.
    """
    assert recovery.disagreement("torch", "legacy") < AGREEMENT


@pytest.mark.parametrize("backend", BACKENDS)
def test_it_recovers_as_much_as_nu_correct_did(recovery, backend):
    """The stronger check: no shared code with either implementation.

    N3 leaves about 0.9% of the field behind on this data whichever
    implementation runs it, so an implementation can be held not to an absolute
    residual but to agreement with the original about which field is present.
    """
    assert recovery.disagreement(backend, BINARY) < AGREEMENT


@pytest.mark.parametrize("backend", BACKENDS + [BINARY])
def test_a_stiffer_spline_recovers_a_smooth_field_better(recoveries, backend):
    """At the default regularization, freedom the field does not need hurts.

    The planted field is smooth by construction, as a real one is, so the
    stiffest spline in the sweep can already represent it.  Additional
    coefficients cannot help and measurably hurt: the extra degrees of freedom
    follow tissue contrast, which is returned as field that was never planted.

    **At the default lambda.**  This is a property of leaving ``-lambda`` at
    ``1e-7`` while ``-distance`` shrinks, not a property of B-splines; the next
    test shows the same fit recovering once the penalty is raised to match.  The
    sweep here runs at the default because that is what a user gets.

    Stated as a bare ordering between the ends of the sweep, with no factor to
    choose.  It is not monotone at every step and is not asserted to be.
    """
    for log_range in AMPLITUDES:
        stiffest = recoveries[(log_range, max(DISTANCES))].residual(backend)
        floppiest = recoveries[(log_range, min(DISTANCES))].residual(backend)
        assert stiffest < floppiest


@pytest.mark.parametrize("log_range", AMPLITUDES)
@pytest.mark.parametrize("lam", LAMBDAS)
def test_more_regularization_recovers_what_a_finer_spline_lost(
        regularized, log_range, lam):
    """The two knobs trade off, so the loss above is not the spline's fault.

    ``-distance`` sets how many coefficients describe the field; ``-lambda``
    sets how much bending is permitted between them.  Halving the spacing
    without touching the penalty spends the extra freedom on anatomy; raising
    the penalty to match recovers the fit.  Roughly a decade of lambda per
    halving of distance, on this data:

    ========  ==========  =========  =========
    residual  d = 200 mm  d = 100 mm  d = 50 mm
    ========  ==========  =========  =========
    1e-7        0.0031      0.0061     0.0151
    1e-6        0.0013      0.0022     0.0101
    1e-5        0.0033      0.0017     0.0025
    1e-4        0.0085      0.0054     0.0035
    ========  ==========  =========  =========

    Asserted as a bare ordering, at the finest spacing, for every weight above
    the default rather than for one selected value.
    """
    assert regularized[(log_range, lam)] < regularized[(log_range, DEFAULT_LAMBDA)]


@pytest.mark.parametrize("log_range", AMPLITUDES)
def test_the_correction_restores_the_reference_volume(recoveries, log_range):
    """End to end at the shipped protocol, in the terms a user would put it in.

    Correcting the artificial volume must bring it back towards the volume it
    was made from.  Asserted as a bare ordering, with no factor: at 200 mm it
    removes three quarters of the deviation at 20% and seven eighths at 40%.

    Only at 200 mm, because it does not hold at every spacing; the next test
    records what that costs.
    """
    recovery = recoveries[(log_range, DEFAULT_DISTANCE)]

    before = recovery.distance_from_reference(recovery.reference
                                              * recovery.planted)
    after = recovery.distance_from_reference(recovery.corrected)

    assert after < before


@pytest.mark.parametrize("log_range", AMPLITUDES)
def test_too_fine_a_spline_undoes_the_correction(recoveries, log_range):
    """The same thing the residual says, in voxels rather than field.

    At the finest spacing in the sweep, and the default ``-lambda``, the
    correction leaves the volume *further* from the truth than the planted
    field left it -- 1.10x the original deviation at 20%.  N3 removes the
    field and puts back anatomy it mistook for one.

    This is not a defect in the port; it is why ``-distance`` is the parameter
    N3 documents most carefully, and why 200 mm is the shipped default.
    Asserted as a bare ordering so the sweep records the cost rather than
    averaging over it.  It is recoverable; see the next test.
    """
    at_default = recoveries[(log_range, DEFAULT_DISTANCE)]
    at_finest = recoveries[(log_range, min(DISTANCES))]

    assert (at_default.distance_from_reference(at_default.corrected)
            < at_finest.distance_from_reference(at_finest.corrected))

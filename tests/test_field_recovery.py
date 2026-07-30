"""Can it recover a bias field we planted ourselves?

Every other test here asks whether the port reproduces the original N3.  This
one asks the question a user actually has: given a volume with a known
non-uniformity, does N3 hand the non-uniformity back?

The substrate is ``brain_nu_ref.mnc.gz`` -- real anatomy that has already been
through ``nu_correct``, so what little non-uniformity remains in it is small
compared with what we are about to add.  A smooth multiplicative field of a
set amplitude goes on top, the result is written out as
``brain_nu_artificial.mnc``, and both implementations are asked to correct it.

Two things are then checked: that the recovered field tracks the planted one,
and that the port recovers *as much of it* as the installed ``nu_correct``
does.  The second matters more than the first.  N3 does not recover a field
perfectly -- it leaves about 0.9% here, whoever runs it -- so an absolute
tolerance would be a statement about N3, not about this code; running the
original on the same file says exactly how much of the residual is ours.
"""

import math
from dataclasses import dataclass

import pytest
import torch

from tests.conftest import Workspace, legacy_data
from torch_n3.pipeline import nu_estimate
from torch_n3.volume import Volume, load_volume

#: Log peak-to-peak amplitude of the planted fields.  A field spanning ``0.2``
#: in the log domain ranges over a factor of ``exp(0.2) = 1.22``, i.e. roughly
#: +-10% about its mean -- what the BrainWeb phantoms call "20% RF", and about
#: what a 1.5 T head coil produces.  ``0.4`` is their harder case.
AMPLITUDES = [0.2, 0.4]


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
    """One planted-field experiment, corrected by both implementations."""

    log_range: float
    inside: torch.Tensor
    planted: torch.Tensor      # the field we put on
    residual: torch.Tensor     # what N3 finds in the untouched reference
    ours: torch.Tensor         # the field torch_n3 estimated
    theirs: torch.Tensor       # the field the installed nu_correct estimated
    reference: Volume          # brain_nu_ref, the volume we started from
    corrected: Volume          # our correction of the artificial volume

    def unexplained(self, field):
        """The part of ``field`` that is not the planted field.

        Divided by ``residual`` as well, because the reference is not perfectly
        uniform to begin with and N3 removes that too; without dividing it out
        the comparison would charge this code for non-uniformity it corrected
        successfully.  Renormalised, since only the shape of a field means
        anything.
        """
        ratio = (field / self.residual / self.planted)[self.inside]
        return ratio / ratio.mean()


@pytest.fixture(scope="module", params=AMPLITUDES,
                ids=["%d%%" % (a * 100) for a in AMPLITUDES])
def recovery(request, tmp_path_factory, brain_reference, model_mask):
    log_range = request.param
    workspace = Workspace(tmp_path_factory.mktemp("recovery"))
    reference = brain_reference
    inside = model_mask.resample_like(reference).data != 0

    planted = synthetic_bias_field(reference, inside, log_range)

    # Write the artificial volume as a real MINC file, in the reference's own
    # 16-bit storage, so that both implementations read exactly the same
    # numbers -- otherwise the comparison would be partly about quantisation.
    artificial = workspace.write("brain_nu_artificial.mnc",
                                 reference.like(reference.data * planted),
                                 like=legacy_data("brain_nu_ref.mnc.gz"),
                                 store_dtype="int16")
    volume = load_volume(artificial)

    mask = workspace.write("mask.mnc", reference.like(inside.to(torch.float64)),
                           store_dtype="int16")
    workspace.run("nu_correct", "-clobber", "-quiet", "-mapping_dir",
                  workspace.at(""), "-mask", mask, artificial,
                  workspace.at("nu.mnc"))
    theirs = volume.data / workspace.read("nu.mnc").data

    field = nu_estimate(volume, mask=model_mask)
    ours = field.evaluate_on(volume)

    return Recovery(log_range=log_range, inside=inside, planted=planted,
                    residual=nu_estimate(reference,
                                         mask=model_mask).evaluate_on(reference),
                    ours=ours, theirs=theirs, reference=reference,
                    corrected=volume.like(volume.data / ours))


def test_the_planted_field_has_the_amplitude_asked_for(recovery):
    """The experiment is only as good as its setup, so check the setup."""
    field = recovery.planted[recovery.inside]

    assert math.log(float(field.max() / field.min())) == pytest.approx(
        recovery.log_range, rel=1e-9)
    assert float(field.mean()) == pytest.approx(1.0, rel=1e-9)


def test_the_planted_field_is_recovered(recovery):
    """Most of the non-uniformity we put in comes back out."""
    planted = non_uniformity(recovery.planted, recovery.inside)
    left_over = float(recovery.unexplained(recovery.ours).std(unbiased=False))

    assert left_over < planted / 4


def test_it_recovers_as_much_as_the_installed_nu_correct(recovery):
    """The comparison that means something: the same field, to a few 1e-5.

    N3 leaves about 0.9% of the field behind on this data no matter who runs
    it, so what a port can be held to is not an absolute residual but whether
    it agrees with the original about which field is there.
    """
    ours = recovery.unexplained(recovery.ours)
    theirs = recovery.unexplained(recovery.theirs)

    assert float(ours.std(unbiased=False)) <= 1.05 * float(
        theirs.std(unbiased=False))

    difference = ours - theirs
    assert float((difference ** 2).mean().sqrt()) < 1e-3
    assert float(difference.abs().max()) < 5e-3


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

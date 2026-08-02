"""What a simulated bias-field experiment is made of.

Everything the driver needs to build one trial and score it: a random smooth
field, Gaussian noise at a stated SNR, and the residual non-uniformity left
after the pipeline has had its go.  No files are written here and nothing is
printed; :mod:`experiments.recovery` does both.

Two things are deliberately *not* shared with ``tests/``:

* the random generators, because ``tests/inputs.py`` says in its own docstring
  that it holds no random numbers: a recorded answer is usable only if
  the question can be asked again exactly, and these questions cannot;
* the file round trip (``tests.inputs.as_stored``), because that exists so a
  test sees the volume quantised the way ``nu_correct`` saw it through a
  16-bit MINC file.  There is no legacy oracle in this experiment, so
  everything stays float64 in memory.

The score, on the other hand, *is* the one ``tests/tables.py`` publishes, so
the two experiments can be read against each other.
"""

import math
import os

import torch

from torch_n3.blocks.spline import BSplineField
from torch_n3.pipeline import DEFAULTS
from torch_n3.volume import load_volume

#: Where the colin27 volumes live.  Not checked in -- ``.gitignore`` excludes
#: ``*.mnc`` -- see ``experiments/README.md`` for where to get them.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT = os.path.join(DATA, "colin27_t1_tal_lin.mnc")
HEAD_MASK = os.path.join(DATA, "colin27_t1_tal_lin_headmask.mnc")
BRAIN_MASK = os.path.join(DATA, "colin27_t1_tal_lin_mask.mnc")

#: Shortest wavelength, in mm, of the planted field.  A receive coil's
#: sensitivity varies on the scale of the coil, so the field a scanner puts on
#: a head is smooth over hundreds of mm; ``tests.inputs.synthetic_bias_field``
#: is ``cos(0.9u)`` across a normalised axis, about 620 mm on this volume.
#: Below ~300 mm the planted field stops being something a B-spline at any of
#: the shipped knot spacings can represent, and every cell of the sweep
#: measures representation error instead of what it set out to measure.
FIELD_SCALE = 400.0

#: How many cosine waves are summed.  Enough that no single one dominates the
#: shape, few enough that a seed still produces a recognisably smooth field.
FIELD_TERMS = 8

#: Added to a trial's seed to make the noise draw, so that the field and the
#: noise of one trial are independent and either can be reproduced alone.
NOISE_SEED_OFFSET = 1000000


def load_experiment(input_path=INPUT, mask_path=HEAD_MASK, device=None):
    """The volume, its mask, and the mask as a boolean on the volume's grid."""
    volume = load_volume(input_path)
    mask = load_volume(mask_path)
    if device:
        volume, mask = volume.to(device), mask.to(device)
    return volume, mask, mask.resample_like(volume).data != 0


def random_bias_field(volume, inside, log_range, seed, scale=FIELD_SCALE,
                      terms=FIELD_TERMS):
    """A random smooth field of exactly ``log_range`` log peak-to-peak.

    A sum of ``terms`` cosine waves in *world* coordinates, with directions
    uniform on the sphere, wavelengths no shorter than ``scale`` mm, random
    phases, and amplitudes falling off with frequency.  Written in mm rather
    than in normalised axes so that the difficulty of a trial is a property of
    the field rather than of how big the volume happens to be.

    Deliberately not a B-spline: N3's basis should have to approximate this,
    not reproduce it.  Normalised to mean 1 inside ``inside``, since a bias
    field is only ever defined up to a global scale -- the same convention as
    ``tests.inputs.synthetic_bias_field``.

    The draw happens on the CPU whatever device ``volume`` is on, so a seed
    means the same field everywhere.
    """
    generator = torch.Generator().manual_seed(int(seed))
    axes = _world_axes(volume)

    shape = torch.zeros([len(axis) for axis in axes], dtype=torch.float64)
    for _ in range(terms):
        direction = torch.randn(3, generator=generator, dtype=torch.float64)
        direction = direction / direction.norm()
        frequency = float(torch.rand((), generator=generator,
                                     dtype=torch.float64)) / scale
        phase = float(torch.rand((), generator=generator,
                                 dtype=torch.float64)) * 2.0 * math.pi
        amplitude = float(torch.randn((), generator=generator,
                                      dtype=torch.float64))

        projection = torch.zeros_like(shape)
        for index, (component, axis) in enumerate(zip(direction, axes)):
            projection = projection + float(component) * axis.reshape(
                _broadcast(index, len(axes)))

        # Lower frequencies get more weight, so the shape is dominated by the
        # smooth part however the wavelengths happen to fall.
        wave = torch.cos(2.0 * math.pi * frequency * projection + phase)
        shape = shape + (amplitude / (1.0 + scale * frequency)) * wave

    shape = shape.to(volume.data.device)
    spread = float(shape[inside].max() - shape[inside].min())
    field = torch.exp(shape * (log_range / spread))
    return field / field[inside].mean()


def noise_sigma(volume, inside, snr):
    """The noise standard deviation that gives ``snr`` inside ``inside``.

    SNR is the mean signal over the mask divided by the noise standard
    deviation, and the mean is taken from the *clean, unbiased* volume so that
    it does not drift with the amplitude of whatever field is planted:  SNR 20
    means the same thing in every cell of the sweep.  ``inf`` means no noise.
    """
    if math.isinf(snr):
        return 0.0
    return float(volume.data[inside].mean()) / snr


def add_noise(data, sigma, seed):
    """``data`` plus white Gaussian noise, floored at zero.

    Drawn on the CPU for the same reason the field is.  The floor is what a
    magnitude image does anyway; inside the mask it never bites (the mean is
    242702 and the largest sigma here is 12135), and outside it only keeps
    ``log()`` in the pipeline away from negative numbers.
    """
    if sigma == 0.0:
        return data
    generator = torch.Generator().manual_seed(int(seed))
    draw = torch.randn(tuple(data.shape), generator=generator,
                       dtype=torch.float64)
    return (data + sigma * draw.to(data.device)).clamp(min=0.0)


def unexplained(recovered, baseline, planted):
    """The part of a recovered field that is not the field that was planted.

    All three arguments are already restricted to the mask and flattened.
    Divided by ``baseline`` -- the same configuration's answer on the
    untouched volume -- because colin27 is not perfectly uniform to begin with
    and N3 removes that too; without it this would charge the code for
    non-uniformity it corrected successfully.  Renormalised to mean 1, since
    only a field's shape means anything.

    This is ``tests/tables.py``'s ratio, and :func:`residual_percent` of it is
    the number that file publishes.
    """
    ratio = recovered / baseline / planted
    return ratio / ratio.mean()


def residual_percent(ratio):
    """How much non-uniformity is left, as a percentage.  Lower is better."""
    return 100.0 * float(ratio.std(unbiased=False))


def log_rms(ratio):
    """The same residual in log units: RMS of ``log(ratio)`` about its mean.

    Differences in log intensity are what N3 actually works in, and unlike the
    percentage above this one does not care that the ratio was normalised.
    """
    logs = torch.log(ratio)
    return float((logs - logs.mean()).pow(2).mean().sqrt())


def non_uniformity_percent(field):
    """A field's coefficient of variation -- what was there to remove."""
    return 100.0 * float(field.std(unbiased=False) / field.mean())


def estimation_grid(volume, mask, shrink, background=DEFAULTS["background"]):
    """The grid and mask ``nu_estimate`` will work on, without running it.

    Mirrors ``pipeline.nu_estimate`` steps 1 and 3 (``pipeline.py:83-92``).
    Only :func:`basis_floor` needs this, and only so that it fits the planted
    field over exactly the samples the pipeline would have fitted it over.
    """
    grid = volume if shrink == 1 else volume.shrink(shrink)
    inside = grid.data > background
    if mask is not None:
        inside &= mask.resample_like(grid).data != 0
    return grid, inside


def basis_spline(volume, mask, planted, distance, lam, solver,
                 shrink=DEFAULTS["shrink"]):
    """``planted`` fitted by the spline the pipeline ends with.

    Same grid, same knot spacing, same weight, same solver, same samples as
    step 7 of ``nu_estimate`` (``pipeline.py:118-124``) -- so the difference
    between this and ``planted`` is representation error: the part of the field
    this basis cannot express at all, no matter how well the rest of the
    pipeline estimates it.

    The fit is of the field, not of its log, because that is what step 7 fits.

    This is also the *oracle estimator* of ``experiments.recovery``: an
    estimator handed the answer, which then does nothing but express it in the
    basis.  Nothing that reads a voxel intensity can do better, so what it
    scores is the ceiling for a given ``distance`` and ``lam``.
    """
    grid, inside_grid = estimation_grid(volume, mask, shrink)
    return BSplineField(grid, distance, lam, solver=solver).fit(
        _resample(planted, volume, grid), inside_grid)


def basis_field(volume, mask, planted, distance, lam, solver,
                shrink=DEFAULTS["shrink"]):
    """:func:`basis_spline`, evaluated back on ``volume``'s grid."""
    return basis_spline(volume, mask, planted, distance, lam, solver,
                        shrink).evaluate_on(volume)


def basis_floor(volume, mask, planted, inside, distance, lam, solver,
                shrink=DEFAULTS["shrink"]):
    """The best score the basis could have got on this field, over ``inside``.

    :func:`basis_field` scored the way a trial is scored, against a flat
    baseline.  A trial sitting near its floor is limited by the basis; one far
    above it is limited by the estimation.
    """
    return score(basis_field(volume, mask, planted, distance, lam, solver,
                             shrink), torch.ones_like(planted), planted,
                 inside)[0]


def score(recovered, baseline, planted, region):
    """``(unexplained_percent, log_rms)`` over one region.

    All three fields are given on the whole grid and restricted here, so that
    a single estimate can be scored over several regions -- the head the
    estimation ran in, and the brain it is actually meant to help.
    """
    ratio = unexplained(recovered[region], baseline[region], planted[region])
    return residual_percent(ratio), log_rms(ratio)


def _resample(field, volume, grid):
    """``field``, given on ``volume``'s grid, sampled onto ``grid``."""
    if grid is volume:
        return field
    return volume.like(field).resample_like(grid).data


def _world_axes(volume):
    """World coordinates in mm along each data axis, slowest first."""
    return [torch.arange(length, dtype=torch.float64) * float(step)
            + float(start)
            for length, start, step in zip(volume.shape, volume.start,
                                           volume.step)]


def _broadcast(index, rank):
    """A shape that broadcasts a 1-D axis along data axis ``index``."""
    return [-1 if axis == index else 1 for axis in range(rank)]

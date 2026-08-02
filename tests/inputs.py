"""The inputs the original N3 programs were run on.

The suite does not run those programs any more -- their answers are recorded
in ``tests/reference/`` and ``tests/regenerate_reference.py`` puts them there.
For that to mean anything, the input a test uses has to be the input the
program was given, so both sides build it from here.

Everything in this module is a deterministic function of the volumes in
``legacy/N3/testing/``: no random numbers, because a stored answer is only
usable only if the question can be posed again exactly.
"""

import os

import torch

from torch_n3.volume import load_volume, save_volume

#: The protocol ``tests/data/brain_nu_ref_legacy.mnc`` was produced with, and
#: the only one it means anything at.  One iteration with the early stop
#: disabled: every stage of the pipeline runs, and the whole of
#: ``nu_evaluate``, while the answer is still a continuous function of
#: rounding error.  ``tests/test_reproducibility.py`` explains what happens
#: after that, and why this used to say two.
PLATFORM_PROTOCOL = dict(iterations=(1,), stop=(0.0,))

#: The protocol ``tests/data/brain_nu_ref_legacy_30.mnc`` was produced with:
#: thirty iterations, again with the early stop disabled so that every run
#: does the same work.  This one is past the divergence threshold by design:
#: it is the converged pipeline, and no two builds agree on it to better than
#: about a part in a thousand.  It is a coarse regression net rather than a
#: sensitive check, and the bound it is held to states as much.  See ``tests/test_reproducibility.py``.
CONVERGED_PROTOCOL = dict(iterations=(30,), stop=(0.0,))


def as_stored(directory, name, volume, like):
    """``volume`` after a round trip through a 16-bit MINC file.

    ``nu_correct`` was handed a file, not an array, so it saw its contents
    quantised.  The regeneration script and the tests both come through here,
    which is the only way to be sure they are looking at the same numbers --
    modelling MINC's scaling in Python instead gets it wrong by a whole step.
    """
    path = os.path.join(str(directory), name)
    save_volume(path, volume, like=like, store_dtype="int16")
    return load_volume(path)


def masked_log(volume, mask):
    """The masked log volume the estimation loop actually works on.

    Returns ``(values, inside)``.  This is step 2 and 3 of
    ``nu_estimate_np_and_em.in``: clamp, take the log, and zero everything
    outside the mask.
    """
    inside = mask.data != 0
    values = torch.log(volume.data.clamp(min=1.0))
    return torch.where(inside, values, torch.zeros_like(values)), inside


def probe_values(volume, mask, stride=47):
    """Every ``stride``-th masked voxel, as a flat list of intensities.

    ``minclookup`` maps a volume, but what is being checked is an
    interpolation rule, and a few thousand real intensities spread across the
    range test that as well as a quarter of a million do -- at a fraction of
    the size on disk.
    """
    values, inside = masked_log(volume, mask)
    return values[inside][::stride].contiguous()


def smooth_bumps(volume, inside):
    """A masked field with structure at several scales, as an iteration makes.

    Analytic rather than random: ``spline_smooth`` was run on exactly this,
    once, and its answer is on disk.
    """
    z, y, x = _voxel_grid(volume)
    values = (0.05 * torch.cos(z / 7.0) - 0.03 * torch.sin(y / 5.0)
              + 0.02 * torch.cos(x / 11.0 + 0.7)
              + 0.004 * torch.sin(z / 1.7) * torch.cos(y / 2.3))
    return torch.where(inside, values, torch.zeros_like(values))


def tilted_plane(volume, inside=None, offset=1.0, slopes=(0.01, -0.004, 0.003)):
    """A plane across the voxel grid, optionally zeroed outside ``inside``.

    Stands in for a field: smooth, and simple enough that a reader can tell at
    a glance what the program under test should do with it.
    """
    z, y, x = _voxel_grid(volume)
    plane = offset + slopes[0] * z + slopes[1] * y + slopes[2] * x
    if inside is None:
        return plane
    return torch.where(inside, plane, torch.zeros_like(plane))


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


def two_tissue_histogram(bins=200, value_range=(4.0, 6.0)):
    """Two overlapping tissue peaks in log intensity, as a histogram.

    A clean stand-in for the thing ``sharpen_hist`` exists to deconvolve, with
    no volume behind it.
    """
    centres = torch.linspace(value_range[0], value_range[1], bins,
                             dtype=torch.float64)
    return sum(torch.exp(-0.5 * ((centres - peak) / 0.25) ** 2) * height
               for peak, height in [(4.6, 3000.0), (5.4, 5000.0)])


def _voxel_grid(volume):
    return torch.meshgrid(*[torch.arange(n, dtype=torch.float64)
                            for n in volume.shape], indexing="ij")

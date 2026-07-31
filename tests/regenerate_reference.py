"""Run the original N3 programs once and record what they said.

    python3 -m tests.regenerate_reference

Needs the MINC toolkit and the N3 programs on ``PATH``; nothing else in the
suite does.  The programs are deterministic, so running this again when
nothing has changed leaves ``git diff`` empty -- which is the check that the
recorded answers are still their answers.

Every case here is the input side of one test.  If you add a comparison
against a legacy program, add the case here and read it back through the
``legacy_output`` fixture rather than shelling out from the test.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

from tests import inputs, reference
from tests.conftest import MODEL_MASK, legacy_data
from torch_n3 import blocks
from torch_n3.volume import Volume, load_volume, save_volume

#: Everything this script needs to be able to run.
PROGRAMS = ["volume_hist", "sharpen_hist", "sharpen_volume", "spline_smooth",
            "evaluate_field", "correct_field", "minclookup", "mincstats",
            "mincresample", "resample_labels", "nu_correct"]

#: The two planted-field amplitudes ``test_field_recovery`` uses.
AMPLITUDES = [0.2, 0.4]

#: The domain of the synthetic histogram ``test_sharpen`` works on.
TWO_TISSUE_RANGE = (4.0, 6.0)


def main():
    missing = [p for p in PROGRAMS if shutil.which(p) is None]
    if missing:
        sys.exit("not on PATH: %s" % ", ".join(missing))

    workspace = Workspace(pathlib.Path(
        tempfile.mkdtemp(prefix="n3_reference_")))
    arrays, scalars = {}, {}

    for case in (histogram_cases, lookup_cases, resampling_cases,
                 pipeline_cases, recovery_cases):
        case(workspace, arrays, scalars)

    reference.save(arrays, scalars, produced_by=_versions())

    print("wrote %s" % reference.ARRAYS)
    for key in sorted(arrays):
        print("  %-44s %s" % (key, tuple(arrays[key].shape)))
    for key in sorted(scalars):
        print("  %-44s %s" % (key, scalars[key]))


# ------------------------------------------------------------------- the cases

def histogram_cases(workspace, arrays, scalars):
    """``volume_hist`` and ``sharpen_hist``."""
    chunk = load_volume(legacy_data("chunk.mnc"))
    mask = load_volume(legacy_data("chunk_mask.mnc"))

    source = workspace.write("chunk.mnc", chunk)
    masked = workspace.write("chunk_mask.mnc", mask, store_dtype="int16")
    workspace.run("volume_hist", "-bins", 200, "-auto_range", "-mask", masked,
                  "-clobber", "-text", "-select", 1, "-quiet", "-window",
                  source, workspace.at("hist.txt"))
    arrays["volume_hist.chunk"] = torch.as_tensor(_text(workspace.at("hist.txt")))

    # sharpen_hist is driven from a histogram, so record the histogram too:
    # the test then feeds it exactly what the program was given.
    values, inside = inputs.masked_log(chunk, mask)
    value_range = blocks.histogram_range(values[inside],
                                         initial=(values.max(), values.min()))
    counts = blocks.histogram(values[inside], 200, value_range)
    arrays["sharpen_hist.chunk_counts"] = counts
    arrays["sharpen_hist.chunk_range"] = torch.tensor(value_range)
    arrays["sharpen_hist.chunk_lut"] = torch.as_tensor(
        _sharpen_hist(workspace, counts, value_range))

    # The same program on a clean two-peak histogram, for the block test.
    two_tissues = inputs.two_tissue_histogram()
    arrays["sharpen_hist.two_tissues_lut"] = torch.as_tensor(
        _sharpen_hist(workspace, two_tissues, TWO_TISSUE_RANGE))


def lookup_cases(workspace, arrays, scalars):
    """``minclookup -continuous`` and ``mincstats -biModalT``."""
    chunk = load_volume(legacy_data("chunk.mnc"))
    mask = load_volume(legacy_data("chunk_mask.mnc"))

    # A few thousand real intensities, laid out as a volume so that minclookup
    # will accept them.  The table is recorded alongside, so the test does not
    # have to reproduce the code that built it.
    probes = inputs.probe_values(chunk, mask)
    if "sharpen_hist.chunk_lut" not in arrays:
        raise RuntimeError("histogram_cases has to run first: it builds the table")
    table = arrays["sharpen_hist.chunk_lut"]
    value_range = tuple(float(v) for v in arrays["sharpen_hist.chunk_range"])

    arrays["minclookup.values"] = probes
    arrays["minclookup.lut"] = table
    arrays["minclookup.range"] = torch.tensor(value_range)
    arrays["minclookup.mapped"] = _minclookup(workspace, probes, table,
                                              value_range)

    source = workspace.write("chunk.mnc", chunk)
    scalars["mincstats.bimodal_threshold_chunk"] = float(
        workspace.run("mincstats", "-quiet", "-biModalT", source))


def resampling_cases(workspace, arrays, scalars):
    """``mincresample -nearest_neighbour`` and ``resample_labels``."""
    chunk = load_volume(legacy_data("chunk.mnc"))
    model_mask = load_volume(MODEL_MASK)

    shrunk = chunk.shrink(3)
    source = workspace.write("chunk.mnc", chunk)
    workspace.run("mincresample", "-nearest_neighbour", "-clobber",
                  "-nelements", *[str(n) for n in reversed(shrunk.shape)],
                  "-step", *[repr(float(s)) for s in reversed(shrunk.step)],
                  source, workspace.at("shrunk.mnc"))
    resampled = workspace.read("shrunk.mnc")
    arrays["mincresample.chunk_shrink3"] = reference.as_volume(resampled.data)
    scalars["mincresample.chunk_shrink3_start"] = [float(v) for v in resampled.start]
    scalars["mincresample.chunk_shrink3_step"] = [float(v) for v in resampled.step]

    grid = chunk.shrink(4)
    mask = workspace.write("model_mask.mnc", model_mask, store_dtype="int16")
    like = workspace.write("grid.mnc", grid)
    workspace.run("resample_labels", "-clobber", "-quiet", "-resample",
                  "-like %s" % like, mask, workspace.at("labels.mnc"))
    arrays["resample_labels.model_mask_on_chunk_shrink4"] = reference.as_volume(
        workspace.read("labels.mnc").data)


def pipeline_cases(workspace, arrays, scalars):
    """One stage at a time, and then ``nu_correct`` itself."""
    chunk = load_volume(legacy_data("chunk.mnc"))
    chunk_mask = load_volume(legacy_data("chunk_mask.mnc"))
    values, inside = inputs.masked_log(chunk, chunk_mask)

    mask = workspace.write("mask.mnc", chunk.like(inside.to(torch.float64)),
                           store_dtype="int16")

    source = workspace.write("log.mnc", chunk.like(values))
    workspace.run("sharpen_volume", "-parzen", "-bins", 200, "-fwhm", 0.15,
                  "-noise", 0.01, "-clobber", "-quiet", mask, source,
                  workspace.at("sharp.mnc"))
    arrays["sharpen_volume.chunk"] = reference.as_volume(
        workspace.read("sharp.mnc").data)

    bumpy = workspace.write("bumpy.mnc",
                            chunk.like(inputs.smooth_bumps(chunk, inside)))
    workspace.run("spline_smooth", "-full_support", "-clobber", "-quiet",
                  "-distance", 200, "-b_spline", "-lambda", 1e-7,
                  "-subsample", 1, "-mask", mask, bumpy,
                  workspace.at("residue.mnc"))
    arrays["spline_smooth.chunk"] = reference.as_volume(
        workspace.read("residue.mnc").data)

    arrays["evaluate_field.chunk"] = _evaluate_field(workspace, chunk,
                                                     chunk_mask, inside)

    field = inputs.tilted_plane(chunk, inside)
    plane = workspace.write("field.mnc", chunk.like(field))
    float_mask = workspace.write("float_mask.mnc",
                                 chunk.like(inside.to(torch.float64)))
    workspace.run("correct_field", plane, float_mask,
                  workspace.at("extended.mnc"))
    arrays["correct_field.chunk"] = reference.as_volume(
        workspace.read("extended.mnc").data)

    stored = workspace.write("chunk_short.mnc", chunk, store_dtype="int16")
    for iterations, shrink, fwhm in [(1, 3, 0.2), (3, 4, 0.15)]:
        workspace.run("nu_correct", "-clobber", "-quiet", "-mapping_dir",
                      workspace.at(""), "-fwhm", fwhm, "-shrink", shrink,
                      "-stop", 0.001, "-iterations", iterations, "-mask", mask,
                      stored, workspace.at("nu.mnc"))
        arrays["nu_correct.chunk_i%d_s%d" % (iterations, shrink)] = \
            reference.as_volume(workspace.read("nu.mnc").data)


def recovery_cases(workspace, arrays, scalars):
    """``nu_correct`` on the planted-field volumes, inside the mask only.

    ``test_field_recovery`` compares fields over the model mask and nowhere
    else, so only those voxels are recorded -- a quarter of the volume.
    """
    brain = load_volume(legacy_data("brain_nu_ref.mnc"))
    model_mask = load_volume(MODEL_MASK)
    inside = model_mask.resample_like(brain).data != 0

    mask = workspace.write("brain_mask.mnc", brain.like(inside.to(torch.float64)),
                           store_dtype="int16")

    volumes = {"reference": brain.data}
    for amplitude in AMPLITUDES:
        planted = inputs.synthetic_bias_field(brain, inside, amplitude)
        volumes["planted_%d" % (amplitude * 100)] = brain.data * planted

    for name, data in volumes.items():
        stored = inputs.as_stored(workspace.path, "%s.mnc" % name,
                                  brain.like(data),
                                  legacy_data("brain_nu_ref.mnc"))
        path = workspace.at("%s.mnc" % name)
        workspace.run("nu_correct", "-clobber", "-quiet", "-mapping_dir",
                      workspace.at(""), "-mask", mask, path,
                      workspace.at("nu_%s.mnc" % name))
        # The field it divided out.  Derived from the *stored* volume, which
        # is what the test rebuilds, so quantisation sits on the same side of
        # the comparison in both places.
        field = stored.data / workspace.read("nu_%s.mnc" % name).data
        arrays["nu_correct.%s_field" % name] = reference.as_volume(field[inside])


# ------------------------------------------------------------------- the tools

class Workspace:
    """A temporary directory plus the verbs this script needs there.

    File arguments are always absolute: several of the legacy programs derive
    their own temporary file names from the output path, and do not
    consistently resolve a relative one against the working directory.
    """

    def __init__(self, path):
        self.path = path

    def at(self, name):
        """The absolute path of ``name`` inside the workspace."""
        return str(self.path / name)

    def write(self, name, volume, like=None, store_dtype="float64"):
        """Put ``volume`` on disk where a legacy program can read it."""
        target = self.at(name)
        save_volume(target, volume, like=like, store_dtype=store_dtype)
        return target

    def read(self, name):
        return load_volume(self.at(name))

    def run(self, *command):
        """Run an installed N3/MINC program, failing loudly if it does."""
        result = subprocess.run([str(c) for c in command], cwd=str(self.path),
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit("%s failed (%d)\n%s\n%s"
                             % (command[0], result.returncode,
                                result.stdout[-2000:], result.stderr[-2000:]))
        return result.stdout


def _sharpen_hist(workspace, counts, value_range):
    centres = blocks.bin_centers(counts.numel(), value_range)
    with open(workspace.at("hist.in"), "w") as handle:
        handle.write("#  domain: %.15g  %.15g\n" % value_range)
        for centre, count in zip(centres, counts):
            handle.write("  %.15g       %.15g\n" % (centre, count))
    workspace.run("sharpen_hist", "-clobber", "-fwhm", 0.15, "-noise", 0.01,
                  "-quiet", workspace.at("hist.in"), workspace.at("hist.out"))
    return _text(workspace.at("hist.out"))[:, 1]


def _minclookup(workspace, probes, table, value_range):
    """Push ``probes`` through ``table`` with ``minclookup -continuous``."""
    strip = Volume(probes.reshape(1, 1, -1), start=(0.0, 0.0, 0.0),
                   step=(1.0, 1.0, 1.0))
    source = workspace.write("probes.mnc", strip)

    positions = torch.linspace(0.0, 1.0, table.numel(), dtype=torch.float64)
    with open(workspace.at("table.txt"), "w") as handle:
        for position, value in zip(positions, table):
            handle.write("%.15g  %.15g\n" % (position, value))

    workspace.run("minclookup", "-continuous", "-clobber", "-range",
                  repr(value_range[0]), repr(value_range[1]), "-lookup_table",
                  workspace.at("table.txt"), source, workspace.at("mapped.mnc"))
    return workspace.read("mapped.mnc").data.reshape(-1)


def _evaluate_field(workspace, chunk, chunk_mask, inside):
    """``spline_smooth -compact`` then ``evaluate_field``: the .imp round trip."""
    grid = chunk.shrink(4)
    coarse_inside = chunk_mask.resample_like(grid).data != 0
    field = inputs.tilted_plane(grid, offset=1.0,
                                slopes=(0.01, -0.005, 0.002))

    source = workspace.write("coarse_field.mnc", grid.like(field))
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
    workspace.run("evaluate_field", "-clobber", "-mask", fine_mask, "-like",
                  fine, workspace.at("field.imp"), workspace.at("evaluated.mnc"))
    return reference.as_volume(workspace.read("evaluated.mnc").data)


def _text(path):
    """A legacy program's two-column text output, comments and all."""
    return np.loadtxt(path)


def _versions():
    """What produced these numbers, for the record."""
    version = subprocess.run(["mincinfo", "-version"], capture_output=True,
                             text=True).stdout.strip()
    return version.replace("\n", "; ") or "unknown MINC build"


if __name__ == "__main__":
    main()

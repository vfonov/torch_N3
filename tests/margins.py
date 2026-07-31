"""Where every comparison in the suite currently sits against its bound.

    python3 -m tests.margins

Prints the table at the bottom of ``PROBLEMS.md``.  That table is the evidence
for the claims in it -- which bounds are tight enough to flake, which have so
much headroom that they have stopped asking a question -- and it is worth
nothing if nobody can reproduce it, so this is the thing that produces it.

Not a test.  Nothing here asserts; it measures, and the assertions live in the
test modules.  The two are kept in step by hand: when a comparison changes, its
row here has to change with it.  Each row names the test it mirrors, so a
mismatch is visible.

Needs no MINC program -- everything the legacy said is read from
``tests/reference/``, exactly as the tests read it.  It does need the CFFI
shim, since half the rows are parity against it.
"""

import torch

from tests import inputs, reference
from tests.conftest import DATA, MODEL_MASK, relative_rms, span
from tests.inputs import PLATFORM_PROTOCOL
from tests.regenerate_reference import PLATFORM_REFERENCE
from torch_n3 import blocks
from torch_n3.backends import legacy
from torch_n3.blocks.spline import DIRECT_SOLVERS
from torch_n3.minc_tools import apply_lut, bimodal_threshold
from torch_n3.pipeline import (DEFAULTS, _sharpen, _smooth, evaluate_field,
                               nu_correct, nu_estimate)
from torch_n3.volume import load_volume

BACKENDS = ["torch", "legacy"]


def main():
    recorded = reference.load()
    rows = []
    for group in (block_rows, pipeline_rows, platform_rows):
        rows.append(None)                      # a blank line between groups
        rows.extend(group(recorded))

    print("%-42s %-11s %-11s %s"
          % ("comparison", "measured", "bound", "of bound"))
    for row in rows:
        if row is None:
            print()
            continue
        name, measured, bound = row
        print("%-42s %-11.3g %-11.3g %6.1f%%"
              % (name, measured, bound, 100 * measured / bound))


# ------------------------------------------------------------------- the rows

def block_rows(recorded):
    """``test_histogram``, ``test_sharpen``, ``test_minc_tools``, ``test_spline``,
    ``test_field``."""
    chunk = load_volume(DATA + "/chunk.mnc")
    chunk_mask = load_volume(DATA + "/chunk_mask.mnc")
    values, inside = inputs.masked_log(chunk, chunk_mask)
    log_range = blocks.histogram_range(values[inside],
                                       initial=(values.max(), values.min()))
    rows = []

    # test_histogram.py
    for parzen in (True, False):
        ours = blocks.histogram(values[inside], 200, log_range, parzen)
        theirs = legacy.histogram(values[inside], 200, log_range, parzen)
        rows.append(("histogram parzen=%s vs shim" % parzen,
                     _worst(ours, theirs), 1e-9))

    # volume_hist was run on the raw intensities, not the log volume.
    raw = chunk.data[inside]
    raw_range = blocks.histogram_range(
        raw, initial=(chunk.data.max(), chunk.data.min()))
    counts = blocks.histogram(raw, 200, raw_range)
    table = recorded["volume_hist.chunk"]
    rows.append(("histogram vs volume_hist (counts)",
                 _worst(counts, table[:, 1]), 1e-6))
    rows.append(("bin centres vs volume_hist",
                 _worst(blocks.bin_centers(200, raw_range), table[:, 0]), 1e-6))

    # test_sharpen.py
    two_tissues = inputs.two_tissue_histogram()
    for deblur in (False, True):
        ours = blocks.sharpen_lut(two_tissues, (4.0, 6.0), 0.15, 0.01, deblur)
        theirs = legacy.sharpen_lut(two_tissues, (4.0, 6.0), 0.15, 0.01, deblur)
        rows.append(("sharpen_lut deblur=%s vs shim" % deblur,
                     _worst(ours, theirs), 1e-11))
    # ...and on the log histogram the pipeline itself produces.
    log_counts = blocks.histogram(values[inside], 200, log_range)
    rows.append(("sharpen_lut vs shim (real histogram)",
                 _worst(blocks.sharpen_lut(log_counts, log_range, 0.15, 0.01),
                        legacy.sharpen_lut(log_counts, log_range, 0.15, 0.01)),
                 1e-9))
    rows.append(("sharpen_lut vs sharpen_hist",
                 _worst(blocks.sharpen_lut(two_tissues, (4.0, 6.0), 0.15, 0.01),
                        recorded["sharpen_hist.two_tissues_lut"]), 1e-6))

    # test_minc_tools.py
    mapped = apply_lut(recorded["minclookup.values"], recorded["minclookup.lut"],
                       tuple(float(v) for v in recorded["minclookup.range"]))
    rows.append(("apply_lut vs minclookup",
                 _worst(mapped, recorded["minclookup.mapped"]), 1e-9))
    threshold = bimodal_threshold(chunk.data)
    reference_threshold = recorded.scalar("mincstats.bimodal_threshold_chunk")
    rows.append(("bimodal_threshold vs mincstats",
                 abs(threshold - reference_threshold), 1e-4))

    # test_spline.py -- the same bumpy field the test fits.
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)
                               for n in chunk.shape], indexing="ij")
    torch.manual_seed(11)
    bumpy = torch.where(inside,
                        0.05 * torch.cos(z / 7.0) - 0.03 * torch.sin(y / 5.0)
                        + 0.02 * x / chunk.shape[2]
                        + 0.01 * torch.randn(chunk.shape, dtype=torch.float64),
                        torch.zeros_like(z))
    for distance, subsample in [(200.0, 1), (200.0, 2), (50.0, 1)]:
        theirs = legacy.BSplineField(chunk, distance, 1e-7).fit(
            bumpy, inside, subsample).evaluate()
        for solver in DIRECT_SOLVERS:
            ours = blocks.BSplineField(chunk, distance, 1e-7,
                                       solver=solver).fit(
                bumpy, inside, subsample).evaluate()
            rows.append(("spline[%s] d=%-3g sub=%d vs shim"
                         % (solver, distance, subsample),
                         _worst(ours, theirs), 1e-6 * span(ours)))

    # test_the_qr_fit_is_the_same_on_the_gpu, which only runs where there is
    # one.  Only the QR row appears: the normal equations are not held to this
    # bound and never were -- failing it by five orders is the reason the other
    # solver exists -- and a row here means an assertion somewhere.
    # `tests/convergence.py` is where both are measured side by side.
    if torch.cuda.is_available():
        def fitted(grid, data, mask):
            return blocks.BSplineField(grid, 200.0, 1e-7, solver="qr").fit(
                data, mask).evaluate()

        here = fitted(chunk, bumpy, inside)
        there = fitted(chunk.like(chunk.data.cuda()), bumpy.cuda(),
                       inside.cuda()).cpu()
        rows.append(("spline[qr] d=200 cpu vs cuda",
                     _worst(here, there), 1e-11 * span(here)))

    # test_field.py
    plane = inputs.tilted_plane(chunk, inside)
    ours = blocks.correct_field(plane, inside, chunk.step)
    theirs = legacy.correct_field(plane, inside, chunk.step)
    rows.append(("correct_field vs shim", _worst(ours, theirs),
                 1e-4 * span(theirs)))
    rows.append(("correct_field vs binary",
                 _worst(ours, recorded["correct_field.chunk"]),
                 1e-4 * span(recorded["correct_field.chunk"])))

    # test_volume.py
    shrunk = chunk.shrink(3)
    rows.append(("shrink vs mincresample",
                 _worst(shrunk.data, recorded["mincresample.chunk_shrink3"]),
                 span(recorded["mincresample.chunk_shrink3"]) / 4095))
    return rows


def pipeline_rows(recorded):
    """``test_pipeline``: one stage at a time, then the whole thing."""
    chunk = load_volume(DATA + "/chunk.mnc")
    chunk_mask = load_volume(DATA + "/chunk_mask.mnc")
    brain = load_volume(DATA + "/brain.mnc")
    brain_reference = load_volume(DATA + "/brain_nu_ref.mnc")
    model_mask = load_volume(MODEL_MASK)
    values, inside = inputs.masked_log(chunk, chunk_mask)
    rows = []

    sharpened = recorded["sharpen_volume.chunk"]
    sharpened = torch.where(inside, sharpened, torch.zeros_like(sharpened))
    smoothed = recorded["spline_smooth.chunk"]
    bumpy = inputs.smooth_bumps(chunk, inside)

    for backend in BACKENDS:
        options = dict(DEFAULTS, bins=200, backend=backend)
        rows.append(("_sharpen[%s] vs sharpen_volume" % backend,
                     _worst(_sharpen(values, inside, options), sharpened),
                     span(sharpened[inside]) / 65535))
    for backend in BACKENDS:
        options = dict(DEFAULTS, backend=backend)
        rows.append(("_smooth[%s] vs spline_smooth" % backend,
                     _worst(_smooth(bumpy, inside, chunk, options), smoothed),
                     span(smoothed) / 65535))

    for backend in BACKENDS:
        for iterations, shrink, fwhm in [(1, 3, 0.2), (3, 4, 0.15)]:
            corrected = nu_correct(chunk, mask=chunk_mask,
                                   evaluation_mask=chunk_mask, fwhm=fwhm,
                                   shrink=shrink, backend=backend,
                                   iterations=(iterations,), stop=(0.001,))
            rows.append(("nu_correct[%s] chunk i%d s%d"
                         % (backend, iterations, shrink),
                         relative_rms(corrected.data,
                                       recorded["nu_correct.chunk_i%d_s%d"
                                                % (iterations, shrink)]),
                         1e-3))

    for backend in BACKENDS:
        corrected = nu_correct(brain, mask=model_mask, backend=backend)
        rows.append(("nu_correct[%s] vs brain_nu_ref" % backend,
                     relative_rms(corrected.data, brain_reference.data), 1e-2))

    # test_the_iteration_amplifies_small_differences
    def divisor(backend, iterations):
        field = nu_estimate(brain, mask=model_mask, backend=backend,
                            iterations=(iterations,), stop=(0.0,))
        return evaluate_field(brain, field, backend=backend).data

    early = relative_rms(divisor("torch", 1), divisor("legacy", 1))
    late = relative_rms(divisor("torch", 10), divisor("legacy", 10))
    rows.append(("amplification, early (bound is a maximum)", early, 1e-6))
    rows.append(("amplification, late / early (a minimum)", 100.0, late / early))
    return rows


def platform_rows(recorded):
    """``test_reproducibility``: every backend and device, against the file."""
    brain = load_volume(DATA + "/brain.mnc")
    model_mask = load_volume(MODEL_MASK)
    reference_volume = load_volume(DATA + "/" + PLATFORM_REFERENCE).data

    runs = [("torch", "cpu"), ("legacy", "cpu")]
    if torch.cuda.is_available():
        runs.append(("torch", "cuda"))

    rows = []
    for backend, device in runs:
        result = nu_correct(brain.to(device), mask=model_mask.to(device),
                            backend=backend, **PLATFORM_PROTOCOL)
        rows.append(("platform reference, %s/%s" % (backend, device),
                     relative_rms(result.data.cpu(), reference_volume),
                     1 / 65535))
    return rows


# ------------------------------------------------------------------ the tools

def _worst(ours, theirs):
    """The largest absolute difference, whatever the two arrived as."""
    ours = torch.as_tensor(ours, dtype=torch.float64)
    theirs = torch.as_tensor(theirs, dtype=torch.float64)
    return float((ours - theirs).abs().max())


if __name__ == "__main__":
    main()

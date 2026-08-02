"""What ``--denoise`` does to N3, measured against the window it competes with.

Not a test: it asserts nothing and prints tables.  ``--denoise`` is the second
modification this port carries -- ``--parzen-sigma`` is the first -- and like
that one it is off by default, has no oracle, and can only be judged by
measurement.  ``tests/parzen.py`` is its twin and should be read beside it.

**Why the two are measured together.**  ``tests/parzen.py`` establishes that
the histogram window's benefit tracks *noise* rather than field amplitude: with
no noise the windows are within a few percent of each other, while at SNR 20
N3's linear split leaves 4.63% against ``sigma 4``'s 1.71%.  The window
suppresses noise-driven variance in the histogram; this filter suppresses the
same variance in the volume, before the histogram is ever built.  They are
therefore **candidate substitutes, not complements**, and the question this
script exists to answer is whether denoising buys anything over windowing
alone.  Measuring either without the other would answer a question nobody
asked.

**Why noise is planted here and not in ``tests/parzen.py``'s sweep.**  A
noiseless sweep cannot show this filter in a good light and should not be
quoted as if it could: with no noise to remove, a spatial filter can only take
away structure the estimate was using.  A run at ``snr=inf`` is included for
exactly that reason -- it is the control that says how much the filter costs
when there is nothing for it to do.

**What is measured.**  The first table is ``tests/tables.py``'s experiment at
the shipped spacing: plant a smooth field of known amplitude on
``brain_nu_ref.mnc``, add white noise at a stated SNR, correct it, and report
the non-uniformity left in the recovered field once the same configuration's
answer on the untouched reference has been divided out.  Lower is better.
Thirty iterations with the early stop disabled, so that a cell measures the fit
rather than which side of the stopping rule a run fell on.

The ``off``/``linear (N3)``/``snr inf`` cell is ``tables.py``'s published cell
at the same spacing and weight, so a mismatch there means something moved
elsewhere and no other number here should be believed.

**What this cannot tell you.**  One analytic field, on one volume, at one seed.
``tests/parzen.py`` carries the same caveat and ``experiments/`` is the answer
to it: 450 random fields per configuration, which is what a claim about real
data needs.  This is the fast orientation, not the verdict.
"""

import argparse
import contextlib
import io
import math
import sys
import tempfile
import time

import torch

from experiments.simulation import add_noise, noise_sigma
from tests.conftest import MODEL_MASK, legacy_data, relative_rms
from tests.inputs import as_stored, synthetic_bias_field
from tests.tables import PUBLISHED
from torch_n3.blocks.denoise import denoise
from torch_n3.pipeline import DEFAULTS, nu_correct, nu_estimate
from torch_n3.volume import load_volume

#: The planted field, as a log peak-to-peak amplitude.  One amplitude only:
#: the question here is about noise, and ``tests/tables.py`` already sweeps
#: amplitude against the smoothness parameters.
AMPLITUDE = 0.2

#: The shipped spacing and weight.  ``PUBLISHED`` is indexed by lambda and
#: then by position in ``DISTANCES``, so these two must agree with it.
DISTANCE = 200.0
LAMBDA = 1e-7
PUBLISHED_INDEX = 0

#: Signal-to-noise ratios.  ``inf`` is the control: no noise to remove, so
#: whatever the filter costs there it costs for nothing.
SNRS = [math.inf, 40.0, 20.0]

#: Histogram windows, from ``tests/parzen.py``.  ``None`` is N3's own linear
#: split; the two Gaussians are the ones that sweep found useful.
SIGMAS = [None, 2.0, 4.0]

#: Whether the volume is filtered before the estimate.
DENOISING = [False, True]

#: Fixed iteration count with the early stop disabled, as ``tests/tables.py``.
PROTOCOL = dict(iterations=(30,), stop=(0.0,))

#: One solver, matching the published tables.
SOLVER = "normal"

#: The seed the noise is drawn with.  Fixed, so a rerun asks the same question.
SEED = 20260802


def _window(sigma):
    return "linear (N3)" if sigma is None else "sigma %g" % sigma


def _snr(snr):
    return "inf" if math.isinf(snr) else "%g" % snr


def measure(verbose=False):
    """Every cell.  Returns ``{(denoising, sigma, snr): percent}``."""
    directory = tempfile.mkdtemp(prefix="n3-denoise-")

    reference_volume = load_volume(legacy_data("brain_nu_ref.mnc"))
    model_mask = load_volume(MODEL_MASK)
    inside = model_mask.resample_like(reference_volume).data != 0

    planted = synthetic_bias_field(reference_volume, inside, AMPLITUDE)
    biased = reference_volume.data * planted

    cells, baselines = {}, {}
    for snr in SNRS:
        # The noise level is set from the clean, unbiased volume so that a
        # given SNR means the same thing whatever was planted, and the same
        # draw is added to the baseline volume as to the biased one -- the
        # baseline exists to divide out whatever the configuration does to an
        # image that has no planted field, and it has to be the same image.
        sigma_noise = noise_sigma(reference_volume, inside, snr)
        noisy = as_stored(
            directory, "biased_%s.mnc" % _snr(snr),
            reference_volume.like(add_noise(biased, sigma_noise, SEED)),
            legacy_data("brain_nu_ref.mnc"))
        clean = as_stored(
            directory, "plain_%s.mnc" % _snr(snr),
            reference_volume.like(add_noise(reference_volume.data,
                                            sigma_noise, SEED)),
            legacy_data("brain_nu_ref.mnc"))

        for denoising in DENOISING:
            for sigma in SIGMAS:
                settings = dict(distance=DISTANCE, lam=LAMBDA, solver=SOLVER,
                                parzen_sigma=sigma, denoise=denoising,
                                **PROTOCOL)

                key = (denoising, sigma, snr)
                baselines[key] = nu_estimate(
                    clean, mask=model_mask,
                    **settings).evaluate_on(clean)[inside]
                got = nu_estimate(noisy, mask=model_mask,
                                  **settings).evaluate_on(noisy)[inside]

                ratio = got / baselines[key] / planted[inside]
                ratio = ratio / ratio.mean()
                cells[key] = 100.0 * float(ratio.std(unbiased=False))

                if verbose:
                    print("  denoise %-3s %-12s snr %-4s %.4f%%"
                          % ("on" if denoising else "off", _window(sigma),
                             _snr(snr), cells[key]), flush=True)
    return cells


def protocol(verbose=False):
    """The shipped protocol on ``brain.mnc``, denoised and not.

    Returns ``{denoising: (iterations, rms_vs_reference, rms_vs_plain)}``.  The
    iteration count is reported because ``-stop`` is a threshold on how far the
    field moved: a volume whose noise has been taken off settles differently,
    and that is a different run rather than merely a different answer
    (CLAUDE.md, "The stopping rule quantises everything downstream").

    ``rms_vs_reference`` is *not* an accuracy verdict.  N3 produced
    ``brain_nu_ref.mnc``, so anything that changes N3 moves away from it; the
    number establishes that the option is not cosmetic.
    """
    brain = load_volume(legacy_data("brain.mnc"))
    model_mask = load_volume(MODEL_MASK)
    reference = load_volume(legacy_data("brain_nu_ref.mnc"))

    results, plain = {}, None
    for denoising in DENOISING:
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            corrected = nu_correct(brain, mask=model_mask, verbose=True,
                                   solver=SOLVER, denoise=denoising)
        iterations = log.getvalue().count("iteration ")
        if verbose:
            print("  denoise %s\n%s" % ("on" if denoising else "off",
                                        log.getvalue()), flush=True)

        if plain is None:
            plain = corrected.data
        results[denoising] = (iterations,
                              relative_rms(corrected.data, reference.data),
                              relative_rms(corrected.data, plain))
    return results


def cost():
    """What the filter costs, against the estimation it feeds.

    Reproduced here rather than copied into prose, because it is the number
    that decides whether the option is usable at all.
    """
    brain = load_volume(legacy_data("brain.mnc"))
    model_mask = load_volume(MODEL_MASK)

    timings = {}
    start = time.perf_counter()
    nu_estimate(brain, mask=model_mask)
    timings["estimate, cpu"] = time.perf_counter() - start

    start = time.perf_counter()
    denoise(brain.data)
    timings["denoise, cpu"] = time.perf_counter() - start

    if torch.cuda.is_available():
        on_gpu = brain.data.to("cuda")
        denoise(on_gpu)  # warm up, so the figure is not a first-call artefact
        torch.cuda.synchronize()
        start = time.perf_counter()
        denoise(on_gpu)
        torch.cuda.synchronize()
        timings["denoise, cuda"] = time.perf_counter() - start
    return brain.data.numel(), timings


def report(cells, runs, voxels, timings):
    print("\nNon-uniformity left after correcting a %d%% planted field, at "
          "--distance %g --lambda %g" % (AMPLITUDE * 100, DISTANCE, LAMBDA))
    print("(lower is better; %d iterations, early stop disabled, solver %r)\n"
          % (PROTOCOL["iterations"][0], SOLVER))

    header = "  %-9s %-13s" % ("denoise", "window")
    print(header + "".join("%10s" % ("snr " + _snr(s)) for s in SNRS))
    for denoising in DENOISING:
        for sigma in SIGMAS:
            row = "  %-9s %-13s" % ("on" if denoising else "off",
                                    _window(sigma))
            print(row + "".join("%9.2f%%" % cells[(denoising, sigma, snr)]
                                for snr in SNRS))

    published = PUBLISHED[AMPLITUDE][LAMBDA][PUBLISHED_INDEX]
    control = cells[(False, None, math.inf)]
    status = "matches" if round(control, 2) == published else "DOES NOT MATCH"
    print("\n  control cell (off, linear, snr inf): %.2f%% against "
          "tables.py's %.2f%% -- %s" % (control, published, status))
    if status != "matches":
        print("  Something moved elsewhere; no number above should be "
              "believed until that is explained.")

    print("\nShipped protocol on brain.mnc:\n")
    print("  %-9s %11s %14s %12s" % ("denoise", "iterations",
                                     "rms vs ref", "rms vs off"))
    for denoising in DENOISING:
        iterations, versus_reference, versus_plain = runs[denoising]
        print("  %-9s %11d %14.3e %12.3e"
              % ("on" if denoising else "off", iterations, versus_reference,
                 versus_plain))

    print("\nCost on brain.mnc (%d voxels):\n" % voxels)
    for label, seconds in timings.items():
        print("  %-16s %7.2f s" % (label, seconds))
    if "denoise, cpu" in timings and "estimate, cpu" in timings:
        print("\n  the filter is %.0fx the whole estimation it feeds, on a CPU"
              % (timings["denoise, cpu"] / timings["estimate, cpu"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python3 -m tests.denoise",
        description="Measure --denoise against --parzen-sigma, over SNR.")
    parser.add_argument("--verbose", action="store_true",
                        help="print each cell and each iteration as it is run")
    parser.add_argument("--skip-protocol", action="store_true",
                        help="the sweep only, without the end-to-end runs")
    arguments = parser.parse_args(argv)

    assert DEFAULTS["denoise"] is False, "the filter must be off by default"

    cells = measure(arguments.verbose)
    runs = {} if arguments.skip_protocol else protocol(arguments.verbose)
    voxels, timings = cost()
    if not runs:
        runs = {d: (0, float("nan"), float("nan")) for d in DENOISING}
    report(cells, runs, voxels, timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())

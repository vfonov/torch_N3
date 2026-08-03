"""What a real Parzen window does to N3, measured.

    python3 -m tests.parzen

Not a test.  Nothing here asserts; it measures and prints, in the shape
``tests/tables.py`` does.  That module measures the spline solver; this one
measures the *histogram kernel*.

**The modification.**  N3's ``-parzen``/``-window`` is not a Parzen window.
``WHistogram::add`` splits each sample linearly between the two bin centres it
falls between: a triangular kernel exactly one bin wide, whose width is set by
``-bins`` and by wherever ``-auto_range`` put the range this iteration, rather
than by any property of the measurement.  ``--parzen-sigma`` replaces it with
the estimator the name denotes: a Gaussian of a stated width, normalised per
sample, reaching as many bins as it needs
(:func:`torch_n3.blocks.histogram.histogram`).  Nothing downstream changes.

**The two competing effects.**  The histogram is the whole of N3's data term,
and it is the noisy part: the deconvolution divides by a Wiener filter, so
high-variance counts become a high-variance mapping and the field acquires
whatever survives the spline.  Smoothing the counts before deconvolving is the
obvious remedy.  Against that, ``sharpen_hist`` already assumes the histogram is
a blurred version of the true intensity distribution and undoes a Gaussian of
``--fwhm``; a Parzen window adds a second Gaussian blur, of width ``sigma``
bins, of which the deconvolution has not been informed.  A wide window should
therefore appear as under-sharpening, and the question is where between the two
the useful width lies, if one exists.

**What is measured.**  Two things, both at ``solver="normal"``, which is the
default and what every published number in the repository holds:

1. The recovery sweep, cell for cell as ``tests/tables.py`` runs it: plant a
   smooth field of a known amplitude on ``brain_nu_ref.mnc``, correct it, and
   report the non-uniformity left in the recovered field once the same
   configuration's answer on the *untouched* reference has been divided out.
   Thirty iterations, early stop disabled.  Lower is better, and the ``None``
   row is N3 as it ships.
2. The shipped protocol end to end on ``brain.mnc``: how many iterations the
   run takes to meet ``-stop``, how far the corrected volume lands from
   ``brain_nu_ref.mnc`` (the legacy suite's regression target), and how far it
   lands from the linear split's own answer.  The last number says whether the
   window changed the result at all; the second says whether it changed it for
   the better, as far as a single reference volume can.

The sweep is ``len(SIGMAS)`` times ``tables.py``'s work, about a minute per
window on a GPU.  Only ``normal`` is swept: the question here concerns the
histogram, and sweeping solvers as well would answer neither question faster.

**Measured 2026-08-02.**  The ``None`` rows reproduce the published tables cell
for cell, which is the check that nothing else moved.  Residual non-uniformity
at the *shipped* ``--lambda 1e-7``, 20% planted:

    window          200 mm    100 mm     50 mm
    linear (N3)      0.31%     0.61%     1.51%
    sigma 0.5        0.31%     0.61%     1.52%
    sigma 1          0.28%     0.53%     1.46%
    sigma 2          0.22%     0.29%     0.77%
    sigma 4          0.22%     0.24%     0.34%

and the best cell in each column once ``--lambda`` is swept as well:

    window          200 mm    100 mm     50 mm
    linear (N3)      0.13%     0.17%     0.25%
    sigma 1          0.12%     0.17%     0.23%
    sigma 2          0.12%     0.15%     0.17%
    sigma 4          0.16%     0.17%     0.19%

Both halves matter.  At a fixed weight the window reduces the residual, by a
third at the shipped setting and by 4x at 50 mm; against a weight tuned for the
spacing it yields almost no reduction at 200 mm and a third at 50 mm.  It
therefore acts largely as a *substitute for regularization*: it improves the
cells that were under-penalised, which is the axis ``--lambda`` already operates
on, and the two effects are not additive.  ``sigma 4`` is where that becomes
visible as harm: it is the best window at 50 mm and the worst at
200 mm/``1e-6`` (1.21x the linear split) and at every ``1e-4`` cell.

**This measurement is noiseless, which is its blind spot.**  One analytic field
on ``brain_nu_ref.mnc``, no noise added.  ``experiments/`` asks the same question
of 450 random fields per window with Gaussian noise at a stated SNR, and the
window's gain there tracks the *noise* rather than the field: at 20% planted and
SNR 20 under the shipped protocol the linear split leaves 4.63% against
``sigma 4``'s 1.71%, where noiseless they are within a few percent of each
other.  See ``experiments/README.md``, "The histogram kernel", before drawing
conclusions about a real volume from the tables above.

A width of 2 bins is 0.052 log units here against ``--fwhm 0.15``'s sigma of
0.064, so the useful window is just under the blur the deconvolution already
removes, and ``sigma 4`` exceeds it.  That is consistent with the mechanism --
the window adds a Gaussian of which ``sharpen_hist`` has not been informed, so a
wide one under-sharpens -- but it is one volume and one synthetic field, and
nothing here tests the next step, which is to take the added width back out of
``--fwhm`` in quadrature.

End to end on ``brain.mnc`` under the shipped protocol, for scale: the window
changes the corrected volume by 6.2e-3 relative RMS at ``sigma 0.5`` and 8.8e-2
at ``sigma 4``, and moves it away from ``brain_nu_ref.mnc`` (3.0e-3 → 4.9e-3 →
8.6e-2).  The second number is not an accuracy verdict, since N3 produced that
reference and anything that changes N3 moves away from it, but it establishes
that the modification is not cosmetic.
"""

import argparse
import contextlib
import io
import math
import sys
import tempfile

import torch

from tests.conftest import MODEL_MASK, legacy_data, relative_rms
from tests.inputs import as_stored, synthetic_bias_field
from torch_n3.pipeline import nu_correct, nu_estimate
from torch_n3.volume import load_volume

#: Gaussian window widths, in bin widths.  ``None`` is N3's own linear split,
#: which is a triangle of half-width one bin -- standard deviation
#: ``1/sqrt(6)`` = 0.41 bins -- so the sweep starts just above it and doubles.
SIGMAS = [None, 0.5, 1.0, 2.0, 4.0]

#: The sweep, matching ``tests/tables.py`` so the two can be read side by side.
AMPLITUDES = [0.2, 0.4]
DISTANCES = [200.0, 100.0, 50.0]
LAMBDAS = [1e-7, 1e-6, 1e-5, 1e-4]

#: Fixed iteration count with the early stop disabled, so that a cell measures
#: the fit rather than which side of the stopping rule a run landed on.
PROTOCOL = dict(iterations=(30,), stop=(0.0,))

#: The one solver swept.  See the module docstring.
SOLVER = "normal"


def _label(sigma):
    return "linear (N3)" if sigma is None else "sigma %g" % sigma


def measure(sigmas, verbose=False):
    """Every cell, for every window.  Returns ``{(sigma, a, d, lam): percent}``."""
    directory = tempfile.mkdtemp(prefix="n3-parzen-")

    brain_reference = load_volume(legacy_data("brain_nu_ref.mnc"))
    model_mask = load_volume(MODEL_MASK)
    inside = model_mask.resample_like(brain_reference).data != 0

    reference = as_stored(directory, "reference.mnc", brain_reference,
                          legacy_data("brain_nu_ref.mnc"))

    planted, volumes = {}, {}
    for amplitude in AMPLITUDES:
        planted[amplitude] = synthetic_bias_field(brain_reference, inside,
                                                  amplitude)
        volumes[amplitude] = as_stored(
            directory, "artificial_%d.mnc" % (amplitude * 100),
            brain_reference.like(brain_reference.data * planted[amplitude]),
            legacy_data("brain_nu_ref.mnc"))

    cells, baselines = {}, {}
    for sigma in sigmas:
        for distance in DISTANCES:
            for lam in LAMBDAS:
                settings = dict(distance=distance, lam=lam, solver=SOLVER,
                                parzen_sigma=sigma, **PROTOCOL)

                key = (distance, lam, sigma)
                if key not in baselines:
                    baselines[key] = nu_estimate(
                        reference, mask=model_mask,
                        **settings).evaluate_on(reference)[inside]

                for amplitude in AMPLITUDES:
                    volume = volumes[amplitude]
                    got = nu_estimate(volume, mask=model_mask,
                                      **settings).evaluate_on(volume)[inside]

                    # Divided by the baseline because the reference is not
                    # perfectly uniform to begin with and N3 removes that too;
                    # renormalised, since only a field's shape means anything.
                    ratio = got / baselines[key] / planted[amplitude][inside]
                    ratio = ratio / ratio.mean()

                    cells[(sigma, amplitude, distance, lam)] = 100.0 * float(
                        ratio.std(unbiased=False))
                    if verbose:
                        print("  %-12s %3d%% %5gmm lambda=%-6g %.4f%%"
                              % (_label(sigma), amplitude * 100, distance, lam,
                                 cells[(sigma, amplitude, distance, lam)]),
                              flush=True)
    return cells


def protocol(sigmas, verbose=False):
    """The shipped protocol on ``brain.mnc``, once per window.

    Returns ``{sigma: (iterations, rms_vs_reference, rms_vs_linear)}``.  The
    iteration count is informative on its own: ``-stop`` is a threshold on how
    far the field moved, so a smoother histogram that settles sooner is a
    different run rather than only a different answer (CLAUDE.md, "The stopping
    rule quantises everything downstream").
    """
    brain = load_volume(legacy_data("brain.mnc"))
    model_mask = load_volume(MODEL_MASK)
    reference = load_volume(legacy_data("brain_nu_ref.mnc"))

    results, linear = {}, None
    for sigma in sigmas:
        # `verbose` prints one line per iteration and there is no other handle
        # on where the run stopped, so the lines are counted rather than the
        # stopping rule re-implemented here.
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            corrected = nu_correct(brain, mask=model_mask, verbose=True,
                                   solver=SOLVER, parzen_sigma=sigma)
        iterations = log.getvalue().count("iteration ")
        if verbose:
            print("  %-14s\n%s" % (_label(sigma), log.getvalue()), flush=True)

        if linear is None:
            linear = corrected.data
        results[sigma] = (iterations,
                          relative_rms(corrected.data, reference.data),
                          relative_rms(corrected.data, linear))
    return results


def _bin_width():
    """One bin, in log-intensity units, on the volume the sweep uses.

    Printed so that a window in bins can be read against ``--fwhm``, which is in
    log units: the two Gaussians, the one the window adds and the one the
    deconvolution removes, are otherwise quoted in different units.
    """
    volume = load_volume(legacy_data("brain_nu_ref.mnc"))
    mask = load_volume(MODEL_MASK)
    values = torch.log(volume.data.clamp(min=1.0))[
        mask.resample_like(volume).data != 0]
    return float(values.max() - values.min()) / (200 - 1)


def _weight(lam):
    """``1e-7`` rather than ``%g``'s ``1e-07``, as the tables are written."""
    return "1e-%d" % round(-math.log10(lam))


def _table(cells, sigma, amplitude):
    """One table, in the shape ``README.md`` and ``tables.py`` print it."""
    print("  %d%% planted, --parzen-sigma %s" % (amplitude * 100, _label(sigma)))
    print("                                       --distance")
    print("      --lambda        " + "".join("%10s" % ("%g mm" % d)
                                             for d in DISTANCES))
    for lam in LAMBDAS:
        label = _weight(lam) + (" (default)" if lam == LAMBDAS[0] else "")
        print("      %-16s" % label
              + "".join("%9.2f%%" % cells[(sigma, amplitude, d, lam)]
                        for d in DISTANCES))
    print()


def _against_linear(cells, sigmas, amplitude, lam):
    """One row per window at a fixed weight, as a ratio to the linear split.

    Every cell divided by the same cell under N3's kernel, so a value below 1.00
    means the window reduced the residual there.
    """
    print("  %d%% planted, lambda=%s -- cell / same cell under N3's split"
          % (amplitude * 100, _weight(lam)))
    print("      window          " + "".join("%10s" % ("%g mm" % d)
                                             for d in DISTANCES))
    for sigma in sigmas:
        row = []
        for distance in DISTANCES:
            key = (amplitude, distance, lam)
            row.append(cells[(sigma,) + key] / cells[(None,) + key])
        print("      %-16s" % _label(sigma)
              + "".join("%10.2f" % value for value in row))
    print()


def _best(cells, sigmas, amplitude):
    """The best window and weight in each column."""
    return [min(((s, l) for s in sigmas for l in LAMBDAS),
                key=lambda k: cells[(k[0], amplitude, d, k[1])])
            for d in DISTANCES]


def main(argv=None):
    args = _parse(argv)
    sigmas = SIGMAS if args.sigma is None else [None] + args.sigma

    width = _bin_width()
    print("On brain_nu_ref.mnc inside the model mask, one of 200 bins is "
          "%.4f log units,\nso --fwhm 0.15 is %.1f bins wide and a window of "
          "sigma s adds a blur of %.4f*s.\nN3's own linear split is a triangle "
          "of standard deviation 0.41 bins.\n" % (width, 0.15 / width, width))

    if args.protocol:
        print("the shipped protocol on brain.mnc, --solver %s:\n" % SOLVER)
        print("  %-14s %-6s %-24s %s"
              % ("window", "iters", "rms vs brain_nu_ref.mnc",
                 "rms vs the linear split"))
        for sigma, (iterations, versus_reference, versus_linear) in \
                protocol(sigmas, verbose=args.verbose).items():
            print("  %-14s %-6d %-24.2e %.2e"
                  % (_label(sigma), iterations, versus_reference, versus_linear))
        print("\n  The middle column is the legacy suite's own measure"
              " (compare_nu_result.pl);\n  it is a single reference volume,"
              " which N3 itself produced, so read it as\n  'did this stay N3'"
              " rather than as an accuracy score.  The sweep below is\n  the"
              " accuracy measurement.\n")

    if not args.sweep:
        return

    cells = measure(sigmas, verbose=args.verbose)
    if args.verbose:
        print()

    for sigma in sigmas:
        for amplitude in AMPLITUDES:
            _table(cells, sigma, amplitude)

    print("relative to N3's linear split (< 1.00 is better than N3):\n")
    for amplitude in AMPLITUDES:
        for lam in LAMBDAS:
            _against_linear(cells, sigmas, amplitude, lam)

    print("best (window, weight) per column:\n")
    for amplitude in AMPLITUDES:
        print("  %d%%: " % (amplitude * 100)
              + "   ".join("%g mm -> %s, %s" % (d, _label(s), _weight(l))
                           for d, (s, l) in zip(DISTANCES,
                                                _best(cells, sigmas, amplitude))))
    print("\nThe published --lambda x --distance tables hold the `linear (N3)`"
          " rows and are\nnot affected by any of this: --parzen-sigma is off by"
          " default.")


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="python3 -m tests.parzen",
        description="Measure what a Gaussian Parzen window does to N3.")
    parser.add_argument("--sigma", type=float, action="append",
                        help="sweep this window width in bins; repeatable "
                             "(default: %s).  N3's linear split is always "
                             "included as the baseline."
                             % ", ".join(str(s) for s in SIGMAS[1:]))
    parser.add_argument("--no-sweep", dest="sweep", action="store_false",
                        help="skip the recovery sweep, the slow half")
    parser.add_argument("--no-protocol", dest="protocol",
                        action="store_false",
                        help="skip the end-to-end run on brain.mnc")
    parser.add_argument("--verbose", action="store_true",
                        help="print each cell, and each iteration, as it goes")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

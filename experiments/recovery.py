"""Bias-field recovery over many random trials.

    python3 -m experiments.recovery --seeds 50

Not a test.  Nothing here asserts; it plants a random smooth field on
colin27, adds Gaussian noise at a stated SNR, estimates the field, and writes
one CSV row per trial saying how much of it came back and how long it took.
``experiments/README.md`` defines every column and ``experiments.summarize``
reads them.

Four estimators, chosen with ``--method`` (see :data:`METHODS`), the last of
which is a ceiling rather than an estimator: ``oracle`` is *given* the field
that was planted and does nothing but express it in the spline basis, so no
method that has to work it out from the intensities can score better.

**Why over many trials.**  ``tests/test_field_recovery.py`` and
``tests/tables.py`` plant *one* analytic field on *one* volume with no noise,
and every number they publish is a single draw.  CLAUDE.md is blunt about what
that is worth ("If you find yourself relying on a cell, the honest fix is to
widen ``LAMBDAS`` and assert the shape being claimed").  Fifty seeds give each
configuration a distribution instead, so the summary can report an IQR and a
difference between two solvers can be read against the spread within one.

**Resumable, by design.**  The full sweep is hours.  Every row is flushed as
it is produced and a re-run skips any trial already in the file, so the job
can be killed and restarted, or extended with more seeds, without losing or
repeating work.  The key is every column that defines a trial -- see
:data:`KEY`.

**Two regions.**  Every trial is scored twice, over the mask the estimation
ran in and over the brain inside it -- see :func:`_regions` for the measured
reason why the difference is worth carrying.

**Where it runs.**  On the GPU by default, falling back to the CPU on a
machine without one; ``--device cpu`` forces it.  ``device`` is one of the key
columns, so the two never mix in a summary and a sweep is resumable only
against rows from the same device.

**Cost.**  The default sweep is 3,600 trials.  ``--dry-run`` counts them
before you commit to it, and the ``seconds`` column measures what they
actually took.  Give the sweep the machine either way: a second job does not
change the *scores* -- they are deterministic given the seed -- but it does
make ``seconds`` meaningless.
"""

import argparse
import contextlib
import csv
import datetime
import io
import math
import os
import socket
import subprocess
import sys
import time

import torch

from experiments import simulation
from torch_n3.blocks.spline import DIRECT_SOLVERS
from torch_n3.optimize import DEFAULTS as OPTIMIZE_DEFAULTS
from torch_n3.optimize import OBJECTIVES, PENALTY, nu_optimize
from torch_n3.pipeline import DEFAULTS, nu_estimate
from torch_n3.volume import load_volume

#: Where a run's rows go unless ``--out`` says otherwise.
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                       "recovery.csv")

#: The two iteration protocols, both swept by default.  ``fixed30`` makes
#: every trial do the same work, so a difference between two cells is the fit;
#: ``default`` is what ``nu_correct`` ships, where noise moves the stopping
#: point and the run time and iteration count become the measurement.
PROTOCOLS = {
    "fixed30": dict(iterations=(30,), stop=(0.0,)),
    "default": dict(iterations=(50,), stop=(0.001,)),
}

#: What defines a trial.  Two rows with the same key are the same experiment,
#: which is what makes a re-run resumable rather than duplicating work.
#:
#: ``method`` joined this list after the first 3,600-trial sweep had been
#: produced, so a row without it is read as ``n3`` -- see :func:`_completed`.
#: That keeps those rows resumable rather than orphaning them.
#: ``penalty`` is here for the same reason ``lam`` is: it changes the answer,
#: so two rows differing only in it are two experiments.  Leaving it out made
#: a penalty sweep silently skip every value after the first -- the resume
#: check matched on everything else and declared the trials already done.
#: ``sample_size`` and ``max_iterations`` are here for the same reason:
#: a descent stopped at a different budget is a different experiment, and a
#: budget sweep would otherwise skip every value after the first.  So is
#: ``backend``: the legacy blocks and the torch ones are two implementations
#: of the same arithmetic and do not agree past three digits (CLAUDE.md), so
#: a row from one is not a row from the other.  Only ``device`` kept those
#: apart before, and only because the legacy backend happens to be CPU-only.
KEY = ("seed", "amplitude", "snr", "method", "backend", "solver", "protocol",
       "distance", "lam", "penalty", "sample_size", "max_iterations",
       "shrink", "device")

#: How the field is estimated.  ``n3`` is ``pipeline.nu_estimate``, the
#: shipped alternating iteration; ``hoyer`` and ``tightness`` are
#: ``optimize.nu_optimize`` descending on the named objective.  ``protocol``
#: applies to ``n3`` alone -- see :func:`_protocols`.
#:
#: ``oracle`` is not an estimator at all: it is handed the field that was
#: planted and fits it with the same penalized spline the others end with.
#: It reads no voxel intensity, so noise and amplitude cannot mislead it and
#: nothing that *does* read one can beat it.  What it scores is the ceiling
#: for a ``(distance, lam, solver, shrink)`` -- the representation error of the
#: basis, and nothing else.  Swept like any other method so that the ceiling
#: gets a distribution over seeds, and a run time, on the same terms as the
#: estimators being measured against it.
ORACLE = "oracle"
METHODS = ("n3", ORACLE) + OBJECTIVES

#: Methods that take neither ``penalty`` nor a descent budget: their rows carry
#: those key columns empty, the way ``n3``'s always have.
UNWEIGHTED = ("n3", ORACLE)

#: The voxel subset ``nu_optimize`` draws is held fixed across a whole cell --
#: the baseline and every trial in it see the same voxels -- rather than
#: tracking the trial's own seed.  The baseline is computed once and shared by
#: every seed, so it could not track them anyway; holding both fixed is what
#: makes the ratio a comparison of *fields* rather than of two different
#: subsets.  The trial's seed still governs the planted field and the noise,
#: which is what the sweep is measuring.
OPTIMIZE_SEED = 0

#: Every column, in the order they are written.  The ``_brain`` ones repeat
#: the score over the brain rather than over the whole head the estimation ran
#: in; see :func:`_regions` for why they are worth carrying.
COLUMNS = KEY + (
    "field_scale", "field_terms", "loss",
    "unexplained_pct", "rms_log", "planted_cv_pct", "floor_pct",
    "unexplained_brain_pct", "rms_log_brain", "planted_cv_brain_pct",
    "floor_brain_pct",
    "iterations", "seconds", "baseline_seconds",
    "timestamp", "git_commit", "torch_version", "host")

#: The default sweep.
SEEDS = 50
AMPLITUDES = [0.2, 0.4, 0.8]
SNRS = [math.inf, 40.0, 20.0]
DISTANCES = [75.0]
LAMBDAS = [DEFAULTS["lam"]]


def main(argv=None):
    args = _parse(argv)
    args.device = _device(args.device)
    seeds = args.seed if args.seed else list(range(1, args.seeds + 1))
    methods = args.method or ["n3"]
    solvers = args.solver or list(DIRECT_SOLVERS)
    protocols = args.protocol or list(PROTOCOLS)

    trials = (sum(len(_protocols(method, protocols)) for method in methods)
              * len(solvers) * len(args.distance)
              * len(args.lam) * len(seeds) * len(args.amplitude)
              * len(args.snr))
    done = _completed(args.out)
    print("%d trials in this sweep, %d rows already in %s"
          % (trials, len(done), args.out), flush=True)
    if args.dry_run:
        return 0

    volume, mask, inside = simulation.load_experiment(
        args.input, args.mask, args.device)
    regions = _regions(volume, inside, args)
    print("%s on %s: %s voxels, %s"
          % (os.path.basename(args.input), args.device,
             "x".join(map(str, volume.shape)),
             ", ".join("%d in the %s" % (int(region.sum()), name)
                       for name, region in regions.items())), flush=True)

    stamp = _provenance(args)
    written = 0
    with _rows(args.out) as write:
        for method in methods:
            for solver in solvers:
                for protocol in _protocols(method, protocols):
                    for distance in args.distance:
                        for lam in args.lam:
                            cell = dict(stamp, method=method, solver=solver,
                                        protocol=protocol, distance=distance,
                                        lam=lam,
                                        penalty=_weight(method, args.penalty),
                                        **_budget(method, args))
                            written += _sweep(write, done, volume, mask,
                                              regions, _settings(cell, args),
                                              cell, seeds, args)
    print("\n%d rows written to %s" % (written, args.out))
    return 0


def _device(requested):
    """The device to run on: the GPU unless told otherwise.

    Every tensor in this experiment is float64 and the volumes are large, so
    the machine's GPU is the right default when it has one -- but ``device``
    is a key column, and a sweep is only comparable within one device (see
    CLAUDE.md on how far CPU and GPU drift over thirty iterations).  Falls
    back to the CPU rather than failing when there is no GPU, so the same
    command works on a machine without one.
    """
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _protocols(method, protocols):
    """The protocols worth running for one method.

    ``protocol`` is N3's iteration rule.  A descent does not have one and the
    oracle does not iterate at all, so sweeping it over them would multiply the
    work and write the same trial twice under two labels.  Decided per method
    rather than once for the whole run, so that ``--method n3 oracle`` sweeps
    both protocols for N3 without duplicating the ceiling.
    """
    return protocols if method == "n3" else protocols[:1]


def _weight(method, requested):
    """The penalty a cell actually runs at, resolved so the key records it.

    Empty for the methods in :data:`UNWEIGHTED`, which have no such parameter
    -- writing ``None`` there would make it look like a value that was left
    unset.  For a descent, the number that will be used, taken from
    ``optimize.PENALTY`` when the command line did not say: a key that recorded
    "the default" rather than the default's value would stop meaning anything
    the day that changes.
    """
    if method in UNWEIGHTED:
        return ""
    return PENALTY[method] if requested is None else float(requested)


def _budget(method, args):
    """The descent's budget, blank for the methods that have none of it."""
    if method in UNWEIGHTED:
        return dict(sample_size="", max_iterations="")
    return dict(sample_size=args.sample_size,
                max_iterations=args.max_iterations)


def _settings(cell, args):
    """The keyword arguments the estimator named by ``cell["method"]`` takes.

    All three families share ``distance``, ``shrink`` and ``solver`` -- the
    field model is the same -- and share little else: ``iterations``/``stop``
    are N3's stopping rule, ``penalty``/``sample_size``/``seed`` belong to the
    descent, and ``lam`` and ``penalty`` are weights on incomparable scales.
    The oracle takes ``lam``, because it is fitting a spline and that is the
    weight it fits under; it is the same ``lam`` N3's final fit uses.
    """
    common = dict(distance=cell["distance"], shrink=args.shrink,
                  solver=cell["solver"])
    if cell["method"] == ORACLE:
        return dict(lam=cell["lam"], **common)
    if cell["method"] == "n3":
        return dict(PROTOCOLS[cell["protocol"]], lam=cell["lam"],
                    backend=args.backend, **common)
    return dict(objective=cell["method"], penalty=cell["penalty"],
                sample_size=cell["sample_size"],
                max_iterations=cell["max_iterations"], **common)


def _regions(volume, inside, args):
    """Where a trial is scored: the estimation mask, and the brain inside it.

    N3 estimates over whatever mask it was given -- here the whole head -- but
    a head mask takes in scalp, skull and neck, where the field is worst
    determined and where nobody is going to use the correction anyway.
    Measured at 75 mm on colin27, one field scores 2.1% over the head and
    0.96% over the brain, and the analytic field of
    ``tests.inputs.synthetic_bias_field`` splits the same way (1.9% / 0.80%),
    so it is the region rather than the generator.  Both are recorded: the
    first is what the estimation was asked to do, the second is what it is
    for.
    """
    regions = {"mask": inside}
    if args.brain_mask and os.path.exists(args.brain_mask):
        regions["brain"] = load_volume(args.brain_mask).to(
            args.device).resample_like(volume).data != 0
    return regions


def _sweep(write, done, volume, mask, regions, settings, cell, seeds, args):
    """Every trial at one ``(solver, protocol, distance, lam)``.

    The baseline -- this configuration's answer on the clean, unbiased volume
    -- does not depend on the seed, the amplitude or the SNR, so it is
    computed once here and held only for as long as this configuration lasts.
    That is what fixes the loop order: a grid search over ``--distance`` and
    ``--lambda`` then pays one baseline per cell rather than one per trial,
    and never holds more than one masked volume of them at a time.
    """
    if all(_key(dict(cell, seed=seed, amplitude=amplitude, snr=snr,
                     shrink=args.shrink, device=args.device))
           in done
           for seed in seeds for amplitude in args.amplitude
           for snr in args.snr):
        print("\n%s: already complete" % _label(cell), flush=True)
        return 0

    print("\n%s" % _label(cell), flush=True)
    baseline, baseline_seconds, _ = _estimate(volume, mask, cell["method"],
                                              settings)
    print("  baseline on the untouched volume: %.1f s" % baseline_seconds,
          flush=True)

    written = 0
    for seed in seeds:
        for amplitude in args.amplitude:
            # Built on first use and shared across the SNRs of this trial:
            # the field a seed plants does not depend on how much noise is
            # about to be added to it, and neither does the basis floor.
            planted, floor = None, {}
            for snr in args.snr:
                row = dict(cell, seed=seed, amplitude=amplitude, snr=snr,
                           shrink=args.shrink, device=args.device,
                           backend=args.backend, field_scale=args.field_scale,
                           field_terms=args.field_terms,
                           baseline_seconds=round(baseline_seconds, 3))
                if _key(row) in done:
                    continue

                if planted is None:
                    planted = simulation.random_bias_field(
                        volume, regions["mask"], amplitude, seed,
                        scale=args.field_scale, terms=args.field_terms)
                    floor = _score(
                        simulation.basis_field(
                            volume, mask, planted, cell["distance"],
                            cell["lam"], settings["solver"], args.shrink),
                        torch.ones_like(planted), planted, regions)
                row.update(floor_pct=floor["mask"][0],
                           floor_brain_pct=floor.get("brain", ("",))[0])

                write(_trial(volume, mask, regions, planted, settings, row,
                             baseline, args))
                written += 1
    return written


def _trial(volume, mask, regions, planted, settings, row, baseline, args):
    """One trial: plant, add noise, estimate, score.  Fills in ``row``."""
    sigma = simulation.noise_sigma(volume, regions["mask"], row["snr"])
    data = simulation.add_noise(volume.data * planted, sigma,
                                row["seed"] + simulation.NOISE_SEED_OFFSET)
    trial = volume.like(data)

    field, seconds, report = _estimate(trial, mask, row["method"], settings,
                                       truth=planted)
    scores = _score(field, baseline, planted, regions)

    row.update(unexplained_pct=scores["mask"][0], rms_log=scores["mask"][1],
               planted_cv_pct=simulation.non_uniformity_percent(
                   planted[regions["mask"]]),
               iterations=report["iterations"], loss=report.get("loss", ""),
               seconds=round(seconds, 3))
    if "brain" in regions:
        row.update(unexplained_brain_pct=scores["brain"][0],
                   rms_log_brain=scores["brain"][1],
                   planted_cv_brain_pct=simulation.non_uniformity_percent(
                       planted[regions["brain"]]))
    if args.verbose:
        print("  seed %-4d %3d%% snr=%-5s mask %6.3f%% brain %6.3f%% "
              "(floor %.3f%%) %2d iterations %5.1f s"
              % (row["seed"], row["amplitude"] * 100, _snr(row["snr"]),
                 row["unexplained_pct"],
                 row.get("unexplained_brain_pct", float("nan")),
                 row["floor_pct"], report["iterations"], seconds), flush=True)
    return row


def _score(recovered, baseline, planted, regions):
    """``{region: (unexplained_percent, log_rms)}`` for one estimate."""
    return {name: simulation.score(recovered, baseline, planted, region)
            for name, region in regions.items()}


def _estimate(volume, mask, method, settings, truth=None):
    """Estimate the field by ``method``, timed.  Returns the field values.

    ``nu_estimate`` returns only the fitted spline, so its iteration count is
    read back off its ``verbose`` output rather than by changing
    ``torch_n3/pipeline.py`` for the sake of an experiment; ``nu_optimize``
    reports its own, along with the loss it reached.

    ``truth`` is the planted field, and only :data:`ORACLE` is given it -- that
    is what makes it the ceiling rather than an estimator.  ``None`` means the
    untouched volume, whose field is unity by definition, which is how the
    oracle's baseline is computed on the same terms as everyone else's.  Its
    ``iterations`` is 1: one spline fit, no loop.
    """
    buffer = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        if method == ORACLE:
            planted = (torch.ones_like(volume.data) if truth is None
                       else truth)
            field = simulation.basis_spline(volume, mask, planted, **settings)
            report = {"iterations": 1}
        elif method == "n3":
            field = nu_estimate(volume, mask=mask, verbose=True, **settings)
            report = {"iterations": buffer.getvalue().count("iteration ")}
        else:
            field = nu_optimize(volume, mask=mask, seed=OPTIMIZE_SEED,
                                **settings)
            report = {"iterations": field.optimize_info["iterations"],
                      "loss": field.optimize_info["loss"][-1]}
    values = field.evaluate_on(volume)
    if values.device.type == "cuda":
        torch.cuda.synchronize()
    return values, time.perf_counter() - start, report


@contextlib.contextmanager
def _rows(path):
    """Append rows to ``path``, flushing each one as it is written.

    Flushed because the sweep is hours long and the interesting failure mode
    is being killed part way through.  The header is written only when the
    file is new, so appending to an existing sweep just extends it.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    if not fresh:
        _check_header(path)

    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        if fresh:
            writer.writeheader()
            handle.flush()

        def write(row):
            writer.writerow({column: _format(row.get(column, ""))
                             for column in COLUMNS})
            handle.flush()

        yield write


def _check_header(path):
    """Refuse to append rows in an order the file's header does not describe.

    A header is written once, when the file is created; every later run
    appends under it.  So if :data:`COLUMNS` has changed since -- a column
    added, or moved, as ``method``, ``penalty`` and ``loss`` all were -- the
    new rows are written in the new order beneath the old header, and every
    field after the first change silently lands in the wrong column.  It is
    not detectable by reading the file: the rows parse, the numbers are
    plausible, and the summary is wrong.

    It happened here, to 450 rows, and this is the guard that stops it
    happening quietly again.  The fix for an existing file is to rewrite it
    under the current header, not to widen this check.
    """
    with open(path, newline="") as handle:
        header = next(csv.reader(handle), [])
    if tuple(header) != COLUMNS:
        raise SystemExit(
            "%s was written with a different set of columns, so appending to "
            "it would put values under the wrong headings.\n  file:    %s\n  "
            "current: %s\nRewrite the file under the current columns (or "
            "write to a new --out) before adding to it."
            % (path, ",".join(header), ",".join(COLUMNS)))


def _completed(path):
    """The keys of every trial already in ``path``.

    A row from before a key column existed is read at that column's default,
    which is what keeps an older sweep resumable instead of silently repeating
    all of it under a new key.
    """
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as handle:
        return {tuple(row.get(column) or KEY_DEFAULTS.get(column, "")
                      for column in KEY)
                for row in csv.DictReader(handle)}


#: What a key column means when an older file does not carry it.
#: ``n3`` rows carry no penalty at all -- it is not one of its parameters --
#: so an empty string is the value, not a missing one.
KEY_DEFAULTS = {"method": "n3", "penalty": "", "sample_size": "",
                "max_iterations": "", "backend": "torch"}


def _key(row):
    """A row's key, as the strings it will be written and read back as."""
    return tuple(_format(row[column]) for column in KEY)


def _format(value):
    """One canonical spelling per number, so keys compare as text."""
    if isinstance(value, float):
        return "%g" % value
    return str(value)


def _snr(snr):
    return "inf" if math.isinf(snr) else "%g" % snr


def _label(cell):
    if cell["method"] == ORACLE:
        return ("--method oracle --solver %s --distance %g --lambda %g"
                % (cell["solver"], cell["distance"], cell["lam"]))
    if cell["method"] == "n3":
        return ("--method n3 --backend %s --solver %s --protocol %s "
                "--distance %g --lambda %g"
                % (cell["backend"], cell["solver"], cell["protocol"],
                   cell["distance"], cell["lam"]))
    return ("--method %s --solver %s --distance %g --penalty %g"
            % (cell["method"], cell["solver"], cell["distance"],
               cell["penalty"]))


def _provenance(args):
    """What produced these rows, repeated on every one of them.

    Duplicated per row rather than kept in a header because rows from several
    runs -- and several machines -- end up in the same file, and a summary is
    only trustworthy if it can tell them apart.
    """
    return dict(backend=args.backend, torch_version=torch.__version__,
                host=socket.gethostname(), git_commit=_commit(),
                timestamp=datetime.datetime.now().isoformat(timespec="seconds"))


def _commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="python3 -m experiments.recovery",
        description="Recover randomly planted bias fields, many times over.")
    parser.add_argument("--out", default=RESULTS,
                        help="CSV to append to (default: %(default)s).  "
                             "Trials already in it are skipped.")
    parser.add_argument("--input", default=simulation.INPUT,
                        help="volume to plant fields on (default: colin27)")
    parser.add_argument("--mask", default=simulation.HEAD_MASK,
                        help="mask for the estimation and the main score")
    parser.add_argument("--brain-mask", default=simulation.BRAIN_MASK,
                        help="second region to score over, recorded in the "
                             "'_brain' columns; '' to skip it")

    parser.add_argument("--seeds", type=int, default=SEEDS,
                        help="run seeds 1..N (default: %(default)s)")
    parser.add_argument("--seed", type=int, nargs="+",
                        help="run exactly these seeds instead")
    parser.add_argument("--amplitude", type=float, nargs="+",
                        default=AMPLITUDES,
                        help="log peak-to-peak of the planted field "
                             "(default: %(default)s)")
    parser.add_argument("--snr", type=float, nargs="+", default=SNRS,
                        help="signal-to-noise ratios; 'inf' is the noiseless "
                             "control (default: inf 40 20)")

    parser.add_argument("--distance", type=float, nargs="+",
                        default=DISTANCES,
                        help="knot spacing in mm (default: %(default)s)")
    parser.add_argument("--lambda", type=float, nargs="+", default=LAMBDAS,
                        dest="lam",
                        help="bending-energy weight (default: %(default)s)")
    parser.add_argument("--method", nargs="+", choices=METHODS,
                        help="estimators to sweep (default: n3).  'hoyer' and "
                             "'tightness' are torch_n3.optimize descending on "
                             "a sharpness objective; they ignore --protocol "
                             "and take --penalty rather than --lambda.  "
                             "'oracle' is handed the planted field and only "
                             "fits it -- the best any estimator could score at "
                             "a given --distance and --lambda")
    parser.add_argument("--penalty", type=float,
                        help="bending-energy weight for the descent methods "
                             "(default: per objective, optimize.PENALTY)")
    parser.add_argument("--sample-size", type=int,
                        default=OPTIMIZE_DEFAULTS["sample_size"],
                        help="voxels the descent's objective sees per "
                             "evaluation (default: %(default)s)")
    parser.add_argument("--max-iterations", type=int,
                        default=OPTIMIZE_DEFAULTS["max_iterations"],
                        help="descent iteration cap (default: %(default)s)")
    parser.add_argument("--solver", nargs="+", choices=DIRECT_SOLVERS,
                        help="spline solvers to sweep (default: all of %s).  "
                             "'sparse' is excluded -- it does not converge."
                             % ", ".join(DIRECT_SOLVERS))
    parser.add_argument("--protocol", nargs="+", choices=sorted(PROTOCOLS),
                        help="iteration protocols to sweep (default: both)")

    parser.add_argument("--shrink", type=int, default=DEFAULTS["shrink"],
                        help="estimation-grid subsampling (default: "
                             "%(default)s)")
    parser.add_argument("--backend", default="torch",
                        choices=("torch", "legacy"))
    parser.add_argument("--device",
                        help="torch device (default: cuda when the machine "
                             "has one, else cpu).  Pass 'cpu' to force it; "
                             "'device' is a key column, so rows from the two "
                             "never mix.")
    parser.add_argument("--field-scale", type=float,
                        default=simulation.FIELD_SCALE,
                        help="shortest wavelength of the planted field, mm "
                             "(default: %(default)s)")
    parser.add_argument("--field-terms", type=int,
                        default=simulation.FIELD_TERMS,
                        help="cosine waves summed (default: %(default)s)")

    parser.add_argument("--dry-run", action="store_true",
                        help="count the trials and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="print each trial as it finishes")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

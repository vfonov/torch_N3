"""torch_n3's own pipeline against ``nu_correct_cxx``, the compiled in-memory
C++ driver (``legacy/N3/src/N3Pipeline``), on the same simulated recovery
problem :mod:`experiments.recovery` measures.

    python3 -m experiments.compare_cxx --seeds 50

Not a test.  Plants one random field per seed (:mod:`experiments.simulation`,
the same generator ``recovery.py`` uses), estimates it with both
implementations under matching options, and writes one row per
``(seed, implementation, protocol)`` to ``results/compare_cxx.csv`` plus one
row per ``(seed, protocol)`` of *direct* pointwise comparisons to
``results/compare_cxx_direct.csv``.

Two protocols, both swept by default (``--protocol``): ``v1.0`` runs
``nu_correct_cxx -V1.0`` against ``torch_n3.pipeline.V1_0`` -- N3's own
fwhm, linear-split histogram and staged stop, what every recorded reference
in ``tests/`` and every table in ``README.md`` was produced under.  ``v1.1``
runs ``nu_correct_cxx -V1.1`` against ``torch_n3.pipeline.DEFAULTS``, this
port's own protocol since 2026-08-06 (tighter stop, a Gaussian Parzen
window).  Both sides always run with rounding off (``-nolegacy_rounding`` /
``legacy_rounding=False``): that flag emulates the *Perl's* file round trips,
and neither implementation here goes through a file, so leaving it off
isolates the arithmetic the two share rather than a precision loss neither
was asked to reproduce.

``nu_correct_cxx`` is a file-based program.  Each trial's noisy volume is
written to a temporary MINC2 file at ``float64`` storage (no quantisation),
corrected, and read back; the field it applied is recovered as
``input / output``, ``nu_evaluate``'s own definition, extended outside the
mask by ``correct_field``.  ``BSplineField.evaluate_on`` -- what the torch
side scores -- does not extend the field outside its domain (CLAUDE.md, "the
spline evaluates to exactly 0 outside its domain"), so scoring is restricted
to the estimation mask, where the two agree regardless of the extension.

**The direct comparison** (``results/compare_cxx_direct.csv``) is a pointwise
relative RMS between the two implementations' *corrected* fields --
``field / baseline``, renormalised to mean 1 over the region, the same
quantity :mod:`experiments.simulation`'s ``unexplained_pct`` is a statistic
of -- rather than a summary statistic of each computed separately.  It answers
three questions per trial: how far the two implementations sit from each
other, and how far each sits from the planted ground truth, on the same
scale (CLAUDE.md, "Compare RMS, not extremes").

Resumable like ``recovery.py``: rows already in ``--out``/``--out-direct``
are skipped.
"""

import argparse
import contextlib
import csv
import datetime
import io
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import torch

from experiments import simulation
from torch_n3.pipeline import DEFAULTS, V1_0, nu_estimate
from torch_n3.volume import load_volume, save_volume

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                       "compare_cxx.csv")
DIRECT_RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results",
    "compare_cxx_direct.csv")

#: The compiled driver.  Built by legacy/N3's own CMake, not this repository's;
#: overridden with ``--driver`` on a machine where the build directory differs.
DRIVER = "/app/legacy/_build/n3/nu_correct_cxx"

#: The two protocols, both swept by default.  Only the four options that
#: differ between ``-V1.0``/``-V1.1`` are named here; ``distance``/``lam``/
#: ``shrink`` are shared and come from the CLI, and ``legacy_rounding`` is
#: always off on both sides -- see the module docstring.
PROTOCOLS = {
    "v1.0": dict(fwhm=V1_0["fwhm"], iterations=V1_0["iterations"],
                stop=V1_0["stop"], parzen_sigma=V1_0["parzen_sigma"]),
    "v1.1": dict(fwhm=DEFAULTS["fwhm"], iterations=DEFAULTS["iterations"],
                stop=DEFAULTS["stop"], parzen_sigma=DEFAULTS["parzen_sigma"]),
}

#: What defines a per-implementation row.  Two rows with the same key are the
#: same experiment, which is what makes a re-run resumable.
KEY = ("seed", "amplitude", "snr", "protocol", "implementation")
DIRECT_KEY = ("seed", "amplitude", "snr", "protocol")

COLUMNS = KEY + (
    "unexplained_pct", "rms_log", "planted_cv_pct",
    "unexplained_brain_pct", "rms_log_brain", "planted_cv_brain_pct",
    "iterations", "seconds",
    "distance", "lam", "shrink", "field_scale", "field_terms",
    "timestamp", "git_commit", "host")

DIRECT_COLUMNS = DIRECT_KEY + (
    "rel_rms_torch_cxx_mask", "rel_rms_torch_truth_mask",
    "rel_rms_cxx_truth_mask",
    "rel_rms_torch_cxx_brain", "rel_rms_torch_truth_brain",
    "rel_rms_cxx_truth_brain",
    "distance", "lam", "shrink", "field_scale", "field_terms",
    "timestamp", "git_commit", "host")

SEEDS = 50
AMPLITUDE = 0.4
SNR = math.inf


def main(argv=None):
    args = _parse(argv)
    args.torch_device = _device(args.device)
    seeds = args.seed if args.seed else list(range(1, args.seeds + 1))
    protocols = args.protocol or list(PROTOCOLS)

    trials = len(seeds) * len(args.amplitude) * len(args.snr) * len(protocols)
    done = _completed(args.out, KEY)
    done_direct = _completed(args.out_direct, DIRECT_KEY)
    print("%d trials in this sweep (x2 implementations), %d rows already in "
          "%s, %d in %s"
          % (trials, len(done), args.out, len(done_direct), args.out_direct),
          flush=True)
    if args.dry_run:
        return 0

    volume, mask, inside = simulation.load_experiment(
        args.input, args.mask, args.torch_device)
    regions = {"mask": inside}
    if args.brain_mask and os.path.exists(args.brain_mask):
        regions["brain"] = load_volume(args.brain_mask).to(
            args.torch_device).resample_like(volume).data != 0
    print("%s: %s voxels, %s"
          % (os.path.basename(args.input),
             "x".join(map(str, volume.shape)),
             ", ".join("%d in the %s" % (int(region.sum()), name)
                       for name, region in regions.items())), flush=True)

    stamp = _provenance(args)
    workdir = tempfile.mkdtemp(prefix="compare_cxx_")
    try:
        written = 0
        with _rows(args.out, COLUMNS) as write, \
             _rows(args.out_direct, DIRECT_COLUMNS) as write_direct:
            for protocol in protocols:
                settings = dict(distance=args.distance, lam=args.lam,
                               shrink=args.shrink, legacy_rounding=False,
                               **PROTOCOLS[protocol])
                print("\n--protocol %s: %s" % (protocol, PROTOCOLS[protocol]),
                     flush=True)
                print("  baseline (clean, unbiased volume)", flush=True)
                baseline_torch, _, _ = _estimate_torch(volume, mask, settings)
                baseline_cxx, _, _ = _estimate_cxx(
                    volume, mask, args, settings, protocol, workdir,
                    "%s_baseline" % protocol)

                for amplitude in args.amplitude:
                    for snr in args.snr:
                        key0 = dict(amplitude=amplitude, snr=snr,
                                   protocol=protocol)
                        for seed in seeds:
                            row = dict(key0, seed=seed)
                            written += _trial(
                                write, write_direct, done, done_direct,
                                volume, mask, regions, settings, row,
                                baseline_torch, baseline_cxx, args, workdir,
                                stamp)
        print("\n%d rows written to %s" % (written, args.out))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


def _trial(write, write_direct, done, done_direct, volume, mask, regions,
          settings, row, baseline_torch, baseline_cxx, args, workdir, stamp):
    planted = simulation.random_bias_field(
        volume, regions["mask"], row["amplitude"], row["seed"],
        scale=args.field_scale, terms=args.field_terms)
    sigma = simulation.noise_sigma(volume, regions["mask"], row["snr"])
    data = simulation.add_noise(volume.data * planted, sigma,
                                row["seed"] + simulation.NOISE_SEED_OFFSET)
    trial = volume.like(data)

    fields, written = {}, 0
    for implementation, baseline in (("torch", baseline_torch),
                                     ("cxx", baseline_cxx)):
        this_row = dict(row, implementation=implementation)
        need_row = _key(this_row, KEY) not in done
        need_direct = _key(row, DIRECT_KEY) not in done_direct
        if not need_row and not need_direct:
            continue
        if implementation == "torch":
            field, seconds, iterations = _estimate_torch(trial, mask, settings)
        else:
            field, seconds, iterations = _estimate_cxx(
                trial, mask, args, settings, row["protocol"], workdir,
                "%s_seed%d" % (row["protocol"], row["seed"]))
        fields[implementation] = field
        if not need_row:
            continue
        scores = {name: simulation.score(field, baseline, planted, region)
                 for name, region in regions.items()}
        this_row.update(
            unexplained_pct=scores["mask"][0], rms_log=scores["mask"][1],
            planted_cv_pct=simulation.non_uniformity_percent(
                planted[regions["mask"]]),
            iterations=iterations, seconds=round(seconds, 3), **stamp)
        if "brain" in regions:
            this_row.update(unexplained_brain_pct=scores["brain"][0],
                            rms_log_brain=scores["brain"][1],
                            planted_cv_brain_pct=simulation.non_uniformity_percent(
                                planted[regions["brain"]]))
        if args.verbose:
            print("  seed %-4d %3d%% snr=%-5s %-5s mask %6.3f%% brain %6.3f%% "
                  "%2s iterations %5.1f s"
                  % (row["seed"], row["amplitude"] * 100, _snr(row["snr"]),
                     implementation, this_row["unexplained_pct"],
                     this_row.get("unexplained_brain_pct", float("nan")),
                     iterations, seconds), flush=True)
        write(this_row)
        written += 1

    if "torch" in fields and "cxx" in fields \
            and _key(row, DIRECT_KEY) not in done_direct:
        direct = dict(row, **stamp)
        for name, region in regions.items():
            corrected_torch = _corrected(fields["torch"], baseline_torch,
                                         region)
            corrected_cxx = _corrected(fields["cxx"], baseline_cxx, region)
            truth = planted[region] / planted[region].mean()
            direct["rel_rms_torch_cxx_" + name] = _rel_rms(corrected_torch,
                                                            corrected_cxx)
            direct["rel_rms_torch_truth_" + name] = _rel_rms(corrected_torch,
                                                              truth)
            direct["rel_rms_cxx_truth_" + name] = _rel_rms(corrected_cxx,
                                                            truth)
        if args.verbose:
            print("  seed %-4d %3d%% snr=%-5s direct: torch<->cxx %.2e  "
                  "torch<->truth %.2e  cxx<->truth %.2e"
                  % (row["seed"], row["amplitude"] * 100, _snr(row["snr"]),
                     direct["rel_rms_torch_cxx_mask"],
                     direct["rel_rms_torch_truth_mask"],
                     direct["rel_rms_cxx_truth_mask"]), flush=True)
        write_direct(direct)
        written += 1
    return written


def _corrected(field, baseline, region):
    """``field`` divided by its own baseline, renormalised to mean 1.

    The quantity :func:`experiments.simulation.score`'s ``unexplained_pct`` is
    a statistic of: dividing out ``baseline`` charges neither implementation
    for colin27's own non-uniformity, and renormalising means only the
    field's *shape* is compared, since a bias field is defined up to a global
    scale.
    """
    ratio = field[region] / baseline[region]
    return ratio / ratio.mean()


def _rel_rms(a, b):
    """Relative RMS of ``a - b``, against ``b``'s own RMS.

    CLAUDE.md: compare RMS, not the largest single difference, which is an
    extreme-value statistic dominated by a handful of mask-edge voxels.
    """
    return float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt())


def _estimate_torch(volume, mask, settings):
    buffer = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        field = nu_estimate(volume, mask=mask, verbose=True, **settings)
    values = field.evaluate_on(volume)
    if values.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    iterations = buffer.getvalue().count("iteration ")
    return values, seconds, iterations


def _estimate_cxx(volume, mask, args, settings, protocol, workdir, tag):
    """Run ``nu_correct_cxx -V1.0``/``-V1.1`` on ``volume``.

    The field is not written anywhere on its own in ``-correct`` mode: it is
    ``input / output``, exactly ``nu_evaluate``'s definition, extended outside
    the mask by ``correct_field`` and floored -- neither of which
    :func:`_estimate_torch`'s raw ``evaluate_on`` does.  Scoring is restricted
    to the estimation mask everywhere this module uses it, where the two
    agree regardless.  ``-nolegacy_rounding`` always: see the module
    docstring.
    """
    input_path = os.path.join(workdir, "%s_in.mnc" % tag)
    output_path = os.path.join(workdir, "%s_out.mnc" % tag)
    save_volume(input_path, volume, like=args.input, store_dtype="float64")

    cmd = [args.driver, "-V1.0" if protocol == "v1.0" else "-V1.1",
          "-nolegacy_rounding", "-clobber", "-mask", args.mask,
          "-distance", str(settings["distance"]),
          "-lambda", str(settings["lam"]),
          "-shrink", str(settings["shrink"]),
          "-verbose", input_path, output_path]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError("nu_correct_cxx failed (%d):\n%s"
                          % (result.returncode, result.stderr))
    iterations = result.stdout.count("CV for change in field estimate")

    output = load_volume(output_path).to(volume.data.device)
    field = torch.where(output.data != 0, volume.data / output.data,
                        torch.ones_like(volume.data))
    os.remove(input_path)
    os.remove(output_path)
    return field, seconds, iterations


def _device(requested):
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@contextlib.contextmanager
def _rows(path, columns):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if fresh:
            writer.writeheader()
            handle.flush()

        def write(row):
            writer.writerow({column: _format(row.get(column, ""))
                             for column in columns})
            handle.flush()

        yield write


def _completed(path, key):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as handle:
        return {tuple(row.get(column, "") for column in key)
                for row in csv.DictReader(handle)}


def _key(row, key):
    return tuple(_format(row[column]) for column in key)


def _format(value):
    if isinstance(value, float):
        return "%g" % value
    return str(value)


def _snr(snr):
    return "inf" if math.isinf(snr) else "%g" % snr


def _provenance(args):
    return dict(distance=args.distance, lam=args.lam, shrink=args.shrink,
                field_scale=args.field_scale, field_terms=args.field_terms,
                git_commit=_commit(), host=socket.gethostname(),
                timestamp=datetime.datetime.now().isoformat(
                    timespec="seconds"))


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
        prog="python3 -m experiments.compare_cxx",
        description="Compare torch_n3 against nu_correct_cxx on randomly "
                    "planted bias fields, under matching options.")
    parser.add_argument("--out", default=RESULTS)
    parser.add_argument("--out-direct", default=DIRECT_RESULTS,
                        help="where the direct torch-vs-cxx-vs-truth "
                             "comparison goes (default: %(default)s)")
    parser.add_argument("--driver", default=DRIVER,
                        help="path to the compiled nu_correct_cxx")
    parser.add_argument("--input", default=simulation.INPUT)
    parser.add_argument("--mask", default=simulation.HEAD_MASK)
    parser.add_argument("--brain-mask", default=simulation.BRAIN_MASK)

    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--amplitude", type=float, nargs="+",
                        default=[AMPLITUDE])
    parser.add_argument("--snr", type=float, nargs="+", default=[SNR])
    parser.add_argument("--protocol", nargs="+", choices=sorted(PROTOCOLS),
                        help="which nu_correct_cxx/torch protocol pairs to "
                             "run (default: both v1.0 and v1.1)")

    parser.add_argument("--distance", type=float, default=DEFAULTS["distance"])
    parser.add_argument("--lambda", type=float, default=DEFAULTS["lam"],
                        dest="lam")
    parser.add_argument("--shrink", type=int, default=DEFAULTS["shrink"])

    parser.add_argument("--field-scale", type=float,
                        default=simulation.FIELD_SCALE)
    parser.add_argument("--field-terms", type=int,
                        default=simulation.FIELD_TERMS)
    parser.add_argument("--device", help="torch device for the torch side "
                                        "(default: cuda if available)")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

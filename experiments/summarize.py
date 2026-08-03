"""What a sweep of :mod:`experiments.recovery` says, in one table.

    python3 -m experiments.summarize --group solver snr

Reads the CSV, groups the trials, and reports the distribution of one metric
per group: how many trials, mean, median, the interquartile range, and the
extremes, alongside the run time.  Plain stdlib -- no pandas, no plotting.

**Why the IQR rather than a standard deviation.**  Every group here is a sample
of random fields, and the question asked of it is whether another draw would
give a different answer.  The quartiles answer that without assuming the spread
is symmetric, which across seeds it is not: a seed whose field lies badly for
the basis produces an outlier that moves the mean and not the median.  Both are
printed, and a gap between them indicates that a configuration's average is set
by a few hard draws.

``floor`` is carried alongside as the mean of the per-trial basis floor: the
score the spline could not have beaten on those fields at that knot spacing.  A
group near its floor is limited by the basis; one far above it is limited by the
estimation.
"""

import argparse
import csv
import os
import statistics
import sys

from experiments.recovery import KEY_DEFAULTS, RESULTS

#: What a group is, unless ``--group`` says otherwise.  These four are the
#: axes of the headline sweep; ``--group distance lam`` is the grid search.
GROUP = ["solver", "protocol", "amplitude", "snr"]

#: Columns that can be summarised.  All are per-trial numbers.  The ``_brain``
#: pair is the same score restricted to the brain rather than to the whole
#: head the estimation ran in -- see ``experiments/README.md``.
METRICS = ("unexplained_pct", "rms_log", "unexplained_brain_pct",
           "rms_log_brain", "seconds", "iterations", "planted_cv_pct",
           "planted_cv_brain_pct", "floor_pct", "floor_brain_pct")

#: The basis floor belonging to each metric, printed beside it.
FLOOR = {"unexplained_brain_pct": "floor_brain_pct",
         "rms_log_brain": "floor_brain_pct"}


def main(argv=None):
    args = _parse(argv)
    rows = _select(_read(args.path), args.where)
    if not rows:
        print("no rows in %s" % args.path
              + (" matching %s" % " ".join(args.where) if args.where else ""))
        return 1

    groups = _group(rows, args.group)
    if args.csv:
        _emit(groups, args.group, args.metric)
    else:
        _print(groups, args.group, args.metric, args.path)
    return 0


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _select(rows, constraints):
    """Rows matching every ``column=value``, compared as written.

    One file holds several experiments -- four solvers, three devices, three
    estimators, two histogram kernels -- and a group pooling two of them answers
    no question.  ``--group`` can separate them, but only by printing every
    combination; this drops the ones not being asked about.  Values compare as
    text, as ``recovery.py`` wrote them, and an empty value is a value:
    ``--where parzen_sigma=`` selects N3's own linear split.
    """
    for constraint in constraints or ():
        column, _, value = constraint.partition("=")
        rows = [row for row in rows if row.get(column, "") == value]
    return rows


def _group(rows, columns):
    """``{(value, ...): [row, ...]}``, in the order the keys first appear.

    A row written before a column existed is read at that column's documented
    default (``recovery.KEY_DEFAULTS``), so an older sweep can be grouped
    against a newer one instead of raising.  Grouping on a column never written
    gives one group labelled ``-``, which is the correct answer to a question
    the data cannot resolve.
    """
    groups = {}
    for row in rows:
        key = tuple(row.get(column) or KEY_DEFAULTS.get(column, "-")
                    for column in columns)
        groups.setdefault(key, []).append(row)
    return groups


def _statistics(rows, metric):
    """``n``, mean, median, quartiles and extremes of one column.

    Quartiles by ``statistics.quantiles``, which interpolates.  With fewer than
    two trials there is nothing to interpolate, so the IQR is reported as the
    single value repeated rather than as an error.
    """
    # ``get``, because a file may hold rows from a run that predates a column
    # -- an older sweep is still worth summarising over the columns it has.
    values = sorted(float(row[metric]) for row in rows if row.get(metric))
    if not values:
        return None
    if len(values) < 2:
        quartiles = [values[0], values[0], values[0]]
    else:
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return dict(n=len(values), mean=statistics.fmean(values),
                median=quartiles[1], q1=quartiles[0], q3=quartiles[2],
                low=values[0], high=values[-1])


def _print(groups, columns, metric, path):
    print("%s: %d groups, %d trials\n"
          % (os.path.basename(path), len(groups),
             sum(len(rows) for rows in groups.values())))

    width = [max(len(column), max(len(key[index]) for key in groups))
             for index, column in enumerate(columns)]
    header = "  ".join("%-*s" % (width[index], column)
                       for index, column in enumerate(columns))
    print("%s   %4s %9s %9s %19s %9s %9s   %8s %8s"
          % (header, "n", "mean", "median", "IQR", "min", "max",
             "sec/mean", "floor"))
    print("-" * (len(header) + 84))

    for key, rows in sorted(groups.items(), key=_ordering):
        stats = _statistics(rows, metric)
        if stats is None:
            continue
        seconds = _statistics(rows, "seconds")
        floor = _statistics(rows, FLOOR.get(metric, "floor_pct"))
        label = "  ".join("%-*s" % (width[index], value)
                          for index, value in enumerate(key))
        print("%s   %4d %9.4f %9.4f %9.4f-%-9.4f %9.4f %9.4f   %8.1f %8.4f"
              % (label, stats["n"], stats["mean"], stats["median"],
                 stats["q1"], stats["q3"], stats["low"], stats["high"],
                 seconds["mean"] if seconds else float("nan"),
                 floor["mean"] if floor else float("nan")))

    print("\nmetric: %s.  IQR is the middle half of the trials; a mean well "
          "above\nthe median means a few hard draws are setting it.  'floor' "
          "is the mean\nbest the spline basis could have done on those "
          "fields." % metric)


def _emit(groups, columns, metric):
    """The same summary as CSV, for pasting into a document."""
    writer = csv.writer(sys.stdout)
    writer.writerow(list(columns) + ["metric", "n", "mean", "median", "q1",
                                     "q3", "min", "max", "seconds_mean",
                                     "floor_pct_mean"])
    for key, rows in sorted(groups.items(), key=_ordering):
        stats = _statistics(rows, metric)
        if stats is None:
            continue
        seconds = _statistics(rows, "seconds")
        floor = _statistics(rows, FLOOR.get(metric, "floor_pct"))
        writer.writerow(list(key) + [
            metric, stats["n"],
            *("%.6g" % stats[name]
              for name in ("mean", "median", "q1", "q3", "low", "high")),
            "%.6g" % seconds["mean"] if seconds else "",
            "%.6g" % floor["mean"] if floor else ""])


def _ordering(item):
    """Sort numerically where a key component is a number, else as text."""
    return tuple((0, _number(value), "") if _number(value) is not None
                 else (1, 0.0, value) for value in item[0])


def _number(value):
    try:
        return float(value)
    except ValueError:
        return None


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="python3 -m experiments.summarize",
        description="Mean, median, IQR and run time per configuration.")
    parser.add_argument("path", nargs="?", default=RESULTS,
                        help="the CSV to read (default: %(default)s)")
    parser.add_argument("--where", nargs="+", metavar="COLUMN=VALUE",
                        help="keep only rows whose COLUMN is exactly VALUE, "
                             "as written in the file; repeatable.  'x=' keeps "
                             "the rows whose x is empty, which is how to "
                             "ask for N3's own histogram (parzen_sigma=) or "
                             "for a method that takes no penalty")
    parser.add_argument("--group", nargs="+", default=GROUP,
                        help="columns that define a group (default: %s)"
                             % " ".join(GROUP))
    parser.add_argument("--metric", default=METRICS[0], choices=METRICS,
                        help="column to summarise (default: %(default)s)")
    parser.add_argument("--csv", action="store_true",
                        help="write the summary as CSV instead of a table")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

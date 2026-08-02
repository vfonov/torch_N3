"""The sweep as pictures.

    python3 -m experiments.figures

Reads ``results/recovery.csv`` and writes five PNGs beside it.  Violin plots
throughout, because that is what this experiment produces: every cell is 50
random fields, and the thing worth seeing is the *shape* of the 50 -- whether a
method's advantage is the whole distribution moving or a few lucky draws, and
whether a tail reaches somewhere the median does not say.  A bar of medians
would hide exactly the two findings the sweep is for (``hoyer``'s failure tail,
and the outlier trials where two backends disagree by 18%).

**Densities are estimated in log space** wherever the axis is logarithmic --
``numpy.log10`` of the scores, with the ticks relabelled afterwards.  Violins
are kernel density estimates, so a KDE fitted in linear space and then drawn on
a log axis is a picture of the wrong distribution; at these dynamic ranges
(0.002% to 15%, four decades) it is badly wrong.

Nothing here computes a number that is not already in the CSV, and nothing is
smoothed beyond the KDE.  ``experiments.summarize`` remains the place to read
an actual value off.
"""

import argparse
import csv
import os
import sys

import numpy
from matplotlib import pyplot
from matplotlib.lines import Line2D

from experiments.recovery import RESULTS

#: Where the PNGs go.
FIGURES = os.path.dirname(RESULTS)

#: One colour per estimator, used in every figure so the eye carries across.
COLOUR = {"n3": "#3a6ea5", "hoyer": "#d1701c", "oracle": "#3f8f5b",
          "uncorrected": "#999999"}

#: The matched configuration: what every method is compared at.  ``hoyer`` is
#: pinned to its converged budget and its swept penalty, ``n3`` to the protocol
#: that gives every trial the same work.  Anything not listed is free.
#: ``parzen_sigma=""`` pins ``n3`` to N3's own linear split.  It is not
#: cosmetic: the file also holds rows run with a Gaussian Parzen window, and
#: without this constraint every ``n3`` panel would pool four
#: histograms into one violin and report four times the trials.
MATCHED = {
    "n3": dict(backend="torch", solver="normal", protocol="fixed30",
               device="cuda", parzen_sigma=""),
    "hoyer": dict(backend="torch", solver="normal", device="cuda",
                  penalty="0.001", max_iterations="400"),
    "oracle": dict(backend="torch", solver="normal", device="cuda"),
}

#: The sweep's axes, in the order they are drawn.
AMPLITUDES = ["0.2", "0.4", "0.8"]
SNRS = ["inf", "40", "20"]


def main(argv=None):
    args = _parse(argv)
    rows = _read(args.path)
    print("%d rows in %s" % (len(rows), args.path))

    for name, draw in (("recovery", _recovery), ("runtime", _runtime),
                       ("implementation", _implementation),
                       ("solvers", _solvers), ("cells", _cells),
                       ("windows", _windows)):
        if args.figure not in ("all", name):
            continue
        figure = draw(rows)
        path = os.path.join(args.out, "%s.png" % name)
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        pyplot.close(figure)
        print("wrote %s" % path)
    return 0


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _select(rows, **constraints):
    """Rows matching every ``column=value``.  Values compare as written."""
    return [row for row in rows
            if all(row.get(column) == value
                   for column, value in constraints.items())]


def _values(rows, column):
    return numpy.array([float(row[column]) for row in rows if row[column]])


# --------------------------------------------------------------------------
# The headline: how much non-uniformity each method leaves, and the ceiling.
# --------------------------------------------------------------------------

def _recovery(rows):
    """Score per method over the nine cells, with the ceiling under them.

    Two panels because the region changes the answer by about 2x and the
    experiment records both: the head is the problem as posed to the
    estimation, the brain is what a correction is for.
    """
    figure, axes = pyplot.subplots(2, 1, figsize=(12, 9), sharex=True)

    # One range for both panels, so that the ~2x the region is worth can be
    # read *between* them.  On per-panel limits the two look alike and the
    # difference the experiment went to the trouble of recording disappears.
    drawn = [_recovery_panel(axis, rows, region, label)
             for axis, region, label
             in ((axes[0], "unexplained_pct", "head mask"),
                 (axes[1], "unexplained_brain_pct", "brain"))]
    for axis in axes:
        _log_axis(axis, sum(drawn, []))

    axes[1].set_xlabel("planted field (log peak-to-peak) and SNR")
    figure.suptitle("Non-uniformity left after correction -- colin27, 50 "
                    "random fields per cell, 75 mm knots", fontsize=13,
                    y=1.0)
    figure.tight_layout()
    # Above the axes rather than inside them: the top panel's only empty
    # corner is where the 80% "uncorrected" line runs.
    figure.legend(handles=_legend(), loc="lower center", ncol=4, fontsize=9,
                  bbox_to_anchor=(0.5, 1.0), frameon=False)
    return figure


def _recovery_panel(axis, rows, column, label):
    methods = ["n3", "hoyer", "oracle"]
    offsets = numpy.linspace(-0.26, 0.26, len(methods))
    drawn = []

    centres, ticks = [], []
    for index, (amplitude, snr) in enumerate(
            (a, s) for a in AMPLITUDES for s in SNRS):
        centres.append(index)
        ticks.append("%d%%\nSNR %s" % (float(amplitude) * 100,
                                       "∞" if snr == "inf" else snr))
        for method, offset in zip(methods, offsets):
            cell = _select(rows, method=method, amplitude=amplitude, snr=snr,
                           **MATCHED[method])
            _violin(axis, _values(cell, column), index + offset,
                    COLOUR[method], width=0.22, drawn=drawn)

    # What doing nothing scores: the planted non-uniformity itself.  Drawn per
    # amplitude rather than per cell because noise does not change it.
    uncorrected = column.replace("unexplained", "planted_cv")
    for index in range(0, len(centres), len(SNRS)):
        amplitude = AMPLITUDES[index // len(SNRS)]
        level = numpy.log10(numpy.median(_values(
            _select(rows, method="n3", amplitude=amplitude, **MATCHED["n3"]),
            uncorrected)))
        drawn.append(numpy.array([level]))
        axis.plot([index - 0.45, index + len(SNRS) - 0.55], [level] * 2,
                  color=COLOUR["uncorrected"], linestyle="--", linewidth=1.4,
                  zorder=1)

    for boundary in range(len(SNRS), len(centres), len(SNRS)):
        axis.axvline(boundary - 0.5, color="0.85", linewidth=1, zorder=0)

    axis.set_xticks(centres)
    axis.set_xticklabels(ticks, fontsize=9)
    axis.set_ylabel("unexplained, %% of mean (%s)" % label)
    return drawn


def _legend():
    return [Line2D([], [], color=COLOUR["n3"], linewidth=8, alpha=0.65,
                   label="n3 (alternating iteration)"),
            Line2D([], [], color=COLOUR["hoyer"], linewidth=8, alpha=0.65,
                   label="hoyer (gradient descent)"),
            Line2D([], [], color=COLOUR["oracle"], linewidth=8, alpha=0.65,
                   label="oracle (given the field; the basis ceiling)"),
            Line2D([], [], color=COLOUR["uncorrected"], linestyle="--",
                   label="uncorrected (what was planted)")]


# --------------------------------------------------------------------------
# What each configuration costs.
# --------------------------------------------------------------------------

def _runtime(rows):
    figure, axis = pyplot.subplots(figsize=(11, 5))

    cells = [("n3 normal/fixed30", "n3",
              dict(MATCHED["n3"])),
             ("n3 normal/default", "n3",
              dict(MATCHED["n3"], protocol="default")),
             ("n3 blocked", "n3",
              dict(MATCHED["n3"], solver="blocked")),
             ("n3 qr", "n3", dict(MATCHED["n3"], solver="qr")),
             ("n3 dr", "n3", dict(MATCHED["n3"], solver="dr")),
             ("n3 legacy/cpu", "n3",
              dict(MATCHED["n3"], backend="legacy", device="cpu")),
             ("n3 torch/cpu", "n3", dict(MATCHED["n3"], device="cpu")),
             ("hoyer cap 400", "hoyer", dict(MATCHED["hoyer"])),
             ("hoyer cap 50", "hoyer",
              dict(MATCHED["hoyer"], max_iterations="50")),
             ("oracle", "oracle", dict(MATCHED["oracle"]))]

    drawn = []
    for index, (_, method, constraints) in enumerate(cells):
        selected = _select(rows, method=method, **constraints)
        _violin(axis, _values(selected, "seconds"), index, COLOUR[method],
                width=0.7, drawn=drawn)

    axis.set_xticks(range(len(cells)))
    axis.set_xticklabels([label for label, _, _ in cells], rotation=30,
                         ha="right", fontsize=9)
    axis.set_ylabel("seconds per estimate")
    _log_axis(axis, drawn)
    axis.set_title("Wall time per estimate -- GPU unless the label says "
                   "otherwise", fontsize=12)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Solver and backend: the things that should not change the answer.
# --------------------------------------------------------------------------

def _implementation(rows):
    """Three questions the sweep answers negatively, which is the useful part.

    Does the spline solver change what N3 recovers; does the backend or the
    device; and does the solver move the ceiling?  All at one cell, so 50
    violins are 50 answers to the same question.

    Every panel is on a **linear axis scaled to its own data**, and both of
    those choices are the point.  These differences are parts in a hundred or
    smaller; on the decade axis the other figures use they are one flat line,
    which would be a picture of the axis and not of the measurement.  Read the
    spread *within* a violin against the gap *between* them: that ratio is the
    whole answer, and it is why the axis is allowed to be this tight.
    """
    figure, axes = pyplot.subplots(1, 3, figsize=(13, 4.6))
    cell = dict(amplitude="0.4", snr="40")
    solvers = ["normal", "qr", "dr", "blocked"]

    panels = [
        (axes[0], "n3", "spline solver",
         [(solver, dict(solver=solver)) for solver in solvers]),
        (axes[1], "n3", "backend and device",
         [("legacy\ncpu", dict(backend="legacy", device="cpu")),
          ("torch\ncpu", dict(backend="torch", device="cpu")),
          ("torch\ncuda", dict(backend="torch", device="cuda"))]),
        (axes[2], "oracle", "spline solver",
         [(solver, dict(solver=solver)) for solver in solvers]),
    ]

    for axis, method, xlabel, variants in panels:
        drawn = []
        for index, (_, constraints) in enumerate(variants):
            selected = _select(rows, method=method,
                               **dict(MATCHED[method], **constraints), **cell)
            _violin(axis, _values(selected, "unexplained_brain_pct"), index,
                    COLOUR[method], width=0.6, log=False, drawn=drawn)
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels([label for label, _ in variants], fontsize=9)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("unexplained, % of mean (brain)")
        _linear_axis(axis, drawn)

    axes[0].set_title("n3: four solvers", fontsize=11)
    axes[1].set_title("n3: same arithmetic, three ways", fontsize=11)
    axes[2].set_title("oracle: the ceiling, four solvers", fontsize=11)
    figure.suptitle("What does not change the answer -- 40%% planted field, "
                    "SNR 40, note the axis on each panel" % (), fontsize=13)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# The four spline solvers: what they agree on and what they cost.
# --------------------------------------------------------------------------

#: The order the solvers are drawn in: the normal equations first, then the
#: three that factorize the stacked system ``[A; sqrt(lambda N) D]``.  That
#: split is the finding -- the three stacked solvers agree with each other far
#: more closely than any of them agrees with ``normal``, because they are the
#: same arithmetic and ``normal`` squares the condition number to form ``AtA``.
SOLVERS = ("normal", "qr", "dr", "blocked")


def _solvers(rows):
    """The effect of the solver on the answer, and its cost.

    Left: the per-trial relative difference from ``normal``, which is the only
    way the question can be posed.  The four distributions of the score itself
    coincide (``implementation.png``), so the difference must be taken *within*
    a trial before anything is resolvable.  Right: seconds.

    The point of putting them side by side is that they are on wildly
    different scales: the accuracy panel spans 1e-7 to 1e-1 and is centred at
    2e-5, the cost panel is a factor of 3.8 top to bottom.
    """
    figure, axes = pyplot.subplots(1, 2, figsize=(12, 5))
    protocol = "fixed30"

    # `parzen_sigma=""` for the same reason MATCHED carries it, and here it
    # matters twice over: the rows are keyed by trial, so a second histogram's
    # row would not pool into the violin but *replace* the one being compared.
    keyed = {solver: {_trial_key(row): row
                      for row in _select(rows, method="n3", solver=solver,
                                         protocol=protocol, backend="torch",
                                         device="cuda", parzen_sigma="")}
             for solver in SOLVERS}
    shared = sorted(set.intersection(*(set(rows) for rows in keyed.values())))

    drawn = []
    for index, solver in enumerate(SOLVERS[1:]):
        difference = numpy.array(
            [_relative(float(keyed[solver][key]["unexplained_brain_pct"]),
                       float(keyed["normal"][key]["unexplained_brain_pct"]))
             for key in shared])
        _violin(axes[0], difference, index, COLOUR["n3"], width=0.6,
                drawn=drawn)
    axes[0].set_xticks(range(len(SOLVERS) - 1))
    axes[0].set_xticklabels(SOLVERS[1:])
    axes[0].set_xlabel("compared against `normal`, per trial")
    axes[0].set_ylabel("relative difference in the score (brain)")
    axes[0].set_title("Accuracy: %d matched trials each" % len(shared),
                      fontsize=11)
    _log_axis(axes[0], drawn)

    drawn = []
    for index, solver in enumerate(SOLVERS):
        _violin(axes[1], numpy.array([float(keyed[solver][key]["seconds"])
                                      for key in shared]),
                index, COLOUR["n3"], width=0.6, log=False, drawn=drawn)
    axes[1].set_xticks(range(len(SOLVERS)))
    axes[1].set_xticklabels(SOLVERS)
    axes[1].set_xlabel("spline solver")
    axes[1].set_ylabel("seconds per estimate")
    axes[1].set_title("Cost: same trials, same GPU", fontsize=11)
    _linear_axis(axes[1], drawn)

    figure.suptitle("N3's four spline solvers -- 30 iterations, colin27, "
                    "75 mm knots", fontsize=13)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# The histogram kernel, one panel per cell.
# --------------------------------------------------------------------------

#: The windows drawn, in order: N3's own linear split first, then the Gaussian
#: Parzen widths in bin widths.  ``""`` is the linear split -- see
#: ``recovery.KEY``.  Spelled as ``recovery._format`` writes them (``"%g"``),
#: since ``_select`` compares the text: ``"1.0"`` selects nothing.
WINDOWS = ("", "1", "2", "4")

#: One colour per iteration protocol, light to dark with the work done.  The
#: protocols are drawn together because their *disagreement* is the finding:
#: under N3's own histogram more iterations make a noisy cell worse, and under
#: a wide window they do not.
PROTOCOL_COLOUR = {"fixed30": "#9dbdd8", "default": "#1f4a75"}
PROTOCOL_LABEL = {"fixed30": "30 iterations", "default": "50 (the shipped -stop)"}


def _windows(rows):
    """One panel per (amplitude, SNR), one violin pair per histogram kernel.

    The question this figure exists for: N3's ``-parzen`` is linear
    interpolation into two bins, ``--parzen-sigma`` makes it a real Gaussian
    kernel, and ``tests/parzen.py`` measures the difference on a single
    noiseless analytic field.  Here it is 50 random fields per cell, at three
    amplitudes and three SNRs, so the noise axis -- the one a smoother
    histogram ought to matter most on -- is swept rather than assumed.

    Both protocols are drawn, which is the one place this figure departs from
    the rest of the file.  Everything else pins ``fixed30`` so that every
    trial does the same work; here the *additional* work is half the result,
    since the reduction a wide window yields grows with the iteration count,
    and at 80% planted it determines whether the window helps or harms.

    Same layout and shared range as :func:`_cells`, and the same dashed
    uncorrected level, so the two can be laid side by side.
    """
    figure, axes = pyplot.subplots(len(AMPLITUDES), len(SNRS),
                                   figsize=(14, 11), sharex=True, sharey=True)
    drawn = []
    protocols = ("fixed30", "default")
    offsets = (-0.19, 0.19)

    for row_index, amplitude in enumerate(AMPLITUDES):
        for column_index, snr in enumerate(SNRS):
            axis = axes[row_index][column_index]
            cell = dict(amplitude=amplitude, snr=snr)
            for index, window in enumerate(WINDOWS):
                for protocol, offset in zip(protocols, offsets):
                    selected = _select(
                        rows, method="n3",
                        **dict(MATCHED["n3"], parzen_sigma=window,
                               protocol=protocol), **cell)
                    _violin(axis, _values(selected, "unexplained_brain_pct"),
                            index + offset, PROTOCOL_COLOUR[protocol],
                            width=0.34, drawn=drawn)

            level = numpy.log10(numpy.median(_values(
                _select(rows, method="n3", **MATCHED["n3"], **cell),
                "planted_cv_brain_pct")))
            drawn.append(numpy.array([level]))
            axis.axhline(level, color=COLOUR["uncorrected"], linestyle="--",
                         linewidth=1.4, zorder=1)

            axis.set_title("%d%% planted, SNR %s"
                           % (float(amplitude) * 100,
                              "∞" if snr == "inf" else snr), fontsize=11)
            if column_index == 0:
                axis.set_ylabel("unexplained, % of mean (brain)")

    for axis in axes[-1]:
        axis.set_xticks(range(len(WINDOWS)))
        axis.set_xticklabels(["linear\n(N3)"]
                             + ["sigma %g" % float(window)
                                for window in WINDOWS[1:]], fontsize=9)
    for row in axes:
        for axis in row:
            _log_axis(axis, drawn)

    figure.suptitle("The histogram kernel, cell by cell -- N3's linear split "
                    "against a Gaussian Parzen window\ncolin27, 50 random "
                    "fields each, 75 mm knots, --solver normal",
                    fontsize=13, y=1.0)
    figure.tight_layout()
    figure.legend(handles=[Line2D([], [], color=PROTOCOL_COLOUR[protocol],
                                  linewidth=8, alpha=0.65,
                                  label=PROTOCOL_LABEL[protocol])
                           for protocol in protocols]
                  + [Line2D([], [], color=COLOUR["uncorrected"],
                            linestyle="--",
                            label="uncorrected (what was planted)")],
                  loc="lower center", ncol=3, fontsize=9,
                  bbox_to_anchor=(0.5, 1.0), frameon=False)
    return figure


# --------------------------------------------------------------------------
# Every estimator, one panel per cell.
# --------------------------------------------------------------------------

#: What each panel holds, left to right: N3 under each solver, then the
#: descent, then the ceiling.  ``(label, method, extra constraints)``.
VARIANTS = ([("n3\n%s" % solver, "n3", dict(solver=solver))
             for solver in SOLVERS]
            + [("hoyer", "hoyer", {}), ("oracle", "oracle", {})])


def _cells(rows):
    """One panel per (amplitude, SNR), every estimator inside it.

    ``recovery.png`` pools the four solvers into a single ``n3`` violin, which
    is the right summary and the wrong picture for two questions: whether the
    solvers separate anywhere in particular, and how each cell's spread
    compares with the gap between methods.  Both need the cell on its own axis
    with the solvers drawn apart.

    One shared y range across all nine, because the comparison *between* cells
    -- the point of sweeping amplitude and SNR at all -- is most of what there
    is to read here.  The brain only; ``recovery.png`` carries both regions.
    """
    figure, axes = pyplot.subplots(len(AMPLITUDES), len(SNRS),
                                   figsize=(14, 11), sharex=True, sharey=True)
    drawn = []

    for row_index, amplitude in enumerate(AMPLITUDES):
        for column_index, snr in enumerate(SNRS):
            axis = axes[row_index][column_index]
            cell = dict(amplitude=amplitude, snr=snr)
            for index, (_, method, extra) in enumerate(VARIANTS):
                selected = _select(rows, method=method,
                                   **dict(MATCHED[method], **extra), **cell)
                _violin(axis, _values(selected, "unexplained_brain_pct"),
                        index, COLOUR[method], width=0.7, drawn=drawn)

            level = numpy.log10(numpy.median(_values(
                _select(rows, method="n3", **MATCHED["n3"], **cell),
                "planted_cv_brain_pct")))
            drawn.append(numpy.array([level]))
            axis.axhline(level, color=COLOUR["uncorrected"], linestyle="--",
                         linewidth=1.4, zorder=1)

            axis.set_title("%d%% planted, SNR %s"
                           % (float(amplitude) * 100,
                              "∞" if snr == "inf" else snr), fontsize=11)
            if column_index == 0:
                axis.set_ylabel("unexplained, % of mean (brain)")

    for axis in axes[-1]:
        axis.set_xticks(range(len(VARIANTS)))
        axis.set_xticklabels([label for label, _, _ in VARIANTS], fontsize=8)
    for row in axes:
        for axis in row:
            _log_axis(axis, drawn)

    figure.suptitle("Every estimator, cell by cell -- colin27, 50 random "
                    "fields each, 75 mm knots", fontsize=13, y=1.0)
    figure.tight_layout()
    figure.legend(handles=_legend(), loc="lower center", ncol=4, fontsize=9,
                  bbox_to_anchor=(0.5, 1.0), frameon=False)
    return figure


def _trial_key(row):
    return (row["seed"], row["amplitude"], row["snr"])


def _relative(one, other):
    """Difference relative to the larger, so it is symmetric and bounded."""
    largest = max(abs(one), abs(other))
    return abs(one - other) / largest if largest else 0.0


# --------------------------------------------------------------------------
# Drawing.
# --------------------------------------------------------------------------

def _violin(axis, values, position, colour, width, log=True, drawn=None):
    """One violin, with its median and its full extent marked.

    Drawn against ``log10(values)`` when ``log``, which is what puts the KDE
    in the space the axis is read in -- see the module docstring.  ``drawn``
    collects what was plotted so the axis can be scaled to the data instead of
    to a limit fixed in advance; a hard-coded limit removed the whole
    ``oracle`` runtime violin from view the first time this ran.

    Draws nothing, without raising, for an empty selection: a figure is regenerated
    from whatever the CSV holds, and a configuration that has not been swept
    yet should leave a gap rather than raise.  Fewer than two points, or a
    constant, has no density, so those get the marker without the body.
    """
    if log:
        values = values[values > 0]
    if len(values) == 0:
        return
    plotted = numpy.log10(values) if log else values
    if drawn is not None:
        drawn.append(plotted)

    if len(plotted) > 1 and numpy.ptp(plotted) > 0:
        parts = axis.violinplot([plotted], positions=[position], widths=width,
                                showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(colour)
            body.set_edgecolor(colour)
            body.set_alpha(0.55)
            body.set_zorder(2)

    axis.plot([position - width / 2.4, position + width / 2.4],
              [numpy.median(plotted)] * 2, color=colour, linewidth=2, zorder=4)
    axis.plot([position, position], [plotted.min(), plotted.max()],
              color=colour, linewidth=0.8, alpha=0.8, zorder=3)


#: Labelled positions within each decade.  1-2-5 rather than the decade alone
#: so that a violin always has a labelled gridline close to it: these span four
#: decades, and on decade-only ticks the ``oracle`` runtime violin came out
#: between an unlabelled 0.1 and the bottom of the frame, from which no number
#: can be read.  Snapping the limits outward to whole decades also resolves
#: this, at a cost of up to a decade of empty panel.
TICKS = (1.0, 2.0, 5.0)


def _log_axis(axis, drawn, pad=0.05):
    """A 1-2-5 log axis over data that is already ``log10``, scaled to fit it.

    The KDE is fitted in log space, so the data on this axis is logarithms and
    the ticks have to be put back by hand.  The range comes from what was
    actually drawn -- so a figure regenerated from a longer sweep, or from one
    method fewer, still shows all of it.
    """
    values = numpy.concatenate(drawn)
    low, high = float(values.min()), float(values.max())
    margin = max((high - low) * pad, 0.03)
    low, high = low - margin, high + margin

    decades = range(int(numpy.floor(low)), int(numpy.ceil(high)) + 1)
    labelled = [(power + numpy.log10(step), step * 10.0 ** power)
                for power in decades for step in TICKS
                if low <= power + numpy.log10(step) <= high]
    axis.set_yticks([position for position, _ in labelled])
    axis.set_yticklabels([_tick(value) for _, value in labelled])
    axis.set_yticks([power + numpy.log10(step) for power in decades
                     for step in range(2, 10)], minor=True)
    axis.set_ylim(low, high)
    _grid(axis)


def _linear_axis(axis, drawn, pad=0.12):
    """Scaled to the data, for panels whose whole point is a small difference.

    Four solvers on colin27 differ by less than a part in a hundred; on a
    decade axis wide enough to hold the oracle as well they are one flat line,
    which is a picture of the axis rather than of the measurement.
    """
    values = numpy.concatenate(drawn)
    low, high = float(values.min()), float(values.max())
    margin = (high - low) * pad or abs(high) * 0.05 or 1.0
    axis.set_ylim(low - margin, high + margin)
    _grid(axis)


def _grid(axis):
    axis.grid(axis="y", which="major", color="0.9", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)


def _tick(value):
    """A tick label: decimal where that is short, an exponent where it is not.

    ``0.000001`` is nine characters of mostly zeros and hard to tell from
    ``0.00001`` at a glance, which is the whole difficulty on the solver
    figure's six-decade axis.  Below a thousandth the labels switch to ``1e-6``
    -- the ticks are always 1, 2 or 5 times a power of ten, so the mantissa is
    a single digit and the label stays short.
    """
    if value >= 1:
        return "%g" % value
    if value >= 1e-3:
        return "%.*f" % (int(numpy.ceil(-numpy.log10(value))), value)
    power = int(numpy.floor(numpy.log10(value) + 1e-9))
    return "%de%d" % (round(value / 10.0 ** power), power)


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="python3 -m experiments.figures",
        description="Violin plots of a recovery sweep.")
    parser.add_argument("path", nargs="?", default=RESULTS,
                        help="the CSV to read (default: %(default)s)")
    parser.add_argument("--out", default=FIGURES,
                        help="directory for the PNGs (default: %(default)s)")
    parser.add_argument("--figure", default="all",
                        choices=("all", "recovery", "runtime",
                                 "implementation", "solvers", "cells",
                                 "windows"),
                        help="which figure to draw (default: all)")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

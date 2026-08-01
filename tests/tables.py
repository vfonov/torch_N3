"""The ``--lambda`` x ``--distance`` tables, re-measured.

    python3 -m tests.tables

Not a test.  Nothing here asserts; it measures and prints, and compares what
it measured against the copies published in ``README.md`` and
``torch_n3/cli.py`` so that a stale cell shows up as a mismatch.
``tests/margins.py`` does the same job for ``PROBLEMS.md``'s bounds; this one
does it for the numbers that, as CLAUDE.md puts it, are "the most cited in the
repository" and are checked by almost nothing.

**Why they need re-measuring rather than adjusting.**  Each cell is two whole
``nu_estimate`` runs of thirty iterations -- well past the histogram knife-edge
that makes end-to-end volumes incomparable -- so a change to any block, to the
spline solver, or to which LAPACK the shim links can move them.  The rule in
CLAUDE.md is to re-measure and to change both published copies together, never
to nudge one.  This module is how you re-measure.

**What is measured.**  Exactly what ``test_field_recovery.py``'s ``regularized``
fixture computes, over the full sweep rather than at one spacing: plant a
smooth field of a known amplitude on ``brain_nu_ref.mnc``, correct it, and
report the non-uniformity left in the recovered field once the same
implementation's answer on the *untouched* reference has been divided out.
Thirty iterations with the early stop disabled, so every run does the same
work.  Lower is better.

The one departure is that the baseline run is cached on ``(distance, lam,
solver)`` rather than recomputed per amplitude.  It does not depend on the
amplitude and the call is deterministic, so the cached value is what the
fixture would have computed.

**Across solvers.**  Every solver in ``DIRECT_SOLVERS`` minimises the same
objective, so the tables should not depend on which one ran -- and to two
decimals they nearly do not.  Measured here on 2026-07-31, against the
published (``normal``) copies:

    solver     cells differing at 2 dp   largest relative move
    normal            0 of 24                   --
    qr                2 of 24                  3.2%
    dr                2 of 24                  3.2%
    blocked           3 of 24                  4.7%

Every conclusion the prose draws survives all four: the same interior minimum
in each column (``1e-6`` at 200 mm, ``1e-5`` at 100 and 50 mm), identically at
both amplitudes, the decade-per-halving rule, and the asymmetry at 50 mm.  The
mobile cells are the same few each time, and they are the ones sitting on a
shallow part of the surface.

``sparse`` is not swept.  It does not converge (``PROBLEMS.md`` §10), and at
~18 s per spline fit the sweep would take hours rather than the minute the
direct solvers need.
"""

import argparse
import math
import sys
import tempfile

from tests.conftest import MODEL_MASK, legacy_data
from tests.inputs import as_stored, synthetic_bias_field
from torch_n3.blocks.spline import DIRECT_SOLVERS
from torch_n3.pipeline import nu_estimate
from torch_n3.volume import load_volume

#: The sweep, matching ``test_field_recovery.py``'s constants.
AMPLITUDES = [0.2, 0.4]
DISTANCES = [200.0, 100.0, 50.0]
LAMBDAS = [1e-7, 1e-6, 1e-5, 1e-4]

#: Fixed iteration count with the early stop disabled, so that a cell measures
#: the fit rather than which side of the stopping rule a run landed on.
PROTOCOL = dict(iterations=(30,), stop=(0.0,))

#: What ``README.md`` and ``torch_n3/cli.py`` currently print, as percentages,
#: indexed by amplitude then weight, one entry per spacing in ``DISTANCES``.
#: ``cli.py`` carries the 20% half only.  These are ``normal``'s numbers.
PUBLISHED = {
    0.2: {1e-7: [0.31, 0.61, 1.51], 1e-6: [0.13, 0.22, 1.01],
          1e-5: [0.33, 0.17, 0.25], 1e-4: [0.85, 0.54, 0.35]},
    0.4: {1e-7: [0.58, 1.00, 1.96], 1e-6: [0.27, 0.44, 1.30],
          1e-5: [0.67, 0.35, 0.48], 1e-4: [1.71, 1.10, 0.71]},
}

#: Half of the last digit the published copies print, so a cell counts as
#: moved when it would be written down differently.
PRINTED = 0.005


def measure(solvers, verbose=False):
    """Every cell, for every solver.  Returns ``{(solver, a, d, lam): percent}``."""
    directory = tempfile.mkdtemp(prefix="n3-tables-")

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
    for solver in solvers:
        for distance in DISTANCES:
            for lam in LAMBDAS:
                settings = dict(distance=distance, lam=lam, solver=solver,
                                **PROTOCOL)

                key = (distance, lam, solver)
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

                    cells[(solver, amplitude, distance, lam)] = 100.0 * float(
                        ratio.std(unbiased=False))
                    if verbose:
                        print("  %-8s %3d%% %5gmm lambda=%-6g %.4f%%"
                              % (solver, amplitude * 100, distance, lam,
                                 cells[(solver, amplitude, distance, lam)]),
                              flush=True)
    return cells


def _weight(lam):
    """``1e-7`` rather than ``%g``'s ``1e-07``, as the published copies write it."""
    return "1e-%d" % round(-math.log10(lam))


def _table(cells, solver, amplitude):
    """One table, in the shape ``README.md`` and ``cli.py`` print it."""
    print("  %d%% planted, --solver %s" % (amplitude * 100, solver))
    print("                                       --distance")
    print("      --lambda        " + "".join("%10s" % ("%g mm" % d)
                                             for d in DISTANCES))
    for lam in LAMBDAS:
        label = _weight(lam) + (" (default)" if lam == LAMBDAS[0] else "")
        print("      %-16s" % label
              + "".join("%9.2f%%" % cells[(solver, amplitude, d, lam)]
                        for d in DISTANCES))
    print()


def _drift(cells, solver):
    """Cells that would be written down differently from the published copy.

    Only the 2 dp comparison is made against ``PUBLISHED``, because that is
    all the published copies record -- taking a *relative* difference against a
    rounded number measures the rounding, not the code.  The relative question
    is asked of ``normal`` instead, in :func:`_move`.
    """
    moved = []
    for amplitude in AMPLITUDES:
        for lam in LAMBDAS:
            for index, distance in enumerate(DISTANCES):
                got = cells[(solver, amplitude, distance, lam)]
                published = PUBLISHED[amplitude][lam][index]
                if abs(got - published) >= PRINTED:
                    moved.append((amplitude, distance, lam, published, got))
    return moved


def _move(cells, solver):
    """Largest relative move against ``normal``, at full precision.

    This is the cross-solver question: every direct solver minimises the same
    objective, so how far can the choice of one move a cell?  Measured against
    ``normal``'s own measurement rather than against the rounded table.

    Returns ``(0.0, None)`` when ``normal`` was not among the solvers swept:
    there is then nothing to measure the move against.
    """
    if ("normal", AMPLITUDES[0], DISTANCES[0], LAMBDAS[0]) not in cells:
        return 0.0, None

    worst = (0.0, None)
    for amplitude in AMPLITUDES:
        for lam in LAMBDAS:
            for distance in DISTANCES:
                key = (amplitude, distance, lam)
                reference = cells[("normal",) + key]
                relative = abs(cells[(solver,) + key] - reference) / reference
                if relative > worst[0]:
                    worst = (relative, key)
    return worst


def _minima(cells, solver, amplitude):
    """The best weight in each column -- the claim the prose actually makes."""
    return [min(LAMBDAS, key=lambda l: cells[(solver, amplitude, d, l)])
            for d in DISTANCES]


def main(argv=None):
    args = _parse(argv)
    solvers = args.solver or list(DIRECT_SOLVERS)

    cells = measure(solvers, verbose=args.verbose)
    if args.verbose:
        print()

    for solver in solvers:
        for amplitude in AMPLITUDES:
            _table(cells, solver, amplitude)

    print("against the copies in README.md and torch_n3/cli.py"
          " (both hold `normal`'s numbers):\n")
    print("  %-9s %-24s %s" % ("solver", "cells differing at 2 dp",
                               "largest move vs normal"))
    for solver in solvers:
        moved = _drift(cells, solver)
        relative, where = _move(cells, solver)
        located = ("--" if where is None or solver == "normal" else
                   "%5.1f%%  at %d%%/%gmm/%s"
                   % (relative * 100, where[0] * 100, where[1],
                      _weight(where[2])))
        print("  %-9s %-24s %s"
              % (solver, "%d of 24" % len(moved), located))
        for amplitude, distance, lam, published, got in moved:
            print("            %d%% %gmm %s: %.2f%% -> %.2f%%"
                  % (amplitude * 100, distance, _weight(lam), published, got))

    print("\nbest weight per column (the interior minimum the prose claims):\n")
    for solver in solvers:
        for amplitude in AMPLITUDES:
            print("  %-9s %d%%: " % (solver, amplitude * 100)
                  + "   ".join("%g mm -> %s" % (d, _weight(best))
                               for d, best
                               in zip(DISTANCES, _minima(cells, solver,
                                                         amplitude))))

    print("\nRe-measure these, do not adjust them, and change both published"
          "\ncopies together.  See CLAUDE.md, \"Published numbers no test"
          " checks\".")


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="python3 -m tests.tables",
        description="Re-measure the --lambda x --distance tables.")
    parser.add_argument("--solver", action="append", choices=DIRECT_SOLVERS,
                        help="sweep this solver; repeatable (default: all of "
                             "%s).  'sparse' is excluded -- it does not "
                             "converge." % ", ".join(DIRECT_SOLVERS))
    parser.add_argument("--verbose", action="store_true",
                        help="print each cell as it is measured")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

"""Where the two backends stop agreeing, on *this* machine's LAPACK.

    python3 -m tests.convergence

Not a test.  Nothing here asserts; it measures and prints a table intended for
comparison against the same table from another machine.  ``tests/margins.py``
measures every bound in the suite; this module measures the one quantity that
is a property of the platform rather than of the code.

**Purpose.**  The spline fit solves normal equations conditioned around
``1e13`` by calling LAPACK's ``dsysv``.  Different LAPACK implementations
answer that differently, none of them incorrectly, and the difference is far
below any bound at block level.  The *pipeline* behaves otherwise: N3's
histogram range is taken from the data and then rounded to the six decimals its
text interchange prints, so a voxel on a bin boundary can fall either side of
it.  Once the loop's output is fed back in, that becomes a step change.

End-to-end agreement between two implementations therefore does not decay.  It
holds constant and is then lost entirely, at an iteration count that depends on
which LAPACK was linked.  Measured on ``brain.mnc`` here, ``legacy`` against
``torch``:

    iterations        1         2         3         4         5         6
    EBTKS clapack   5.49e-08  5.55e-08  5.65e-08  5.73e-08  5.82e-08  8.37e-04
    system LAPACK   5.52e-08  1.17e-03  2.12e-03  1.69e-03  5.40e-04  5.07e-04

That is what this module measures: the **divergence threshold**, the iteration
count at which two implementations cease to agree, moved from the sixth
iteration to the second because a fitted field moved by ``3.1e-11``.  The same
behaviour appears between a CPU and a GPU running identical code, with the
threshold at the third.

**The solver moves it too.**  ``--solver qr`` fits the spline through the
stacked least-squares system rather than the normal equations, which are
conditioned around ``1e13`` at the shipped knot spacing; the same fit comes out
of a system conditioned around ``1e6``.  Measured here on ``brain.mnc``, with
the system LAPACK in the shim:

    divergence at        normal      qr
    torch cpu vs cuda       3         7
    legacy vs torch         2         4

and agreement between CPU and GPU before the threshold tightens from
``6.3e-11`` to ``1.5e-13``.  Note the second row: the QR formulation tracks the
*C++ oracle* for longer than the port's own normal equations do, although the
oracle solves the normal equations itself.  The more accurate solve is worth
more here than matching the other implementation's formulation.  Neither row is
a property of the code alone; re-measure both columns on any new platform.

**Reading the output.**  The quantity to report is the divergence threshold:
the first iteration count at which agreement is lost.  The values before it are
informative only in being small, and those after it are not comparable between
machines, since they record which side of a rounding boundary one voxel fell
on, which is not a property anything can be held to.

``tests/inputs.py::PLATFORM_PROTOCOL`` must remain below the smallest
divergence threshold on any platform this is expected to run on.  It is
currently 1.  If this script reports a threshold of 2 anywhere, 1 is the only
safe count and there is no margin; if every platform reports 6, it could be
raised.  That is the decision this script exists to inform.

See ``README.md`` for how to build against a different LAPACK, and
``PROBLEMS.md`` §8 for what moved last time one changed.
"""

import argparse
import os
import platform
import re
import subprocess
import sys

import torch

from tests.conftest import DATA, MODEL_MASK, relative_rms
from torch_n3.blocks.spline import SOLVERS
from torch_n3.pipeline import nu_correct
from torch_n3.volume import load_volume

#: Agreement is treated as lost once it is this many times worse than at one
#: iteration.  The step, when it occurs, is four orders of magnitude, so the
#: exact factor is immaterial: this detects a step, not a threshold crossing.
DIVERGENCE = 100.0


def main(argv=None):
    options = _parse(argv)

    volume = load_volume(os.path.join(DATA, options.volume + ".mnc"))
    mask = load_volume(MODEL_MASK if options.volume == "brain"
                       else os.path.join(DATA, options.volume + "_mask.mnc"))

    print(_provenance(options.solver))
    print()

    solver = options.solver
    runs = [("legacy vs torch",
             _pair("torch", "cpu", solver, "legacy", "cpu", "normal"))]
    if options.device != "cpu":
        runs.append(("torch cpu vs " + options.device,
                     _pair("torch", "cpu", solver,
                           "torch", options.device, solver)))

    counts = list(range(1, options.max_iterations + 1))
    print("%-22s %s" % ("relative RMS", "".join("%11d" % n for n in counts)))
    for label, pair in runs:
        values = [pair(volume, mask, n) for n in counts]
        print("%-22s %s" % (label, "".join("%11.3g" % v for v in values)))
        print("%-22s %s" % ("", _divergence_note(values, counts)))
    print()
    print("Divergence at N means agreement survives N-1 iterations, no more.")
    print("tests/inputs.py::PLATFORM_PROTOCOL is currently %d iteration(s)."
          % _platform_iterations())


def _pair(backend_a, device_a, solver_a, backend_b, device_b, solver_b):
    """A function of ``(volume, mask, iterations)`` comparing two runs."""
    def compare(volume, mask, iterations):
        protocol = dict(iterations=(iterations,), stop=(0.0,))
        a = nu_correct(volume.to(device_a), mask=mask.to(device_a),
                       backend=backend_a, solver=solver_a, **protocol).data.cpu()
        b = nu_correct(volume.to(device_b), mask=mask.to(device_b),
                       backend=backend_b, solver=solver_b, **protocol).data.cpu()
        return relative_rms(a, b)
    return compare


def _divergence_note(values, counts):
    """Where agreement was lost, in words."""
    if not values:
        return ""
    first = values[0]
    for value, count in zip(values, counts):
        if first > 0 and value > DIVERGENCE * first:
            return "^ divergence at %d iterations (%.3g -> %.3g)" % (
                count, first, value)
    return "^ no divergence up to %d iterations" % counts[-1]


def _platform_iterations():
    from tests.inputs import PLATFORM_PROTOCOL
    return PLATFORM_PROTOCOL["iterations"][0]


def _provenance(solver):
    """Everything needed to make a reported table reproducible."""
    lines = ["platform:  %s, python %s, torch %s"
             % (platform.platform(), platform.python_version(),
                torch.__version__)]
    lines.append("shim links: %s" % (_linked_libraries() or "unknown"))
    lines.append("solver:    %s (torch runs; the legacy backend is always "
                 "'normal')" % solver)
    if torch.cuda.is_available():
        lines.append("cuda:      %s" % torch.cuda.get_device_name(0))
    return "\n".join(lines)


def _linked_libraries():
    """The LAPACK/BLAS the CFFI extension actually resolved against.

    Best effort, and the reason a reported table is worth anything: the
    library named in the build is not always the one the loader finds.

    Only ``N3_LAPACK_LIBS`` overrides give the shim a LAPACK/BLAS dependency
    of its own -- the default build (see ``build_legacy.py``) links none at
    all and instead leaves ``dgemm_``/``dsysv_`` undefined, resolved at
    import time against whatever PyTorch already loaded into the process. So:
    look for a direct dependency first (an explicit override, e.g. MKL); if
    there is none, tell the shared-with-PyTorch default apart from an
    ``N3_LAPACK_LIBS="EBTKS"`` build -- which also shows no dependency here,
    because it is a static archive -- by whether ``dsysv_`` is *defined* in
    the shim's own symbol table (statically linked in) or still undefined
    (left for PyTorch to resolve). Only in the latter case did PyTorch's
    choice actually run.
    """
    from torch_n3._legacy import _n3legacy

    path = getattr(_n3legacy, "__file__", None)
    wanted = ("lapack", "blas", "mkl", "accelerate", "flexi")
    if path is not None:
        command = ["otool", "-L", path] if sys.platform == "darwin" else ["ldd", path]
        try:
            output = subprocess.run(command, capture_output=True, text=True,
                                    check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            output = ""
        found = [line.split()[0].strip()
                 for line in output.splitlines()
                 if any(name in line.lower() for name in wanted)]
        if found:
            return ", ".join(found)

    if path is not None and _defines_own_lapack(path):
        return "no dynamic LAPACK (static?)"

    config = torch.__config__.show()
    shared = re.findall(r"\b((?:BLAS|LAPACK)_INFO=\S+?),?\b", config)
    if shared:
        return "%s (via PyTorch, shim links none of its own)" % ", ".join(shared)
    return "no dynamic LAPACK (static?)"


def _defines_own_lapack(path):
    """Whether ``dsysv_`` is defined in ``path`` rather than left undefined.

    Distinguishes a static LAPACK linked into the shim (e.g.
    ``N3_LAPACK_LIBS="EBTKS"``) from the shared-with-PyTorch default, which
    otherwise both look identical to :func:`_linked_libraries`'s first check
    -- neither has a dynamic dependency to name.
    """
    symbol = "_dsysv_" if sys.platform == "darwin" else "dsysv_"
    try:
        output = subprocess.run(["nm", path], capture_output=True, text=True,
                                check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[-1] == symbol:
            # nm prints "<address> <type> <name>"; undefined symbols have no
            # address and type "U" (or "u").
            return len(fields) == 3 and fields[-2].upper() != "U"
    return False


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="python3 -m tests.convergence",
        description="Where two backends stop agreeing, as iterations grow.")
    parser.add_argument("--max-iterations", type=int, default=6, metavar="N",
                        help="sweep 1..N (default 6; divergence usually "
                             "occurs well before that)")
    parser.add_argument("--volume", default="brain", choices=["brain", "chunk"],
                        help="brain is the volume the published numbers use; "
                             "chunk is 20x smaller, for a quick check that "
                             "this runs at all (default: brain)")
    parser.add_argument("--device", default="cpu",
                        help="also compare torch on cpu against this device, "
                             "e.g. cuda (default: cpu, which skips it)")
    parser.add_argument("--solver", default="normal", choices=SOLVERS,
                        help="which spline solver the torch runs use; the "
                             "legacy backend always has 'normal' (default: "
                             "normal, the shipped one)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

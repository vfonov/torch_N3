"""Where the two backends stop agreeing, on *this* machine's LAPACK.

    python3 -m tests.convergence

Not a test.  Nothing here asserts; it measures, and prints a table meant to be
compared against the same table from another machine.  ``tests/margins.py`` is
the sibling that measures every bound in the suite; this one measures the one
thing that is a property of the platform rather than of the code.

**What it is for.**  The spline fit solves normal equations conditioned around
``1e13`` and calls LAPACK's ``dsysv`` to do it.  Different LAPACKs answer that
differently -- none of them wrongly -- and the difference is far below anything
the blocks care about.  The *pipeline* is another matter: N3's histogram range
is taken from the data and then rounded to the six decimals its text
interchange prints, so a voxel sitting on a bin boundary can fall either side
of it.  Feed the loop's output back in and that becomes a step change.

So end-to-end agreement between two implementations does not decay.  It holds
flat, and then goes, all at once, at an iteration count that depends on which
LAPACK was linked.  Measured on ``brain.mnc`` here, ``legacy`` against
``torch``:

    iterations        1         2         3         4         5         6
    EBTKS clapack   5.49e-08  5.55e-08  5.65e-08  5.73e-08  5.82e-08  8.37e-04
    system LAPACK   5.52e-08  1.17e-03  2.12e-03  1.69e-03  5.40e-04  5.07e-04

That is the whole point of the exercise: the cliff moved from the sixth
iteration to the second because a fitted field moved by ``3.1e-11``.  The same
shape appears between a CPU and a GPU running identical code, with the cliff on
the third.

**Reading the output.**  The number to report is the *cliff*: the first
iteration count at which agreement is lost.  The pre-cliff values are not
interesting beyond being small, and the post-cliff ones are not comparable
between machines at all -- they record which side of a rounding boundary one
voxel fell on, which is not a property anything can be held to.

``tests/inputs.py::PLATFORM_PROTOCOL`` must stay below the smallest cliff of
any platform this is expected to run on.  It is currently 1.  If this script
reports a cliff at 2 anywhere, 1 is the only safe count and there is no margin
left; if every platform reports 6, it could afford to be higher.  That is the
decision this script exists to inform.

See ``README.md`` for how to build against a different LAPACK, and
``PROBLEMS.md`` §8 for what moved last time one changed.
"""

import argparse
import os
import platform
import subprocess
import sys

import torch

from tests.conftest import DATA, MODEL_MASK, relative_rms
from torch_n3.pipeline import nu_correct
from torch_n3.volume import load_volume

#: Agreement is "lost" once it is this many times worse than at one iteration.
#: The step is four orders of magnitude when it comes, so nothing here is
#: sensitive to the exact factor -- it is a cliff detector, not a threshold.
CLIFF = 100.0


def main(argv=None):
    options = _parse(argv)

    volume = load_volume(os.path.join(DATA, options.volume + ".mnc"))
    mask = load_volume(MODEL_MASK if options.volume == "brain"
                       else os.path.join(DATA, options.volume + "_mask.mnc"))

    print(_provenance())
    print()

    runs = [("legacy vs torch", _pair("torch", "cpu", "legacy", "cpu"))]
    if options.device != "cpu":
        runs.append(("torch cpu vs " + options.device,
                     _pair("torch", "cpu", "torch", options.device)))

    counts = list(range(1, options.max_iterations + 1))
    print("%-22s %s" % ("relative RMS", "".join("%11d" % n for n in counts)))
    for label, pair in runs:
        values = [pair(volume, mask, n) for n in counts]
        print("%-22s %s" % (label, "".join("%11.3g" % v for v in values)))
        print("%-22s %s" % ("", _cliff_note(values, counts)))
    print()
    print("A cliff at N means agreement survives N-1 iterations, no more.")
    print("tests/inputs.py::PLATFORM_PROTOCOL is currently %d iteration(s)."
          % _platform_iterations())


def _pair(backend_a, device_a, backend_b, device_b):
    """A function of ``(volume, mask, iterations)`` comparing two runs."""
    def compare(volume, mask, iterations):
        protocol = dict(iterations=(iterations,), stop=(0.0,))
        a = nu_correct(volume.to(device_a), mask=mask.to(device_a),
                       backend=backend_a, **protocol).data.cpu()
        b = nu_correct(volume.to(device_b), mask=mask.to(device_b),
                       backend=backend_b, **protocol).data.cpu()
        return relative_rms(a, b)
    return compare


def _cliff_note(values, counts):
    """Where agreement was lost, in words."""
    if not values:
        return ""
    first = values[0]
    for value, count in zip(values, counts):
        if first > 0 and value > CLIFF * first:
            return "^ cliff at %d iterations (%.3g -> %.3g)" % (
                count, first, value)
    return "^ no cliff up to %d iterations" % counts[-1]


def _platform_iterations():
    from tests.inputs import PLATFORM_PROTOCOL
    return PLATFORM_PROTOCOL["iterations"][0]


def _provenance():
    """Everything needed to make a reported table reproducible."""
    lines = ["platform:  %s, python %s, torch %s"
             % (platform.platform(), platform.python_version(),
                torch.__version__)]
    lines.append("shim links: %s" % (_linked_libraries() or "unknown"))
    if torch.cuda.is_available():
        lines.append("cuda:      %s" % torch.cuda.get_device_name(0))
    return "\n".join(lines)


def _linked_libraries():
    """The LAPACK/BLAS the CFFI extension actually resolved against.

    Best effort, and the reason a reported table is worth anything: the
    library named in the build is not always the one the loader finds.  A
    statically linked LAPACK will not show up here at all, which is itself
    worth reporting -- say so by hand if you built that way.
    """
    from torch_n3._legacy import _n3legacy

    path = getattr(_n3legacy, "__file__", None)
    if path is None:
        return None
    if sys.platform == "darwin":
        command = ["otool", "-L", path]
    else:
        command = ["ldd", path]
    try:
        output = subprocess.run(command, capture_output=True, text=True,
                                check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None

    wanted = ("lapack", "blas", "mkl", "accelerate", "flexi")
    found = [line.split()[0].strip()
             for line in output.splitlines()
             if any(name in line.lower() for name in wanted)]
    return ", ".join(found) if found else "no dynamic LAPACK (static?)"


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="python3 -m tests.convergence",
        description="Where two backends stop agreeing, as iterations grow.")
    parser.add_argument("--max-iterations", type=int, default=6, metavar="N",
                        help="sweep 1..N (default 6; the cliff is usually "
                             "found well before that)")
    parser.add_argument("--volume", default="brain", choices=["brain", "chunk"],
                        help="brain is the volume the published numbers use; "
                             "chunk is 20x smaller, for a quick check that "
                             "this runs at all (default: brain)")
    parser.add_argument("--device", default="cpu",
                        help="also compare torch on cpu against this device, "
                             "e.g. cuda (default: cpu, meaning don't)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

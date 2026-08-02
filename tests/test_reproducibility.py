"""Does this build still produce the volume it produced when it was recorded?

Every other end-to-end test here compares one implementation against another.
This one compares the package against *itself*: against
``tests/data/brain_nu_ref_legacy.mnc``, a volume checked into the repository,
which is what this pipeline produced on ``brain.mnc`` with the original C++
blocks driving it.  Another machine, another BLAS, a newer ``torch``, a GPU
instead of a CPU -- all of them have to land on that volume, and both backends
have to land on it too.

That makes it the one test here that can fail for a reason that has nothing to
do with the code in this repository, which is the point: it is where a change
in the floor shows up as a change in the floor rather than as a puzzling
one-in-a-thousand drift somewhere else.

The reference is stored ``float64``, which is what every other volume here is
not.  N3's own files are 16-bit and the rest of the suite is happy to be held
to one part in 65535, but that is orders of magnitude above the difference
between the two backends on this run: rounding the reference on its way to disk
would have meant every comparison below measuring that rounding instead of the
code.  At
6.7 MB it is the largest thing in the repository, and worth it -- the numbers
these tests report are now the implementations' own.

**Why one iteration and not the shipped fifty.**  N3's loop feeds its own
output back in, and it does so through a step that is not continuous: the
histogram range is taken from the data, then rounded to the six decimals the
legacy's text interchange prints, and a voxel sitting on a bin boundary can
fall either side of it.  Measured on ``brain.mnc``, the backends agree to
``5.5e-08`` relative RMS after one iteration; at two, one whole count moves
between bins -- out of the 3,724 samples the shrunken estimation grid
contributes -- and they end up ``1.17e-3`` apart, four orders of magnitude
worse.  That is a property of the algorithm: the legacy implementation has the
same divergence threshold, and this suite's own
``test_the_iteration_amplifies_small_differences`` covers what follows it.
Holding a converged run to a tight bound is therefore not something any
implementation could satisfy.

One iteration still runs every stage of the pipeline, and the whole of
``nu_evaluate``.  What it does *not* cover is convergence, or a second trip
round the loop; the default protocol is tested against N3's own output, three
digits at a time, in
``test_pipeline.py::test_matches_the_legacy_reference_volume``.

This said *two* until 2026-07-31, when the shim stopped linking EBTKS's
bundled LAPACK, which moved the bin-boundary transition from the sixth
iteration to the second, so two is now past the divergence threshold rather
than short of it.  The iteration at which it occurs is a property of the build
rather than of the algorithm -- CPU against GPU places it at the third -- so
one is the only count safe for every run this test covers.  ``README.md`` has the measurements,
``PROBLEMS.md`` §8 the reasoning.

**The converged run is here too, and it is a different kind of evidence.**
``brain_nu_ref_legacy_30.mnc`` is the same volume at thirty iterations, and it
exists because one iteration does not exercise convergence, the stopping rule,
or thirty trips round a loop that feeds its own output back in.  It is recorded
the same way and held to the same bound, ``1/65535``.

Measured against each recorded volume, as relative RMS:

    iterations    torch/cpu    legacy/cpu    torch/cuda    bound
    1             5.5e-08      0             5.5e-08       1.53e-05
    30            1.9e-03      0             2.5e-03       1.53e-05

At one iteration both torch runs are inside the bound by a factor of 277.  At
thirty they are over it by factors of 121 and 166, on identical code.
"""

import pytest
import torch

from tests.conftest import assert_close, legacy_data, relative_rms
from tests.inputs import CONVERGED_PROTOCOL, PLATFORM_PROTOCOL
from tests.regenerate_reference import CONVERGED_REFERENCE, PLATFORM_REFERENCE
from torch_n3.pipeline import nu_correct
from torch_n3.volume import load_volume

#: Every way of running the pipeline that is available here.  The legacy
#: backend is CPU-only by construction -- it hands its arrays to C.
RUNS = [("torch", "cpu"), ("legacy", "cpu")]
if torch.cuda.is_available():
    RUNS.append(("torch", "cuda"))


#: The two recorded runs: the protocol each was made with, the volume it went
#: to, and the bound a rerun is held to.  Both are ``1/65535``, N3's own
#: working precision -- the legacy passes every intermediate through a 16-bit
#: MINC file, so agreement finer than that is not something the algorithm
#: defines, whoever implements it.
PROTOCOLS = [
    ("one", PLATFORM_PROTOCOL, PLATFORM_REFERENCE, 1 / 65535),
    ("converged", CONVERGED_PROTOCOL, CONVERGED_REFERENCE, 1 / 65535),
]


@pytest.fixture(scope="module")
def platform_reference():
    """The recorded volume, as it comes back off disk."""
    return load_volume(legacy_data(PLATFORM_REFERENCE))


@pytest.fixture(scope="module")
def recorded_volumes():
    """Both recorded volumes, keyed by the name :data:`PROTOCOLS` gives them."""
    return {name: load_volume(legacy_data(path)).data
            for name, _, path, _ in PROTOCOLS}


def corrected_brain(brain, model_mask, backend, device,
                    protocol=PLATFORM_PROTOCOL):
    """``nu_correct`` under a recorded protocol, on one backend and device."""
    result = nu_correct(brain.to(device), mask=model_mask.to(device),
                        backend=backend, **protocol)
    return result.data.cpu()


@pytest.mark.parametrize("backend,device", RUNS)
def test_reproduces_the_recorded_volume(brain, model_mask, platform_reference,
                                        backend, device):
    """The whole pipeline, held to the recorded volume by relative RMS.

    The bound is not a measurement.  N3's working precision is one part in
    ``65535``: the legacy hands every intermediate volume to the next program
    through a 16-bit MINC file, so agreement finer than that is not something
    the algorithm defines, whoever implements it.  Applied to relative RMS --
    RMS difference over mean signal, ``compare_nu_result.pl``'s own measure --
    that is a bound of ``1/65535``, or ``0.001526%``.

    RMS and not the largest difference: a maximum over 900k voxels is decided
    by a handful at the mask edge and says nothing about the volume, which is
    the same reason ``CLAUDE.md`` asks for RMS everywhere else.

    Where each run currently sits, as relative RMS:

        legacy, cpu     0            bit-identical -- it wrote the file
        torch,  cpu     5.516e-08    the port's own difference, 0.36% of bound
        torch,  cuda    5.516e-08

    Almost all of that is the blocks disagreeing, not the platform: CPU and
    GPU differ from each other far less than either differs from the legacy,
    which is why the last two rows read the same to four figures.

    There is a factor of 277 in hand, which is a lot; the alternative was a
    bound drawn around the 5.516e-08 that was measured, and a test that
    records what the code does rather than what it must.  The number to watch
    is the measured one, not the margin -- if it moves off 5.516e-08,
    something changed, whether or not this still passes.

    The zero is deliberately *not* asserted.  Bit-equality holds for the
    machine that recorded the file and would fail on another LAPACK for a
    reason that is nobody's fault, which is exactly how a good bound gets
    loosened into a bad one.  That is not hypothetical here: see
    ``README.md`` on what swapping LAPACK does to this volume.
    """
    reference = platform_reference.data

    result = corrected_brain(brain, model_mask, backend, device)

    assert relative_rms(result, reference) < 1 / 65535


@pytest.mark.parametrize("backend,device", RUNS)
def test_reproduces_the_recorded_converged_volume(brain, model_mask,
                                                  recorded_volumes, backend,
                                                  device):
    """The same question after thirty iterations, at the same bound.

    One iteration leaves the loop untested: no convergence, no stopping rule,
    and no chance for a difference to be fed back in and grown.  This runs the
    protocol out to thirty and holds the answer to ``1/65535``, exactly as the
    one-iteration test does.

    Where each run currently sits, as relative RMS:

        legacy, cpu     0            bit-identical -- it wrote the file
        torch,  cpu     1.855e-03    121x the bound
        torch,  cuda    2.540e-03    166x the bound

    The two torch runs do not meet it, and this test fails for them.  The same
    code is inside the bound by a factor of 277 at one iteration; see
    ``python3 -m tests.convergence`` for where agreement is lost on this build,
    and ``PROBLEMS.md`` §11 for the record.
    """
    result = corrected_brain(brain, model_mask, backend, device,
                             CONVERGED_PROTOCOL)

    assert relative_rms(result, recorded_volumes["converged"]) < 1 / 65535


@pytest.mark.parametrize("name,protocol,path,bound", PROTOCOLS)
def test_the_two_recorded_runs_are_not_the_same_volume(recorded_volumes, name,
                                                       protocol, path, bound):
    """Guard against the pair becoming one file recorded twice.

    They are produced by the same function from the same input and differ only
    in the iteration count, so a mistake in wiring the protocols through would
    leave two identical volumes and two tests that agree for the wrong reason.
    Thirty iterations move the answer a great deal -- 45% relative RMS from
    one -- so simply requiring them to differ is enough to catch it.
    """
    other = "converged" if name == "one" else "one"

    assert not torch.equal(recorded_volumes[name], recorded_volumes[other])


def test_the_reference_is_on_the_grid_it_was_made_from(brain,
                                                       platform_reference):
    """A volume the geometry did not survive is not a reference for anything."""
    assert platform_reference.shape == brain.shape
    assert_close(platform_reference.start, brain.start)
    assert_close(platform_reference.step, brain.step)
    assert_close(platform_reference.dir_cos, brain.dir_cos)


@pytest.mark.parametrize("name,protocol,path,bound", PROTOCOLS)
@pytest.mark.parametrize("backend,device", RUNS)
def test_running_it_twice_gives_the_same_bits(brain, model_mask, backend,
                                              device, name, protocol, path,
                                              bound):
    """The precondition for any of the above to mean anything.

    A recorded volume is only a reference if the thing that produced it is
    deterministic.  This is the strongest statement in the suite -- bit
    equality, no tolerance at all -- and it is affordable because it asks
    nothing of the platform except that it do the same thing twice.

    Asked at both iteration counts, and the converged one is the version that
    is worth running: the divergence threshold above makes the loop *sensitive*
    rather than random, and thirty passes through it is where any genuine
    non-determinism -- an unordered reduction, a race in a GPU kernel -- would
    be amplified into view rather than remaining within the last bits.  Unlike every other
    comparison in this module, this one keeps full strength at thirty.
    """
    once = corrected_brain(brain, model_mask, backend, device, protocol)
    twice = corrected_brain(brain, model_mask, backend, device, protocol)

    assert torch.equal(once, twice)

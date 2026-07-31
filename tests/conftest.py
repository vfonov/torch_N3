"""Shared fixtures: the legacy test data, and what the legacy programs said.

Most tests here are the ones ``legacy/N3/testing/CMakeLists.txt`` defines,
re-expressed as comparisons -- but against *recorded* answers rather than a
live subprocess.  The programs are deterministic, so the answers are recorded
once by ``python3 -m tests.regenerate_reference`` and read back through the
``legacy_output`` fixture; see :mod:`tests.reference` for why.

The volumes themselves are in ``tests/data/``: byte-for-byte the same images
as ``legacy/N3/testing/`` and the installed model mask, converted once to
MINC2 so that ``minc2_simple`` can open them directly.  The originals are
gzipped MINC1, which it cannot, and converting them on every run meant every
test needed ``mincconvert``.  See ``tests/data/README.md``.
"""

import os
import shutil

import pytest
import torch

from tests import reference
from torch_n3.volume import load_volume

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

#: Where the same images live in their original form, for the record.
ORIGINALS = os.path.join(os.path.dirname(HERE), "legacy", "N3", "testing")

#: The average brain mask ``nu_correct -auto_mask`` reaches for, and the one
#: ``legacy/N3/testing/CMakeLists.txt`` passes to its reference comparison.
MODEL_MASK = os.path.join(DATA,
                          "icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.mnc")


def legacy_data(name):
    """A test volume, by the name the legacy suite knows it as."""
    return os.path.join(DATA, name)


def original_data(name):
    """The same volume as it ships: gzipped MINC1, needing ``mincconvert``."""
    return os.path.join(ORIGINALS, name)


def program_available(name):
    """Whether ``name`` can be run."""
    return shutil.which(name) is not None


def requires_program(name):
    """Skip the test unless ``name`` is on ``PATH``."""
    return pytest.mark.skipif(not program_available(name),
                              reason="%s is not on PATH" % name)


def assert_close(actual, expected, atol=0.0, rtol=0.0):
    """Fail unless two arrays agree, whatever they arrived as.

    Values reach these tests as tensors, as ``numpy`` arrays out of the
    recorded reference, or as plain Python lists; this puts them on the same
    footing first.
    """
    torch.testing.assert_close(torch.as_tensor(actual, dtype=torch.float64),
                               torch.as_tensor(expected, dtype=torch.float64),
                               rtol=rtol, atol=atol)


def span(values):
    """The range a legacy program's output covers, for scaling tolerances."""
    values = torch.as_tensor(values, dtype=torch.float64)
    return float(values.max() - values.min())


def relative_rms(result, expected):
    """``compare_nu_result.pl``'s measure: RMS difference over mean signal.

    The suite's one measure of whole-volume agreement, and the only one worth
    a fixed bound.  The obvious alternative -- the largest difference anywhere
    -- is an extreme-value statistic over hundreds of thousands of voxels,
    decided by a handful at the mask edge, and it moves for reasons that have
    nothing to do with whether two implementations agree.
    """
    result = torch.as_tensor(result, dtype=torch.float64)
    expected = torch.as_tensor(expected, dtype=torch.float64)
    return float(((result - expected) ** 2).mean().sqrt() / expected.mean())


@pytest.fixture(scope="session")
def legacy_output():
    """What the original N3 programs answered, recorded once."""
    return reference.load()


@pytest.fixture(scope="session")
def chunk():
    """The small 91x52x50 volume -- fast enough for per-block tests."""
    return load_volume(legacy_data("chunk.mnc"))


@pytest.fixture(scope="session")
def chunk_mask():
    return load_volume(legacy_data("chunk_mask.mnc"))


@pytest.fixture(scope="session")
def brain():
    return load_volume(legacy_data("brain.mnc"))


@pytest.fixture(scope="session")
def brain_reference():
    """``nu_correct``'s own output, kept as the regression target."""
    return load_volume(legacy_data("brain_nu_ref.mnc"))


@pytest.fixture(scope="session")
def model_mask():
    return load_volume(MODEL_MASK)

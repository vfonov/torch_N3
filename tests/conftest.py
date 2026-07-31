"""Shared fixtures: the legacy test data, and what the legacy programs said.

Most tests here are the ones ``legacy/N3/testing/CMakeLists.txt`` defines,
re-expressed as comparisons -- but against *recorded* answers rather than a
live subprocess.  The programs are deterministic, so the answers are recorded
once by ``python3 -m tests.regenerate_reference`` and read back through the
``legacy_output`` fixture; see :mod:`tests.reference` for why.

What the suite still needs from the MINC toolkit is ``mincconvert``, because
the volumes in ``legacy/N3/testing/`` are gzipped MINC1 and ``minc2_simple``
reads neither.  Without it, the tests that need those volumes skip.
"""

import os
import shutil

import pytest
import torch

from tests import reference
from torch_n3.volume import load_volume

TESTING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "legacy", "N3", "testing")

#: The average brain mask ``nu_correct -auto_mask`` reaches for, and the one
#: ``legacy/N3/testing/CMakeLists.txt`` passes to its reference comparison.
MODEL_MASK = os.path.join("/opt/minc/1.9.18.13/share/N3",
                          "icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.mnc.gz")

#: Reading the test data at all needs this one program.
CONVERTER = "mincconvert"


def legacy_data(name):
    return os.path.join(TESTING, name)


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


@pytest.fixture(scope="session")
def legacy_output():
    """What the original N3 programs answered, recorded once."""
    return reference.load()


@pytest.fixture(scope="session")
def chunk():
    """The small 91x52x50 volume -- fast enough for per-block tests."""
    return _testing_volume("chunk.mnc.gz")


@pytest.fixture(scope="session")
def chunk_mask():
    return _testing_volume("chunk_mask.mnc.gz")


@pytest.fixture(scope="session")
def brain():
    return _testing_volume("brain.mnc.gz")


@pytest.fixture(scope="session")
def brain_reference():
    """``nu_correct``'s own output, kept as the regression target."""
    return _testing_volume("brain_nu_ref.mnc.gz")


@pytest.fixture(scope="session")
def model_mask():
    if not os.path.exists(MODEL_MASK):
        pytest.skip("average brain mask not installed: %s" % MODEL_MASK)
    return _load(MODEL_MASK)


def _testing_volume(name):
    return _load(legacy_data(name))


def _load(path):
    if not program_available(CONVERTER):
        pytest.skip("%s is not on PATH: the test volumes are gzipped MINC1"
                    % CONVERTER)
    return load_volume(path)

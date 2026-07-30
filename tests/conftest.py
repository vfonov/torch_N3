"""Shared fixtures: the legacy test data, and a way to run the legacy programs.

Most tests here are the ones ``legacy/N3/testing/CMakeLists.txt`` defines,
re-expressed as comparisons: run the installed N3 program, run our Python, and
require the two to agree.
"""

import os
import subprocess

import pytest

from torch_n3.volume import load_volume, save_volume

TESTING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "legacy", "N3", "testing")

#: The average brain mask ``nu_correct -auto_mask`` reaches for, and the one
#: ``legacy/N3/testing/CMakeLists.txt`` passes to its reference comparison.
MODEL_MASK = os.path.join("/opt/minc/1.9.18.13/share/N3",
                          "icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.mnc.gz")


def testing_file(name):
    return os.path.join(TESTING, name)


@pytest.fixture(scope="session")
def chunk():
    """The small 91x52x50 volume -- fast enough for per-block tests."""
    return load_volume(testing_file("chunk.mnc.gz"))


@pytest.fixture(scope="session")
def chunk_mask():
    return load_volume(testing_file("chunk_mask.mnc.gz"))


@pytest.fixture(scope="session")
def brain():
    return load_volume(testing_file("brain.mnc.gz"))


@pytest.fixture(scope="session")
def brain_reference():
    """``nu_correct``'s own output, kept as the regression target."""
    return load_volume(testing_file("brain_nu_ref.mnc.gz"))


@pytest.fixture(scope="session")
def model_mask():
    if not os.path.exists(MODEL_MASK):
        pytest.skip("average brain mask not installed: %s" % MODEL_MASK)
    return load_volume(MODEL_MASK)


@pytest.fixture
def workspace(tmp_path):
    """Somewhere to exchange MINC files with the legacy programs."""
    return Workspace(tmp_path)


class Workspace:
    """A temporary directory plus the verbs the tests need there.

    File arguments are always absolute: several of the legacy programs derive
    their own temporary file names from the output path, and do not
    consistently resolve a relative one against the working directory.
    """

    def __init__(self, path):
        self.path = path

    def at(self, name):
        """The absolute path of ``name`` inside the workspace."""
        return str(self.path / name)

    def write(self, name, volume, like=None, store_dtype="float64"):
        """Put ``volume`` on disk where a legacy program can read it."""
        target = self.at(name)
        save_volume(target, volume, like=like, store_dtype=store_dtype)
        return target

    def read(self, name):
        return load_volume(self.at(name))

    def run(self, *command):
        """Run an installed N3/MINC program, failing loudly if it does."""
        result = subprocess.run([str(c) for c in command],
                                cwd=str(self.path), capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError("%s failed (%d)\n%s\n%s"
                                 % (command[0], result.returncode,
                                    result.stdout[-2000:], result.stderr[-2000:]))
        return result.stdout

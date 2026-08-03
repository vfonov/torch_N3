"""What the original N3 programs answered, recorded once.

The programs are deterministic -- re-running ``sharpen_volume`` on the same file
gives the same bytes -- so running them on every test run gains nothing and
costs a good deal: the suite would need the whole MINC toolkit installed, would
spend most of its time in ``subprocess``, and could not distinguish a failure of
this port from a different build of the original.

Their answers therefore live in ``tests/reference/legacy.npz`` and are loaded
from there.  ``python3 -m tests.regenerate_reference`` rebuilds the file; if the
programs still give the same answers, ``git diff`` is empty.

Whole volumes are stored as ``float32``: they came out of 16-bit MINC files in
the first place, and 1e-7 relative is a hundred times finer than the tightest
tolerance anything here is compared against (``span / 65535``).  The small
arrays -- lookup tables, histograms, the values probed through ``minclookup`` --
are kept at ``float64``, since those comparisons are held to 1e-6 and 1e-9 and
cost nothing to store exactly.  :func:`as_volume` marks the former.

``tests/data/brain_nu_ref_legacy.mnc`` is deliberately not held here.  It is an
answer of this pipeline rather than of the legacy programs, it is compared
against far more tightly than anything above, and it is a volume, so it lives in
``tests/data/`` as a ``float64`` MINC file that any tool can open.  See
:func:`tests.regenerate_reference.platform_reference_case`.
"""

import json
import os

import numpy as np
import torch

DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
ARRAYS = os.path.join(DIRECTORY, "legacy.npz")
SCALARS = os.path.join(DIRECTORY, "legacy.json")

MISSING = ("%s is not recorded in tests/reference/. Run\n"
           "    python3 -m tests.regenerate_reference\n"
           "with the MINC toolkit on PATH to produce it.")


def _numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


class LegacyOutputs:
    """The recorded answers, keyed by ``program.case``."""

    def __init__(self, arrays, scalars):
        self._arrays = arrays
        self._scalars = scalars

    def __getitem__(self, key):
        if key not in self._arrays:
            raise KeyError(MISSING % key)
        return torch.as_tensor(self._arrays[key], dtype=torch.float64)

    def scalar(self, key):
        if key not in self._scalars:
            raise KeyError(MISSING % key)
        return self._scalars[key]

    def keys(self):
        return sorted(list(self._arrays) + list(self._scalars))


def load():
    if not os.path.exists(ARRAYS):
        raise FileNotFoundError(MISSING % "tests/reference/legacy.npz")
    with np.load(ARRAYS) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with open(SCALARS) as handle:
        scalars = json.load(handle)["values"]
    return LegacyOutputs(arrays, scalars)


def as_volume(values):
    """Mark an array as a whole volume, storable at ``float32``."""
    return np.asarray(_numpy(values), dtype=np.float32)


def save(arrays, scalars, produced_by):
    """Write the recorded answers.  Used only by the regeneration script."""
    os.makedirs(DIRECTORY, exist_ok=True)
    np.savez_compressed(ARRAYS, **{key: _numpy(value)
                                   for key, value in arrays.items()})
    with open(SCALARS, "w") as handle:
        json.dump({"produced_by": produced_by, "values": scalars}, handle,
                  indent=2, sort_keys=True)
        handle.write("\n")

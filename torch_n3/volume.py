"""MINC volumes as plain arrays plus the geometry N3 needs.

N3 only ever asks three things of a volume: the voxel values, the voxel size
along each axis, and where the grid sits in world space.  :class:`Volume`
carries exactly that, in *standard order* -- a C-ordered ``torch`` tensor
whose axes run slowest to fastest with positive steps -- so that the rest of
the package can be written as ordinary tensor code.  The geometry stays in
``numpy``: it is three numbers per axis and it describes the grid rather than
travelling with it to a device.

Note that ``minc2_simple`` reads MINC2 (HDF5) files only, while much older MINC
data -- including ``legacy/N3/testing`` -- is MINC1, optionally gzipped.
:func:`load_volume` converts those on the fly with ``mincconvert``.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch
from minc2_simple import minc2_file
from minc2_simple.minc2_simple import minc2_dim


class Volume:
    """Voxel data on a regular, axis-aligned grid.

    ``data[i, j, k]`` sits at world coordinate ``start + (i, j, k) * step``
    along the axes named by ``dir_cos``.  Axis 0 is the slowest-varying one,
    matching C order.
    """

    def __init__(self, data, start, step, dir_cos=None):
        self.data = torch.as_tensor(data, dtype=torch.float64).contiguous()
        self.start = np.asarray(start, dtype=np.float64)
        self.step = np.asarray(step, dtype=np.float64)
        if dir_cos is None:
            dir_cos = np.eye(3)[::-1]  # axis 0 is Z, axis 2 is X
        self.dir_cos = np.asarray(dir_cos, dtype=np.float64)

    @property
    def shape(self):
        return tuple(self.data.shape)

    def like(self, data):
        """The same grid, holding ``data`` instead."""
        return Volume(data, self.start, self.step, self.dir_cos)

    def to(self, device):
        """The same volume, with its data on ``device``."""
        return self.like(self.data.to(device))

    def resample_like(self, grid):
        """Sample this volume on ``grid``, as ``mincresample -nearest_neighbour``.

        Both grids are axis-aligned and share a world space, so "nearest
        neighbour" is one rounded index per axis.  Points of ``grid`` that
        fall outside this volume become zero, which is ``mincresample``'s
        default fill value.
        """
        device = self.data.device
        picks, inside = [], []
        for axis in range(3):
            world = grid.start[axis] + np.arange(grid.shape[axis]) * grid.step[axis]
            index = np.rint((world - self.start[axis]) / self.step[axis]).astype(int)
            inside.append(torch.as_tensor((index >= 0) & (index < self.shape[axis]),
                                          device=device))
            picks.append(torch.as_tensor(np.clip(index, 0, self.shape[axis] - 1),
                                         device=device))

        keep = (inside[0][:, None, None] & inside[1][None, :, None]
                & inside[2][None, None, :])
        data = self.data[picks[0][:, None, None], picks[1][None, :, None],
                         picks[2][None, None, :]]
        return Volume(torch.where(keep, data, torch.zeros_like(data)),
                      grid.start, grid.step, grid.dir_cos)

    def shrink(self, factor):
        """Sub-sample onto a coarser grid, as ``ShrinkVolume`` does.

        The legacy driver keeps ``start``, multiplies the step by ``factor``
        and asks for ``ceil((n - 1) / factor) + 1`` samples per axis
        (`nu_estimate_np_and_em.in:954`).  Only axes finer than
        ``factor * min(step)`` are thinned, so a volume that is already coarse
        along one axis keeps its resolution there.
        """
        target_step = abs(min(self.step, key=abs)) * factor
        thin = [abs(s) < target_step for s in self.step]

        counts = [int(np.ceil((n - 1) / factor)) + 1 if do else n
                  for n, do in zip(self.shape, thin)]
        steps = [s * factor if do else s for s, do in zip(self.step, thin)]

        return self.resample_like(
            Volume(torch.zeros(counts, device=self.data.device),
                   self.start, steps, self.dir_cos))


def load_volume(path):
    """Read a MINC file into a :class:`Volume` (as ``float64``)."""
    with _as_minc2(path) as readable:
        handle = minc2_file(readable)
        handle.setup_standard_order()
        data = torch.from_numpy(handle.load_complete_volume("float64"))
        # representation_dims() lists dimensions fastest-varying first, the
        # reverse of the array's axes.
        dims = list(reversed(handle.representation_dims()))
        handle.close()

    return Volume(data,
                  start=[d.start for d in dims],
                  step=[d.step for d in dims],
                  dir_cos=[d.dir_cos for d in dims])


def save_volume(path, volume, like=None, store_dtype=None):
    """Write ``volume`` to ``path``.

    ``like`` is a MINC file whose header (dimension names, storage type,
    metadata) the output should copy -- the equivalent of ``mincmath``'s
    ``-copy_header``, which is what the legacy drivers rely on to keep the
    output looking like the input.
    """
    data = np.ascontiguousarray(volume.data.detach().cpu().numpy(),
                                dtype=np.float64)
    out = minc2_file()
    template = None

    if like is not None:
        with _as_minc2(like) as readable:
            template = minc2_file(readable)
            out.imitate(template, store_type=store_dtype,
                        representation_type="float64")
            out.create(path)
            out.copy_metadata(template)
            template.close()
    else:
        out.define(_dims_of(volume), store_type=store_dtype or "float32",
                   representation_type="float64")
        out.create(path)

    _set_range_for_integer_storage(out, data)
    out.setup_standard_order()
    out.save_complete_volume(data)
    out.close()


def _set_range_for_integer_storage(handle, data):
    """Give integer-typed output the full dynamic range of the data.

    MINC stores integers together with a real range and rescales on read.
    Without this the values would be clipped to whatever range the template
    happened to carry -- the same bookkeeping ``mincmath`` does for its output.
    """
    if handle.store_dtype() not in ("float32", "float64"):
        handle.set_volume_range(float(data.min()), float(data.max()))


def _dims_of(volume):
    """Dimension records describing ``volume``, in minc2_simple's order."""
    axis_ids = [minc2_file.MINC2_DIM_Z, minc2_file.MINC2_DIM_Y,
                minc2_file.MINC2_DIM_X]
    dims = [minc2_dim(id=axis_id, length=int(n), start=float(s), step=float(d),
                      have_dir_cos=1, dir_cos=np.asarray(cos, dtype=np.float64))
            for axis_id, n, s, d, cos in zip(axis_ids, volume.shape,
                                             volume.start, volume.step,
                                             volume.dir_cos)]
    return list(reversed(dims))


class _as_minc2:
    """Context manager yielding a path ``minc2_simple`` can open.

    MINC1 and gzipped files are converted into a temporary MINC2 copy; MINC2
    files are passed straight through.
    """

    def __init__(self, path):
        self.path = path
        self._tmpdir = None

    def __enter__(self):
        try:
            minc2_file(self.path).close()
            return self.path
        except Exception:
            pass

        self._tmpdir = tempfile.mkdtemp(prefix="torch_n3_")
        converted = os.path.join(self._tmpdir, "minc2.mnc")
        subprocess.run(["mincconvert", "-2", "-clobber", self.path, converted],
                       check=True, capture_output=True)
        return converted

    def __exit__(self, *_exc):
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        return False

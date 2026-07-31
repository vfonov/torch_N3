"""Volume geometry and resampling, against the MINC tools N3 uses for them."""

import numpy as np
import torch

from tests.conftest import assert_close, requires_program, span
from torch_n3.volume import Volume, load_volume, save_volume


def test_load_puts_the_volume_in_standard_order(chunk):
    """chunk.mnc is stored xspace/zspace/yspace; we always see (Z, Y, X)."""
    assert chunk.shape == (52, 50, 91)
    np.testing.assert_allclose(chunk.step, [3.0, 2.0, 2.0])
    np.testing.assert_allclose(chunk.start, [-72.0, -126.0, -90.0])


def test_save_then_load_round_trips(tmp_path, chunk):
    save_volume(str(tmp_path / "copy.mnc"), chunk, store_dtype="float64")
    again = load_volume(str(tmp_path / "copy.mnc"))

    assert_close(again.data, chunk.data, atol=1e-9)
    np.testing.assert_allclose(again.step, chunk.step)
    np.testing.assert_allclose(again.start, chunk.start)


def test_shrink_matches_the_legacy_estimation_grid(legacy_output, chunk):
    """`ShrinkVolume` is `mincresample -nearest_neighbour` onto a coarser grid.

    ``nu_estimate_np_and_em.in:954`` picks the count and step and keeps the
    start; the recorded run was given exactly those.
    """
    recorded = legacy_output["mincresample.chunk_shrink3"]

    shrunk = chunk.shrink(3)

    assert shrunk.shape == tuple(recorded.shape)
    np.testing.assert_allclose(
        shrunk.step, legacy_output.scalar("mincresample.chunk_shrink3_step"))
    np.testing.assert_allclose(
        shrunk.start, legacy_output.scalar("mincresample.chunk_shrink3_start"))
    # mincresample stores its result in the input's 12-bit type, so it can be
    # half a level off; the voxels it picked are what matters.
    assert_close(shrunk.data, recorded, atol=span(recorded) / 4095)


def test_shrink_leaves_already_coarse_axes_alone():
    """An axis coarser than factor * min(step) keeps its resolution."""
    volume = Volume(torch.zeros((4, 20, 20)), start=(0, 0, 0), step=(20.0, 1.0, 1.0))

    shrunk = volume.shrink(4)

    assert shrunk.shape == (4, 6, 6)
    np.testing.assert_allclose(shrunk.step, [20.0, 4.0, 4.0])


def test_resample_like_matches_resample_labels(legacy_output, chunk, model_mask):
    """The driver moves a mask onto the estimation grid with `resample_labels`."""
    recorded = legacy_output["resample_labels.model_mask_on_chunk_shrink4"]

    resampled = model_mask.resample_like(chunk.shrink(4))

    assert torch.equal(resampled.data != 0, recorded != 0)


def test_resample_like_fills_outside_with_zero():
    """mincresample's default fill value is zero, and so is ours."""
    source = Volume(torch.ones((4, 4, 4)), start=(0, 0, 0), step=(1.0, 1.0, 1.0))
    grid = Volume(torch.zeros((6, 4, 4)), start=(0, 0, 0), step=(1.0, 1.0, 1.0))

    resampled = source.resample_like(grid)

    assert bool(resampled.data[:4].all())
    assert not bool(resampled.data[4:].any())


@requires_program("mincconvert")
def test_gzipped_minc1_input_is_readable():
    """The test data is MINC1 and gzipped; minc2_simple reads neither directly."""
    volume = load_volume("legacy/N3/testing/block.mnc.gz")

    assert volume.data.numel() > 0

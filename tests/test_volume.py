"""Volume geometry and resampling, against the MINC tools N3 uses for them."""

import numpy as np

from torch_n3.volume import Volume, load_volume


def test_load_puts_the_volume_in_standard_order(chunk):
    """chunk.mnc is stored xspace/zspace/yspace; we always see (Z, Y, X)."""
    assert chunk.shape == (52, 50, 91)
    np.testing.assert_allclose(chunk.step, [3.0, 2.0, 2.0])
    np.testing.assert_allclose(chunk.start, [-72.0, -126.0, -90.0])


def test_save_then_load_round_trips(workspace, chunk):
    workspace.write("copy.mnc", chunk)
    again = workspace.read("copy.mnc")

    np.testing.assert_allclose(again.data, chunk.data)
    np.testing.assert_allclose(again.step, chunk.step)
    np.testing.assert_allclose(again.start, chunk.start)


def test_shrink_matches_the_legacy_estimation_grid(workspace, chunk):
    """`ShrinkVolume` is `mincresample -nearest_neighbour` onto a coarser grid."""
    shrunk = chunk.shrink(3)
    source = workspace.write("chunk.mnc", chunk)

    # nu_estimate_np_and_em.in:954 -- same step and count, keeping start.
    lengths = [str(n) for n in reversed(shrunk.shape)]
    steps = [repr(float(s)) for s in reversed(shrunk.step)]
    workspace.run("mincresample", "-nearest_neighbour", "-clobber",
                  "-nelements", *lengths, "-step", *steps, source, workspace.at("shrunk.mnc"))
    reference = workspace.read("shrunk.mnc")

    assert shrunk.shape == reference.shape
    np.testing.assert_allclose(shrunk.step, reference.step)
    np.testing.assert_allclose(shrunk.start, reference.start)
    # mincresample stores its result in the input's 12-bit type, so it can be
    # half a level off; the voxels it picked are what matters.
    span = reference.data.max() - reference.data.min()
    np.testing.assert_allclose(shrunk.data, reference.data, rtol=0,
                               atol=span / 4095)


def test_shrink_leaves_already_coarse_axes_alone():
    """An axis coarser than factor * min(step) keeps its resolution."""
    volume = Volume(np.zeros((4, 20, 20)), start=(0, 0, 0), step=(20.0, 1.0, 1.0))

    shrunk = volume.shrink(4)

    assert shrunk.shape == (4, 6, 6)
    np.testing.assert_allclose(shrunk.step, [20.0, 4.0, 4.0])


def test_resample_like_matches_resample_labels(workspace, chunk, model_mask):
    """The driver moves a mask onto the estimation grid with `resample_labels`."""
    grid = chunk.shrink(4)
    resampled = model_mask.resample_like(grid)

    source = workspace.write("mask.mnc", model_mask, store_dtype="int16")
    like = workspace.write("grid.mnc", grid)
    workspace.run("resample_labels", "-clobber", "-quiet",
                  "-resample", "-like %s" % like, source, workspace.at("resampled.mnc"))

    reference = workspace.read("resampled.mnc")
    np.testing.assert_array_equal(resampled.data != 0, reference.data != 0)


def test_resample_like_fills_outside_with_zero():
    """mincresample's default fill value is zero, and so is ours."""
    source = Volume(np.ones((4, 4, 4)), start=(0, 0, 0), step=(1.0, 1.0, 1.0))
    grid = Volume(np.zeros((6, 4, 4)), start=(0, 0, 0), step=(1.0, 1.0, 1.0))

    resampled = source.resample_like(grid)

    assert resampled.data[:4].all()
    assert not resampled.data[4:].any()


def test_gzipped_minc1_input_is_readable():
    """The test data is MINC1 and gzipped; minc2_simple reads neither directly."""
    volume = load_volume("legacy/N3/testing/block.mnc.gz")

    assert volume.data.size > 0

"""`torch_n3.minc_tools` against the MINC programs it stands in for."""

import numpy as np

from torch_n3.backends import legacy
from torch_n3.minc_tools import apply_lut, bimodal_threshold


def test_apply_lut_matches_minclookup(workspace, chunk, chunk_mask):
    """Our continuous lookup is `minclookup -continuous`.

    Built the way the pipeline builds it: a sharpened histogram of the masked
    log volume, applied to every voxel.
    """
    inside = chunk_mask.data != 0
    log_volume = np.where(inside, np.log(np.clip(chunk.data, 1.0, None)), 0.0)

    value_range = legacy.histogram_range(log_volume[inside],
                                         initial=(log_volume.max(),
                                                  log_volume.min()))
    counts = legacy.histogram(log_volume[inside], 200, value_range)
    lut = legacy.sharpen_lut(counts, value_range, fwhm=0.15, noise=0.01)

    table = workspace.path / "table.txt"
    with open(table, "w") as fp:
        for position, value in zip(np.linspace(0.0, 1.0, lut.size), lut):
            fp.write("%.15g  %.15g\n" % (position, value))

    source = workspace.write("log.mnc", chunk.like(log_volume))
    workspace.run("minclookup", "-continuous", "-clobber",
                  "-range", repr(value_range[0]), repr(value_range[1]),
                  "-lookup_table", table, source, workspace.at("lut.mnc"))

    from_binary = workspace.read("lut.mnc").data
    from_python = apply_lut(log_volume, lut, value_range)

    np.testing.assert_allclose(from_python, from_binary, rtol=0, atol=1e-9)


def test_apply_lut_clamps_outside_the_domain():
    """Values past either end map to the end of the table, as minclookup does."""
    lut = np.array([10.0, 20.0, 30.0])
    values = np.array([-5.0, 0.0, 0.5, 1.0, 7.0])

    mapped = apply_lut(values, lut, (0.0, 1.0))

    np.testing.assert_allclose(mapped, [10.0, 10.0, 20.0, 30.0, 30.0])


def test_bimodal_threshold_matches_mincstats(workspace, chunk):
    """`nu_evaluate`'s automatic mask comes from `mincstats -biModalT`."""
    source = workspace.write("chunk.mnc", chunk)
    from_binary = float(workspace.run("mincstats", "-quiet", "-biModalT", source))

    from_python = bimodal_threshold(chunk.data)

    # mincstats prints four decimals of a value in the hundreds of thousands.
    assert from_python == np.float64(from_binary).astype(float) or \
        abs(from_python - from_binary) < 1e-3 * max(1.0, abs(from_binary))


def test_bimodal_threshold_separates_two_clusters():
    """The threshold falls in the gap, so it classifies both groups correctly."""
    rng = np.random.default_rng(7)
    background = rng.normal(10.0, 1.0, 5000)
    foreground = rng.normal(50.0, 3.0, 5000)

    threshold = bimodal_threshold(np.concatenate([background, foreground]))

    assert (background < threshold).mean() > 0.99
    assert (foreground > threshold).mean() > 0.99

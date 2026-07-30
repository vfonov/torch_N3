"""`torch_n3.minc_tools` against the MINC programs it stands in for."""

import torch

from tests.conftest import assert_close
from torch_n3 import blocks
from torch_n3.minc_tools import apply_lut, bimodal_threshold


def test_apply_lut_matches_minclookup(workspace, chunk, chunk_mask):
    """Our continuous lookup is `minclookup -continuous`.

    Built the way the pipeline builds it: a sharpened histogram of the masked
    log volume, applied to every voxel.
    """
    inside = chunk_mask.data != 0
    log_volume = torch.log(chunk.data.clamp(min=1.0))
    log_volume = torch.where(inside, log_volume, torch.zeros_like(log_volume))

    value_range = blocks.histogram_range(log_volume[inside],
                                         initial=(log_volume.max(),
                                                  log_volume.min()))
    counts = blocks.histogram(log_volume[inside], 200, value_range)
    lut = blocks.sharpen_lut(counts, value_range, fwhm=0.15, noise=0.01)

    table = workspace.path / "table.txt"
    with open(table, "w") as fp:
        positions = torch.linspace(0.0, 1.0, lut.numel(), dtype=torch.float64)
        for position, value in zip(positions, lut):
            fp.write("%.15g  %.15g\n" % (position, value))

    source = workspace.write("log.mnc", chunk.like(log_volume))
    workspace.run("minclookup", "-continuous", "-clobber",
                  "-range", repr(value_range[0]), repr(value_range[1]),
                  "-lookup_table", table, source, workspace.at("lut.mnc"))

    from_binary = workspace.read("lut.mnc").data
    from_python = apply_lut(log_volume, lut, value_range)

    assert_close(from_python, from_binary, atol=1e-9)


def test_apply_lut_clamps_outside_the_domain():
    """Values past either end map to the end of the table, as minclookup does."""
    lut = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    values = torch.tensor([-5.0, 0.0, 0.5, 1.0, 7.0], dtype=torch.float64)

    mapped = apply_lut(values, lut, (0.0, 1.0))

    assert_close(mapped, [10.0, 10.0, 20.0, 30.0, 30.0], atol=1e-12)


def test_bimodal_threshold_matches_mincstats(workspace, chunk):
    """`nu_evaluate`'s automatic mask comes from `mincstats -biModalT`."""
    source = workspace.write("chunk.mnc", chunk)
    from_binary = float(workspace.run("mincstats", "-quiet", "-biModalT", source))

    from_python = bimodal_threshold(chunk.data)

    # mincstats prints four decimals of a value in the hundreds of thousands.
    assert abs(from_python - from_binary) < 1e-3 * max(1.0, abs(from_binary))


def test_bimodal_threshold_separates_two_clusters():
    """The threshold falls in the gap, so it classifies both groups correctly."""
    generator = torch.Generator().manual_seed(7)
    background = 10.0 + torch.randn(5000, dtype=torch.float64,
                                    generator=generator)
    foreground = 50.0 + 3.0 * torch.randn(5000, dtype=torch.float64,
                                          generator=generator)

    threshold = bimodal_threshold(torch.cat([background, foreground]))

    assert float((background < threshold).to(torch.float64).mean()) > 0.99
    assert float((foreground > threshold).to(torch.float64).mean()) > 0.99

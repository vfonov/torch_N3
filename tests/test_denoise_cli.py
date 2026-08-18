"""``torch_n3.denoise_cli``: the standalone entry point around ``denoise()``.

The CLI adds no arithmetic of its own -- it is argument parsing plus the same
``load_volume`` / ``denoise`` / ``save_volume`` calls available from Python --
so its output is required to be exactly, not approximately, what those calls
produce directly. Exact comparisons pass ``--store-dtype float64``: without
it, ``save_volume``'s default inherits ``chunk.mnc``'s own (16-bit integer)
storage type via ``like=``, which quantizes the written data (CLAUDE.md,
"Pitfalls when porting" -- a program handed a file sees its contents
quantized) and is a property of MINC storage, not of this wrapper.
"""

import torch

from tests.conftest import legacy_data
from torch_n3 import denoise_cli
from torch_n3.blocks.denoise import denoise
from torch_n3.pipeline import DEFAULTS
from torch_n3.volume import load_volume


def test_defaults_match_the_pipeline():
    """The two tools must not drift apart silently (CLAUDE.md)."""
    args = denoise_cli.build_parser().parse_args(["in.mnc", "out.mnc"])
    assert args.search == DEFAULTS["denoise_search"]
    assert args.patch == DEFAULTS["denoise_patch"]
    assert args.strength == DEFAULTS["denoise_strength"]


def test_default_run_matches_a_direct_call(chunk, tmp_path):
    output = str(tmp_path / "out.mnc")
    denoise_cli.main([legacy_data("chunk.mnc"), output,
                      "--store-dtype", "float64"])

    expected = denoise(chunk.data)
    result = load_volume(output)
    assert torch.equal(result.data, expected)


def test_geometry_round_trips(chunk, tmp_path):
    output = str(tmp_path / "out.mnc")
    denoise_cli.main([legacy_data("chunk.mnc"), output])

    result = load_volume(output)
    assert result.shape == chunk.shape
    assert torch.allclose(torch.as_tensor(result.start),
                          torch.as_tensor(chunk.start))
    assert torch.allclose(torch.as_tensor(result.step),
                          torch.as_tensor(chunk.step))


def test_search_patch_strength_reach_the_filter(chunk, tmp_path):
    output = str(tmp_path / "out.mnc")
    denoise_cli.main([legacy_data("chunk.mnc"), output,
                      "--search", "2", "--patch", "2", "--strength", "0.5",
                      "--store-dtype", "float64"])

    expected = denoise(chunk.data, search=2, patch=2, strength=0.5)
    result = load_volume(output)
    assert torch.equal(result.data, expected)


def test_zero_strength_is_the_identity(chunk, tmp_path):
    output = str(tmp_path / "out.mnc")
    denoise_cli.main([legacy_data("chunk.mnc"), output, "--strength", "0",
                      "--store-dtype", "float64"])

    result = load_volume(output)
    assert torch.equal(result.data, chunk.data)

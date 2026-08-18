"""Command line front end for the standalone denoiser:
``python3 -m torch_n3.denoise_cli input.mnc output.mnc``.

This runs :func:`torch_n3.blocks.denoise.denoise` alone and writes its result
to a file. It is not part of N3: unlike ``torch_n3/cli.py``'s ``--denoise``,
which filters a copy of the volume that feeds the field *estimate* only, this
writes the filtered volume itself. There is one implementation of the filter
(``blocks/denoise.py``); there is no ``--backend`` here.
"""

import argparse
import sys
import time

from torch_n3.blocks.denoise import denoise
from torch_n3.pipeline import DEFAULTS
from torch_n3.volume import load_volume, save_volume


def build_parser():
    parser = argparse.ArgumentParser(
        prog="torch_n3.denoise_cli",
        description="Filter a MINC volume with one non-local-means pass "
                    "(Manjon et al. 2010, torch_n3/blocks/denoise.py).")
    parser.add_argument("input", help="MINC volume to filter")
    parser.add_argument("output", help="where to write the filtered volume")

    parser.add_argument("--search", type=int, default=DEFAULTS["denoise_search"],
                        help="search radius in voxels; the cost is cubic in "
                             "this and in nothing else (default: %(default)s)")
    parser.add_argument("--patch", type=int, default=DEFAULTS["denoise_patch"],
                        help="patch half-width in voxels; larger is a "
                             "stricter similarity test and so less smoothing "
                             "(default: %(default)s)")
    parser.add_argument("--strength", type=float,
                        default=DEFAULTS["denoise_strength"],
                        help="multiplies the estimated noise level; 0 is "
                             "exactly the identity (default: %(default)s)")
    parser.add_argument("--device", help="run on this torch device, e.g. "
                                         "cuda -- the filter costs some forty "
                                         "times a full N3 estimation on the "
                                         "CPU, and this is the way to make it "
                                         "practical")
    parser.add_argument("--store-dtype",
                        help="storage type of the output, overriding the one "
                             "inherited from --input's header")
    parser.add_argument("--verbose", action="store_true",
                        help="report elapsed time and device")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    volume = load_volume(args.input)
    if args.device:
        volume = volume.to(args.device)

    start = time.time()
    result = volume.like(denoise(volume.data, search=args.search,
                                 patch=args.patch, strength=args.strength))
    elapsed = time.time() - start

    save_volume(args.output, result, like=args.input,
               store_dtype=args.store_dtype)

    if args.verbose:
        print("denoised in %.2fs (device=%s)"
              % (elapsed, args.device or "cpu"))

    return 0


if __name__ == "__main__":
    sys.exit(main())

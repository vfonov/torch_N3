"""Command line front end: ``python3 -m torch_n3 input.mnc output.mnc``.

The options mirror ``nu_correct``'s, so an existing invocation mostly
translates across.  What is *not* here is the ``.imp`` mapping file:
:func:`torch_n3.pipeline.nu_estimate` returns the fitted spline as an object,
and ``--field`` writes it out as a volume, which is the useful form from
Python.  Writing N3's compact spline format would only matter for handing the
field back to the legacy tools.
"""

import argparse
import sys

from torch_n3.pipeline import DEFAULTS, evaluate_field, nu_estimate, nu_evaluate
from torch_n3.volume import load_volume, save_volume


def build_parser():
    parser = argparse.ArgumentParser(
        prog="torch_n3",
        description="Remove the smooth intensity non-uniformity from an MRI "
                    "volume (N3, Sled/Zijdenbos/Evans 1998).")
    parser.add_argument("input", help="MINC volume to correct")
    parser.add_argument("output", help="where to write the corrected volume")

    parser.add_argument("--mask", help="restrict the estimation to this region; "
                                       "strongly recommended")
    parser.add_argument("--evaluation-mask",
                        help="region where the field is used as fitted before "
                             "being extended outwards [default: derived from "
                             "the data, as nu_evaluate does]")
    parser.add_argument("--field", help="also write the estimated field here")

    protocol = parser.add_argument_group(
        "protocol", "defaults are what `nu_correct` uses with no options")
    protocol.add_argument("--distance", type=float, default=DEFAULTS["distance"],
                          help="B-spline knot spacing in mm (default: %(default)s)")
    protocol.add_argument("--fwhm", type=float, default=DEFAULTS["fwhm"],
                          help="assumed histogram blur (default: %(default)s)")
    protocol.add_argument("--noise", type=float, default=DEFAULTS["noise"],
                          help="Wiener constant (default: %(default)s)")
    protocol.add_argument("--bins", type=int, default=DEFAULTS["bins"],
                          help="histogram bins (default: %(default)s)")
    protocol.add_argument("--shrink", type=int, default=DEFAULTS["shrink"],
                          help="estimation-grid coarsening (default: %(default)s)")
    protocol.add_argument("--lambda", dest="lam", type=float,
                          default=DEFAULTS["lam"],
                          help="spline regularization (default: %(default)s)")
    protocol.add_argument("--iterations", type=int, nargs="+",
                          default=list(DEFAULTS["iterations"]),
                          help="iteration count, one per stopping stage")
    protocol.add_argument("--stop", type=float, nargs="+",
                          default=list(DEFAULTS["stop"]),
                          help="stopping threshold, one per stage")
    protocol.add_argument("--field-floor", type=float, default=0.1,
                          help="smallest field value allowed before dividing")

    parser.add_argument("--verbose", action="store_true",
                        help="report the field change at every iteration")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    volume = load_volume(args.input)
    mask = load_volume(args.mask) if args.mask else None
    evaluation_mask = (load_volume(args.evaluation_mask)
                       if args.evaluation_mask else None)

    field = nu_estimate(volume, mask=mask, verbose=args.verbose,
                        distance=args.distance, fwhm=args.fwhm, noise=args.noise,
                        bins=args.bins, shrink=args.shrink, lam=args.lam,
                        iterations=tuple(args.iterations), stop=tuple(args.stop))

    corrected = nu_evaluate(volume, field, mask=evaluation_mask,
                            field_floor=args.field_floor)
    save_volume(args.output, corrected, like=args.input)

    if args.field:
        save_volume(args.field,
                    evaluate_field(volume, field, mask=evaluation_mask,
                                   field_floor=args.field_floor),
                    store_dtype="float32")

    return 0


if __name__ == "__main__":
    sys.exit(main())

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


#: Shown after the options.  ``--distance`` and ``--lambda`` between them set
#: one thing -- how much the field is allowed to bend -- and changing either
#: alone is the commonest way to get a worse answer than the defaults give.
SMOOTHNESS_NOTE = """\
how --distance and --lambda interact:

  They are two halves of one setting.  --distance decides how many
  coefficients describe the field; --lambda decides how much bending is
  allowed between them.  Halve the spacing without raising the penalty and
  the extra coefficients are spent following tissue contrast, which comes
  back as field that was never there.

  Non-uniformity left behind after correcting a volume with a known 20%
  field planted on it, over both knobs (lower is better):

                                         --distance
      --lambda           200 mm    100 mm     50 mm
      1e-7 (default)      0.31%     0.61%     1.51%
      1e-6                0.13%     0.22%     1.01%
      1e-5                0.33%     0.17%     0.25%
      1e-4                0.85%     0.54%     0.35%

  So: raise --lambda by about a decade each time you halve --distance, and
  err high rather than low.  The best cell per column is 1e-6, 1e-5, 1e-5 --
  one decade for the first halving and none for the second -- so that rule
  overshoots at 50 mm on purpose.  It is the right way to be wrong: there,
  1e-4 costs 0.10 points against the best cell, while the default 1e-7 costs
  1.26 and leaves the volume further from the truth than it started.

  Measured on one synthetic field (tests/test_field_recovery.py), which is
  smoother than a real coil profile.  A 40% field gives the same table
  roughly doubled, with the same minima; both are in README.md.  Read them
  as the shape of the trade-off, not as a table to tune from.
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="torch_n3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Remove the smooth intensity non-uniformity from an MRI "
                    "volume (N3, Sled/Zijdenbos/Evans 1998).",
        epilog=SMOOTHNESS_NOTE)
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
                          help="B-spline knot spacing in mm, the scale below "
                               "which the field cannot vary; raise --lambda "
                               "with it if you lower this (default: %(default)s)")
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
                          help="penalty on the field's bending energy; about a "
                               "decade more per halving of --distance, see the "
                               "note below (default: %(default)s)")
    protocol.add_argument("--iterations", type=int, nargs="+",
                          default=list(DEFAULTS["iterations"]),
                          help="iteration count, one per stopping stage")
    protocol.add_argument("--stop", type=float, nargs="+",
                          default=list(DEFAULTS["stop"]),
                          help="stopping threshold, one per stage")
    protocol.add_argument("--field-floor", type=float, default=0.1,
                          help="smallest field value allowed before dividing")

    parser.add_argument("--backend", choices=("torch", "legacy"),
                        default=DEFAULTS["backend"],
                        help="which implementation of the blocks to run "
                             "(default: %(default)s)")
    parser.add_argument("--device", help="run on this torch device, e.g. cuda")
    parser.add_argument("--verbose", action="store_true",
                        help="report the field change at every iteration")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    def read(path):
        volume = load_volume(path)
        return volume.to(args.device) if args.device else volume

    volume = read(args.input)
    mask = read(args.mask) if args.mask else None
    evaluation_mask = read(args.evaluation_mask) if args.evaluation_mask else None

    field = nu_estimate(volume, mask=mask, verbose=args.verbose,
                        distance=args.distance, fwhm=args.fwhm, noise=args.noise,
                        bins=args.bins, shrink=args.shrink, lam=args.lam,
                        iterations=tuple(args.iterations), stop=tuple(args.stop),
                        backend=args.backend)

    corrected = nu_evaluate(volume, field, mask=evaluation_mask,
                            field_floor=args.field_floor, backend=args.backend)
    save_volume(args.output, corrected, like=args.input)

    if args.field:
        save_volume(args.field,
                    evaluate_field(volume, field, mask=evaluation_mask,
                                   field_floor=args.field_floor,
                                   backend=args.backend),
                    store_dtype="float32")

    return 0


if __name__ == "__main__":
    sys.exit(main())

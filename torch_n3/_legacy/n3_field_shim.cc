/*
 * n3_field_shim.cc -- the `correct_field` half of the shim.
 *
 * Kept apart from n3_shim.cc because volume_io and EBTKS both define VIO_ROUND
 * and friends; correctField.cc has the same problem and solves it the same
 * way.  The Laplace solve itself is *not* reimplemented here: we build a pair
 * of volume_io volumes around the caller's arrays and hand them to the legacy
 * smooth().
 */

#include <config.h>

/* correctField.cc is a command-line program, so it is pulled in textually with
 * its entry point renamed rather than linked as an object -- that keeps the
 * legacy tree untouched while still compiling the original smooth().  (It
 * cannot simply be added to the source list: the build renames every main() to
 * the same symbol, and sharpen_hist.cc has already claimed it.) */
#undef main
#define main n3_correct_field_unused_main
#include "correctField.cc"
#undef main

/* A double-precision volume_io volume with an identity voxel<->real mapping,
 * so that get/set_volume_real_value round-trip exactly. */
static VIO_Volume make_volume(const int count[3], const double step[3],
                              double lo, double hi)
{
  static VIO_STR names[] = { (VIO_STR) MIxspace, (VIO_STR) MIyspace,
                             (VIO_STR) MIzspace };
  int sizes[VIO_MAX_DIMENSIONS];
  VIO_Real seps[VIO_MAX_DIMENSIONS];

  for (int i = 0; i < 3; i++) {
    sizes[i] = count[i];
    seps[i] = step[i];
  }

  VIO_Volume v = create_volume(3, names, NC_DOUBLE, TRUE, 0.0, 0.0);
  set_volume_sizes(v, sizes);
  set_volume_separations(v, seps);
  alloc_volume_data(v);
  set_volume_voxel_range(v, lo, hi);
  set_volume_real_range(v, lo, hi);
  return v;
}

extern "C" {

int n3_correct_field(double *field, const unsigned char *mask,
                     const int *count, const double *step)
{
  int sizes[VIO_MAX_DIMENSIONS];
  VIO_Real seps[VIO_MAX_DIMENSIONS];
  int i, j, k;

  double lo = field[0], hi = field[0];
  long total = (long) count[0] * count[1] * count[2];
  for (long at = 1; at < total; at++) {
    if (field[at] < lo) lo = field[at];
    if (field[at] > hi) hi = field[at];
  }
  if (hi <= lo)
    hi = lo + 1.0;

  VIO_Volume volume = make_volume(count, step, lo, hi);
  VIO_Volume mask_volume = make_volume(count, step, 0.0, 1.0);

  long at = 0;
  for (i = 0; i < count[0]; i++)
    for (j = 0; j < count[1]; j++)
      for (k = 0; k < count[2]; k++, at++) {
        set_volume_real_value(volume, i, j, k, 0, 0, field[at]);
        set_volume_real_value(mask_volume, i, j, k, 0, 0, mask[at] ? 1.0 : 0.0);
      }

  for (i = 0; i < 3; i++) {
    sizes[i] = count[i];
    seps[i] = step[i];
  }
  smooth(sizes, seps, volume, mask_volume);

  at = 0;
  for (i = 0; i < count[0]; i++)
    for (j = 0; j < count[1]; j++)
      for (k = 0; k < count[2]; k++, at++)
        field[at] = get_volume_real_value(volume, i, j, k, 0, 0);

  delete_volume(volume);
  delete_volume(mask_volume);
  return 0;
}

}  /* extern "C" */

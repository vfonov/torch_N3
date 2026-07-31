/*
 * n3_field_shim.cc -- the `correct_field` half of the shim.
 *
 * Kept apart from n3_shim.cc because volume_io and EBTKS both define VIO_ROUND
 * and friends; correctField.cc has the same problem and solves it the same
 * way.  The Laplace solve itself is *not* reimplemented here: we build a pair
 * of volumes around the caller's arrays and hand them to the legacy smooth().
 *
 * "volume_io" here is `compat/volume_io.h`, not MINC's -- a plain double
 * buffer behind the handful of accessors smooth() calls.  See that file for
 * why.
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

extern "C" {

int n3_correct_field(double *field, const unsigned char *mask,
                     const int *count, const double *step)
{
  int sizes[VIO_MAX_DIMENSIONS];
  VIO_Real seps[VIO_MAX_DIMENSIONS];
  int i, j, k;

  VIO_Volume volume = n3_create_volume(count, step);
  VIO_Volume mask_volume = n3_create_volume(count, step);
  if (volume == NULL || mask_volume == NULL) {
    delete_volume(volume);
    delete_volume(mask_volume);
    return 1;
  }

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

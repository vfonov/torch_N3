/*
 * time_stamp.h -- stand-in for MINC's, so correctField.cc compiles unmodified.
 *
 * The only caller is its command-line main(), which the build renames away;
 * the result is only ever written into a MINC header.  Returning NULL is what
 * the caller already tolerates -- it guards the matching free().
 */

#ifndef N3_COMPAT_TIME_STAMP_H
#define N3_COMPAT_TIME_STAMP_H

#include <stddef.h>

static char *time_stamp(int argc, char **argv)
{
  (void) argc;
  (void) argv;
  return NULL;
}

#endif /* N3_COMPAT_TIME_STAMP_H */

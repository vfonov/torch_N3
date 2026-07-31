/*
 * volume_io.h -- the sliver of MINC's volume_io that correctField.cc uses.
 *
 * `smooth()` does its entire multigrid SOR solve on two flat arrays it
 * allocates itself; volume_io appears only in the loop that copies the caller's
 * volume in, the loop that copies the answer back out, and the command-line
 * `main()` that the build renames away.  Rather than link libminc2 for that --
 * which also drags in a second HDF5 alongside the one minc2_simple already
 * uses -- this header stands in for it, so that
 * `torch_n3/_legacy/n3/CorrectField/correctField.cc` still compiles byte for
 * byte as N3 shipped it.
 *
 * A VIO_Volume here is a plain double buffer with an identity voxel<->real
 * mapping, which is what the shim was constructing out of the real volume_io
 * anyway.  Values therefore round-trip exactly instead of through a rescale.
 *
 * This is deliberately not a general volume_io: it is 3D only, ignores the
 * time and vector axes, and the file-I/O entry points abort if anything ever
 * reaches them.
 */

#ifndef N3_COMPAT_VOLUME_IO_H
#define N3_COMPAT_VOLUME_IO_H

#include <stdio.h>
#include <stdlib.h>

#define VIO_MAX_DIMENSIONS 5

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/* volume_io spells these itself; EBTKS defines its own, hence the guards. */
#ifndef MIN
#define MIN(x, y) (((x) <= (y)) ? (x) : (y))
#endif
#ifndef VIO_ABS
#define VIO_ABS(x) (((x) > 0) ? (x) : (-(x)))
#endif

#define MI_ORIGINAL_TYPE 0
#define VIO_OK 0

typedef double VIO_Real;
typedef char *VIO_STR;

struct n3_volume_struct {
  int sizes[VIO_MAX_DIMENSIONS];
  VIO_Real seps[VIO_MAX_DIMENSIONS];
  VIO_Real real_min, real_max;
  double *data;
};

typedef struct n3_volume_struct *VIO_Volume;

static long n3_volume_index(VIO_Volume v, int i, int j, int k)
{
  return ((long) i * v->sizes[1] + j) * v->sizes[2] + k;
}

static VIO_Real get_volume_real_value(VIO_Volume v, int i, int j, int k,
                                      int t, int c)
{
  (void) t; (void) c;
  return v->data[n3_volume_index(v, i, j, k)];
}

static void set_volume_real_value(VIO_Volume v, int i, int j, int k,
                                  int t, int c, VIO_Real value)
{
  (void) t; (void) c;
  v->data[n3_volume_index(v, i, j, k)] = value;
}

/* The data is already real-valued doubles, so the range is bookkeeping only.
 * (The real volume_io would rescale the voxel values through it here; that
 * rescale is exactly the round-off this header removes.) */
static void set_volume_real_range(VIO_Volume v, VIO_Real lo, VIO_Real hi)
{
  v->real_min = lo;
  v->real_max = hi;
}

static void get_volume_sizes(VIO_Volume v, int sizes[])
{
  int i;
  for (i = 0; i < VIO_MAX_DIMENSIONS; i++)
    sizes[i] = v->sizes[i];
}

static void get_volume_separations(VIO_Volume v, VIO_Real seps[])
{
  int i;
  for (i = 0; i < VIO_MAX_DIMENSIONS; i++)
    seps[i] = v->seps[i];
}

static int get_volume_n_dimensions(VIO_Volume v)
{
  (void) v;
  return 3;
}

/* Allocate a zeroed 3D volume.  Not part of volume_io's API -- the shim calls
 * this instead of create_volume/alloc_volume_data/set_volume_*. */
static VIO_Volume n3_create_volume(const int count[3], const double step[3])
{
  int i;
  long total = (long) count[0] * count[1] * count[2];
  VIO_Volume v = (VIO_Volume) calloc(1, sizeof(struct n3_volume_struct));

  if (v == NULL)
    return NULL;
  for (i = 0; i < VIO_MAX_DIMENSIONS; i++) {
    v->sizes[i] = (i < 3) ? count[i] : 1;
    v->seps[i] = (i < 3) ? step[i] : 1.0;
  }
  v->data = (double *) calloc((size_t) total, sizeof(double));
  if (v->data == NULL) {
    free(v);
    return NULL;
  }
  return v;
}

static void delete_volume(VIO_Volume v)
{
  if (v != NULL) {
    free(v->data);
    free(v);
  }
}

/* Reached only from correctField.cc's renamed-away main(); nothing in the
 * extension reads or writes a file. */
static int n3_no_file_io(const char *what)
{
  fprintf(stderr, "torch_n3 legacy shim: %s is not available\n", what);
  abort();
  return 1;
}

static int input_volume(const char *path, int n_dims, void *dim_names,
                        int type, int signed_flag, double lo, double hi,
                        int create, VIO_Volume *volume, void *options)
{
  (void) path; (void) n_dims; (void) dim_names; (void) type;
  (void) signed_flag; (void) lo; (void) hi; (void) create;
  (void) volume; (void) options;
  return n3_no_file_io("input_volume");
}

static int output_modified_volume(const char *path, int type, int signed_flag,
                                  double lo, double hi, VIO_Volume volume,
                                  const char *original, const char *history,
                                  void *options)
{
  (void) path; (void) type; (void) signed_flag; (void) lo; (void) hi;
  (void) volume; (void) original; (void) history; (void) options;
  return n3_no_file_io("output_modified_volume");
}

#endif /* N3_COMPAT_VOLUME_IO_H */

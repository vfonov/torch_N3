/*
 * n3_shim.h -- C ABI over the original N3 C++ code.
 *
 * This header is compiled as C++ *and* fed verbatim to cffi's cdef(), so it
 * must stay free of preprocessor directives and C++ constructs.
 *
 * Every function here delegates to code in legacy/N3/; nothing numerical is
 * reimplemented.  The only thing the shim adds is an array-in / array-out
 * calling convention, so that the legacy behaviour can be used as an oracle
 * from Python without going through text files and subprocesses.
 *
 * Return convention: 0 on success, negative on failure.
 */

/* ---------------------------------------------------------------- histogram
 * Reuses DHistogram / WHistogram (legacy/N3/src/VolumeHist).
 *
 * Bin i is centred on  min_val + i * (max_val - min_val) / (nbins - 1).
 * parzen != 0 selects WHistogram, which splits each sample linearly between
 * the two nearest bin centres; otherwise each sample lands in one bin.
 * Samples outside the outer half-bins are discarded.
 */
int n3_histogram(const double *values, long n, int nbins,
                 double min_val, double max_val, int parzen,
                 double *counts);

/* Range scan used by `volume_hist -auto_range`, replicated exactly,
 * including its else-if quirk.  Writes {min, max}. */
int n3_histogram_range(const double *values, long n, double *out_min_max);

/* ------------------------------------------------------------------ sharpen
 * Reuses gaussian(), weiner() and non_negative() from
 * legacy/N3/src/SharpenHist/sharpen_hist.cc, and drives them with the same
 * sequence of operations as that program's main().
 *
 * counts: nbins histogram counts;  lut: nbins output mapping values.
 * deblur != 0 skips the Wiener deconvolution (the program's -blur flag).
 */
int n3_sharpen_lut(const double *counts, int nbins,
                   double min_bin, double max_bin,
                   double fwhm, double noise, int deblur,
                   double *lut);

/* ---------------------------------------------------------------- B-splines
 * Wraps TBSplineVolume (legacy/N3/src/Splines/TBSpline.cc) directly: the
 * object is created, fed data points, fitted and evaluated by the legacy
 * code itself.
 *
 * start/step are the world coordinate of voxel (0,0,0) and the voxel
 * separations; count is the volume shape.  distance is the knot spacing and
 * lambda the bending-energy weight.
 */
void *n3_spline_create(const double *start, const double *step,
                       const int *count, double distance, double lambda);
int   n3_spline_add(void *handle, int x, int y, int z, double value);
int   n3_spline_fit(void *handle);
int   n3_spline_n_coefficients(void *handle);
int   n3_spline_coefficients(void *handle, double *out);
int   n3_spline_evaluate_grid(void *handle, double *out);
void  n3_spline_free(void *handle);

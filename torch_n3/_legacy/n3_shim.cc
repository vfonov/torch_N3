/*
 * n3_shim.cc -- C ABI over the original N3 C++ code.  See n3_shim.h.
 *
 * Nothing here recomputes N3's mathematics.  Each entry point either calls a
 * legacy class (DHistogram, WHistogram, TBSplineVolume) or calls the legacy
 * free functions gaussian()/weiner()/non_negative() in the same order as
 * sharpen_hist.cc's main().
 */

#include <config.h>
#include <math.h>
#include <string.h>

#include <EBTKS/Matrix.h>

#include "DHistogram.h"
#include "WHistogram.h"
#include "TBSpline.h"

/* Defined in legacy/N3/src/SharpenHist/sharpen_hist.cc, which we compile
 * alongside this file (with main() renamed out of the way). */
DblMat gaussian(double fwhm, int size);
CompMat weiner(CompMat &blur, double noise);
void non_negative(DblMat *X);

extern "C" {

int n3_histogram(const double *values, long n, int nbins,
                 double min_val, double max_val, int parzen,
                 double *counts)
{
  if (nbins <= 0 || n < 0)
    return -1;

  DHistogram *hist = parzen
      ? (DHistogram *) new WHistogram(min_val, max_val, (unsigned) nbins)
      : new DHistogram(min_val, max_val, (unsigned) nbins);

  for (long i = 0; i < n; i++)
    hist->add(values[i]);

  for (int i = 0; i < nbins; i++)
    counts[i] = (*hist)[i];

  delete hist;
  return 0;
}

int n3_histogram_range(const double *values, long n, double *out_min_max)
{
  if (n <= 0)
    return -1;

  /* This mirrors minchist.cc:163-186 including its `else if`: a value can
   * only update one of the two bounds per visit.  Replicated deliberately so
   * that the PyTorch port can be compared against it exactly. */
  double lo = values[0], hi = values[0];
  for (long i = 0; i < n; i++) {
    if (values[i] < lo)
      lo = values[i];
    else if (values[i] > hi)
      hi = values[i];
  }
  if (hi <= lo)
    hi = lo + 1.0;

  out_min_max[0] = lo;
  out_min_max[1] = hi;
  return 0;
}

int n3_sharpen_lut(const double *counts, int nbins,
                   double min_bin, double max_bin,
                   double fwhm, double noise, int deblur,
                   double *lut)
{
  if (nbins <= 1 || min_bin == max_bin)
    return -1;

  DblMat X(nbins, 1, 0.0);
  for (int i = 0; i < nbins; i++)
    X(i, 0) = counts[i];

  /* sharpen_hist.cc:113-186, verbatim in structure. */
  int padded_size = int(pow(2, ceil(log((double) nbins) / log(2.0)) + 1) + .5);
  int offset = (padded_size - nbins) / 2;

  double slope = (max_bin - min_bin) / double(nbins - 1);
  CompMat blur = fft(gaussian(fwhm / slope, padded_size), 0, 1);
  CompMat filter = weiner(blur, noise);

  DblMat X_padded(padded_size, 1, 0.0);
  X_padded.insert(X, offset, 0);

  DblMat f;
  if (!deblur) {
    f = real(ifft(pmultEquals(asCompMat(X_padded).fft(0, 1), filter), 0, 1));
    non_negative(&f);
  } else {
    f = X_padded;
  }

  DblMat moment(padded_size, 1);
  for (int i = 0; i < padded_size; i++)
    moment(i, 0) = (min_bin + (i - offset) * slope) * f(i, 0);

  DblMat Y_padded =
      pdiv(real(ifft(pmultEquals(asCompMat(moment).fft(0, 1), blur), 0, 1)),
           real(ifft(pmultEquals(asCompMat(f).fft(0, 1), blur), 0, 1)));

  for (int i = 0; i < nbins; i++) {
    double v = Y_padded(offset + i, 0);
    lut[i] = isfinite(v) ? v : 0.0;
  }
  return 0;
}

/* A fitted spline plus the grid it was defined on, so that evaluate_grid()
 * does not need the caller to repeat the geometry. */
struct n3_spline_handle {
  TBSplineVolume *spline;
  int count[3];
};

void *n3_spline_create(const double *start, const double *step,
                       const int *count, double distance, double lambda)
{
  n3_spline_handle *h = new n3_spline_handle;
  h->spline = new TBSplineVolume(start, step, count, distance, lambda);
  for (int i = 0; i < 3; i++)
    h->count[i] = count[i];
  return (void *) h;
}

int n3_spline_add(void *handle, int x, int y, int z, double value)
{
  n3_spline_handle *h = (n3_spline_handle *) handle;
  return h->spline->addDataPoint(x, y, z, value) ? 0 : -1;
}

int n3_spline_fit(void *handle)
{
  n3_spline_handle *h = (n3_spline_handle *) handle;
  return h->spline->fit() ? 0 : -1;
}

int n3_spline_n_coefficients(void *handle)
{
  n3_spline_handle *h = (n3_spline_handle *) handle;
  return (int) h->spline->getCoefficients().size();
}

int n3_spline_coefficients(void *handle, double *out)
{
  n3_spline_handle *h = (n3_spline_handle *) handle;
  DblArray coef = h->spline->getCoefficients();
  for (unsigned i = 0; i < coef.size(); i++)
    out[i] = coef[i];
  return 0;
}

int n3_spline_evaluate_grid(void *handle, double *out)
{
  n3_spline_handle *h = (n3_spline_handle *) handle;
  long at = 0;
  for (int x = 0; x < h->count[0]; x++)
    for (int y = 0; y < h->count[1]; y++)
      for (int z = 0; z < h->count[2]; z++)
        out[at++] = (*h->spline)(x, y, z);
  return 0;
}

void n3_spline_free(void *handle)
{
  n3_spline_handle *h = (n3_spline_handle *) handle;
  delete h->spline;
  delete h;
}

}  /* extern "C" */

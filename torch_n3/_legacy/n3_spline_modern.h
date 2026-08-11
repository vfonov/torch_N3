/*
 * n3_spline_modern.h -- factory for TBSplineVolumeModern (n3_spline_modern.cc),
 * the equilibration+Cholesky solve path reachable through the oracle backend
 * as solver="equilibrated".  See n3_spline_modern.cc for the algorithm and
 * why this is a subclass rather than a re-vendored TBSpline.cc.
 */
#ifndef N3_SPLINE_MODERN_H
#define N3_SPLINE_MODERN_H

#include "TBSpline.h"

TBSplineVolume *n3_make_modern_spline(const DblMat &domain,
                                      const double start[VDIM],
                                      const double step[VDIM],
                                      const int count[VDIM],
                                      double distance, double lambda,
                                      int allocate_flag);

#endif

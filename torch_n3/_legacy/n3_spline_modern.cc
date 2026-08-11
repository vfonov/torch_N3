/*
 * n3_spline_modern.cc -- TBSplineVolumeModern, the equilibration+Cholesky
 * solve path from legacy/N3 commit 7d84753 ("n3pipeline_core: solve spline
 * normal equations by equilibration + Cholesky"), reachable through the
 * oracle backend as solver="equilibrated".
 *
 * TBSpline::fit() (TBSpline.cc:328-342) is virtual and touches only
 * `protected` members (_haveData, _n, _lambda, _nsamples, _AtA, _AtF, _coef,
 * _fitted, bendingEnergyTensor), so this subclass overrides fit() to call a
 * differently-named solve method instead of the inherited, non-virtual
 * solveSymmetricSystem -- no vendored byte in n3/Splines/TBSpline.{cc,h}
 * changes.  See torch_n3/_legacy/n3/README.md ("If a file needs to change to
 * build, the fix belongs in the shim").
 *
 * solveEquilibrated below transcribes the algorithm from TBSpline.cc's
 * N3_SPLINE_MODERN_SOLVE block.  It calls dsysv_/dposv_ directly -- the
 * plain Fortran LAPACK symbols -- rather than through EBTKS_dsysv/
 * EBTKS_dposv: there is no LAPACKE indirection to route around here, since
 * build_legacy.py already resolves dsysv_ as an undefined symbol at build
 * time, satisfied at import time by whatever LAPACK PyTorch has already
 * loaded (see build_legacy.py's LAPACK_LIBS comment).  dposv_ is part of the
 * same base Fortran interface every LAPACK distribution ships, unlike
 * LAPACKE (the optional C wrapper Debian's OpenBLAS package omits, which is
 * why legacy/EBTKS's own "lapacke" backend is unbuildable on that machine).
 *
 * Integer width: TBSpline.cc declares these same two symbols with its own
 * `_integer` (a `long`, matching f2c's INTEGER mapping, which is what
 * EBTKS_dsysv/EBTKS_dposv and the bundled clapack expect).  That widening is
 * harmless there because solveSymmetricSystem only ever narrows the result
 * to `(int)` once at the very end and never inspects the raw value.  This
 * file *does* branch on the raw `info` (Cholesky success vs. fall back to
 * dsysv), and the symbols this build actually links -- a real Fortran LAPACK,
 * either shared with PyTorch's or the system one build_legacy.py names via
 * N3_LAPACK_LIBS -- use the standard LP64 Fortran INTEGER, 4 bytes, not 8.
 * Declaring the scalars `long` here would leave the upper 4 bytes of `info`
 * uninitialized after a call that only writes 4, so a successful dposv_
 * reads back as a nonzero garbage value and every fit silently takes the
 * dsysv_ fallback -- measured while bringing this up: the modern path always
 * printed "not numerically positive definite" and fell back, even for a
 * matrix that is SPD.  Plain `int` is the correct width for the ABI this
 * file actually calls.
 */

#include <config.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "TBSpline.h"
#include "n3_spline_modern.h"

extern "C" {
int dsysv_(char *uplo, int *n, int *nrhs, double *a, int *lda, int *ipiv,
           double *b, int *ldb, double *work, int *lwork, int *info);
int dposv_(char *uplo, int *n, int *nrhs, double *a, int *lda,
           double *b, int *ldb, int *info);
}

class TBSplineVolumeModern : public TBSplineVolume {
public:
  TBSplineVolumeModern(const double start[VDIM], const double step[VDIM],
                       const int count[VDIM], double distance,
                       double lambda = _default_lambda,
                       int allocate_flag = TRUE)
    : TBSplineVolume(start, step, count, distance, lambda, allocate_flag) {}

  TBSplineVolumeModern(const DblMat &domain, const double start[VDIM],
                       const double step[VDIM], const int count[VDIM],
                       double distance, double lambda = _default_lambda,
                       int allocate_flag = TRUE)
    : TBSplineVolume(domain, start, step, count, distance, lambda,
                     allocate_flag) {}

  virtual Boolean fit() override
  {
    if (_haveData == FALSE)
      return FALSE;

    int info;
    DblMat A;
    bendingEnergyTensor(_n, A);
    A *= (_lambda * _nsamples);
    A += _AtA;

    _coef = solveEquilibrated(A, _AtF, &info).array();
    _fitted = (info == 0);
    return _fitted;
  }

private:
  // Same contract as TBSpline::solveSymmetricSystem: A is undefined after
  // the call, *info == 0 on success.
  DblMat solveEquilibrated(DblMat &A, DblMat b, int *info)
  {
    if (A.getrows() != A.getcols() || A.getrows() != b.getrows() ||
       b.getcols() != 1)
      {
        DblMat x;
        *info = -1;
        return x;
      }

    int n = (int) A.getcols();
    int nrhs = 1;
    int lda = n;
    int ldb = n;
    int w_info;

    double *Ael = (double *) *A.getEl();
    double *bel = (double *) *b.getEl();

    // Symmetric (Jacobi) equilibration: (D A D)(D^-1 x) = D b with
    // D = diag(1/sqrt(Aii)).  A non-positive diagonal entry means the
    // matrix is not positive definite; leave it unscaled in that case and
    // let dsysv handle it, exactly as TBSpline.cc does.
    std::vector<double> d((size_t) n);
    bool scaled = true;
    for (int i = 0; i < n; i++)
      {
        double aii = Ael[(size_t) i * (size_t) n + (size_t) i];
        if (!(aii > 0.0)) { scaled = false; break; }
        d[(size_t) i] = std::sqrt(aii);
      }
    if (scaled)
      {
        for (int r = 0; r < n; r++)
          for (int c = 0; c < n; c++)
            Ael[(size_t) r * (size_t) n + (size_t) c] /=
              (d[(size_t) r] * d[(size_t) c]);
        for (int i = 0; i < n; i++)
          bel[i] /= d[(size_t) i];
      }

    int solved = 0;
    {
      // Cholesky first; on failure (matrix not numerically SPD) fall back
      // to dsysv on the same equilibrated system, restoring it from the
      // copy dposv_ overwrote.
      std::vector<double> Acopy(Ael, Ael + (size_t) n * (size_t) n);
      std::vector<double> bcopy(bel, bel + (size_t) n);

      dposv_((char *) "U", &n, &nrhs, Ael, &lda, bel, &ldb, &w_info);

      if (w_info == 0)
        solved = 1;
      else
        {
          std::fprintf(stderr, "TBSplineVolumeModern: dposv info=%d, "
                       "matrix is not numerically positive definite; "
                       "falling back to dsysv\n", w_info);
          std::memcpy(Ael, &Acopy[0], sizeof(double) * (size_t) n * (size_t) n);
          std::memcpy(bel, &bcopy[0], sizeof(double) * (size_t) n);
        }
    }

    if (!solved)
      {
        int *ipiv = new int[n];
        double work;
        int lwork = 1;
        dsysv_((char *) "U", &n, &nrhs, Ael, &lda, ipiv, bel, &ldb,
               &work, &lwork, &w_info);
        delete [] ipiv;
      }

    if (scaled)   // recover x from the scaled unknown y = D^-1 x
      for (int i = 0; i < n; i++)
        bel[i] /= d[(size_t) i];

    *info = (int) w_info;
    return b;
  }
};

// Factory, declared in n3_spline_modern.h: n3_shim.cc constructs through
// this rather than naming TBSplineVolumeModern directly, so the class stays
// private to this translation unit.
TBSplineVolume *n3_make_modern_spline(const DblMat &domain,
                                      const double start[VDIM],
                                      const double step[VDIM],
                                      const int count[VDIM],
                                      double distance, double lambda,
                                      int allocate_flag)
{
  return new TBSplineVolumeModern(domain, start, step, count, distance,
                                  lambda, allocate_flag);
}

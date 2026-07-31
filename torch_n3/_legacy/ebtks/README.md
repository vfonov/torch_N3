# Vendored EBTKS sources

The subset of EBTKS the legacy backend compiles, copied **byte for byte** from
`legacy/EBTKS/` so that `build_legacy.py` needs no EBTKS installation. All the
headers are here (they are small, and pruning headers buys nothing); the `.cc`
list is pruned to what the extension actually pulls in, checked by dropping each
one and looking for new undefined symbols:

| Source | Pulled in by |
|---|---|
| `templates/Matrix.cc` | `Mat<double>`, out of `TBSpline.cc` and `sharpen_hist.cc` |
| `templates/MatrixSupport.cc` | `fft` / `ifft`, the deconvolution in `sharpen_hist.cc` |
| `templates/ValueMap.cc` | `LUT<double>`, out of `DHistogram.cc` |
| `src/MString.cc`, `src/Path.cc`, `src/FileIO.cc`, `src/OrderedCltn.cc` | reached from those |

`src/dcomplex.cc`, `src/fcomplex.cc`, `templates/Pool.cc` and
`templates/SimpleArray.cc` are *not* compiled: everything needed from them is
template or inline code that lands in the including translation unit. The rest
of EBTKS — `Histogram`, `Polynomial`, `TrainingSet`, `backProp`, `amoeba`,
`popen`, `CachedArray`, `Dictionary`, `Matrix3D` — is not reachable from N3's
blocks at all.

**`clapack/` is deliberately not vendored.** EBTKS bundles an f2c'd LAPACK, and
`TBSpline.cc`'s `dsysv_` used to resolve to it; the build links the system
LAPACK/BLAS instead. Both solve the same near-singular normal equations and
neither is wrong, but they do not agree, and the difference survives the
iteration — see the LAPACK section of the top-level `README.md` for the
measurements and `PROBLEMS.md` §8 for the two bounds that moved because of it.

**Do not modify these files.** Same rule as `../n3/`: they are half of the
oracle the PyTorch blocks are checked against. Anything that has to change to
make the build work belongs in `../compat/` or in the generated `config.h`.

`COPYING` is EBTKS's licence, carried over with the code.

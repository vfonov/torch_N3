# Vendored N3 sources

The subset of the original N3 C++ that the legacy backend compiles, copied here
**byte for byte** from `legacy/N3/src/` so that `build_legacy.py` needs nothing
but the MINC toolkit (EBTKS, LAPACK) to build. The directory names are the
original ones, so every `legacy/N3/src/<Dir>/<file>` reference in the docs maps
straight onto `torch_n3/_legacy/n3/<Dir>/<file>`.

**Do not modify these files.** They are the oracle the PyTorch blocks in
`torch_n3/blocks/` are checked against; editing them would make the comparison
meaningless. If a file needs to change to build, the fix belongs in the shim
(`n3_shim.cc`, `n3_field_shim.cc`) or in the generated `config.h`/`version.h`
that `build_legacy.py` writes.

| File | Used for |
|---|---|
| `Splines/{Spline,TBSpline}.{cc,h}` | the regularized tensor cubic B-spline fit |
| `VolumeHist/{DHistogram,WHistogram}.{cc,h}` | the masked histogram and its Parzen variant |
| `SharpenHist/sharpen_hist.cc` | histogram deconvolution and the intensity mapping |
| `SharpenHist/args.{cc,h}`, `MincProg/print_version.c` | referenced by `sharpen_hist.cc`'s unused `main()`; linked only to satisfy the linker |
| `CorrectField/correctField.cc` | SOR extension of the field outside the mask (`#include`d by `n3_field_shim.cc`) |

Nothing else from N3 is needed: the rest of the pipeline is Perl drivers and
standalone programs, transcribed into `torch_n3/pipeline.py` rather than linked.

`COPYING` is N3's licence, carried over with the code.

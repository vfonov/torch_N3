# Plan: PyTorch reimplementation of N3

## Strategy

N3 is decomposed into a handful of **pure functional blocks**. Every block gets two
interchangeable implementations behind one signature:

- `legacy` — calls the original N3 C/C++ through a CFFI extension (Stage 1)
- `torch` — pure PyTorch (Stage 2)

The pipeline is written once, in Python, against the block interfaces. Because the two
backends are swappable, the legacy backend is a **numerical oracle**: every Stage-2 test is
"torch block output == legacy block output on the same input". That is the red-green loop, and
it is what keeps the port honest without needing a human to eyeball volumes.

Readability is a requirement, not a nice-to-have: each block is one small module, named for
the mathematics it performs, with the N3 paper's notation in the docstring and a pointer to
the corresponding lines of `legacy/N3/`.

## Feasibility check (done — no installs needed)

- `libEBTKS.a` + `EBTKS/*.h` are installed under `/opt/minc/1.9.18.13`; `liblapack.so` /
  `libblas.so` are present system-wide.
- `legacy/N3/src/Splines/{Spline,TBSpline}.cc` **compile as-is** against them (verified). The
  `MRegion.h` include that appears to be missing is behind `#ifdef HAVE_INTERFACE`, which N3
  never defines.
- CFFI compiles C++ via `source_extension='.cpp'`, so the shim can instantiate
  `TBSplineVolume` directly — real reuse, no copied code.
- `torch` 2.13 (CUDA available), `numpy`, `scipy`, `minc2_simple` all present.

## Block inventory

| Block | Legacy source | Stage-1 route |
|---|---|---|
| `load_volume` / `save_volume` | — | `minc2_simple` (already CFFI) |
| `histogram(values, mask, bins, range, parzen)` | `src/VolumeHist/WHistogram.h:65` | shim reuses `WHistogram::add` semantics |
| `sharpen_lut(counts, min_bin, max_bin, fwhm, noise, deblur)` | `src/SharpenHist/sharpen_hist.cc:98-190` | shim, EBTKS `fft`/`ifft` |
| `apply_lut(volume, lut)` | `minclookup -continuous` | shim / linear interp |
| `bspline_fit(values, mask, geometry, distance, lambda, subsample)` | `src/Splines/TBSpline.cc:190,290` | shim instantiates `TBSplineVolume` |
| `bspline_evaluate(spline, geometry)` | `src/Splines/TBSpline.cc:310` | same object |
| `bimodal_threshold(volume)` | `src/VolumeStats`, `mincstats -biModalT` | shim / port |
| `shrink(volume, factor)` | `nu_estimate_np_and_em.in:954` + `mincresample` | Python + `minc2_simple` |
| pipeline glue (log/exp, mask, stopping rule) | `nu_estimate_np_and_em.in:60-193` | Python, backend-agnostic |

## Stage 1 — Python pipeline over the legacy C code

**Progress**: steps 1–3 done. `torch_n3/_legacy/` builds a CFFI extension that compiles
`Spline.cc`, `TBSpline.cc`, `DHistogram.cc`, `WHistogram.cc` and `sharpen_hist.cc` straight
from `legacy/N3/src`, and `torch_n3/backends/legacy.py` exposes them as numpy functions.
`tests/test_legacy_backend.py` (8 tests) passes, including a direct comparison against the
installed `sharpen_hist` binary. Steps 4–5 (pipeline + end-to-end reference match) are next.


1. **Skeleton**: `torch_n3/` package, `tests/`, pytest config. No installation; run from `/app`.
2. **CFFI extension** `torch_n3/_legacy/`: `n3_shim.cc` exposing an `extern "C"` API for the
   blocks above, built against `legacy/N3/src/{Splines,SharpenHist,VolumeHist}` +
   `libEBTKS.a` + LAPACK. Built in-place, checked in as a build script.
3. **Legacy backend** `torch_n3/backends/legacy.py`: numpy-array-in / numpy-array-out wrappers.
4. **Pipeline** `torch_n3/pipeline.py`: the N3 loop, exactly as `nu_estimate_np_and_em.in`
   sequences it, parameterized by backend.
5. **End-to-end validation** against the installed binaries and
   `legacy/N3/testing/brain_nu_ref.mnc.gz`. This pins the pipeline before any block is
   rewritten.

Exit criterion: `pytest tests/ -k stage1` green, and the Python pipeline reproduces
`nu_correct` on `chunk.mnc.gz` within the tolerance the legacy suite itself uses (1e-4 RMS).

## Stage 2 — replace blocks with PyTorch, one at a time

Each block, in this order (cheapest and most independently verifiable first):

1. `histogram` (incl. the Parzen/linear-split variant)
2. `apply_lut` (continuous lookup = linear interpolation + clamping)
3. `bimodal_threshold`
4. `sharpen_lut` — `torch.fft`, Wiener filter, the `E[u|v]` ratio
5. `bspline_fit` / `bspline_evaluate` — the hard one: cubic tensor B-spline normal equations
   with the bending-energy penalty, solved with `torch.linalg`
6. `shrink` / resampling

Per block, red-green:

- **Red**: `tests/test_<block>.py` compares `torch` backend to `legacy` backend on fixtures
  drawn from the real test volumes. It fails because the torch implementation does not exist.
- **Green**: implement the smallest readable thing that passes.
- **Refactor**: name things after the maths; docstring cites the N3 paper and the legacy lines.

After each block flips, the end-to-end test re-runs with that block on `torch` and the rest on
`legacy`, so a regression is always attributable to one block.

Exit criterion: all blocks `torch`, the legacy CFFI extension no longer imported by the
pipeline (only by tests), and the end-to-end result still matches `brain_nu_ref.mnc.gz`.

## Stage 3 — make it a PyTorch program, not a transcription

Only after Stage 2 is green:

- batched / GPU execution, `float32` vs `float64` tolerance study
- optional autograd-friendliness of the field fit
- CLI mirroring `nu_correct`'s interface

## Conventions

- Parity tests run in `float64`. Per-block tolerances are recorded in the test, not global.
- `legacy/N3/` is never modified.
- Nothing is installed. If a dependency turns out to be missing, stop and ask.

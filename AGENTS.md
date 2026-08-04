# AGENTS.md

N3 (Non-parametric Non-uniform intensity Normalization, Sled/Zijdenbos/Evans 1998)
reimplemented in PyTorch. Removes the smooth multiplicative intensity inhomogeneity
("bias field") from MRI volumes without a tissue model.

Full reference: `CLAUDE.md` (authoritative, read it before substantive work). This file is a
concise orientation.

## Layout

| Path | Role |
|---|---|
| `torch_n3/` | The PyTorch port. `pipeline.py` is N3; `blocks/` implements each stage (`histogram.py`, `sharpen.py`, `spline.py`, `field.py`); `volume.py` is MINC I/O; `minc_tools.py`; `optimize.py` + `blocks/sharpness.py` are a second estimator (`--method`, Hoyer sparsity or within-cluster variance). |
| `legacy/N3/`, `legacy/EBTKS/` | Original C++/Perl and its numerical library. Source of truth for the algorithm. Separate git repositories; changes to them are made on branch **`aislop`**, never on `master` or `develop-1.9.18`. The installed oracle is `/opt/minc/1.9.18.13/bin` and is not written to; this tree's own build installs to `/app/legacy/_install/bin`. |
| `torch_n3/_legacy/` | Vendored N3 + EBTKS sources; the CFFI backends/legacy backend (compiles the original C++, now only an oracle). **Do not modify** the `n3/`/`ebtks/` sources: they are what every `tests/` comparison is measured against, and must stay byte-identical to the files they vendor from `legacy/N3/src`. |
| `tests/` | `test_*.py` compare PyTorch blocks against recorded C++ answers via the shim. `tests/regenerate_reference.py` is the only thing that shells out. `tests/margins.py`, `convergence.py`, `tables.py`, `parzen.py`, `denoise.py` measure and print (not tests). |
| `experiments/` | Monte-Carlo simulations (`experiments.recovery` / `summarize`), not collected by pytest. |
| `minc2-simple/` | MINC2 binding source; `minc2-simple/USAGE.md` is the Python API reference. |
| `PROBLEMS.md` | Known weaknesses, fitted thresholds, and every comparison's measured margin. Section numbers are load-bearing (cited as "§8", "§9"...). |
| `tests/data/` | Test volumes as MINC2, checked in. |

## The port vs. the legacy

- `torch_n3.backends.resolve("torch" | "legacy")` switches the pipeline; the legacy block
  compiles the original C++ through a CFFI shim.
- **"Legacy" is ambiguous**: the *installed* N3 is a Perl script driving separate executables
  communicating only through files (rounded in transit); the *legacy backend* is the same C++
  called in-process on float64 buffers, and does not reproduce the installed programs.
- End-to-end N3 comparisons are meaningful only to ~3 digits; iteration amplifies rounding.
  Constrain blocks, not pipelines. `python3 -m tests.convergence` reports where backends stop
  agreeing (a property of the machine's LAPACK).

## Algorithm (as implemented by legacy code)

Estimation runs on a shrunk grid; output field is evaluated at full resolution. Everything is
in the **log-intensity domain** until the final `exp`.

1. Optionally `-shrink` (nearest-neighbour) input to a coarser grid.
2. `log(clamp(input, 1, ...))`.
3. Build mask (background threshold, user mask, bimodal threshold, or average brain mask).
4. Iterate: `corrected = log_volume − residue`; sharpen via histogram deconvolution to get `U`;
   `working = log_volume − U`; smooth `working` with a regularized tensor-cubic-B-spline fit
   over the mask → new `residue`. Stop when the population std of the field change < `-stop`.
5. `field = exp(residue)`; optionally normalize by mean in mask.
6. Fit compact splines, write `.imp`; `nu_evaluate` evaluates it, extends outside the mask
   (`correct_field`), clamps to floor, then `output = input / field`.

Key components: histogram (WHistogram, linear/binned split) `volume_hist`; deconvolution
(Gaussian kernel + Wiener filter, Nadaraya-Watson conditional expectation mapping)
`sharpen_hist`; field fit `spline_smooth`/TBSpline; `correct_field` (sequential SOR, not
exactly reproducible). Default protocol: `-distance 200 -fwhm 0.15 -noise 0.01 -bins 200
-iterations 50 -stop 0.001 -shrink 4 -lambda 1e-7 -parzen -log -sharpen`.

## Pitfalls when porting (headline items)

- **float64 everywhere**: `torch.linspace`/`zeros` default to float32; a stray float32 costs six
  digits. `torch.std` defaults to `unbiased=True`; the stopping rule needs the population std
  (`unbiased=False`).
- The FFT length is `2^(ceil(log2(nbins))+1)` — 512 at 200 bins, not 256 — and the histogram
  sits centred at an `offset`, not at index 0. Gaussian kernel is built on the bin grid and wraps.
- `minclookup -continuous` is linear interpolation with clamping outside the domain.
- The spline fit's normal equations are nearly singular (cond ~1e13 at 200 mm); compare fitted
  *fields*, never coefficients. `solver="qr"` (stacked least squares) is better-conditioned
  (~2.3e6) and CPU/GPU-reproducible; `solver="normal"` is the default and what every recorded
  reference was produced with. Five solvers exist but `sparse` (LSQR) does not converge and `dr`
  is only useful for sweeping `lambda`.
- The auto histogram range is rounded to `%lf`'s six decimals; a voxel on a bin boundary flips
  and moves a whole count, which is the start of cross-backend divergence. Don't chase it.
- `-parzen`/`-window` is a linear split, not a Parzen window. `--parzen-sigma` (a Gaussian
  Parzen window, off by default) and `--denoise` (non-local-means prefilter, off, not a port)
  are port *modifications* measured in `README.md`, `tests/parzen.py`/`tests.denoise.py` and
  `experiments/`. Don't quote their `tests/README` analytic tables for a real volume.
- Thresholds on intensities are thresholds on a scale: pin to a *range* (1st-to-99th centile),
  never a level or `max()`. Use `kthvalue`, not `torch.quantile` (refuses >2^24 elements).
  The estimation mask is built by `pipeline.estimation_mask`, shared by `nu_estimate` and
  `nu_optimize`: N3's fixed `background` of 1 when a mask is supplied, Otsu's threshold from
  the data when none is (`bimodal=None`, `--bimodal`/`--no-bimodal`). The masked path is
  bit-identical to what every recorded reference was produced under.
- The installed N3's precision is not float precision: intermediates pass between programs as
  MINC files and are rounded, so a float64 port cannot reproduce `brain_nu_ref.mnc.gz` to 1e-4.

## Testing / measurement protocol

- `python3 -m pytest` runs `tests/`. No test runs an N3 program; answers come from
  `tests/reference/legacy.npz` (regenerated by `tests/regenerate_reference.py`; `git diff`
  should be empty after rerun).
- Measurement scripts (each a module, run as `python3 -m tests.<name>`):
  `margins` (`PROBLEMS.md`'s table), `convergence`, `tables` (`--lambda` × `--distance`),
  `parzen`, `denoise`.
- **Never widen a tolerance or alter inputs to make a test pass.** Fix the defect, remove the
  confound, or deliberately change the (incorrect) requirement in its own commit with the
  measurement recorded. Two known-brittle comparisons: the stopping-rule quantisation (check the
  iteration count first), and whole-volume extreme-value stats (compare RMS, not `max`).
- The `--lambda` × `--distance` tables (in `README.md` and `cli.py`) are the most-cited numbers
  but largely unasserted; `python3 -m tests.tables` re-measures them. Update both copies
  together.

## Environment

- Python 3.12, `torch` 2.13 (CUDA, check device at runtime), `numpy`, `scipy`, `minc2_simple`.
- `minc2_simple` reads **MINC2 (HDF5) only**; things in `legacy/N3/testing/` are MINC1+gzipped
  and must go through `mincconvert -2`. `torch_n3.volume.load_volume` does this transparently.
- `representation_dims()`/shape are reversed vs. numpy axes; `voxel_to_world` takes storage
  order. Pass absolute paths when driving legacy programs (relative paths unreliable).
- Do not install packages; if something is missing, say so and stop.

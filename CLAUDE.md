# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Reimplement **N3** (Non-parametric Non-uniform intensity Normalization, Sled/Zijdenbos/Evans
1998) in **PyTorch**. N3 removes the smooth multiplicative intensity inhomogeneity ("bias
field") from MRI volumes without a tissue model.

`legacy/N3/` is the original C++/Perl implementation — **reference only, do not modify**. It is
also already compiled and installed, so it can be run to produce ground truth for any part of
the pipeline. MINC volume I/O from Python goes through `minc2_simple`, already installed.

## Layout

| Path | Role |
|---|---|
| `legacy/N3/src/NUcorrect/*.in` | Perl drivers — these hold the *pipeline*: `nu_estimate.in` (= `nu_correct`), `nu_estimate_np_and_em.in` (the iteration loop), `sharpen_volume.in`, `nu_evaluate.in`. |
| `legacy/N3/src/SharpenHist/` | `sharpen_hist` — histogram deconvolution and the intensity mapping. The mathematical core. |
| `legacy/N3/src/Splines/`, `legacy/N3/src/SplineSmooth/` | `spline_smooth` — regularized tensor cubic B-spline / thin-plate spline field fit, and the `.imp` compact-field format. |
| `legacy/N3/src/VolumeHist/` | `volume_hist` — masked histogram, with the Parzen (`-window`) variant in `WHistogram.h`. |
| `legacy/N3/src/EvaluateField/`, `src/CorrectField/` | `evaluate_field` (`.imp` → field volume), `correct_field` (extend field outside mask). |
| `legacy/N3/src/VolumeStats/` | `volume_stats` — mean/stddev/biModalT used for masking and the stopping rule. |
| `legacy/N3/testing/` | Test volumes **and a reference result** (`brain_nu_ref.mnc.gz`) — the regression target. |
| `legacy/N3/model_data/N3/` | ICBM/average-305 brain masks used by `-auto_mask` on Talairach-space input. |
| `minc2-simple/` | Source checkout of the MINC2 binding (already installed). `minc2-simple/USAGE.md` is the Python API reference. |
| `torch_n3/` | The port. `pipeline.py` is N3 itself; `blocks/` is the PyTorch implementation of each stage (`histogram.py`, `sharpen.py`, `spline.py`, `field.py`); `volume.py` is MINC I/O and geometry; `minc_tools.py` holds the two MINC utilities N3 leans on; `backends/legacy.py` wraps the original C++ through the CFFI shim in `_legacy/` and is now only an oracle. `backends.resolve("torch"\|"legacy")` switches the pipeline between them. |
| `torch_n3/_legacy/n3/` | The N3 sources the shim compiles, vendored byte for byte from `legacy/N3/src` (see its `README.md`) so the legacy backend builds without an N3 checkout. **Do not modify** — they are the oracle. Everything below still cites `legacy/N3/src` as the *source of truth*; for the files listed there, the two are the same bytes. |
| `torch_n3/_legacy/ebtks/` | Likewise EBTKS, vendored from `legacy/EBTKS`: all headers, seven `.cc` files, no `clapack/`. The build links the **system** LAPACK/BLAS instead — the one thing the extension still needs from outside. What that swap costs is measured in `README.md`; it is not nothing. |
| `torch_n3/_legacy/compat/` | Stand-ins for `<volume_io.h>`, `<time_stamp.h>` and `<ParseArgv.h>`, placed first on the include path so that `correctField.cc` and `args.cc` compile **unmodified** without libminc2. `smooth()` does its whole solve on flat `float`/`char` arrays; volume_io was only ever the marshalling at its edges. |
| `tests/margins.py`, `tests/convergence.py` | Neither is a test; both measure and print. `python3 -m tests.margins` prints `PROBLEMS.md`'s table — where every comparison sits against its bound. `python3 -m tests.convergence` prints where the two backends stop agreeing as iterations grow, which is a property of the machine's LAPACK rather than of this code; run it on any new platform before trusting an end-to-end number there. |
| `PROBLEMS.md` | Known weak spots in the test suite: fitted thresholds, tight margins, dropped assertions, and the measured margin of every comparison. |
| `tests/data/` | The test volumes as MINC2, checked in: byte-for-byte the same images as `legacy/N3/testing/` and the installed model mask. Converted once so that reading them needs nothing installed. |
| `tests/data/brain_nu_ref_legacy.mnc` | Not one of N3's files: this pipeline's own output on `brain.mnc` with the legacy blocks, under `inputs.PLATFORM_PROTOCOL`, checked in so another machine/BLAS/device can be held to it (`tests/test_reproducibility.py`). Written by the regeneration script. Regenerating churns MINC's `ident` header attribute; the voxel data is reproducible. |
| `tests/` | `test_pipeline.py` is `legacy/N3/testing/CMakeLists.txt`'s cases, re-expressed as comparisons, run on both backends. `test_histogram.py`, `test_sharpen.py`, `test_spline.py`, `test_field.py` compare each PyTorch block against the same C++ through the shim. **No test runs an N3 program**: their answers are recorded in `tests/reference/legacy.npz` by `tests/regenerate_reference.py` (the only thing that shells out) and the inputs they were given live in `tests/inputs.py`, shared by both. Re-run the script and `git diff` should be empty (except for `brain_nu_ref_legacy.mnc`'s `ident` header). `test_reproducibility.py` holds every backend and device to a volume checked into the repo — the cross-platform canary. |

## The algorithm as the legacy code actually implements it

Everything except the final division happens in the **log-intensity domain**; the estimated
field `F` is accumulated in log space and only exponentiated at the very end.

Driver: `legacy/N3/src/NUcorrect/nu_estimate_np_and_em.in:60-193`.

1. Optionally resample input to a coarser grid (`-shrink`, nearest-neighbour, `ShrinkVolume`
   at `:954`). Estimation runs on that grid; the *output* is a spline, so the final field is
   evaluated at full resolution.
2. `log_volume = log(clamp(input, 1, 1.7e308))` — the clamp exists to stop `log` of near-zero
   eating the dynamic range.
3. Build the mask: threshold `input > background_threshold` (default 1), intersected with the
   user mask, or a bimodal threshold (`-bimodalT`), or the average brain mask when
   `-auto_mask` and the volume is tagged Talairach.
4. Field estimate `residue` starts at zero (or `log(initial)`).
5. **Iterate** (`for $iter < $iterations`):
   - `corrected = log_volume − residue`
   - **Sharpen**: build the estimate `U` of the corrected volume via histogram deconvolution
     (below). This is `sharpen_estimate` at `:500` → `sharpen_volume` → `volume_hist`,
     `sharpen_hist`, `minclookup -continuous`. Result is re-masked.
   - `working = log_volume − U`  (residual field, log domain)
   - **Smooth** `working` with the regularized B-spline fit over the mask → new `residue`.
   - Stop when the change in the field is small (below).
6. `field = exp(residue)`; optionally divide by its mean in the mask (`-normalize_field`).
7. Fit *compact* splines to the field and write the `.imp` mapping file (`compact_spline_volume`
   at `:627`, format written by `outputCompactField`, `legacy/N3/src/SplineSmooth/fieldIO.cc:81`).
8. `nu_evaluate` (`nu_evaluate.in:46-80`): evaluate the `.imp` on the input grid, extend it
   outside the mask (`correct_field`), clamp to `field_floor`, then
   `output = input / field`.

### Histogram sharpening — `legacy/N3/src/SharpenHist/sharpen_hist.cc:98-190`

Given the masked histogram `X` of the corrected log volume (default 200 bins, auto range):

- Pad to `padded_size = 2^(ceil(log2(nbins)) + 1)`, centred at `offset = (padded − n)/2`.
- `slope = (max_bin − min_bin)/(n − 1)`; the Gaussian kernel width is `fwhm/slope` **in bin
  units**. Kernel is unit-area, centred at index 0 and wrapped (`gaussian()` at `:199`).
- `H = fft(gaussian)`; Wiener restoration filter `G = conj(H) / (conj(H)·H + noise)`
  (`weiner()` at `:222`). `noise` is a bare additive constant, not scaled by signal power.
- `f = max(ifft(fft(X_padded) · G), 0)` — the deconvolved intensity distribution.
  With `-blur`/`$nodeblur_flag` the deconvolution is skipped and `f = X_padded`.
- `moment[i] = (min_bin + (i − offset)·slope) · f[i]`
- **Mapping** `U = ifft(fft(moment)·H) / ifft(fft(f)·H)` — i.e. `E[u | v]`, a Nadaraya–Watson
  conditional expectation under the Gaussian kernel. Non-finite entries → 0.
- Written as a lookup table over the normalized domain and applied with `minclookup
  -continuous` (linear interpolation between table entries).

### Histogram — `legacy/N3/src/VolumeHist/WHistogram.h:65-93`

`-parzen`/`-window` does **not** mean a Parzen kernel density estimate: each sample is split
linearly between the two nearest bin centres (mass `1−offset` / `offset`). Without it each
sample increments a single bin. Samples outside the first/last half-bin are discarded.

### Field smoothing — `legacy/N3/src/Splines/TBSpline.cc`

Tensor product of **cubic B-splines** with uniform knot spacing `distance`, fit by normal
equations with a bending-energy penalty (`fit()` at `:290`):

```
(AᵀA + λ · n_samples · J) c = Aᵀ f
```

`J` is the bending-energy tensor (`bendingEnergyTensor` at `:393`), `λ` is `-lambda`
(default `1e-7` in the N3 protocol). Only voxels inside the mask contribute; `-subsample n`
takes every n-th voxel. The spline **evaluates to exactly 0 outside its domain**; without
`-full_support` the domain is shrunk to the mask bounding box.

`-tp_spline` selects a thin-plate spline instead; `b_spline` is the default and what
`nu_correct` uses.

### Stopping rule

`field_CV` (`nu_estimate_np_and_em.in:701`) is misnamed: it returns the **standard deviation**
(not the coefficient of variation) of the voxelwise change in the log field inside the mask.
Iteration stops when that is below `-stop`. `-iterations a b` / `-stop x y` define staged
thresholds.

## Default N3 protocol

What `nu_correct` passes down with no options (`nu_estimate.in:418-438, :491, :519`):

| Parameter | Default | Note |
|---|---|---|
| `-distance` | 200 mm | B-spline knot spacing; the dominant smoothness parameter |
| `-fwhm` (sharpen) | 0.15 | Gaussian width in log-intensity units |
| noise | 0.01 | Wiener constant |
| `-bins` | 200 | histogram bins |
| `-iterations` | 50 | (`-V0.9` protocol: `10 20`) |
| `-stop` | 0.001 | |
| `-shrink` | 4 | estimation-grid subsampling factor |
| `-lambda` | 1e-7 | spline regularization |
| `-spline_subsample` | 1 | |
| flags | `-parzen -log -sharpen 0.15 0.01` | the EM/tissue-model path is dead code — it `die`s without `-sharpen` |

## Running the legacy reference (ground truth)

All N3 binaries and Perl drivers are installed under `/opt/minc/1.9.18.13/bin` and are on
`PATH`: `nu_correct`, `nu_estimate`, `nu_estimate_np_and_em`, `nu_evaluate`, `sharpen_volume`,
`sharpen_hist`, `volume_hist`, `spline_smooth`, `evaluate_field`, `correct_field`,
`volume_stats`, plus MINC tools (`mincinfo`, `mincmath`, `minclookup`, `mincblur`,
`mincresample`, `mincstats`). Model masks are in `/opt/minc/1.9.18.13/share/N3/`.

Test data lives in `legacy/N3/testing/` (gzipped MINC1 — `minc2_simple` cannot open it; the
suite uses MINC2 copies in `tests/data/`, see its README):
`chunk.mnc.gz` + `chunk_mask.mnc.gz` (small, 91×52×50 — use for fast iteration),
`brain.mnc.gz` + `brain_mask.mnc.gz`, `block.mnc.gz`, and `brain_nu_ref.mnc.gz`, the reference
`nu_correct` output. `legacy/N3/testing/CMakeLists.txt` lists the exact legacy invocations,
including the reference comparison via `compare_nu_result.pl` at tolerance `1e-4`.

Isolating a single stage is the cheapest way to validate a PyTorch component, e.g.:

```bash
volume_hist -bins 200 -auto_range -mask chunk_mask.mnc.gz chunk.mnc.gz h.txt \
    -clobber -text -select 1 -quiet -window     # -window = the Parzen variant
sharpen_hist -clobber -fwhm 0.15 -noise 0.01 -quiet h.txt h.sharp
```

`h.txt` carries `# domain: <min_bin> <max_bin>` and two columns (bin centre, count);
`h.sharp` is the two-column lookup table. Both are plain text, so they diff directly against
tensors.

For anything a *test* needs to compare against, put the case in
`tests/regenerate_reference.py` instead of shelling out from the test — build the input in
`tests/inputs.py` so both sides use the same one, run the script once, and read the answer
back through the `legacy_output` fixture. Storage policy is in `tests/reference.py`: whole
volumes at `float32`, small arrays at `float64`, and record only what the assertion looks at
(the recovery test keeps the masked quarter of the volume, not all of it).

One trap: a program that was handed a *file* saw its contents quantised, and modelling MINC's
16-bit scaling in Python gets it wrong by a whole quantisation step. Round-trip through
`inputs.as_stored()` instead, which both the script and the test call.

## Test tolerances

Known weak spots in the current suite — thresholds that were fitted to the measurement,
one that is far too loose, three that are tight enough to flake, and the assertions that
were dropped rather than satisfied — are listed in **[PROBLEMS.md](PROBLEMS.md)**, with
where every comparison currently sits against its bound. Read it before adjusting a
tolerance, and add to it rather than quietly fixing a bound.

**Never widen a tolerance to make a test pass.** A threshold states what the code is required
to do; moving it after the fact turns the test into a record of what the code happens to do,
and destroys the only evidence that something changed. The same applies to quietly changing a
test's inputs until it goes green.

When an assertion fails, the options are, in order: find the defect; or find the confound and
remove *that* (a comparison between two implementations that stopped at different iterations
is not measuring what it claims to, and controlling the iteration count is a fix — loosening
the bound is not); or, if the requirement was genuinely wrong, change it deliberately, in its
own commit, with the measurement and the reasoning written down.

Thresholds here should be round numbers chosen from something real — the 16-bit quantum of the
MINC file a legacy program wrote (`span(reference) / 65535`), the six decimals `%lf` prints
(`1e-6`), a single shared constant for cross-implementation agreement — and not the measured
difference plus a margin. If a bound cannot be justified without running the code first, prefer
asserting the property that motivated it.

Two known-brittle comparisons, so nobody reaches for the threshold when they fail:

- **The stopping rule quantises everything downstream.** Iteration stops at `change < 0.001`;
  two implementations whose per-iteration change differs in the fifth decimal can land on
  either side of that and run a different number of iterations, which moves the output by far
  more than any block-level difference. Check the iteration count first (`verbose=True`).
- **Extreme-value statistics over a whole volume** (`max |a - b|` across 240k voxels) are
  dominated by a handful of mask-edge voxels and are not worth a fixed bound. Compare RMS.

### Published numbers no test checks

Most measured numbers in the docs are reproduced by `python3 -m tests.margins`, so a stale
one shows up as a mismatch against `PROBLEMS.md`. The **`--lambda` × `--distance` tables are
the exception**, and they are the most cited numbers in the repository:

| Where | What |
|---|---|
| `README.md`, "Does it actually remove a bias field?" | both tables, 20% and 40% planted |
| `torch_n3/cli.py`, `SMOOTHNESS_NOTE` | the 20% table, shown by `--help` |
| `tests/test_field_recovery.py` | asserts the `1e-7` row at every spacing (the recovery sweep), and `LAMBDAS = [1e-5, 1e-4]` at the *finest* spacing only — the `regularized` fixture fixes `distance = min(DISTANCES)` |

That is 10 of the 24 published cells. The other **14 are asserted by nothing** — the whole
`1e-6` row included, which is where the best cell at the default spacing lives. The two
copies are kept in step by hand, and the prose conclusions drawn from them (which cell is
best per column, the decade-per-halving rule, the asymmetry that justifies erring high) are
checked by nobody. They are also *not* invariant: they run the pipeline for 30 iterations,
which is well past the histogram knife-edge, so a change to any block — or to which LAPACK
the shim links — can move them.

Re-measure them, do not adjust them, and change both copies together. It is 24 cells, two
`nu_estimate` calls each, about 15 s in total; drive it over `AMPLITUDES × DISTANCES ×
(1e-7, 1e-6, 1e-5, 1e-4)` exactly as the `regularized` fixture does, and take
`ratio.std(unbiased=False)` of the recovered field over the planted one, each divided by the
same implementation's baseline on the untouched reference. Verified unchanged 2026-07-31,
after the LAPACK swap.

If you find yourself relying on a cell, the honest fix is to widen `LAMBDAS` and assert the
shape being claimed — that each column has an interior minimum, and where — rather than to
keep trusting a table by hand.

## Gotchas when porting

- Do the arithmetic in log space and keep the field there until the final `exp`. Ratios in
  intensity space are differences in log space; the legacy code relies on this everywhere.
- The FFT length is `2^(ceil(log2(nbins))+1)` — at 200 bins that is 512, not 256, and the
  histogram sits centred at `offset`, not at index 0. Off-by-`offset` errors silently shift
  the mapping.
- The Gaussian kernel is built on the *bin* grid (`fwhm/slope`) and wraps around index 0.
- Histogram range is recomputed per iteration (`-auto_range`), so the lookup domain moves
  between iterations.
- `minclookup -continuous` interpolates linearly between table entries and clamps outside the
  domain; a `torch` port must match that, not nearest-neighbour.
- The spline is zero outside its domain, and `nu_evaluate` therefore runs `correct_field` to
  extend the field beyond the mask before dividing.
- Estimation runs on the shrunk grid but the output field is evaluated at full resolution —
  don't collapse those two grids into one.
- `volume_stats -biModalT` (Otsu-style bimodal threshold) is what produces the automatic mask;
  results depend on it when no mask is supplied. `mincstats -biModalT`, which `nu_evaluate`
  uses, is plain Otsu over a 2000-bin histogram returning the winning **bin centre** —
  reproduced exactly by `torch_n3.minc_tools.bimodal_threshold`.
- **`torch.linspace`, `torch.zeros` etc. default to float32.** Every tensor N3 touches must
  be `float64`; a stray float32 grid of table positions costs six digits and shows up as a
  1e-7 disagreement with `minclookup`, which looks like an algorithm bug and is not.
- **`torch.std` defaults to `unbiased=True`, `numpy.std` to `ddof=0`.** The stopping rule uses
  the population standard deviation; pass `unbiased=False`.
- **The B-spline normal equations are nearly singular** — condition number ~1e13 at the
  default 200 mm knot spacing, because the knots are further apart than the volume is wide.
  The coefficients are not determined to better than ~1e-4 by *any* solver (LU, Cholesky and
  the legacy's `dsysv` all differ by that much); the fitted field is determined to ~1e-6.
  Compare fields, never coefficients.
- **`correct_field` cannot be reproduced exactly.** Its SOR relaxation sweeps in raster order,
  which is inherently sequential; `blocks/field.py` sweeps the two checkerboard colours in
  turn, the same Gauss-Seidel iteration reordered. The two agree to ~5e-6 relative, which is
  about the accuracy of the solve itself. Its *prolongation* between levels, on the other
  hand, is exactly reproducible and matters: after the last level (`inc == 2`) the odd voxels
  are never relaxed, only interpolated.
- **The iteration amplifies.** Backends that agree on the field to 1.4e-7 after one iteration
  disagree by 2.5e-4 after ten. Under the shipped protocol they land 1.1e-3 apart end to end,
  and running the *same* backend on a GPU moves it slightly more (1.3e-3). No end-to-end N3
  comparison is meaningful past three digits; pin blocks, not pipelines.
- **The amplification starts at a discontinuity, not at float noise.** The auto histogram
  range is taken from the data and then rounded to `%lf`'s six decimals, so a voxel on a bin
  boundary can fall either side of it and a whole count moves between bins. On `brain.mnc`,
  the backends agree to 5.5e-8 relative RMS after one iteration; at two, one whole count
  flips (out of the 3,724 samples the shrunken grid contributes) and they end up 1.17e-3
  apart — four orders of magnitude worse. This is why
  `tests/test_reproducibility.py` pins one iteration — past that, an end-to-end golden
  volume records which side of a rounding boundary one voxel landed on. Raising the count
  makes that test louder, not stronger. Which iteration the flip lands on is not fixed
  either: it moved from the sixth to the second when the shim switched to the system
  LAPACK, on a change of 3.1e-11 in the fitted field. See `README.md` on that. The same
  step appears between CPU and GPU with no LAPACK change at all, on the third.
- **Measure the flip, do not assume it.** `python3 -m tests.convergence` sweeps the
  iteration count and reports where agreement is lost, along with the LAPACK the extension
  actually resolved against. Run it before raising `PLATFORM_PROTOCOL`, before believing an
  end-to-end number on an unfamiliar machine, and after anything that touches the spline or
  the histogram. `N3_LAPACK_LIBS` / `N3_LAPACK_LIB_DIRS` rebuild the shim against a
  different LAPACK if you want both columns; `README.md` has the recipes. Note that
  `PLATFORM_PROTOCOL` is at 1 with **no margin** — the smallest cliff seen is 2 — so any
  increase needs this run on every platform that matters, not just one.
- **"Legacy" means two different things; keep them apart.** The *installed* N3 is a Perl
  script driving separate executables, which can only talk through files. The `legacy`
  *backend* here is those same C++ routines called through the CFFI shim, on float64
  buffers, in one process — no file, no rounding. So it is the original arithmetic without
  the original's quantisation, and it does not reproduce the installed programs either
  (3.7e-3 from `brain_nu_ref.mnc`, slightly worse than the port). Statements below about
  rounding between stages are about the installed programs only.
- **The installed N3's precision is not float.** Every intermediate passes between programs
  as a MINC file, so it is rounded on the way. Worse, writing and reading disagree: `mincmath`
  writes each *slice* against its own `image-min`/`image-max`, while `volume_io` programs
  (`volume_hist`, `spline_smooth`) hand the reader a volume rescaled onto a *single* grid for
  the whole file. How many distinct values survive depends on which file `mincmath` took its header from —
  the input's `valid_range` (4096 for the 12-bit test data) before the mask is applied, and
  the full 16 bits after, because the driver builds the mask with `-short -signed`. This is
  why a float64 port cannot reproduce `brain_nu_ref.mnc.gz` to the legacy suite's `1e-4`;
  see `tests/test_pipeline.py`.

## Environment

- Python 3.12 with `torch` 2.13 (CUDA build, GPU present and usable), `numpy`, `scipy`.
  Check device availability at runtime rather than assuming CPU-only.
- `minc2_simple` is installed: `minc2_file(path)`, `.setup_standard_order()`,
  `.load_complete_volume(dtype)`, `.save_complete_volume()`, `.representation_dims()`,
  world/voxel transforms. Full reference: `minc2-simple/USAGE.md`. Note that
  `setup_standard_order()` reorders to `TIME → Z → Y → X → VEC` with positive steps, which is
  what makes the array numpy/torch C-order compatible — the legacy MINC files here have
  file order `xspace zspace yspace`.
- Do not install packages; if something is missing, say so and stop.

### minc2_simple gotchas (learned the hard way)

- It reads **MINC2 (HDF5) only**. Everything in `legacy/N3/testing/` is MINC1 *and* gzipped,
  so it must go through `mincconvert -2` first — `torch_n3.volume.load_volume` does this
  transparently. The tests do not rely on that: they read MINC2 copies from `tests/data/`,
  which is why the suite needs no MINC program at all.
- `representation_dims()` and `store_dims()` list dimensions **fastest-varying first**, the
  reverse of the numpy axes. `.shape` is likewise reversed relative to `.data.shape`.
- `voxel_to_world()` takes indices in *storage* order, not standard order. Don't mix it with
  standard-order arrays; compute geometry from `representation_dims()` instead.
- `imitate(other, path=...)` already calls `create()`; calling `create()` again fails.
  `set_volume_range()` must come *after* `create()`.
- Several legacy programs (`spline_smooth`, `sharpen_volume`, `nu_correct -mapping_dir`) do
  not reliably resolve *relative output paths* against the working directory. Pass absolute
  paths when driving them from Python.

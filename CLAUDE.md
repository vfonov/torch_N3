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
| `torch_n3/_legacy/ebtks/` | Likewise EBTKS, vendored from `legacy/EBTKS`: all headers, seven `.cc` files, no `clapack/`. The build links the **system** LAPACK/BLAS instead, which is the one dependency the extension still takes from outside. What that substitution costs is measured in `README.md`: a change of 3.1e-11 in the fitted field, which moves the divergence threshold from the sixth iteration to the second. |
| `torch_n3/_legacy/compat/` | Stand-ins for `<volume_io.h>`, `<time_stamp.h>` and `<ParseArgv.h>`, placed first on the include path so that `correctField.cc` and `args.cc` compile **unmodified** without libminc2. `smooth()` performs its entire solve on flat `float`/`char` arrays; volume_io provided only the marshalling at its edges. |
| `tests/margins.py`, `tests/convergence.py`, `tests/tables.py`, `tests/parzen.py` | None is a test; all four measure and print. `python3 -m tests.margins` prints `PROBLEMS.md`'s table — where every comparison sits against its bound. `python3 -m tests.convergence` prints where the two backends stop agreeing as iterations grow, which is a property of the machine's LAPACK rather than of this code; run it on any new platform before trusting an end-to-end number there. `python3 -m tests.tables` re-measures the `--lambda` × `--distance` tables under each direct solver and diffs them against the copies published in `README.md` and `cli.py`. `python3 -m tests.parzen` measures the one *modification* to the algorithm carried by the port: `--parzen-sigma`, a Gaussian Parzen window in place of N3's linear split (`blocks/histogram.py`). It runs `tables.py`'s sweep once per window at `solver="normal"`, plus the shipped protocol end to end. Its `None` rows are `tables.py`'s published cells, so a mismatch there indicates a change elsewhere. |
| `PROBLEMS.md` | Known weaknesses in the test suite: fitted thresholds, tight margins, dropped assertions, and the measured margin of every comparison. |
| `tests/data/` | The test volumes as MINC2, checked in: byte-for-byte the same images as `legacy/N3/testing/` and the installed model mask. Converted once so that reading them needs nothing installed. |
| `tests/data/brain_nu_ref_legacy.mnc` | Not one of N3's files: this pipeline's own output on `brain.mnc` with the legacy blocks, under `inputs.PLATFORM_PROTOCOL`, checked in so another machine/BLAS/device can be held to it (`tests/test_reproducibility.py`). Written by the regeneration script. Regenerating churns MINC's `ident` header attribute; the voxel data is reproducible. |
| `torch_n3/optimize.py`, `torch_n3/blocks/sharpness.py` | A second estimator, not a port of anything: `nu_optimize` keeps N3's B-spline field and bending-energy penalty but replaces the alternating iteration with gradient descent on a stated objective — Hoyer sparsity of a Gaussian soft histogram (`hoyer`), or soft-assignment within-cluster variance against learned centroids (`tightness`, which is the tissue model N3's own EM branch abandoned). Returns the same `BSplineField` as `nu_estimate`, so `nu_evaluate` and `experiments/` take it unchanged (`--method`). **Both measures are optimal on a constant image** — a field that cancels the volume — and `blocks/sharpness.standardize` is the only thing preventing that; `tests/test_optimize.py` demonstrates the collapse rather than asserting it away. `penalty` is not N3's `lambda`: the data term is dimensionless, so the scales are unrelated. Measured on `tests/tables.py`'s experiment at each method's best weight (20% planted): N3 0.13/0.17/0.25% at 200/100/50 mm against `hoyer`'s 0.28/0.21/**0.17**% — N3 degrades as the field gains freedom, `hoyer` improves. **`tightness` diverges**: its loss falls monotonically while the field's own non-uniformity grows without bound (35%→257% at 200 mm), because within-cluster variance is also minimised by *amplifying* contrast into separated spikes, and standardizing pins only the first two moments. Retained as a recorded negative result, as is `spline.py`'s `sparse` solver. |
| `experiments/` | Not the test suite (`pytest.ini` collects `tests/` only) and not part of the port: the Monte-Carlo version of `test_field_recovery.py`. `python3 -m experiments.recovery` plants a *random* smooth field on colin27, adds Gaussian noise at a stated SNR, recovers it, and writes one CSV row per trial; `python3 -m experiments.summarize` reports mean/median/IQR and run time per configuration. Resumable, hours long, and swept over `--distance`/`--lambda` for the grid search. Its generators are the only part of it covered by the test suite (`tests/test_simulation.py`); `experiments/README.md` defines every column, and `experiments/data/` holds the colin27 volumes (not checked in). |
| `tests/` | `test_pipeline.py` is `legacy/N3/testing/CMakeLists.txt`'s cases, re-expressed as comparisons, run on both backends. `test_histogram.py`, `test_sharpen.py`, `test_spline.py`, `test_field.py` compare each PyTorch block against the same C++ through the shim. **No test runs an N3 program**: their answers are recorded in `tests/reference/legacy.npz` by `tests/regenerate_reference.py` (the only thing that shells out) and the inputs they were given live in `tests/inputs.py`, shared by both. Re-run the script and `git diff` should be empty (except for `brain_nu_ref_legacy.mnc`'s `ident` header). `test_reproducibility.py` holds every backend and device to a volume checked into the repository, and is the cross-platform check. |

## The algorithm as implemented by the legacy code

Everything except the final division happens in the **log-intensity domain**; the estimated
field `F` is accumulated in log space and only exponentiated at the very end.

Driver: `legacy/N3/src/NUcorrect/nu_estimate_np_and_em.in:60-193`.

1. Optionally resample input to a coarser grid (`-shrink`, nearest-neighbour, `ShrinkVolume`
   at `:954`). Estimation runs on that grid; the *output* is a spline, so the final field is
   evaluated at full resolution.
2. `log_volume = log(clamp(input, 1, 1.7e308))` — the clamp prevents `log` of a
   near-zero value from consuming the dynamic range.
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

`blocks/histogram.py` also carries the estimator that name denotes, as a **modification, not a
port**: `histogram(..., sigma=)` / `nu_estimate(parzen_sigma=)` / `--parzen-sigma` spreads each
sample with a Gaussian of `sigma` *bin widths*, truncated at 4σ and renormalized per sample so
a voxel near the range edge still counts as one. Off by default; the legacy backend rejects it;
there is no oracle, so `tests/test_histogram.py` states its properties instead. What it does to the
pipeline is measured by `python3 -m tests.parzen` and written up in `README.md`, "Gaussian Parzen
window". In summary: it substitutes for regularization, reducing the residual non-uniformity by a third
at the shipped defaults and by nothing at 200 mm once `--lambda` is tuned for the spacing; and
past about `sigma 2` it adds more blur than `--fwhm` instructs the deconvolution to remove,
which appears as degradation in the well-regularized cells.

That measurement is **noiseless**, and `experiments/` (450 random fields per window on colin27,
`--parzen-sigma`, `experiments/README.md` "The histogram kernel", `results/windows.png`) establishes
that the window's reduction is against *noise* rather than against field amplitude: at 20% planted and
SNR 20 under the shipped protocol, 4.63% left by N3's split against `sigma 4`'s 1.71%, and
`sigma 4` better on 94% of trials. The reduction also grows with the iteration count: under
N3's own histogram, more iterations make a noisy cell worse. The analytic tables must not be
quoted as if they covered a real volume.

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

Isolating a single stage is the least expensive way to validate a PyTorch component:

```bash
volume_hist -bins 200 -auto_range -mask chunk_mask.mnc.gz chunk.mnc.gz h.txt \
    -clobber -text -select 1 -quiet -window     # -window = the Parzen variant
sharpen_hist -clobber -fwhm 0.15 -noise 0.01 -quiet h.txt h.sharp
```

`h.txt` carries `# domain: <min_bin> <max_bin>` and two columns (bin centre, count);
`h.sharp` is the two-column lookup table. Both are plain text and therefore diff directly
against tensors.

For anything a *test* needs to compare against, put the case in
`tests/regenerate_reference.py` instead of shelling out from the test — build the input in
`tests/inputs.py` so both sides use the same one, run the script once, and read the answer
back through the `legacy_output` fixture. Storage policy is in `tests/reference.py`: whole
volumes at `float32`, small arrays at `float64`, and record only what the assertion looks at
(the recovery test keeps the masked quarter of the volume, not all of it).

One pitfall: a program handed a *file* saw its contents quantised, and modelling MINC's 16-bit
scaling in Python is wrong by a whole quantisation step. Round-trip through
`inputs.as_stored()` instead, which both the script and the test call.

## Test tolerances

Known weaknesses in the current suite — thresholds fitted to the measurement, one that is far
too loose, three tight enough to fail intermittently, and the assertions that were dropped
rather than satisfied — are listed in **[PROBLEMS.md](PROBLEMS.md)**, together with where
every comparison currently sits against its bound. Read it before adjusting a tolerance, and
add to it rather than correcting a bound without record.

**A tolerance must not be widened to make a test pass.** A threshold states what the code is
required to do; moving it after the fact reduces the test to a record of what the code happens
to do, and destroys the only evidence that something changed. The same applies to altering a
test's inputs until it passes.

When an assertion fails, the remedies are, in order: identify the defect; or identify the
confound and remove it (a comparison between two implementations that stopped at different
iteration counts does not measure what it claims to, and controlling the iteration count is a
remedy, whereas loosening the bound is not); or, if the requirement was itself incorrect,
change it deliberately, in its own commit, with the measurement and the reasoning recorded.

Thresholds here are round numbers derived from a physical quantity — the 16-bit quantum of the
MINC file a legacy program wrote (`span(reference) / 65535`), the six decimals `%lf` prints
(`1e-6`), a single shared constant for cross-implementation agreement — and not the measured
difference plus a margin. If a bound cannot be justified without first running the code, assert
instead the property that motivated it.

Two comparisons are known to be brittle, and their thresholds are not the remedy when they
fail:

- **The stopping rule quantises everything downstream.** Iteration stops at `change < 0.001`;
  two implementations whose per-iteration change differs in the fifth decimal can fall on
  either side of it and run different numbers of iterations, which moves the output by far
  more than any block-level difference. Check the iteration count first (`verbose=True`).
- **Extreme-value statistics over a whole volume** (`max |a - b|` across 240k voxels) are
  dominated by a handful of mask-edge voxels and do not support a fixed bound. Compare RMS.

### Published numbers no test checks

Most measured numbers in the documentation are reproduced by `python3 -m tests.margins`, so a
stale one appears as a mismatch against `PROBLEMS.md`. The **`--lambda` × `--distance` tables are
the exception**, and they are the most cited numbers in the repository:

| Where | What |
|---|---|
| `README.md`, "Bias-field recovery on simulated data" | both tables, 20% and 40% planted |
| `torch_n3/cli.py`, `SMOOTHNESS_NOTE` | the 20% table, shown by `--help` |
| `tests/test_field_recovery.py` | asserts the `1e-7` row at every spacing (the recovery sweep), and `LAMBDAS = [1e-5, 1e-4]` at the *finest* spacing only — the `regularized` fixture fixes `distance = min(DISTANCES)` |

That is 10 of the 24 published cells. The remaining **14 are asserted by no test**, including
the whole `1e-6` row, which contains the best cell at the default spacing. The two copies are
kept consistent by hand, and the conclusions drawn from them in prose (which cell is best per
column, the decade-per-halving rule, the asymmetry that justifies erring high) are verified by
nothing. They are also not invariant: they run the pipeline for 30 iterations, well past the
divergence threshold of the histogram, so a change to any block, or to the LAPACK the shim
links, can move them.

Re-measure them rather than adjusting them, and change both copies together. **`python3 -m
tests.tables` performs this**: it drives `AMPLITUDES × DISTANCES × (1e-7, 1e-6, 1e-5, 1e-4)`
exactly as the `regularized` fixture does, takes `ratio.std(unbiased=False)` of the
recovered field over the planted one (each divided by the same implementation's baseline on
the untouched reference), and diffs the result against the published copies, so that a stale
cell appears as a mismatch in the way `tests.margins` reports a stale bound. It is 24 cells and two
`nu_estimate` calls each, about 15 s per solver.

**Re-run 2026-07-31 under every direct solver**: the tables are more stable than the warning
above implies. All 24 cells reproduce exactly under `solver="normal"`, which is what
both copies hold. Against `normal`'s own measurement:

| solver | cells differing at 2 dp | largest relative move |
|---|---|---|
| `normal` | 0 of 24 | — |
| `qr` | 2 of 24 | 3.2% |
| `dr` | 2 of 24 | 3.2% |
| `blocked` | 3 of 24 | 4.7% |

Every move is in the same few shallow cells: 40%/50 mm at `1e-7` (1.96% → 1.95%) and at
`1e-5` (0.48% → 0.47%) for `qr` and `dr`, plus 20%/50 mm at `1e-7` (1.51% → 1.52%) and
20%/100 mm at `1e-6` (0.22% → 0.23%) for `blocked`. The largest relative move is at
20%/100 mm/`1e-6` for all three. Every conclusion drawn from the tables survives all four
solvers: the same interior minimum in each column (`1e-6` at 200 mm, `1e-5` at 100 and 50
mm), identically at both amplitudes, the decade-per-halving rule, and the asymmetry at 50
mm. The tables therefore measure the `lambda`/`distance` trade-off rather than recording which
side of the histogram's rounding boundary a voxel fell on, which is worth establishing given
that they run 30 iterations, well past the divergence threshold.

Note that `dr` is algebraically `qr` at a fixed weight and still moves 0.3% from it in the most
sensitive cell: 30 iterations amplify float64 rounding by that much. That is the scale at which
a cell's last digit should be read.

If a result comes to depend on a particular cell, the correct remedy is to widen `LAMBDAS` and
assert the property being claimed — that each column has an interior minimum, and where it lies
— rather than to continue trusting a hand-maintained table.

`README.md`'s **`--parzen-sigma` tables are in the same category**: published, cited, and
asserted by nothing. They are the same sweep, so the same warning applies, with one mitigating
factor — their `linear (N3)` row is the published `1e-7` row, so `python3 -m tests.parzen`
re-measures the baseline alongside the modification and drift in either becomes visible. Only
the default histogram path has an oracle; the window's own tables rest entirely on that
re-run.

## Pitfalls when porting

- Perform the arithmetic in log space and retain the field there until the final `exp`. Ratios
  in intensity space are differences in log space, and the legacy code relies on this
  throughout.
- The FFT length is `2^(ceil(log2(nbins))+1)` — at 200 bins that is 512, not 256, and the
  histogram sits centred at `offset`, not at index 0. An off-by-`offset` error shifts the mapping
  without any other symptom.
- The Gaussian kernel is built on the *bin* grid (`fwhm/slope`) and wraps around index 0.
- Histogram range is recomputed per iteration (`-auto_range`), so the lookup domain moves
  between iterations.
- `minclookup -continuous` interpolates linearly between table entries and clamps outside the
  domain; a `torch` port must match that, not nearest-neighbour.
- The spline is zero outside its domain, and `nu_evaluate` therefore runs `correct_field` to
  extend the field beyond the mask before dividing.
- Estimation runs on the shrunk grid while the output field is evaluated at full resolution.
  The two grids must not be collapsed into one.
- `volume_stats -biModalT` (Otsu-style bimodal threshold) produces the automatic mask, and the
  results depend on it when no mask is supplied. `mincstats -biModalT`, which `nu_evaluate`
  uses, is plain Otsu over a 2000-bin histogram returning the winning **bin centre** —
  reproduced exactly by `torch_n3.minc_tools.bimodal_threshold`.
- **`torch.linspace`, `torch.zeros` etc. default to float32.** Every tensor N3 touches must
  be `float64`; a stray float32 grid of table positions costs six digits and appears as a
  1e-7 disagreement with `minclookup`, which resembles an algorithmic error and is not one.
- **`torch.std` defaults to `unbiased=True`, `numpy.std` to `ddof=0`.** The stopping rule uses
  the population standard deviation; pass `unbiased=False`.
- **The B-spline normal equations are nearly singular** — condition number ~1e13 at the
  default 200 mm knot spacing, because the knots are further apart than the volume is wide.
  The coefficients are not determined to better than ~1e-4 by *any* solver (LU, Cholesky and
  the legacy's `dsysv` all differ by that much); the fitted field is determined to ~1e-6.
  Compare fields, never coefficients.
- **The fit need not be posed that way.** `BSplineField(..., solver="qr")`
  minimises the same objective through the stacked system `[A; sqrt(lambda N) D] c ~ [f; 0]`,
  where `D'D = J`, and never forms `AtA`. Since the normal equations *are* that matrix's Gram
  matrix, its condition number is their square root: 2.3e6 instead of 5.3e12 on `brain.mnc`'s
  estimation grid, by construction rather than incidentally. What this yields is
  reproducibility, which is worth establishing before investigating a cross-platform
  difference: the fitted field moves 3.0e-13 relative RMS between CPU and GPU instead of
  2.5e-9, and the divergence threshold moves from the third iteration to the seventh (CPU vs
  GPU) and from the second to the fourth (legacy vs torch). The cost is
  holding `A` dense — 25 MB on `brain.mnc` at the default `-shrink 4`, 600 MB at `-shrink 1`
  — and about 8% of the runtime. `solver="normal"` is still the **default** and is what every
  recorded reference in `tests/` was produced with; the legacy backend has no other solver
  and rejects the argument. See `PROBLEMS.md` §9 for what changing the default would cost.
- **`correct_field` cannot be reproduced exactly.** Its SOR relaxation sweeps in raster order,
  which is inherently sequential; `blocks/field.py` sweeps the two checkerboard colours in
  turn, the same Gauss-Seidel iteration reordered. The two agree to ~5e-6 relative, which is
  about the accuracy of the solve itself. Its *prolongation* between levels, by contrast, is
  exactly reproducible and is significant: after the last level (`inc == 2`) the odd voxels are
  never relaxed, only interpolated.
- **The iteration amplifies.** Backends that agree on the field to 1.4e-7 after one iteration
  disagree by 2.5e-4 after ten. Under the shipped protocol they land 1.1e-3 apart end to end,
  and running the *same* backend on a GPU moves it slightly more (1.3e-3). No end-to-end N3
  comparison is meaningful past three digits; constrain blocks rather than pipelines.
- **The amplification starts at a discontinuity, not at float noise.** The auto histogram
  range is taken from the data and then rounded to `%lf`'s six decimals, so a voxel on a bin
  boundary can fall either side of it and a whole count moves between bins. On `brain.mnc`,
  the backends agree to 5.5e-8 relative RMS after one iteration; at two, one whole count
  flips (out of the 3,724 samples the shrunken grid contributes) and they end up 1.17e-3
  apart — four orders of magnitude worse. This is why
  `tests/test_reproducibility.py` runs one iteration: past that, an end-to-end recorded volume
  registers which side of a rounding boundary one voxel fell on. Raising the count makes the
  test noisier, not stricter. The divergence threshold is not fixed either: it moved from the
  sixth iteration to the second when the shim switched to the system LAPACK, on a change of
  3.1e-11 in the fitted field. See `README.md`. The same step appears between CPU and GPU with
  no change of LAPACK, at the third.
- **Measure the divergence threshold rather than assuming it.** `python3 -m tests.convergence`
  sweeps the iteration count and reports where agreement is lost, together with the LAPACK the
  extension resolved against. Run it before raising `PLATFORM_PROTOCOL`, before relying on an
  end-to-end number from an unfamiliar machine, and after any change to the spline or the
  histogram. `N3_LAPACK_LIBS` / `N3_LAPACK_LIB_DIRS` rebuild the shim against a different
  LAPACK where both columns are required; `README.md` gives the procedure. `--solver qr`
  reports the same sweep for the better-conditioned fit, which is where its thresholds above
  were measured. Note that `PLATFORM_PROTOCOL` is at 1 with **no margin** — the smallest
  threshold observed is 2 — so any increase requires this run on every platform of interest,
  not one. That floor is set by the *default* solver; under `qr` the smallest threshold
  measured here is 4.
- **There are five solvers, of which four are usable.** `blocks/spline.py` has `solver=` with
  `normal` (the legacy's normal equations, the default and the reference), `qr` (dense
  stacked least squares), `blocked` (the same stacked system folded in one band at a time),
  `dr` (the same system reparameterized so the penalty is diagonal) and `sparse` (the same
  again through `scipy.sparse` + `lsqr`). `DIRECT_SOLVERS` is the first four;
  parametrise anything asserting an exact fit over that, not over `SOLVERS`.
  `blocked` exploits the fact that `A` is banded once its rows are sorted by first-axis
  knot: a sample with corner `k` touches only columns `[k*n1*n2, (k+4)*n1*n2)`. It equals
  `qr` to 4e-15 and shares its CPU/GPU reproducibility, but never holds `A` — at 12.5 mm
  on `chunk.mnc` (3168 coefficients) it is 3.45 s and 1.55 GB against `qr`'s 12.72 s and
  7.53 GB. At 200 mm there is one band and it reduces to `qr` plus a sort, so there is
  no advantage at the shipped spacing. `sparse` **does not converge**: LSQR is defeated by the
  same conditioning the stacked form reduces, and terminates at `istop=3`, 1.8e-3 from the
  direct answer. It is retained as a recorded negative result, not as a usable option.
- **`dr` is for sweeping `lambda`, and reduces to `qr` at a single weight.** The Demmler–Reinsch
  reparameterization factorizes `[A; sqrt(lambda_0 N) D]` once at an *anchor* weight, so
  `R0'R0 = A'A + lambda_0 N J`, then diagonalizes the penalty in that basis: with
  `D~ = sqrt(N) D R0^-1` and `D~'D~ = U diag(gamma) U'`, every weight satisfies
  `A'A + lambda N J = R0'(I + (lambda - lambda_0) D~'D~)R0`. A fit is then one elementwise
  division and a triangular solve. At `lambda == lambda_0` the divisor is 1 and the answer
  *is* `qr`'s, to rounding — which is why `dr` sits in `DIRECT_SOLVERS` and meets the same
  oracle bounds (identical margins to `qr`: 7.49e-08, 1.52e-08, 2.24e-12). What it provides is
  `BSplineField.refit(lam)`: on `brain.mnc`'s estimation grid an additional weight costs
  0.033 ms at 200 mm and 0.140 ms at 50 mm, against 5.6 ms and 27.7 ms for a fresh `qr` fit, a
  factor of 170–200. This is the solver to use when re-measuring the `--lambda` × `--distance`
  tables.
- **For a *single* weight `dr` is the slowest solver and the most memory-intensive, and should
  not be selected.** The eigendecomposition is `O(k^3)` in addition to everything `qr`
  performs, and yields no benefit until a second weight is requested. One fit on `brain.mnc`'s estimation grid:
  7.3 ms at 200 mm and 43.6 ms at 50 mm, against `qr`'s 5.6 and 27.7 and `normal`'s 4.9 and
  8.3; a whole 30-iteration run is 1.8 s at 50 mm against `qr`'s 1.2 s and `normal`'s 0.4 s.
  It reaches parity at about the second weight. `normal` is also the most memory-frugal by an
  order of magnitude away from the shipped protocol, since it holds `AtA` and never `A`
  (370 MB against `qr`'s 8.75 GB at `-shrink 1`, 25 mm). `blocked`'s memory advantage is
  governed by the sample-to-coefficient ratio rather than by `-distance`: it improves on `qr`
  by 6.4× where samples greatly outnumber coefficients, but is second-worst of the four at
  `-shrink 4`, 12.5 mm, where coefficients (7,581) exceed samples (3,724) and its `O(size^2)`
  running `R` dominates. Figures in `README.md`, "Solver cost on a GPU".
- **Anchor `dr` on the stacked matrix, never on `A` alone.** The textbook Demmler–Reinsch
  takes the QR of the design alone. That is unusable here and fails without any
  diagnostic: `A` is the *masked* design, and at fine spacings the mask leaves basis functions with no data
  under them. Measured on `chunk.mnc`, `cond(A)` is 8.4e7 at 200 mm and 7.5e12 at 50 mm,
  where `A` is rank deficient (243 of 245 columns) — worse than the normal equations there
  — and `D R^-1` overflows into a `gamma` with 83 non-positive entries reaching -5.7e6, so
  `1 + lambda N gamma` goes through zero. Clipping `gamma` at zero does not recover it; the
  eigenvectors are as damaged as the eigenvalues. Anchoring on `[A; sqrt(lambda_0 N) D]`
  costs nothing and removes the failure, because the penalty rows span exactly the directions the data
  leaves empty: `cond(R0)` is 3.5e6 and 2.9e5 at those spacings. The cost is that the basis
  is valid only at or above its anchor, since below it the divisor can reach zero, so
  `anchor=` belongs at the bottom of the grid and `refit()` raises an exception beneath it.
- **"Legacy" denotes two distinct things, which must be kept apart.** The *installed* N3 is a
  Perl script driving separate executables, which communicate only through files. The `legacy`
  *backend* here is those same C++ routines called through the CFFI shim, on float64 buffers,
  within one process, with no file and no rounding. It is therefore the original arithmetic
  without the original's quantisation, and it does not reproduce the installed programs either
  (3.7e-3 from `brain_nu_ref.mnc`, marginally worse than the port). Statements below about
  rounding between stages apply to the installed programs only.
- **The installed N3's precision is not that of float.** Every intermediate passes between
  programs as a MINC file and is rounded in transit. Writing and reading are also mutually
  inconsistent: `mincmath`
  writes each *slice* against its own `image-min`/`image-max`, while `volume_io` programs
  (`volume_hist`, `spline_smooth`) hand the reader a volume rescaled onto a *single* grid for
  the whole file. The number of distinct values that survive depends on the file `mincmath` took its header
  from:
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

### minc2_simple pitfalls

- It reads **MINC2 (HDF5) only**. Everything in `legacy/N3/testing/` is MINC1 *and* gzipped,
  so it must go through `mincconvert -2` first — `torch_n3.volume.load_volume` does this
  transparently. The tests do not rely on that: they read MINC2 copies from `tests/data/`,
  which is why the suite requires no MINC program.
- `representation_dims()` and `store_dims()` list dimensions **fastest-varying first**, the
  reverse of the numpy axes. `.shape` is likewise reversed relative to `.data.shape`.
- `voxel_to_world()` takes indices in *storage* order, not standard order. Do not combine it
  with standard-order arrays; compute geometry from `representation_dims()` instead.
- `imitate(other, path=...)` already calls `create()`; calling `create()` again fails.
  `set_volume_range()` must come *after* `create()`.
- Several legacy programs (`spline_smooth`, `sharpen_volume`, `nu_correct -mapping_dir`) do
  not reliably resolve *relative output paths* against the working directory. Pass absolute
  paths when driving them from Python.

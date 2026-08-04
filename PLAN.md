# Plan: `nu_correct` in C++, entirely in memory

*(This file previously held the PyTorch port's plan. That version is in git at commit
`863988a`; recover it with `git show 863988a:PLAN.md`. The PyTorch plan is also
superseded by `README.md` and `PROBLEMS.md`, which carry the measurements.)*

## Context

N3's pipeline is a Perl layer over separate executables that communicate only through
files. `nu_correct` (`nu_estimate.in`) calls `nu_estimate_np_and_em.in`, which calls
`sharpen_volume.in`, `spline_smooth`, `volume_stats` and a dozen `mincmath` invocations;
`nu_evaluate.in` then calls `evaluate_field`, `correct_field` and `mincmath` again. Every
intermediate is written as a MINC volume and read back, quantised to the file's storage
type in transit, and the histogram and lookup table pass as text written with `%lf`.

This task replaces that layer with one C++ program holding every intermediate in memory at
`double` precision. Most numerical routines are reused by linking the original translation
units, which is what `torch_n3/_legacy/n3_shim.h` already does for the PyTorch port's
oracle. What is written new is the driver logic that lives in Perl today, the handful of
MINC utilities the drivers lean on, and one numerical stage that the original keeps inside
a `main()` (§1).

The work is done test-first: every routine gets a failing CTest case, with its oracle
recorded and its bound justified, before its implementation exists (§6, §7). This is
affordable here because the Perl pipeline is a complete oracle for every stage — including,
via `-save_fields` and `-save_histograms`, each individual iteration.

Scope, agreed with the user:

- **A new binary alongside** the Perl drivers, which stay installed and unchanged. Both
  must run on the same input; that comparison is the only available validation.
- **The N3 path plus thin-plate splines**: np method, `b_spline` and `tp_spline`,
  `-sharpen`, `-parzen`/`-parzen_sigma`, `-shrink`, `-mask`, `-distance`, `-lambda`,
  `-iterations`, `-stop`, `-normalize_field`, `-auto_mask`/`-bimodalT`, `-floor`,
  `-subsample`, `-mapping_dir`. Not the EM branch (it `die`s without `-sharpen`), not the
  WM branch (needs `class_statistics`, `estimate`, `lgmask`, none shipped), not `fir`
  smoothing, `-real`, `-differential`, `-initial` or `-islands`.
- **A double-precision copy of `correct_field`'s `smooth()`**, rather than linking the
  original, which solves on `float` arrays.
- **An estimate-only mode**: the binary writes only the `.imp` when invoked under a name
  matching `nu_estimate`, as `nu_estimate.in:420` decides it.
- Install to `/app/legacy/_install`. `/opt/minc/1.9.18.13` is the oracle and is not written
  to. **Nothing under `/app/torch_n3` is touched**; the vendored copies in
  `torch_n3/_legacy/n3/` stay byte-identical to `legacy/N3/src`, which is automatic because
  the reused sources are linked, never edited.

"No intermediate on disk" means no temporary files and no `TmpDir`. The program still
writes the corrected volume and, on request, the `.imp`.

## 0. Defects found in the first draft

Reviewing the first version against the sources turned up eight errors. Each is corrected
below; they are listed here because several change the design rather than the wording.

1. **"The numerical blocks are not rewritten" was false for sharpening.**
   `sharpen_hist.cc` keeps padding, offset, FFT, Wiener filtering, the moment vector and
   the Nadaraya–Watson mapping inside `main()`. Only `gaussian()`, `weiner()` and
   `non_negative()` (`sharpen_hist.cc:71-74`) are callable. That sequence is genuinely new
   code and needs its own oracle (§7).
2. **Linking two `args` translation units is an ODR clash.** `SharpenHist/args.h:7`,
   `SplineSmooth/splineSmoothArgs.h:47` and `VolumeHist/args.h:26` each define a different
   global `class args` with static members. `-Dmain=` alone is not enough; the class must
   be renamed per object library too (§5).
3. **`-full_support` is not used on both spline paths.** `spline_smooth_volume`
   (`nu_estimate_np_and_em.in:571-579`) passes it for `b_spline` and *omits* it for
   `tp_spline`, so `splineSmooth.cc:124-130` gives the two paths **different domains**:
   `volume_domain(volume)` for B-splines, `reduced_domain(mask_volume)` for thin plates.
4. **The buffer type was over-designed.** `fitSplinesToVolumeLookup`, `smoothVolumeLookup`,
   `volume_domain`, `reduced_domain` and `outputCompactField` all take `VIO_Volume`. An
   in-memory `VIO_Volume` of type `NC_DOUBLE` satisfies both requirements and makes those
   five routines reusable unmodified (§2), removing most of the proposed `VolumeD` and
   `FitField`.
5. **The end-to-end comparisons had no pass/fail criterion.** `-legacy_rounding` isolates
   the `%lf` rounding, but the dominant divergence — the MINC round trip between stages —
   had no instrument, so §7's numbers would have had nothing to be compared against (§4).
6. **The `-parzen_sigma` oracle is not the installed N3.**
   `/opt/minc/1.9.18.13/bin/volume_hist` has no `-gaussian_window`; only
   `/app/legacy/_install` does, and `MNI::Spawn` resolves each program through `PATH`.
7. **The staged stopping rule was never described.** `-iterations a b` / `-stop x y`
   implement a two-clause test at `nu_estimate_np_and_em.in:171-185`, and the `-V0.9`
   protocol uses it.
8. **The output volume's type and history were unspecified.** `nu_evaluate.in:78` writes
   with `mincmath -copy_header -zero -div` and then splices the `.imp`'s history. Writing
   `NC_DOUBLE` output instead would differ from the Perl for reasons unrelated to the
   arithmetic and would corrupt the comparison.

## 1. What is reused, and what is new

Linked unmodified from `legacy/N3/src`, no `args` reference, no `main`:

| Source | Provides |
|---|---|
| `Splines/{Spline,TBSpline}.cc` | `TBSplineVolume`, `TPSpline`; the regularized fit, `getCoefficients`/`setCoefficients` (`Spline.h:131-134`), evaluation through the per-grid lookup tables |
| `SplineSmooth/fieldIO.cc` | `createThinPlateSpline`, `outputCompactField` (the `.imp` writer), `smoothVolume`/`smoothVolumeLookup` |
| `VolumeHist/{DHistogram,WHistogram,GHistogram}.cc` | the three histogram estimators, including the Gaussian window added in the previous task |
| EBTKS `Histogram` | `biModalThreshold()`, which is what `volume_stats -biModalT` reports |

Linked with two renames, because each carries a `main()` and a `class args`:

| Source | Provides |
|---|---|
| `SharpenHist/sharpen_hist.cc` + `args.cc` | `gaussian()`, `weiner()`, `non_negative()` |
| `SplineSmooth/splineSmooth.cc` + `splineSmoothArgs.cc` | `volume_domain()`, `reduced_domain()`, `fitSplinesToVolume()`, `fitSplinesToVolumeLookup()` — the masked, `-subsample`d fit loops |

Written new, under `legacy/N3/src/N3Pipeline/`:

| File | Contents |
|---|---|
| `Buffers.{h,cc}` | in-memory `VIO_Volume` creation (§2), `loadDouble()`, `saveLike()`, buffer arithmetic, `resampleNearest`, `resampleLabel`, `shrink`, and the masked `mean`/`stddev`/`min`/`max` that `volume_stats` prints |
| `MincTools.{h,cc}` | `applyLookup()` (`minclookup -continuous`) and `bimodalThresholdMincstats()` (2000-bin Otsu returning the winning bin centre) |
| `Sharpen.{h,cc}` | `autoRange()` and `sharpenLookup()`: the body of `sharpen_hist.cc:100-190`, over buffers |
| `SmoothField.{h,cc}` | the double-precision transcription of `correctField.cc`'s `smooth()` |
| `FitField.{h,cc}` | thin wrapper choosing spline type, domain and grid, then calling the reused fit; plus evaluation on a second grid |
| `NuEstimate.{h,cc}` | the iteration of `nu_estimate_np_and_em.in:101-205` |
| `NuEvaluate.{h,cc}` | `nu_evaluate.in:46-80` |
| `nu_correct_cxx.cc` | `ParseArgv` argument table (the `src/VolumeHist/args.cc` pattern) and `main()` |

`torch_n3/blocks/` and `torch_n3/minc_tools.py` are a validated specification for the
routines with no C++ original — `apply_lut` at `minc_tools.py:12`, `bimodal_threshold` at
`:48`, `resample_like`/`shrink` at `volume.py:54-96`, and the sharpening sequence in
`blocks/sharpen.py`. Read them; do not modify them.

## 2. The buffer type

Every intermediate is a `VIO_Volume` created with `create_volume(3, dims, NC_DOUBLE,
FALSE, 0.0, 0.0)` + `set_volume_sizes` + `alloc_volume_data`, never written to disk. Input
comes through `input_volume(path, 3, NULL, NC_DOUBLE, FALSE, 0.0, 0.0, TRUE, &vol, NULL)`,
which is `loadFloatVolume`'s call at `fieldIO.cc:101` with the type raised.

The one pitfall: volume_io applies a voxel-to-real scale, so a created volume must have its
real range set to its voxel range for `get_volume_real_value` to be the identity. Assert
that at construction rather than assuming it.

The payoff is that `volume_domain`, `reduced_domain`, `fitSplinesToVolumeLookup`,
`smoothVolumeLookup` and `outputCompactField` take these buffers directly, so the fit loop
— including exactly how `-subsample` indexes and how the mask is tested — is reused rather
than retyped.

## 3. Stage by stage

| Perl | C++ |
|---|---|
| `ShrinkVolume` (`:955`) | keep `start`, multiply `step` by the factor and take `ceil((n-1)/factor)+1` samples, on axes finer than `factor·min\|step\|` only; sample nearest neighbour. The factor is a float. **Check this against `mincresample` before building anything on it** (cycle 1) |
| `CheckSampling(..., isLabel=1)` → `resample_labels` | **trilinear, then threshold at 0.5** (`resample_labels.in:176-177`), not nearest neighbour |
| `log_transform` (`:657`) | `log(max(v, 1.0))`; the clamp is `mincmath -clamp -const2 1 1.7e308` |
| `CreateMask` (`:297`) | `input > background_threshold`, intersected with the user mask; `-bimodalT`, or `-auto_mask` on a non-Talairach volume, takes the threshold from EBTKS `Histogram::biModalThreshold` over `ceil(voxelMax-voxelMin+1)` bins (`volumeStats.cc:286-299`); `-auto_mask` on a Talairach volume uses the average brain mask, label-resampled |
| `mincmath -sub`/`-add`/`-mult` | buffer arithmetic |
| `sharpen_estimate` (`:500`) | `autoRange` → histogram class → `sharpenLookup` → `applyLookup`, then re-masked (`:527`) |
| `spline_smooth_volume` (`:567`) | b_spline: `TBSplineVolume(volume_domain(vol), start={0,0,0}, step, sizes, distance, lambda)`, `fitSplinesToVolumeLookup(..., mask, subsample)`, `fit()`, `smoothVolumeLookup`. tp_spline: `createThinPlateSpline(reduced_domain(mask), distance, lambda)` and `fitSplinesToVolume`. **The two domains differ** — defect 3 above |
| `field_CV` (`:702`) | population standard deviation, inside the mask, of `old_residue − new_residue` (`:166-167`, `mincmath -sub -zero`). The name is wrong: it is not a CV |
| stopping test (`:171-185`) | stop when `change < stop[0]`; additionally, for each stage `s ≥ 1`, stop when `iter ≥ iterations[s-1]` and `change < stop[s]`. Computed whenever `-stop` was given at all, so `-stop 0` still reports the change and never stops — which is what makes a fixed-iteration comparison possible |
| `normalize_field_volume` (`:595`) | divide by the in-mask mean; warn and skip if the mean is 0 |
| `compact_spline_volume` (`:628`) | a **second, fresh** fit, at the same distance and lambda, to `exp(residue)` — not the loop's coefficients. `outputCompactField` runs only when a `.imp` was asked for |
| `nu_evaluate` auto mask (`:49-56`) | `mincstats -biModalT`: 2000-bin Otsu returning the bin **centre**, then `input >= threshold`. A different rule from `volume_stats -biModalT` above; both are needed |
| `evaluate_field` (`:64`) | a second `TBSplineVolume` on the **same domain and distance** over the full-resolution grid, `setCoefficients`, evaluate. The coefficient count comes from `domain` and `distance` alone (`TBSpline` ctor), and the spline's coordinates are `index × step` from voxel (0,0,0) (`splineSmooth.cc:140`), which `ShrinkVolume` preserves by keeping `start`. `inputCompactField` does exactly this and is the proof |
| `correct_field` (`:69`) | the double transcription of `smooth()` |
| field floor + `mincmath -div` (`:71-79`) | clamp the field at `field_floor` only if its minimum falls below it, then `output = input / field`, written with the input's header, storage type and valid_range (`output_modified_volume`), with the `.imp` history spliced in as `CreateHistory` does |

The estimation grid and the output grid stay separate objects throughout.

Memory: all-double, roughly six full buffers in the loop. On a 181×217×181 volume at
`-shrink 1` that is about 57 MB each, plus `TBSplineVolume`'s `AtA` at `_nProduct²`, which
is the cost the Perl already pays.

## 4. Where this cannot match the Perl, and how to tell

Four deliberate divergences, all consequences of the task:

1. **No quantisation between stages.** Every intermediate is `double` rather than a MINC
   file. This is the point of the change and it moves the result.
2. **No `%lf` rounding** of the histogram, its domain, or the lookup table.
3. **`smooth()` in double** rather than float.
4. **`spline_smooth` reads volumes as float** (`loadFloatVolume`, `splineSmooth.cc:101`);
   the C++ fit sees double.

`-legacy_rounding` rounds the histogram counts, the range and the lookup table to six
decimals, reproducing (2) alone. It is a verification instrument, not a feature.

Divergence (2) has a second component, found in cycle 6 and measured there.
`sharpen_hist` writes the lookup table's *entry positions* with six decimals as well:
0.005025 where the exact position is 0.005025125628. `minclookup` interpolates against
those, and doing so instead of against the exact positions moves the result by 5.114e-07
relative RMS on `chunk.mnc` — four orders of magnitude above the 1.121e-16 at which the two
implementations otherwise agree. `-legacy_rounding` must therefore round the positions, not
only the values.

Divergence (1) has no instrument, so the end-to-end comparisons need a configuration where
it is small rather than a bound that assumes it away. The first comparison is therefore
`-shrink 1 -distance 200 -iterations 1 -stop 0`: one iteration, two round trips, expected
agreement of order the quantum of the intermediates over their own range. The bound must be
**derived from the data, never assumed**: the earlier draft assumed 16-bit files and
published 1e-4, but the test volumes are 12-bit (`chunk.mnc` has a `valid_range` of 0..4095),
so the correct bound is the physical quantity

  ``0.5 · (log max − log min) / 4095``,

2.683e-4 relative RMS on `chunk.mnc`. The stages round to a half-voxel midpoint on the way
out and back, so half the in-mask log range over the file's quantum is the width one round
trip can shift a corrected value. That is what cycles 11 and 12 assert (they measure 2.30e-4
and 1.83e-4 against it) and what cycle 14 repeats end to end. If that comparison is
inconclusive, the fallback is a `-legacy_quantise` flag emulating the round trip on the two
buffers that dominate — the corrected volume entering the histogram, and the working field
entering the fit. Do not build it speculatively.

**The stopping rule quantises everything downstream** (`change < 0.001`): two
implementations differing in the fifth decimal can run different iteration counts, which
moves the output by far more than any block difference. Every comparison fixes the
iteration count first with `-stop 0`, and only then repeats at the default protocol.

## 5. Build

- `legacy/N3/CMakeLists.txt` gains two `OBJECT` libraries, each renaming both the `main`
  and the `args` class so the two can coexist in one binary (defect 2):

  ```cmake
  ADD_LIBRARY(sharpenhist_core OBJECT src/SharpenHist/sharpen_hist.cc src/SharpenHist/args.cc)
  TARGET_COMPILE_DEFINITIONS(sharpenhist_core PRIVATE main=n3_sharpen_main args=n3_sharpen_args)

  ADD_LIBRARY(splinesmooth_core OBJECT src/SplineSmooth/splineSmooth.cc src/SplineSmooth/splineSmoothArgs.cc)
  TARGET_COMPILE_DEFINITIONS(splinesmooth_core PRIVATE main=n3_spline_main args=n3_spline_args)
  ```

  The `args.cc` files are linked only to satisfy the references the unused `main()`s make.
  The existing `sharpen_hist` and `spline_smooth` executables keep their own compilations
  and are unaffected. `-Dargs=` is safe because `args` is referenced only inside those two
  pairs: `fieldIO.cc`, `TBSpline.cc`, `Spline.cc` and the three histogram sources mention
  it nowhere.
- `ADD_EXECUTABLE(nu_correct_cxx ...)` over the `N3Pipeline/` sources,
  `Splines/{Spline,TBSpline}.cc`, `SplineSmooth/fieldIO.cc`,
  `VolumeHist/{DHistogram,WHistogram,GHistogram}.cc`, `$<TARGET_OBJECTS:sharpenhist_core>`
  and `$<TARGET_OBJECTS:splinesmooth_core>`.
- `INSTALL` it twice, as `nu_correct_cxx` and `nu_estimate_cxx`, so the argv[0] rule
  (`/nu_estimate/i` → estimate only) works as it does for the Perl. `-estimate_only` and
  `-correct` override it.
- The `LIBMINC_LIBRARIES` repair loop added in the previous task already lets this tree
  configure against the installed libminc.

## 6. Test harness

There is no C++ test framework in this environment — no gtest, no catch2, and nothing may
be installed. CTest is already wired up (`CMakeLists.txt:14, 229-231`) and
`legacy/N3/testing/CMakeLists.txt` holds ~20 cases, so the harness is CTest plus a header:

- `testing/n3pipeline/check.h` — about 40 lines: `CHECK_TRUE`, `CHECK_NEAR(a, b, tol)`,
  `CHECK_RMS(a, b, n, tol)`, a failure counter, and a `report()` that prints each failure
  with its measured margin and returns a non-zero exit code. Printing the margin matters:
  it is what makes the green step informative rather than binary, and it is how
  `tests/margins.py` reports in the PyTorch tree.
- One test executable per block (`test_shrink`, `test_histogram`, …), each an
  `ADD_EXECUTABLE` + `ADD_TEST` in `testing/CMakeLists.txt`, linked against the same object
  libraries as `nu_correct_cxx`.
- `testing/n3pipeline/regenerate_reference.sh` — **the only thing that runs an N3 program**.
  It drives `/opt/minc/1.9.18.13/bin` (and `/app/legacy/_install/bin` for `-parzen_sigma`)
  and writes `testing/n3pipeline/reference/`. This mirrors the policy already in force in
  the PyTorch tree (`tests/regenerate_reference.py`): tests read recorded answers, they do
  not shell out. Re-running the script and finding an empty `git diff` is itself a check.
- `testing/n3pipeline/reference/` — checked in. Small arrays (histograms, lookup tables,
  thresholds, domains) as text; volumes as raw little-endian float64. Record **only what
  the assertion looks at**: for a field comparison that is the masked voxels, not the
  volume. `chunk.mnc` is 91×52×50, so a masked dump is well under a megabyte.

Tolerances are fixed before the implementation exists and are justified without running it:
`exact` where both sides do the same arithmetic, `1e-6` where the oracle passed through
`%lf`, `5e-6` relative RMS for the SOR solve. **Where no bound can be justified in advance,
the test asserts the property that motivated it instead** — see cycle 13. A cycle that
stays red is a defect or a confound; widening the bound is not one of the remedies
(`CLAUDE.md`, "Test tolerances").

The existing ~20 CTest cases are the regression guard and must stay green throughout — in
particular after §5's object libraries are added, since `sharpen_hist` and `spline_smooth`
keep their own compilations of the renamed sources.

## 7. Red-green cycles

Each cycle is: declare in the header and stub the body (`abort()`, or return NaN) → write
the test and record its fixture → `ctest -R` shows **red**, and the failure is the
assertion or the stub, never a missing file → implement → **green** → commit. Stubbing
first is what proves the fixture is readable and the test wired up before any real code
exists.

All cycles run on `tests/data/chunk.mnc` + `chunk_mask.mnc` (91×52×50) unless stated.

| # | Test | Oracle | Bound |
|---|---|---|---|
| 0 | the harness itself: one deliberate `CHECK_NEAR` failure exits non-zero and prints its margin; then delete it | — | red-then-green on the harness, before it is trusted |
| 1 | `shrink` at factors 2, 3, 4 and a non-integer, and on a volume with mixed step sizes | `mincresample -nearest_neighbour -nelements … -step …` | exact. **First cycle after the harness**: a wrong grid invalidates everything after it |
| 2 | `resampleLabel` | `resample_labels -resample -like` | exact |
| 3 | buffers: `get_volume_real_value` is the identity on a created `NC_DOUBLE` volume; a load/save round trip preserves storage type and `valid_range`; masked `mean`/`stddev`/`min`/`max` | `volume_stats -mean -stddev -min -max -mask` | identity exactly; statistics to 1e-6 |
| 4 | `autoRange` and all three histogram estimators (plain, `-window`, `-gaussian_window 2`) | `volume_hist -bins 200 -auto_range -mask …` | 1e-6, the `%lf` limit |
| 5 | `sharpenLookup`, and separately `-blur` (deconvolution skipped) | `sharpen_hist -fwhm 0.15 -noise 0.01 -range …` on cycle 4's histogram | 1e-6 |
| 6 | `applyLookup`, including clamping outside the domain and interpolation between entries | `minclookup -continuous -range` | 1e-6 |
| 7 | both bimodal rules, which must be shown to **disagree** on the same volume as well as to match their own oracles | `volume_stats -biModalT`, `mincstats -biModalT` | exact |
| 8 | spline fit, `b_spline` at 200/100/50 mm and `tp_spline`, each with its own domain rule (defect 3) | `spline_smooth -full_support -b_spline …`, `spline_smooth -tp_spline …` | fields to 1e-6. **Compare fields, never coefficients**: the normal equations are near-singular at 200 mm and no solver determines the coefficients to better than ~1e-4 |
| 9 | `smooth()` in double, and separately the prolongation step, which *is* exactly reproducible | `correct_field` | 5e-6 relative RMS for the solve; exact for the prolongation |
| 10 | the staged stopping rule as a pure function over a table of `(iter, change, iterations[], stop[])` cases, including `-stop 0` never stopping | none — it is logic transcribed from `:171-185` | exact |
| 11 | one iteration of `NuEstimate`: the mask, then `est0`, then `field0` | the Perl with **`-save_fields -save_histograms`**, which writes `${basename}_est$iter.mnc`, `${basename}_field$iter.mnc` and `${basename}_hist$iter.txt` (`:154, :193, :508`) | per-iteration, so the loop is not an opaque end-to-end. 1e-4 relative RMS at `-shrink 1` (§4); report `field_CV` from both |
| 12 | `NuEvaluate` stage by stage: auto mask from `mincstats -biModalT`, `evaluate_field` on the full grid, `correct_field`, the floor clamp, the division | each Perl stage's intermediate | 1e-6 per stage except `correct_field`'s 5e-6 |
| 13 | end-to-end **properties**, asserted before any bound is measured: the field is strictly positive; the in-mask coefficient of variation of the output is below the input's; a volume with no planted non-uniformity yields a field within 1e-3 of constant; iteration counts match the Perl exactly at `-stop 0` | the Perl | properties, not bounds — none of these needs the code to have been run first |
| 14 | end-to-end bounded: `-shrink 1 -distance 200 -iterations 1 -stop 0`, with and without `-legacy_rounding` | `nu_correct` | 0.5·(log max − log min)/4095 relative RMS — the 12-bit quantum of the two intermediates over their own range (§4), 2.683e-4 on `chunk.mnc`; the driver measures 1.8e-4. This is the only end-to-end comparison whose bound is justified in advance |
| 15 | argv[0] and the argument table: `nu_estimate_cxx` writes only the `.imp`; `-estimate_only`/`-correct` override it; every out-of-scope option (`-em`, `-fir`, `-real`, `-differential`, `-initial`, `-islands`) exits non-zero with a message | none | behavioural. **Out-of-scope options must fail loudly, not be ignored** |
| 16 | `-tp_spline` and `-parzen_sigma 2` end to end at a fixed count | the Perl. For `-parzen_sigma` the oracle is **`/app/legacy/_install/bin/nu_correct` with `/app/legacy/_install/bin` first on `PATH`**, not the installed N3 (defect 6): `MNI::Spawn` resolves `volume_hist` through `PATH` and the stock one has no `-gaussian_window` | as cycle 14 |
| 17 | `-estimate_only` against the Perl's `.imp`, compared by evaluating both and diffing the fields, not the text | `evaluate_field` on each | 1e-6 |

Cycles 1-10 are unit cycles and should each close within a session. Cycles 11-17 are
integration cycles: they stay red longer, which is why 13 comes before 14 — a property that
can be asserted without measurement gives the integration work a green signal well before
any bound is available.

Reported, not asserted, once the cycles are green:

- `brain.mnc` at the default protocol against `nu_correct` and against
  `legacy/N3/testing/brain_nu_ref.mnc.gz`. The legacy suite's `1e-4` will not be met: a
  double pipeline cannot reproduce a 16-bit-quantised one, and `torch_n3`'s own legacy
  backend sits 3.7e-3 from it. A number with no defensible bound belongs in a printed table,
  not in an assertion.
- Both iteration counts at the default protocol on each test volume. A difference in count
  is the expected failure mode, not a defect.

## 8. Files

New: `legacy/N3/src/N3Pipeline/{Buffers,MincTools,Sharpen,SmoothField,FitField,NuEstimate,NuEvaluate}.{h,cc}`,
`legacy/N3/src/N3Pipeline/nu_correct_cxx.cc`;
`legacy/N3/testing/n3pipeline/check.h`, `regenerate_reference.sh`, one `test_*.cc` per
cycle, and the recorded `reference/` fixtures.

Edited: `legacy/N3/CMakeLists.txt` and `legacy/N3/testing/CMakeLists.txt`.

Commit at the end of each cycle, with the red output and the measured margin in the
message. A cycle whose test was written after its implementation is not one of these
commits.

Untouched: every existing source in `legacy/N3/src` (linked, not modified), the Perl
drivers, `legacy/EBTKS`, everything under `/app/torch_n3`, `/opt/minc`.

## Open risks

- **`-Dargs=` across a header boundary.** The rename must reach every translation unit that
  includes `args.h` or `splineSmoothArgs.h`. It does here because each header is included
  only by its own pair, but a compile error naming `args` is the signal that this stopped
  being true.
- **EBTKS `Histogram::biModalThreshold` uses `ceil(max−min+1)` bins**, which collapses to
  one or two bins on a float-valued input. This is legacy behaviour: reproduce it, do not
  fix it, and record it.
- **The `brain_nu_ref.mnc.gz` comparison has no bound**, only an expectation, which is why
  §7 reports it rather than asserting it. If it lands far from 3.7e-3, the diagnosis path
  is cycle 14, then the unit cycles — not a tolerance.
- **Cycles 11-17 can stay red for a long time.** The mitigation is cycle 13's properties
  and cycle 11's per-iteration oracle, both of which give a green signal before any
  end-to-end bound exists. If cycle 11 cannot be made to work — for instance if
  `-save_fields`' `exp_transform` loses too much through its own MINC write — the
  integration cycles lose their only intermediate oracle, and the fallback is
  `-legacy_quantise` (§4) brought forward rather than deferred.

## Not in scope, recorded

- Carrying back the port's other findings: the better-conditioned QR spline fit
  (`torch_n3/blocks/spline.py`, `solver="qr"`, cond 2.3e6 against the normal equations'
  5.3e12) and the `--denoise` prefilter. The C++ pipeline keeps `dsysv` on `AtA`.
- The EM and WM branches, `fir` smoothing, `-real`, `-differential`, `-initial`,
  `-islands`.
- Threading or GPU. The program stays single-threaded: the gain sought here is the removal
  of file round trips, and a second change at the same time would make it unmeasurable.
- One divergence found while reading, not fixed and not this task's business:
  `torch_n3/pipeline.py:122` resamples the user mask onto the estimation grid with
  **nearest neighbour**, where the Perl uses `resample_labels`, which is trilinear
  thresholded at 0.5. The C++ program follows the Perl.

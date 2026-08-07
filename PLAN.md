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
| `CreateMask` (`:297`) | `input > background_threshold`, intersected with the user mask; `-bimodalT`, or `-auto_mask` on a non-Talairach volume, takes the threshold from EBTKS `Histogram::biModalThreshold` over `ceil(voxelMax-voxelMin+1)` bins (`volumeStats.cc:286-299`); `-auto_mask` on a Talairach volume uses the average brain mask, label-resampled. **The C++ tool does not port the Talairach branch**: it reads no `xspace:spacetype` and always collapses `-auto_mask` to the bimodal threshold. See "Not in scope" |
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
`nu_correct_cxx.cc` exposes it on the CLI as `-legacy_rounding`/`-nolegacy_rounding`
(2026-08-06); it was reachable only from test code before that. It is now also a version
default: on under `-V1.0`, off under `-V1.1` (§7, cycle 14's row).

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

  ``0.5 · (log max − log min) / valid_steps``,

where `valid_steps` is `hi − lo` of the volume's own recorded `valid_range` — 4095 on
`chunk.mnc`, read from `chunk_valid_range.txt` by `n3fixture::valid_steps`, not a fixed
denominator — giving 2.683e-4 relative RMS there. The stages round to a half-voxel midpoint on the way
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
| 13 | end-to-end **properties**, asserted by driving the built binary (`test_driver_properties`): the field, read from the driver's own `.imp`, is strictly positive and finite in the mask; the in-mask coefficient of variation of the corrected output is below the input's; a volume with no planted non-uniformity (two-tissue phantom + additive noise) yields a **bounded, finite** field whose RMS CV stays below **0.25× the phantom's tissue contrast** — a derived bound proving the estimator does not absorb the structure it is meant to be blind to (neither this port at RMS CV 0.071, nor the legacy at 0.035, reaches a flat field; the gap is a recorded over-correction to chase in cycle 14, not a pass criterion); and `-stop 0` runs the requested staged count (cycle 10 end to end) | none — properties, asserted before any bound is measured, and no test in this cycle runs the Perl | properties, not bounds — none of these needs the code to have been run first. The legacy 0.035 is the installed `nu_correct` on the shipped phantom under the test's own options (`-shrink 2 -iterations 15 -stop 0.0 -distance 100 -mask chunk_mask`), re-measured 2026-08-06; the earlier ~0.0064 was taken on the striped phantom `1ac5192` replaced and does not apply. The criterion is a CV about the mean and therefore does not constrain a uniform gain: the field means are 1.078 (port) and 1.064 (legacy) where a bias-free volume's ideal is 1.0. `test_driver_properties` pins every run to `-V1.0 -nolegacy_rounding` (2026-08-06) so these numbers stay valid regardless of which protocol `nu_correct_cxx`'s own implicit default currently selects |
| 14 | end-to-end bounded: `-shrink 1 -distance 200 -iterations 1 -stop 0`, with and without `-legacy_rounding` | `nu_correct` | 0.5·(log max − log min)/valid_steps relative RMS — the 12-bit quantum of the two intermediates over their own range (§4); valid_steps is `hi − lo` of the volume's recorded valid_range (`chunk_valid_range.txt` = 0..4095 → 4095), 2.683e-4 on `chunk.mnc`. Closed in `test_driver_endtoend` (`99fc4a7`): both `-legacy_rounding` configurations measure 1.840e-04 against `nu_correct_shrink1.f64`, 1.46× inside the bound. **The three comparisons sharing this bound are tight, not roomy**: cycles 11, 12 and 14 sit at 0.86, 0.68 and 0.69 of it, and cycle 11's 2.299e-4 leaves 14% headroom, so a change to the histogram or the spline can put that one red. (`99fc4a7`'s message and an earlier revision of this row said "a full order below the bound", which is wrong by a factor of seven; the commit is published and is corrected here rather than amended.) This is the only end-to-end comparison whose bound is justified in advance, and the only test that reads the corrected volume the driver's `n3::save` writes. It runs at `-V1.0`, which reproduces the Perl's default protocol (`fwhm 0.15`, linear interpolation); `-V1.1` (2026-08-06) is the driver's own implicit default and does not match it, so the command must pass `-V1.0` explicitly |
| 15 | argv[0] and the argument table: `nu_estimate_cxx` writes only the `.imp`; `-estimate_only`/`-correct` override it; every out-of-scope option (`-em`, `-fir`, `-real`, `-differential`, `-initial`, `-islands`) exits non-zero with a message; and **the protocol versions resolve as "Protocol versions" below states** — a bare invocation gives `-V1.1`'s four values, `-V1.0` and `-V0.9` override them, each fills only what the user did not give, and the last `-V` wins | none | behavioural. **Out-of-scope options must fail loudly, not be ignored**. The version rules need no bound: they are the resolved `EstimateOptions`, reported by `-verbose`, against the table below |
| 16 | `-tp_spline` and `-parzen_sigma 2` end to end at a fixed count | the Perl. For `-parzen_sigma` the oracle is **`/app/legacy/_install/bin/nu_correct` with `/app/legacy/_install/bin` first on `PATH`**, not the installed N3 (defect 6): `MNI::Spawn` resolves `volume_hist` through `PATH` and the stock one has no `-gaussian_window` | as cycle 14 |
| 17 | `-estimate_only` against the Perl's `.imp`, compared by evaluating both and diffing the fields, not the text | `evaluate_field` on each | 1e-6 |

Cycles 1-10 are unit cycles and should each close within a session. Cycles 11-17 are
integration cycles: they stay red longer, which is why 13 comes before 14 — a property that
can be asserted without measurement gives the integration work a green signal well before
any bound is available.

### Protocol versions

`-V<x>` selects a set of defaults, each filling only the options the user did not give
(`nu_estimate.in:499-505`); a scalar resolved after parsing, so the flag is
order-independent and the last `-V` wins. Three exist:

| | `-V0.9` | `-V1.0` | `-V1.1` |
|---|---|---|---|
| `-iterations` | `10 20` | 50 | 1000 |
| `-stop` | `0.001 0.005` | 0.001 | 1e-5 |
| `-shrink` | 3 | 4 | 4 |
| `-fwhm` | 0.15 | 0.15 | 0.1 |
| `-parzen_sigma` | off (linear split) | off (linear split) | 4.0 |
| `legacy_rounding` | off | **on** | off |

`-V0.9` and `-V1.0` are ports: `-V0.9` is `nu_estimate.in:499-505`'s original protocol and
`-V1.0` is the Perl's current default (§"Default N3 protocol" in `CLAUDE.md`), with
`legacy_rounding` on because reproducing the Perl's `%lf` rounding between stages is what
that mode is for. **`-V1.1` is not a port**, and since `9154eac` (2026-08-06) it is what a
bare `nu_correct_cxx in out` selects. Its four departures do not stand on equal evidence:

- `-parzen_sigma 4.0` is measured, in the PyTorch tree rather than here:
  `experiments/README.md`, "The histogram kernel" — 450 random fields per window on colin27,
  20% planted at SNR 20, where N3's linear split leaves 4.63% of the field and `sigma 4`
  leaves 1.71%, better on 94% of trials.
- `-fwhm 0.1`, `-iterations 1000` and `-stop 1e-5` are **not measured anywhere**. They are
  recorded here as the shipped values, not as a result.

Two consequences to keep in view. The default depends on `-parzen_sigma`, whose end-to-end
oracle comparison is cycle 16 and still open, so the shipped protocol rests on the one path
with no oracle behind it yet. And every comparison against the Perl must pass `-V1.0`
explicitly: cycles 13, 14 and `test_driver_fwhm` all do.

Reported, not asserted, once the cycles are green ("the default protocol" below means the
Perl's own default, reproduced by `nu_correct_cxx -V1.0` since `-V1.1` (2026-08-06) became the
driver's own implicit default and does not match it):

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

## 9. Instructions for the implementing agent

The open work is enumerated in `/app/TODO.md`, which is the tracker; this section is the
standard each item is closed to. It applies to `legacy/N3/src/N3Pipeline/` and
`legacy/N3/testing/n3pipeline/`, on branch `aislop`.

### Priority

**Algorithm before ergonomics.** A change that alters a corrected voxel, a fitted field or a
recorded measurement outranks one that alters a message, a name or a file layout. The two
classes must not share a commit: a rename bundled with a behavioural fix makes the
behavioural diff unreadable, and a measurement moved inside a refactor cannot be attributed.
`TODO.md`'s "Order of work" applies this rule to the current list.

Within the algorithm class, rank by what a wrong answer costs a reader: silent wrong output
first, crash second, wrong diagnostic third. `-fwhm 0` writing an all-zero volume at exit 0
outranks `-distance 0` segfaulting, which outranks `-version` printing an incomplete string.

### Correctness

- The Perl is the specification. Cite the line a decision comes from in the comment beside
  it, as the existing code does (`:1557`, `:499-505`, `:316-325`, `fieldIO.cc:120-133`).
  A behaviour with no citation is an invention and has to be argued for.
- A value the Perl rejects must be rejected here: non-zero exit and a message on stderr,
  never accepted, never a crash. `nu_estimate_np_and_em.in:1449-1481` is the validation
  block; port it in full rather than one predicate at a time.
- Never edit a file under `/app/torch_n3/_legacy/`; it vendors `legacy/N3/src` byte for
  byte and is the PyTorch port's oracle. Never write to `/opt/minc/1.9.18.13` — the
  installed programs there are the untouched oracle. This tree's own build goes to
  `/app/legacy/_install`.
- The existing translation units in `legacy/N3/src` are linked, not modified.

### Bounds and measurements

- A bound is derived from a physical quantity before the code is run, never fitted to what
  the code happens to do. The quantities available here are the file's own quantum
  (`n3fixture::valid_steps("chunk_valid_range.txt")`, never a literal `4095`), the six
  decimals `%lf` prints (`1e-6`), and exactness.
- **A tolerance is never widened to make a test pass.** When an assertion fails: find the
  defect; or find and remove the confound; or, if the requirement was itself wrong, change
  it deliberately, in its own commit, with the measurement and the reasoning recorded.
- Compare relative RMS over a volume, never `max |a − b|`: whole-volume extremes are decided
  by a handful of mask-edge voxels and do not support a fixed bound.
- Compare fields, never spline coefficients: the normal equations are near-singular at
  200 mm and no solver determines the coefficients better than ~1e-4.
- **When a comparison's inputs change, re-measure both sides in the same session on the same
  input.** The recorded legacy field CV of 0.0064 survived a commit that replaced the
  phantom it was measured on, and stood in `PLAN.md` and `TODO.md` for a day as an 11× gap
  that is really 2.0×. A number carried across a change of input is not a measurement.
- Every number published in `PLAN.md` or `TODO.md` must be reproducible from the committed
  code by a stated command. If it is not, say where it came from in the same sentence.

### Commits

- One item per commit, red before green: the test that fails is committed with, or before,
  the fix that makes it pass. A cycle whose test was written after its implementation is not
  one of these commits.
- The message carries the red output and the measured margin.
- The commit message's list of what it fixed must match what it fixed. `1ac5192` claimed 26
  items and enumerated 24.

### Conventions in this code

- **Arguments.** `ParseArgv` with an `ArgvInfo` table, as `src/SharpenHist/args.cc`,
  `src/SplineSmooth/splineSmoothArgs.cc`, `src/VolumeStats/VolumeStatsArgs.cc`,
  `src/VolumeHist/args.cc` and `src/EvaluateField/evaluateFieldArgs.cc` all do. Until that
  rewrite lands, no new `atof(argv[++i])`: numeric options go through one helper that checks
  `strtod`'s end pointer and dies on a trailing character.
- **One program name.** `die()`, `usage()`, `out_of_scope()` and the positional-argument
  error all report `program_name` (the `argv[0]` basename). No message contains a literal
  `"nu_correct_cxx"`, and none prints the full `argv[0]` path.
- **Library code in `src/N3Pipeline/` does not `exit()`.** New failure paths return a status
  or throw, so a test can assert them in-process. The 17 existing `exit(1)` sites are their
  own commit, not a side effect of another change.
- **A rule implemented twice is a defect.** The `.imp` path rule currently has three
  independent copies. Shared test helpers go in `testing/n3pipeline/fixture.h`, beside
  `must()`, `read_text()`, `valid_steps()`; shared pipeline helpers go in `Buffers.h` or
  `MincTools.h`.
- **Check `snprintf`'s return** wherever a path is composed into a fixed buffer, or use
  `std::string`.
- Verify an assumption rather than trusting it, in the manner of `Buffers.cc:63-107`, which
  checks the contiguity and no-voxel-scaling properties every bulk loop rests on and fails
  loudly if either breaks.
- Layout: GNU braces, two-space indent, `/* … */` comments, ~80 columns. Prose in comments
  and in the two documents follows `/app/CLAUDE.md`'s "Documentation style": declarative,
  no rhetorical framing, no addressing the reader, bold reserved for operational warnings.

### Before reporting an item closed

Run, from `/app/legacy/_build/n3`:

```
make && ctest
```

Report the pass count, and state whether `make` recompiled anything — a green `ctest` over a
stale binary is not evidence. Quote the margins the changed tests printed. If a step was
skipped, say so; do not describe an unverified claim as verified. `legacy/N3` and
`legacy/EBTKS` must be clean apart from the commit under discussion, and
`/app/torch_n3/_legacy/n3/` must still be byte-identical to `legacy/N3/src`.

## 10. External BLAS/LAPACK for `legacy/EBTKS`/`legacy/N3`'s own build

Separate from the `nu_correct_cxx` work above, and orthogonal to it: let this tree's build
(`legacy/EBTKS` + `legacy/N3`, installing to `/app/legacy/_install`) link a BLAS/LAPACK
supplied at build time, compiling the bundled `clapack/` sources only when none is supplied.
Planning only — no change made yet. `/opt/minc/1.9.18.13`, `/app/torch_n3` and
`legacy/N3/src` stay untouched by this item; §8's "Untouched: … `legacy/EBTKS`" line belongs
to the `nu_correct_cxx` feature above and does not apply here — this item edits
`legacy/EBTKS/CMakeLists.txt` and `legacy/N3/CMakeLists.txt`, deliberately.

### Current state

- `legacy/EBTKS/clapack/` is a trimmed, f2c-translated LAPACK/BLAS subset. 20 `.c` files
  on disk; **19** are compiled unconditionally into `libEBTKS.a`, listed literally at
  `EBTKS/CMakeLists.txt:171-189` inside `EBTKS_LIB_SRCS` (`test.c` is the twentieth and is
  not built).
- `dsysv_` is the only symbol out of that subset called from outside `clapack/` itself —
  exactly once, `legacy/N3/src/Splines/TBSpline.cc:648`
  (`TBSpline::solveSymmetricSystem`). Everything else (`dgemm_`, `dsytrf_`, `dsytrs_`,
  `dlasyf_`, the f2c support routines `s_cmp`/`s_copy`, …) is internal to `dsysv_`'s own
  call tree and referenced nowhere else in either tree (grep for all 19 symbol names
  across `N3/src` and `EBTKS/{src,templates,include}` returns only the `dsysv_`
  declaration and call sites). **Nothing in `EBTKS_LIB_SRCS` itself calls LAPACK**: EBTKS's
  own library code is independent of `clapack/`, which is compiled into the archive purely
  so that consumers linking `-lEBTKS` resolve `dsysv_`.
- `legacy/EBTKS/src/TBSpline.cc` is a second, divergent copy of the same file — it calls
  `dsysv_("UU", …)` where N3's calls `dsysv_("U", …)` — and is **not** in
  `EBTKS_LIB_SRCS`, so it is dead and out of scope. It must not be added to the build as
  part of this item.
- `legacy/N3/CMakeLists.txt` never names LAPACK. It gets `dsysv_` transitively: every
  executable that compiles `Splines/TBSpline.cc` directly (`evaluate_field`,
  `spline_smooth`, `nu_correct_cxx`, `nu_estimate_cxx`, and the `testing/n3pipeline`
  binaries, via `n3pipeline_core`) links `${EBTKS_LIBRARIES}` through the old-style,
  directory-wide `LINK_LIBRARIES(mincprog ${EBTKS_LIBRARIES} ${VOLUME_IO_LIBRARIES}
  ${LIBMINC_LIBRARIES})` at `CMakeLists.txt:155`, which resolves against `libEBTKS.a` —
  where the symbol is defined today because `clapack/dsysv.c` is compiled into it.
  `correct_field` compiles no `TBSpline.cc` and needs nothing from LAPACK.
- `EBTKS_LIBRARIES` is exported as the bare string `"EBTKS"` (`EBTKSConfig.cmake.in`),
  resolved by `-lEBTKS` against a directory added with `LINK_DIRECTORIES`
  (`UseEBTKS.cmake.in`) — not a CMake target, so nothing about it propagates
  transitively to a consumer built from a separate `FIND_PACKAGE(EBTKS)` (which is how
  `legacy/N3` finds it, `CMakeLists.txt:43`). **Consequence**: if `clapack/*.c` stop
  being compiled into `libEBTKS.a`, `dsysv_` disappears from that archive, and
  `legacy/N3`'s own executables — which need the symbol independently of EBTKS's C++ —
  fail to link unless `legacy/N3/CMakeLists.txt` is separately told what to link instead.
  The choice of LAPACK cannot be made local to `EBTKS/CMakeLists.txt`; it has to be
  threaded through the exported `EBTKSConfig.cmake` to `legacy/N3/CMakeLists.txt`.

### Precedent already in this repository

`torch_n3/_legacy/build_legacy.py` solved the identical problem for the CFFI shim, which
compiles the same `TBSpline.cc` (byte-identical vendored copy) against a chosen LAPACK:
opt-in only, two variables (`N3_LAPACK_LIBS`, `N3_LAPACK_LIB_DIRS`), nothing
auto-detected, bundled `clapack/` reachable as the explicit fallback
(`N3_LAPACK_LIBS="EBTKS"`). Its own default — link nothing, share whatever LAPACK the
host process (PyTorch) already loaded — is specific to being a Python extension and has
no equivalent for a standalone executable, so it is not reused here; the naming
convention is.

That swap is measured there, not assumed safe: `README.md` ("Which LAPACK the legacy
backend links") and `PROBLEMS.md` §8 record calling the **unmodified** `TBSpline.cc`
(`_integer` = `long int`, 8 bytes; no hidden Fortran string-length argument — an f2c-style
call) directly against a real, separately built system LAPACK (OpenBLAS) with no ABI
adapter, and it works: the two builds' fitted fields differ by 3.1e-11 relative RMS at one
iteration — a difference between two valid solvers of a `~1e13`-condition-number system,
not corruption. Reading why it works out safely, from `solveSymmetricSystem`
(`TBSpline.cc:626-651`): `ipiv` is allocated, passed, and deleted **unread**, so its
assumed element width never matters; `w_info` is narrowed to `int` before use, which is
defined truncation to the low 32 bits regardless of what a callee using 4-byte `INTEGER`
left in the high bytes; every scalar *input* (`n`, `nrhs`, `lda`, `ldb`, `lwork`) is a
small positive value whose low 4 bytes equal the true value on this little-endian
platform, so a callee reading them as 32-bit `INTEGER` still reads correctly; and `ipiv`,
being `_integer[n]` at 8 bytes an element, is strictly larger than the `n`×4 bytes a
32-bit-`INTEGER` callee writes, so the width mismatch cannot overrun it. The **missing
hidden string-length argument** is a separate question that the integer-width reasoning
does not cover, and it is safe for its own reason: `UPLO` is `CHARACTER*1` in LAPACK and
is consumed only through `LSAME`, which compares one character. A Fortran compiler emits
that comparison from the dummy argument's *declared* length, not the passed one, so the
hidden length is never read and the register holding garbage is never consulted. This is
why f2c-style calls into gfortran-built LAPACK are routine rather than exceptional.
(`XERBLA` takes `CHARACTER*6` plus a hidden length and is reached only on an argument
error, which this call site does not produce.) All of this is
checked-in, already-published evidence, not a theoretical argument, and it directly
de-risks doing the same swap here: **no change to `TBSpline.cc` is planned or required.**
Scope note: this reasoning covers LP64 (32-bit Fortran `INTEGER`) libraries — OpenBLAS,
distro reference LAPACK/BLAS, MKL's default `mkl_rt` interface. An ILP64 build was not
measured; document it as unsupported rather than silently assume it works.

### Design

New CMake cache variables in `legacy/EBTKS/CMakeLists.txt`, both empty by default, which
reproduces the current `libEBTKS.a` byte for byte:

- `EBTKS_LAPACK_LIBRARIES` — `;`/space-separated library names or full paths, e.g.
  `openblas` (this machine, verified above — one name, since Debian's OpenBLAS package
  merges LAPACK into the same `.so`), the more portable `lapack;blas`, or an absolute path
  to `libmkl_rt.so`. Non-empty is the only trigger: no `FIND_PACKAGE(BLAS)`/
  `FIND_PACKAGE(LAPACK)` autodetection (see "Rejected alternatives").
- `EBTKS_LAPACK_LIBRARY_DIRS` — optional, added via `LINK_DIRECTORIES` for bare names.

Naming mirrors `N3_LAPACK_LIBS`/`N3_LAPACK_LIB_DIRS` deliberately — same mechanism, same
names, already proven elsewhere in this repository.

`legacy/EBTKS/CMakeLists.txt`:

```cmake
SET(EBTKS_LAPACK_LIBRARIES "" CACHE STRING
    "External LAPACK/BLAS to link, e.g. 'lapack;blas' or an absolute path to \
libmkl_rt.so. Empty (default) builds the bundled clapack/ sources instead.")
SET(EBTKS_LAPACK_LIBRARY_DIRS "" CACHE STRING
    "Extra -L directories for EBTKS_LAPACK_LIBRARIES, when its entries are bare names.")

IF(EBTKS_LAPACK_LIBRARIES)
  IF(EBTKS_LAPACK_LIBRARY_DIRS)
    LINK_DIRECTORIES(${EBTKS_LAPACK_LIBRARY_DIRS})
  ENDIF()
ELSE()
  LIST(APPEND EBTKS_LIB_SRCS
    clapack/dcopy.c  clapack/dgemm.c  clapack/dgemv.c  clapack/dger.c
    clapack/dlasyf.c clapack/dscal.c  clapack/dswap.c  clapack/dsyr.c
    clapack/dsysv.c  clapack/dsytf2.c clapack/dsytrf.c clapack/dsytrs.c
    clapack/idamax.c clapack/ieeeck.c clapack/ilaenv.c clapack/lsame.c
    clapack/s_cmp.c  clapack/s_copy.c clapack/xerbla.c)
ENDIF()
...
ADD_LIBRARY(EBTKS ${LIBRARY_TYPE} ${EBTKS_LIB_SRCS})
IF(EBTKS_LAPACK_LIBRARIES)
  TARGET_LINK_LIBRARIES(EBTKS PUBLIC ${EBTKS_LAPACK_LIBRARIES})
ENDIF()
```

The 19 `clapack/*.c` lines move out of the unconditional literal `EBTKS_LIB_SRCS` list
(`CMakeLists.txt:171-189`) into the `ELSE()` branch above; nothing else in the file
changes. The `TARGET_LINK_LIBRARIES(EBTKS PUBLIC …)` is defensive rather than load-bearing:
`EBTKS` is a **static** library, so it records an interface dependency and embeds nothing,
and no source in `EBTKS_LIB_SRCS` calls LAPACK at all ("Current state"), so the one in-tree
consumer (`EBTKS/testing/ebtks_test_fft`, an FFT test, reached through
`LINK_LIBRARIES(EBTKS)` — a target name, so interface deps do propagate) does not actually
need it. Keep it so that a future EBTKS source calling LAPACK links without a second fix.

Export the decision through `EBTKSConfig.cmake.in`/`UseEBTKS.cmake.in`. **Two** variables
are required, not one, and both must be exported for the same reason `EBTKS_LIBRARIES` and
`EBTKS_LIBRARY_DIRS` both already are — a bare `-l` name needs its `-L` alongside it:

```
set(EBTKS_LAPACK_LIBRARIES    "@EBTKS_LAPACK_LIBRARIES_CONFIG@")     # empty if EBTKS bundled its own
set(EBTKS_LAPACK_LIBRARY_DIRS "@EBTKS_LAPACK_LIBRARY_DIRS_CONFIG@")
```

with `UseEBTKS.cmake.in` extended to `LINK_DIRECTORIES(${EBTKS_LIBRARY_DIRS}
${EBTKS_LAPACK_LIBRARY_DIRS})`, mirroring what it already does for EBTKS's own build
directory. An earlier draft of this plan proposed exporting a single variable holding
"resolved absolute paths" so the consumer needed no `-L`; that is **not implementable** —
CMake performs no configure-time resolution of a bare library name to a file, that being
the linker's job, and there is no supported API that would do it. Passing absolute paths in
`EBTKS_LAPACK_LIBRARIES` (which works, and needs no `-L`) stays available to the user as a
convention, but the build must not depend on it. For the `openblas` case on this machine
neither is needed: `/usr/lib/x86_64-linux-gnu` is a default linker search path, so
`EBTKS_LAPACK_LIBRARY_DIRS` stays empty.

`legacy/N3/CMakeLists.txt` needs exactly one line changed, extending the existing
directory-wide link at `:155`:

```cmake
LINK_LIBRARIES(mincprog ${EBTKS_LIBRARIES} ${EBTKS_LAPACK_LIBRARIES} ${VOLUME_IO_LIBRARIES} ${LIBMINC_LIBRARIES})
```

`${EBTKS_LAPACK_LIBRARIES}` is empty — a no-op — whenever EBTKS built the bundled
fallback, so this line is unchanged in the default configuration. Because this is the
directory-wide link list, every target added afterward (`evaluate_field`,
`spline_smooth`, `nu_correct_cxx`, `nu_estimate_cxx`, and, through
`ADD_SUBDIRECTORY(testing)`, every `n3pipeline` test binary) picks it up with no further
edits to `legacy/N3/testing/CMakeLists.txt`.

Nothing in `legacy/N3/src` or `legacy/EBTKS/{src,templates,include}` changes; no
re-vendoring of `torch_n3/_legacy/n3/` or `torch_n3/_legacy/ebtks/` is triggered.

### Rejected alternatives

- **`FIND_PACKAGE(BLAS)`/`FIND_PACKAGE(LAPACK)` autodetection.** A spike confirms it
  resolves cleanly on this machine with no Fortran compiler present (CMake 3.28,
  `FIND_PACKAGE(LAPACK)` → `/usr/lib/x86_64-linux-gnu/libopenblas.so;-lm;-ldl`), but
  making it the default would make the default build's numerics depend on whatever
  happens to be installed, silently — this is the tree whose own `_install` build backs
  `nu_correct_cxx`'s comparisons against the Perl oracle (§5-§7 above), and `CLAUDE.md`'s
  "Test tolerances" section is explicit that behaviour must not change without a
  deliberate, measured, recorded step. Opt-in only keeps `cmake ..` with no extra flags
  byte-for-byte what it is today.
- **A `dsysv_` ABI adapter** (converting the 8-byte f2c `integer`/no-hidden-length call
  to a real Fortran ABI). Considered because of the width and hidden-argument
  differences, but the checked-in measurement in `README.md`/`PROBLEMS.md` §8 (above)
  shows the direct, unmodified call already works correctly against real system LAPACK.
  Adding indirection would solve an already-solved problem, and — because it would have
  to live in or beside `TBSpline.cc` — would trigger the re-vendoring `CLAUDE.md`
  requires for any edit to a file `torch_n3/_legacy/n3/` mirrors, for no benefit.

### Verification, once implemented

**The existing CTest suite cannot serve as the acceptance gate for this change, and an
earlier draft of this plan wrongly assumed it could.** Two independent reasons, both
checked:

1. *The tests do not exercise the binaries under test.* `nu_reference_1` runs
   `compare_nu_result.pl`, which shells out to bare `nu_estimate`/`nu_evaluate`; those
   Perl drivers resolve programs through `MNI::Spawn` against `$ENV{PATH}`
   (`nu_estimate.in:417`, `SetOptions(strict => 2)` at `:486`). `MINC_TEST_ENVIRONMENT`,
   the variable `testing/CMakeLists.txt` would use to point that at the build tree, is
   referenced four times there and **set nowhere in this tree** — it comes from the MINC
   superbuild, which is not in play. With `/opt/minc/1.9.18.13/bin` on `PATH`, the test
   measures the installed oracle. A green `nu_reference_1` is therefore no evidence about
   a rebuilt `spline_smooth` whatsoever. Confirmed on the current `_install`:
   `nm _install/bin/spline_smooth | grep dsysv_` shows `T dsysv_`, statically bound from
   `clapack/`, while the binary the test actually ran is the one under `/opt/minc`.
2. *Even pointed at the right binaries, its tolerance is the wrong instrument.*
   `compare_nu_result.pl` computes relative RMS against `brain_nu_ref.mnc.gz` at
   `1e-4` across a full 50-iteration `nu_estimate`. `README.md` ("Which LAPACK the legacy
   backend links") records that this same swap changes the fitted field by 3.1e-11 and
   moves the divergence threshold from the sixth iteration to the second; `CLAUDE.md`
   states no end-to-end N3 comparison is meaningful past three digits. A 50-iteration
   comparison at 1e-4 may therefore fail under an external LAPACK **without anything
   being wrong**, and per `CLAUDE.md` widening 1e-4 is not an available remedy.

So the gate is block-level, per `CLAUDE.md`'s "constrain blocks rather than pipelines":

- Default build, no new flags — the only place a strict identity is claimed, and it is
  claimed strictly: `ar t libEBTKS.a` lists the same 19 `clapack` object members as today,
  and the whole archive should be bit-identical to a pre-change build (same sources, same
  order, same flags). `legacy/N3/testing`'s CTest cases stay green, which for the default
  build *is* meaningful, because nothing changed.
- `-DEBTKS_LAPACK_LIBRARIES=openblas` (the confirmed case, "OpenBLAS, checked concretely on
  this machine" above) — link-level checks, which are what this item is actually
  responsible for: `ar t libEBTKS.a` lists **no** `clapack` members; `nm` on
  `spline_smooth`, `evaluate_field`, `nu_correct_cxx`, `nu_estimate_cxx` shows `dsysv_` as
  `U` rather than `T`, and `ldd` lists `libopenblas.so.0`; every executable and the CTest
  suite still builds and runs to completion.
- Numerical gate, **one iteration, not fifty**: run the two builds' own `spline_smooth`
  over the same input and compare the fitted fields directly. Expect a difference near the
  already-measured 3.1e-11 relative RMS; materially larger is a defect in the swap,
  materially smaller means the external library was not actually reached. Drive this with
  explicit absolute paths to the built binaries, not through `PATH`, for reason 1 above.
- `nu_reference_1` under external LAPACK is run as an **observation, not a gate**, and
  only with `PATH` explicitly set to the build tree so it means something. Record the
  relative RMS it reports. If it exceeds 1e-4, that is the documented amplification and
  belongs in `PROBLEMS.md` beside the existing §8 entry, together with the number — it is
  not license to change the 1e-4 in `testing/CMakeLists.txt`.
- `-DEBTKS_LAPACK_LIBRARIES="lapack;blas"` (this machine's `liblapack.so.3`/
  `libblas.so.3`, currently pointing at the same `libopenblas.so.0` via
  `update-alternatives`): link-level checks only; expected to match the `openblas` case
  bit for bit, since it is the same shared object under a different `-l` name. Its value is
  as a test of the multi-name/`-L` path, not of a second implementation.
- The genuinely different implementation is the reference Netlib build: `apt install
  liblapack3` selected via `update-alternatives --config
  liblapack.so.3-x86_64-linux-gnu`, which `README.md:536-538` already names. This is the
  one that would show a larger-than-3.1e-11 field difference legitimately, so measure it
  rather than predicting it. Requires installing a package, which `CLAUDE.md` forbids
  doing unilaterally — ask first, and treat this bullet as optional.
- If `EBTKS_LAPACK_LIBRARIES` is ever turned on for a real `_install` build rather than
  exercised only as a build-system capability, record the effect on `nu_correct_cxx`'s
  numbers in `PROBLEMS.md`, following the existing §8 entry as the template.

### Default posture — resolved

Confirmed by the user: the goal is the ability to **completely replace** `clapack/` with a
system-provided library (OpenBLAS named as the concrete example), the bundled sources used
only when none is supplied — which is a restatement of the original request ("use another
BLAS/LAPACK library provided during the build, only use the included version as a
fall-back") and exactly the mechanism already designed above: when
`EBTKS_LAPACK_LIBRARIES` is set, none of the 19 `clapack/*.c` are compiled and the archive
carries no clapack objects at all (first "Verification" bullet below) — replacement is
complete, not additive. `cmake ..` with no extra flags keeps building the bundled fallback,
matching "only use the included version as a fall-back option" and CLAUDE.md's
no-silent-behaviour-change rule; a system library is selected by passing
`-DEBTKS_LAPACK_LIBRARIES=...` explicitly, not auto-detected. No `FIND_PACKAGE` probing is
added.

### OpenBLAS, checked concretely on this machine

The user named OpenBLAS with LAPACKE as the example, so both were checked directly rather
than assumed:

- `libopenblas-dev`/`libopenblas0-pthread` (0.3.26, this container's installed package)
  builds LAPACK *into* `libopenblas.so.0` under its plain Fortran names — `nm -D
  /usr/lib/x86_64-linux-gnu/libopenblas.so.0` lists `dsysv_` (and `dsysv_aa_`, `dsysv_rk_`,
  `dsysv_rook_`) as `T` (defined). One name resolves the sole symbol this tree needs:
  `-DEBTKS_LAPACK_LIBRARIES=openblas` — no separate `lapack`/`blas` package required, and
  it is the same library `update-alternatives` already points `liblapack.so.3`/
  `libblas.so.3` at on this machine (`README.md`, "Which LAPACK the legacy backend
  links"), so this is not a second code path, just a shorter one.
- LAPACKE (`LAPACKE_dsysv`, the C-calling-convention wrapper — `lapack_int`, an explicit
  row/column-major flag, no hidden Fortran arguments) is a **separate, optional** interface
  some OpenBLAS builds also ship. This package does not: `nm -D libopenblas.so.0 | grep
  LAPACKE_` is empty, and no `lapacke.h` is installed system-wide (only a vendored copy
  under Eigen, unrelated). It does not matter here regardless of whether it is present,
  because `TBSpline.cc:648` calls the raw Fortran-mangled `dsysv_` symbol directly, the
  same call the bundled `clapack/dsysv.c` answers today — not `LAPACKE_dsysv`. Nothing in
  this plan links or requires `liblapacke`; recorded here only so "OpenBLAS, which
  includes LAPACKE" is not misread as this swap needing LAPACKE's headers or a
  `LAPACKE_dsysv` call. Should the ABI-adapter alternative ever be revisited (currently
  rejected, above), LAPACKE would be the natural replacement for the hand-declared
  `extern "C"` prototype — it is the more robust interface — but that is a `TBSpline.cc`
  change and out of scope for this build-system-only item.

`-DEBTKS_LAPACK_LIBRARIES=openblas` becomes the primary case in "Verification, once
implemented" below, ahead of the generic `lapack;blas` alternative, since it is the example
given and already confirmed to export the needed symbol on this machine.

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
- **`-auto_mask` on a Talairach-tagged volume**. The Perl detects Talairach space by reading
  `xspace/yspace/zspace:spacetype` and, when the ICBM average brain mask is readable,
  intersects it (label-resampled) with the background-threshold mask instead of using the
  bimodal threshold (`nu_estimate_np_and_em.in:361-369`). This tree ships the ICBM mask
  (`model_data/N3/icbm_avg_152_*_VI_mask.mnc.gz`) but no Talairach-tagged test volume exists,
  so the branch cannot be cycled red/green and is deliberately not ported. The C++ tool
  always collapses `-auto_mask` to the bimodal threshold, which for a Talairach volume that
  differs from the Perl. Recorded so it is not a *silent* divergence.
- Threading or GPU. The program stays single-threaded: the gain sought here is the removal
  of file round trips, and a second change at the same time would make it unmeasurable.
- One divergence found while reading, not fixed and not this task's business:
  `torch_n3/pipeline.py:122` resamples the user mask onto the estimation grid with
  **nearest neighbour**, where the Perl uses `resample_labels`, which is trilinear
  thresholded at 0.5. The C++ program follows the Perl.

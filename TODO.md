# TODO: `nu_correct` in C++, entirely in memory

Progress tracker for `/app/PLAN.md` (authoritative). The implementation lives in the
nested git repository `legacy/N3/`, on branch `aislop` (per-cycle commits); `/app/TODO.md` and
`/app/PLAN.md` live in the outer repository.

Status: `[x]` done & committed, `[~]` code written but not wired/tested, `[ ]` pending.

## Red-green cycles (PLAN §7)

| # | Test | Status |
|---|---|---|
| 0 | harness itself (red-then-green) | [x] `n3cxx_test_harness` |
| 1 | `shrink` | [x] `n3cxx_test_shrink` |
| 2 | `resampleLabel` | [x] `n3cxx_test_resample_label` |
| 3 | buffers / masked statistics | [x] `n3cxx_test_buffers` |
| 4 | auto range + histogram estimators | [x] `n3cxx_test_histogram` |
| 5 | sharpen / deconvolution | [x] `n3cxx_test_sharpen` |
| 6 | apply lookup | [x] `n3cxx_test_lookup` |
| 7 | both bimodal rules | [x] `n3cxx_test_bimodal` |
| 8 | spline fit (b_spline + tp_spline) | [x] `n3cxx_test_spline` |
| 9 | `smooth()` in double + prolongation | [x] `n3cxx_test_extend` |
| 10 | staged stopping rule | [x] `n3cxx_test_stopping` |
| 11 | one iteration of `NuEstimate` | [x] `n3cxx_test_estimate` |
| 12 | `nu_evaluate` stage by stage | [x] `n3cxx_test_evaluate` (commit `63648fb`) |
| 13 | end-to-end properties | [x] `test_driver_properties` (field from the driver's `.imp` strictly positive & finite in-mask; in-mask output CV < input CV; no-bias two-tissue phantom → bounded field, RMS CV < 0.25× its tissue contrast; `-stop 0` runs the requested staged count) |
| 14 | end-to-end bounded (`-shrink 1 -iterations 1 -stop 0`) | [x] `test_driver_endtoend` (commit `99fc4a7`): drives the binary at `-V1.0` with and without `-legacy_rounding` against `nu_correct_shrink1.f64` under the derived bound `0.5·(log max − log min)/valid_steps` = 2.683e-4; both measure 1.840e-04 (the recorded 1.8e-4, now an assertion, and a full order below the bound). This is the one end-to-end comparison whose bound is justified in advance (the 12-bit quantum of the two MINC round trips, not the 16-bit figure PLAN §4 once assumed), and the only test that reads the corrected volume the driver's `n3::save` writes |
| 15 | argv[0] / argument table | [ ] drive `nu_estimate_cxx` vs `nu_correct_cxx` to pin the argv[0] split, and assert the argument rules: `-V0.9` order-independence, the `-bins`/`-background`/`-distance`/`-iterations` validation, `-clobber` covering the `.imp`, every out-of-scope option exiting non-zero, and the four degenerate-argument cases the 2026-08-06 review found unguarded (items 1-4 below) |
| 16 | `-tp_spline` and `-parzen_sigma` end to end | [ ] the `-parzen_sigma` oracle is `/app/legacy/_install/bin/nu_correct` with `/app/legacy/_install/bin` first on `PATH` (PLAN §7 row 16): `MNI::Spawn` resolves `volume_hist` through `PATH` and the stock one has no `-gaussian_window` |
| 17 | `-estimate_only` vs `.imp` | [~] oracle `estimate.imp` recorded; driver writes the .imp, Domain matches exactly. `estimate.imp` is read by `test_evaluate.cc:62,75` as an *input*; no test evaluates the driver's own `.imp` and the Perl's and diffs the two fields, which is what the cycle specifies |

## Driver (blocks 13–17)

- `legacy/N3/src/N3Pipeline/nu_correct_cxx.cc` (main + ParseArgv) — [x] commit `48a101a`
- `ADD_EXECUTABLE(nu_correct_cxx)` + INSTALL as `nu_correct_cxx`/`nu_estimate_cxx` — [x] (drives argv[0] rule; two targets)
- Smoke CTest `n3cxx_driver_estimate` + `n3cxx_driver_correct` — [x] (CTest 36/36)

## Protocol versions — done (2026-08-06)

- `-V1.1` added as the new implicit default (`-stop 1e-5 -iterations 1000 -fwhm 0.1
  -parzen_sigma 4.0`, `legacy_rounding` off); `-V1.0` now names what used to be the only,
  implicit default (`legacy_rounding` on by default, reproducing the Perl's `%lf` rounding).
  `-legacy_rounding`/`-nolegacy_rounding` is a real CLI flag for the first time — previously
  the `EstimateOptions` field of that name (`NuEstimate.h:49`) was reachable only from test
  code. This finally lets cycle 14's still-unwritten `test_driver_endtoend.cc` (row 14 above)
  toggle it from the command line instead of constructing `EstimateOptions` by hand.
- `test_driver_fwhm.cc` and `test_driver_properties.cc` pin every driver invocation to
  `-V1.0 -nolegacy_rounding`, so their comments' recorded numbers (byte-exact
  default/`-fwhm 0.15` agreement; RMS CV 0.071/quantum 5.4e-4) stay valid independent of the
  implicit default. Re-verified after the change: identical to the previously recorded values.
  `PLAN.md` §4, §7 (cycles 13-14) and its "Reported, not asserted" tail now say `-V1.0` is
  what "the default protocol" means for reproducing a PLAN-documented number.
- `-noparzen` removed from `nu_correct_cxx.cc` entirely (it now dies as an unknown option);
  the histogram is always built with linear interpolation between bin centres
  (`WHistogram.h:65-93`) unless `-parzen_sigma` is given, which was already true internally
  (`Histogram.cc:65-68`) but previously togglable off via a flag with no counterpart once
  `-noparzen` is gone. `-parzen` itself is kept, accepted as a no-op, for compatibility.
  `HistogramOptions.window` and its plain-single-bin branch are untouched — that path is
  still unit-tested at the class level (`test_histogram.cc`), just no longer reachable from
  the driver's CLI.
- `make && ctest`: 36/36, no new warnings.

## Reported, not asserted (PLAN §7 tail)

- `brain.mnc` default-protocol tables vs `nu_correct` and `brain_nu_ref.mnc.gz` — [ ]
- Iteration counts at the default protocol on each test volume — [ ]
- Cycle 13 measured a port **over-correction on a no-bias phantom**: field RMS CV 0.071
  against the legacy's 0.035; neither reaches a flat field. The port stays under cycle 13's
  derived bound (0.25× the phantom's tissue contrast = 0.214), so it passes while the gap is
  a chase item. Record the legacy run as an oracle in cycle 14 — [ ]. The legacy figure was
  re-measured on 2026-08-06; the earlier ~0.0064 was taken on the pre-`1ac5192` striped
  phantom and does not apply to the shipped one (review item 7 below).

## Fixed since the review

- `-fwhm` aliased `-distance` (`nu_correct_cxx.cc:219`), corrected in `40f94d1`. `-fwhm` now
  sets the deconvolution width and `-distance` alone sets the knots, which is what
  `nu_estimate.in:205` maps it to and what `:498` passes on as `-sharpen <width> 0.01`.
  Item 1 above adds an asserting test so it cannot silently regress.

## Found in review (2026-08-04)

All eight items below are fixed; each has its own commit (plus the two PLAN edits are
separate commits per AGENTS.md). CTest is 36/36; a fresh out-of-source configure with the
LIBMINC/EBTKS dirs succeeds after the in-source artifacts were removed.

| # | Resolved in | Item |
|---|---|---|
| 1 | `e6b46b3` | The two `40f94d1` exit-status-only smoke tests were joined (not replaced) by `test_driver_fwhm`, which drives the built binary and compares outputs: `-fwhm 0.15` byte-identical to the default, `-fwhm 0.3` and `-distance 100` each move the correction through their own, different channels. A re-aliasing of `-fwhm` fails an assertion now, not by exhausting memory. `n3cxx_driver_estimate` and `n3cxx_driver_correct` remain registered |
| 2 | `make install` | Re-ran `make install`; `/app/legacy/_install/bin/nu_correct_cxx` is now the 15:30 build. Confirmed the installed copy equals the fresh build byte-for-byte (rms 0.0) and now writes the `.imp` |
| 3 | `8ccdca2` | `-auto_mask`'s Talairach branch recorded as not ported in PLAN §3 and Not-in-scope. The tool always collapses `-auto_mask` to bimodal; the ICBM mask ships but no Talairach-tagged test volume exists, so the branch cannot be cycled red/green. Divergence is now documented, not silent |
| 4 | `e6b46b3` | A correct run now always writes the `.imp` (`imp_path`, relocated only by `-mapping_dir`), matching the comment and the Perl (`nu_estimate.in:57-58`) |
| 5 | `4989c0b` | PLAN §4 and §7 cycle 14 corrected in their own commit: bound is 0.5·(log max − log min)/4095 = 2.683e-4 on `chunk.mnc`, derived, not assumed; cycles 11-12 measure 2.30e-4/1.83e-4; the driver measures 1.8e-4 |
| 6 | `e6b46b3` | `imp_path` now replaces only the final extension (`s/\.[^\.]*$/\.imp/`), so `out.mnc.gz → out.mnc.imp`; the dead `.mnc.gz`/`.mnc.Z` length checks are gone |
| 7 | `2847122` | 17 in-source CMake artifacts `git rm --cached`'d and a `.gitignore` added; `config.h.cmake` (CONFIGURE_FILE'd at CMakeLists.txt:134) correctly kept. Fresh out-of-source configure verified |
| 8 | `e6b46b3` | Dead fragments removed: the identical-arms `resample_label` ternary in `NuEstimate.cc` and the unreachable `!sharpen && parzen_sigma` die (plus the then-unused `A.sharpen` field) |

Verified alongside the gaps: CTest 36/36 including the 18 pre-existing legacy cases;
`legacy/EBTKS` carries no changes of its own; no pre-existing `legacy/N3` source or Perl driver
modified by the C++ task; `torch_n3/_legacy/n3/` still byte-identical to `legacy/N3/src`.

## Resolved from review (2026-08-05)

A second pass over `691d56f` (cycle 13) and `d4ef611` and the driver/test suite raised
26 items; all are now fixed in one batch (commit below). CTest is 36/36. The confirmed
facts from that pass still hold: `imp_path` reproduces `replace_ext` exactly, every correct
run writes the `.imp`, `torch_n3/_legacy/n3/` is byte-identical to `legacy/N3/src`, and
`legacy/EBTKS` carries no changes of its own.

### Behavioural divergences from the Perl — fixed in `nu_correct_cxx.cc`

1. `-background` no longer suppresses the automatic threshold:
   `e.bimodalT = user_mask == NULL && (A.bimodalT || A.auto_mask)`. `CreateMask`
   (`nu_estimate_np_and_em.in:316-325`) computes the bimodal threshold whenever
   `auto_mask` is set and no user mask was given, overwriting `-background`;
   `nu_estimate.in:509-511` always adds `-auto_mask`.
2. `-bimodalT` with `-mask` is overridden by the mask (as the option text and
   `CreateMask`'s `if(!defined $user_mask_volume)` require), and the threshold
   histogram is never masked — the `@VolumeStatsCmd`'s dead `-mask $Mask` is not
   carried over. `NuEstimate.cc` now passes `NULL` to `bimodal_threshold_volume_stats`.
3. `-V0.9` is order-independent: a `version` scalar is resolved after parsing, and the
   0.9 defaults (`iterations 10 20`, `stop 0.001 0.005`, `shrink 3`) fill only the
   options the user did not give (`nu_estimate.in:499-505`). Verified: the corrected
   volume's voxel data (`max|a−b| = 0.0`) and `.imp` coefficients are identical for
   `-shrink 2 -V0.9` and `-V0.9 -shrink 2` — not byte-identical, once the later fix for
   item 6 put the full invocation into the history line (review, 2026-08-06, item 15).
4. `-clobber` now covers the `.imp`: a correct run refuses to proceed when the `.imp`
   already exists, as `nu_estimate_np_and_em`'s `-noclobber` does for its output.
5. Validation matches the Perl: die on `-bins <= 1`, `-background < 0`,
   `-distance < 0`, and any `-iterations < 0` (`nu_estimate_np_and_em.in:1459-1467`).
6. History is the full invocation (`"$0 @ARGV"`, `$Invocation = "$0 @ARGV"`:
   `:1557`): `invocation()` builds `<name> <args…>`, is passed to `nu_estimate` for the
   `.imp`'s command attribute, and is used as the corrected output's appended history.
7. `-version` prints the name the binary was invoked under (`Program <name>`), so
   `nu_estimate_cxx -version` says `nu_estimate_cxx`.

### Cycle 13 and the test suite — fixed

8. The phantom is indexed in `values()`'s storage order (z fastest = `sizes[2]`, then
   `sizes[1]`, then `sizes[0]`), so the region is a real ellipsoid, not aliased
   stripes, and `rx/ry/rz` sit on their own axes.
9. The factor-of-two bound is replaced by a derived one: field RMS CV below 0.25× the
   phantom's tissue contrast (`(250−100)/((250+100)/2) = 0.857`, bound 0.214). Measured
   CV 0.071 leaves a real margin and the bound proves the estimator does not absorb the
   structure it is meant to be blind to.
10. Property 3 measures RMS CV over the mask, not the `lo`/`hi` extreme values.
11. The field CV is now the quantity computed and printed by the test, so the cycle-14
    chase item is re-measurable from the committed code. (Legacy ~0.0064 vs port 0.071:
    a recorded over-correction for cycle 14, asserted by nothing at cycle 13.)
12. The legacy field-0.0064 comparison remains a cycle-14 measurement, not a cycle-13
    oracle: cycle 13 is oracle-free and explicitly does not claim legacy parity. The
    recorded port-vs-legacy gap is kept in PLAN/TODO as a chase item, not asserted.
13. Cycle 13's oracle column no longer says "the Perl" and property 4 asserts the
    staged-count rule end to end (cycle 10), not agreement with a Perl the test never
    runs.
14. Property 1 reads the field from the driver's `.imp` (`evaluate_saved_field`) and
    asserts it is positive and finite in the mask — the field itself, not a
    quotient the correction already forces into agreement.
15. Properties 3 and 4 check the driver's exit status and fail loudly (with cleanup)
    instead of discarding `rc` and misreading a missing file.
16. `cleanup()` uses `imp_of()` (final extension replaced) so the correct `.imp` is
    removed and no `.mnc.imp` accumulates.
17. The stale "within 1e-3 of constant" text is gone from the two comments and from
    the CMake block, whose copy-pasted `harness_red` header was dropped.
18. `test_driver_fwhm` cleanup removes the right `.imp`, calls `cleanup(same)` (the
    `_same.mnc` no longer leaks), and its `.imp` handling is shared wording.
19. `all_rms` is gone; the tests use `n3check::rel_rms`. The three "moves the
    correction" checks now require the rel RMS to exceed one output quantum,
    derived from the data (`(max−min)/valid_steps/mean` — `n3fixture::valid_steps`,
    not a literal `4095` — 5.400e-04 here), not a round 1e-4 that sat below the
    volume's own 12-bit step. Measured: fwhm 0.3 18.7×, dist 100 2.23×.
20. `round_trip_bound` in `test_estimate`/`test_evaluate` derives the quantum from
    `chunk_valid_range.txt` (`n3fixture::valid_steps`, `hi − lo`) instead of a literal
    `4095.0`, and `must()`s the masked minimum is positive so `log(0) → −inf` cannot
    poison the bound.
21. `imp.empty() ? NULL : &imp` is `&imp` (never empty) and `out_of_scope()` is `void`,
    invoked for its exit, not wrapped in an uncallable `if`.
22. `.gitignore` covers an in-source build's clutter: the un-anchored CMake `Makefile`/
    `CTestTestfile.cmake`/`CMakeFiles/`/`install_manifest.txt`/`CPack*`, the
    `config.h`/`version.h`/`epm-header` and launcher scripts, and the built executables
    by basename — none collides with a tracked file (`test_ref.log`, the `.cc` sources,
    the `*.in` templates remain tracked).

### Bookkeeping — fixed in this file and `PLAN.md`

23. All three counts read 36/36, verified by a fresh `ctest` run.
24. The fwhm test is described as added alongside (not replacing) the two smoke tests.
25. Cycle 15's dangling "assert items 1 and 2" is reworded to name the concrete rules to
    pin (`argv[0]` split, `-V0.9` order, validation, `.imp` clobber).
26. PLAN §7 row 13's oracle reads "none — properties…", its tolerance column describes
    the derived 0.25× tissue-contrast criterion, and PLAN row 14's bound is
    `valid_steps`-derived rather than a fixed `4095`.

## Found in review (2026-08-06)

Third pass, over `1ac5192` (the 26-item batch) and `b653c2e`, with the suite built and run.
Nothing below is fixed. Line numbers are `legacy/N3` HEAD's.

Confirmed first. `make` recompiled nothing, so the build tree is HEAD; **CTest 36/36 in
20.5 s**. Items 1-4, 6, 8, 10, 11, 14-16, 18, 20-22 hold in the sources and under probe:
`-V0.9` is order-independent (`-shrink 2 -V0.9` vs `-V0.9 -shrink 2` give `max|a−b| = 0.0`
and identical `.imp` coefficients), the `.imp` is covered by `-clobber`, the bimodal rule
follows `CreateMask`'s `if(!defined $user_mask_volume)` and passes `NULL` for the threshold
histogram, prior MINC history is preserved, and `.gitignore` collides with no tracked file
(`git ls-files | git check-ignore --stdin --no-index` is empty). The phantom is a real
ellipsoid and property 3 is not vacuous: **30.6% of the 132,056 mask voxels fall inside it**,
so both tissues are present in the estimation region.

### Driver defects, reachable from the command line

Items 1-4 fixed in `7016c41`, verified with `make && ctest`: 36/36, no new warnings. Each
of the four command lines below now dies cleanly instead of misbehaving; the guardrail
cases (`-parzen_sigma 2`, `-V0.9` alone, `-distance foo`) still run/die as before.

| # | Where | Status | What needed fixing |
|---|---|---|---|
| 1 | `nu_correct_cxx.cc:309` (pre-fix) | **fixed** | `-distance 0` segfaulted ("Corrected: index 2 into array of size 0 !", SIGSEGV). The Perl exits 1 with `Distance parameter must be positive` from `spline_smooth`. Item 5 ported `$distance < 0` (`nu_estimate_np_and_em.in:1463`) but the spline requires `> 0`. Fixed by dying on `A.distance <= 0`, before any spline is fit — a clean die where the Perl itself reaches the same message only after several iterations, by crashing inside `spline_smooth` |
| 2 | `nu_correct_cxx.cc:241-257` (pre-fix) | **fixed** | `-fwhm 0` and `-sharpen 0 0` wrote an all-zero volume and exited 0. The Perl dies at `:1449-1451`, `Invalid sharpen parameters`. Fixed: `A.fwhm > 0 && A.noise > 0` validated after parsing resolves either option |
| 3 | `nu_correct_cxx.cc:260` (pre-fix) | **fixed** | `-parzen_sigma <= 0` was accepted and ran. The Perl dies at `:1452-1453` only when the user gave the option (`0` means "off" internally otherwise). Fixed with a `user_parzen_sigma` flag, checked only when set |
| 4 | `nu_correct_cxx.cc:303-321` (pre-fix) | **fixed** | The `-iterations`/`-stop` count check ran before the `-V0.9` fill, so `-V0.9 -iterations 5` silently dropped the second stage's threshold (`Stopping.cc:31`'s `stage >= thresholds.size()` guard hid it). The Perl dies (`:1479-1481`) on the fully resolved arguments. Fixed by moving all validation below the fill |
| 5 | `nu_correct_cxx.cc:293-297` | open | `-version` prints `Program nu_correct_cxx` only. The Perl prints `Program nu_correct, built from:` then `N3 1.12.00`. `config.h` carries the version |
| 6 | `nu_correct_cxx.cc:398`, `Buffers.cc:301` | open | The appended history line has no `<date>>>>` prefix. Every legacy program writes one through `time_stamp()` (`brain_nu_ref.mnc.gz` shows it), so an output's history mixes timestamped inherited lines with one bare line. The `.imp` is unaffected — `fieldIO` stamps it itself |

Also closed alongside 1-4, from PLAN §9's "no new `atof(argv[++i])`": all nine directly-typed
numeric options (`-distance`, `-lambda`, `-subsample`/`-spline_subsample`, `-fwhm`,
`-parzen_sigma`, `-bins`, `-shrink`, `-background`, `-floor`) now go through
`parse_double`/`parse_int`, which check `strtod`/`strtol`'s end pointer and die naming the
option and the offending text — `-distance foo` now dies `expected a number, got 'foo'`
instead of becoming `0.0`. `-iterations`/`-stop`'s own value loops were already guarded by
`is_number` and are unchanged.

### Recorded measurements that are wrong

All four fixed, in the source comments and in this file; `PLAN.md` §7 row 13 and §4 were
already corrected in the prior session. Cycle 14 (still open) is where the legacy 0.035
re-measurement becomes a recorded oracle rather than a comment.

| # | Where | Status | What needed fixing |
|---|---|---|---|
| 7 | `test_driver_properties.cc:208-210` | **fixed** | The legacy field CV was recorded as 0.0064, from before item 8 replaced the striped phantom with an ellipsoid. Regenerating exactly the shipped phantom (a standalone copy of the generator reproduces the test's 0.0708) and running the installed `nu_correct` with the test's own options — `-shrink 2 -iterations 15 -stop 0.0 -distance 100 -mask chunk_mask` — gives legacy mean 1.0640, **RMS CV 0.0350** against the port's mean 1.0778, RMS CV 0.0708: a 2.0× gap, not 11×, with near-identical mean gain. The source comment, `PLAN.md` §7 row 13, and this file all now carry 0.035 |
| 8 | `test_driver_properties.cc:208-210` | **fixed** | The same comment also said "Measured field CV is ~0.05 here" (the test prints 0.0708) and "the 0.3 derived bound" (the bound is `0.25 × 0.857 = 0.214`). Item 17 claimed the stale text was gone; this instance had survived. Rewritten together with item 7 |
| 9 | `test_driver_fwhm.cc:100-106`, this file (was `:142-143`) | **fixed** | The quantum was recorded as "~3.9e-4"; the test prints **5.400e-04**. The movements were recorded as "fwhm 0.3 ~10×, dist 100 ~3×"; measured `1.008e-2 / 5.400e-4 = 18.7×` and `1.206e-3 / 5.400e-4 = 2.23×`. Both the source comment and this file's item 19 corrected together. Item 10 below (the literal `4095.0` this quantum was computed from) was fixed in the same pass |

### Test-suite defects

| # | Where | Status | What needs fixing |
|---|---|---|---|
| 10 | `test_driver_fwhm.cc:109` | **fixed** | Item 19's fix reintroduced the literal `4095.0` — `(bs.maximum - bs.minimum) / 4095.0 / bs.mean` — in the same commit in which item 20 removed it from `test_estimate.cc` and `test_evaluate.cc`. Now `n3fixture::valid_steps("chunk_valid_range.txt")`. The quantum is still taken over the *in-mask* range where the two round-trip bounds take it over the volume's recorded `valid_range`; that distinction is real (the fwhm test measures a movement relative to what actually varies, not to the file's nominal range) and is not a defect |
| 11 | `test_driver_properties.cc:211-227` | **documented, not bounded** | Property 3's criterion is a CV about the mean and is therefore blind to a uniform gain of any size. On a volume with no planted bias the ideal field is 1.0; the measured means are 1.078 (port) and 1.064 (legacy) — a larger departure than the 0.0708 the bound does constrain. This is an algorithmic question (what cycle 13 should assert), and no bound derived now would be anything but fitted to this measurement, which the tolerance rule excludes. The comment now states the gap explicitly, including the Jensen's-inequality component it is not explained by (~0.0025 of the observed 6-8%), as a distinct chase item alongside the CV one — no numeric bound added |
| 12 | `test_driver_properties.cc:87,104` | **fixed** | Unused `vi` and `vo`, left over from the removed `lo`/`hi` computation, removed; confirmed silent under `-Wall -Wextra` |
| 13 | `test_driver_properties.cc:96-101` | **fixed** | The first `system()` failure path returns without `cleanup(corr_path)` or the `delete_volume`s, leaving the output and its `.imp` in `$TMPDIR`; now matches the other two failure paths |

### Bookkeeping

| # | Where | Status | What needs fixing |
|---|---|---|---|
| 14 | `1ac5192`'s message, this file `:80` | not fixable | Items **12 and 17 are absent** from the commit message's list, and item 12 is a deferral to cycle 14 rather than a fix, so "all are now fixed" overstates the batch. `1ac5192` is an already-existing commit; per the git-safety rule against amending published commits, this is recorded here rather than rewritten |
| 15 | this file `:98-99` | **fixed** | "the corrected volume is byte-identical for `-shrink 2 -V0.9` and `-V0.9 -shrink 2`" stopped being true when item 6 put the invocation into the history: the files differ at byte 55. Corrected in place to say the voxel data (`max|a−b| = 0.0`) and `.imp` coefficients are identical |

### Style and robustness

Measured over `1ac5192`. None of these changes a corrected voxel; they are ordered last for
that reason, with the exception noted in item 16.

| # | Where | What needs fixing |
|---|---|---|
| 16 | `nu_correct_cxx.cc:226-301` | The driver hand-rolls argument parsing where five sibling programs use `ParseArgv`: `src/{SharpenHist,SplineSmooth,VolumeStats,VolumeHist,EvaluateField}/*Args.{cc,h}` each declare an `ArgvInfo` table. The chain is 31 `else if(tok == …)` branches, 11 copy-pasted `if(i+1>=argc)` guards and 15 bare `atof`/`atoi` calls. **This is the root cause of items 1-4**: `atoi`/`atof` have no error channel, so `-distance foo` becomes `0.0` and `-fwhm foo` becomes `0`. `TODO.md:34` calls the file "main + ParseArgv", which it is not |
| 17 | `nu_correct_cxx.cc:82`, `:94`, `:325`, `:197` | Four spellings of the program's own name, all reachable from `nu_estimate_cxx`: `die()` hardcodes `"nu_correct_cxx: "` (12 call sites), `usage()` hardcodes it, `:325` uses raw `argv[0]` (the full path), and only `out_of_scope()` uses `program_name`. Item 7's fix threaded the name through two of the four |
| 18 | `nu_correct_cxx.cc:177-180` | `struct Parsed { Arguments args; };` is declared and never used — dead code item 21's sweep missed. Eleven lines exceed 100 columns, all in the parse chain, against ~80 elsewhere in `src/N3Pipeline/` |
| 19 | `nu_correct_cxx.cc:262`, `NuEstimate.h` | `A.blur` / `EstimateOptions::blur` is set by `-nodeblur` and means *skip* the deconvolution, so `if(options.blur)` reads as the opposite of what it does. The Perl calls it `$nodeblur_flag` |
| 20 | 17 sites in `src/N3Pipeline/` | The library `exit(1)`s from inside `namespace n3` (`Buffers.cc:12`, `FitField.cc` ×7, `MincTools.cc` ×3, `Sharpen.cc:25`, `SmoothField.cc:149`, `NuEstimate.cc:86`, `Stopping.cc:13`). All of it links into 13 test executables, so no failure path can be asserted without forking — the empty-composite-mask rejection (`NuEstimate.cc:83-87`) is untestable by construction. The legacy programs each *are* a process; this is a library |
| 21 | `test_driver_properties.cc:51-57`, `test_driver_fwhm.cc:37-53` | `snprintf` truncation is never detected: four fixed buffers (`cmd[1600]`, `p[256]`, `path[256]`, `cmd[1024]`), return value discarded at every call. A long `$TMPDIR` truncates the path, and the test then compares the wrong file or reports a driver failure that did not occur |
| 22 | `nu_correct_cxx.cc:132`, `test_driver_properties.cc:64`, `test_driver_fwhm.cc:61` | The `.imp` path rule (`s/\.[^\.]*$/\.imp/`) has three independent `find_last_of('.')` implementations. Items 16 and 18 fixed the same bug in two of them in one commit. A future change to `imp_path` desynchronises the tests silently, and they keep passing while deleting files the driver no longer writes |
| 23 | `test_driver_properties.cc`, `test_driver_fwhm.cc` | The two driver tests bypass `fixture.h`, which the other 11 tests include, and each carry their own `outdir()`, path builder and cleanup. `cleanup()` is `unlink()` in one file and `system("rm -f …")` in the other. `run_driver()`, `outdir()` and `imp_of()` belong beside `valid_steps()` |
| 24 | `nu_correct_cxx.cc:389` | `e.bimodal_bins = (int) ceil(hi - lo + 1)` is unbounded. Faithful to `volumeStats.cc:286`, but the port accepts any file: a wide stored `valid_range` sizes the histogram to that many bins with no cap and no diagnostic |
| 25 | `nu_correct_cxx.cc:347-429` | Every `VIO_Volume` is raw with a hand-placed `delete_volume` and no RAII, so each new early return inherits four cleanup obligations; `test_driver_properties.cc:96-101` already skips its four. `-help` also prints to stderr while `-version` prints to stdout, both returning 0 |

Not a defect, and the model the rest should follow: `Buffers.cc:63-107` verifies the
contiguity and no-voxel-scaling assumptions every bulk loop rests on instead of trusting
them, and fails loudly. The per-decision citations to the Perl line (`:1557`, `:499-505`,
`:316-325`, `fieldIO.cc:120-133`) are what make these reviews checkable and must not be
relaxed.

### Order of work

**Algorithm first.** A change that alters a corrected voxel, a field, or a recorded
measurement outranks one that alters a message, a name or a file layout, and the two must
not share a commit. PLAN §9 states the conventions each step is held to.

1. **Driver defects 1-4 — fixed, `7016c41`.** Each was a wrong answer or a crash on a legal
   command line; `-fwhm 0` writing an all-zero volume at exit 0 was the worst. Closed with
   one `parse_double`/`parse_int` helper (checks `strtod`/`strtol`'s end pointer, used at all
   nine directly-typed numeric options) plus the two Perl-side validations items 2-3 needed
   (`nu_estimate_np_and_em.in:1449-1453`, `:1452-1453`) and moving all validation, including
   the `-iterations`/`-stop` size check, below the `-V0.9` fill. `make && ctest`: 36/36, no
   new warnings. The `ParseArgv` rewrite (item 16) is separate and still open.
2. **Measurement corrections 7-9, and test-suite defects 10, 12, 13 — fixed, `606775e`.**
   7-8 rewrote the stale comment in `test_driver_properties.cc` (legacy CV 0.035, not 0.0064;
   bound 0.214, not "0.3"); 9-10 rewrote `test_driver_fwhm.cc`'s comment and its quantum
   computation (`n3fixture::valid_steps`, not a literal `4095.0`, which needed `fixture.h`
   and `N3_REFERENCE_DIR` added to that CMake target); 12 removed the two unused variables;
   13 fixed the leaking failure path. `PLAN.md` §7 row 13 and §4 were already corrected in
   the prior session. `make && ctest`: 36/36, no new warnings, confirmed under `-Wall -Wextra`
   on both changed test files.
3. **Item 11 — documented, not bounded.** Property 3's criterion is a CV about the mean and
   does not constrain the 6-8% uniform gain both implementations show on a bias-free volume.
   No bound is added: deriving one now from the only measurement in hand would be exactly
   the fitted-tolerance mistake the tolerance rule forbids. The property's comment states the
   gap and rules out Jensen's inequality as more than a tenth of it, as a chase item.
   Bookkeeping items 14-15 also closed: 14 is a commit-message inaccuracy in an
   already-published commit, recorded rather than fixed by amending it; 15's stale claim in
   this file is corrected in place, above.
4. **Cycle 14 — fixed, `99fc4a7`** (`test_driver_endtoend`). Turns the hand-measured
   1.8e-4 into the one end-to-end assertion whose bound is justified in advance. Both
   `-legacy_rounding` configurations measure 1.840e-04 against the recorded oracle
   (`nu_correct_shrink1.f64`), under the derived 2.683e-4 quantum bound — a full order of
   margin, no fix needed (the red was the missing assertion, not a defect). It is now the
   only test that reads the corrected volume the driver's `n3::save` writes. The legacy
   field-CV re-measurement (0.035, item 7) is a separate reported-not-asserted chase item,
   not an oracle here.
5. **Cycle 15, then 17, then 16** (still open) — 15 and 17 have their inputs recorded
   already; 16 needs the `PATH` arrangement of PLAN §7 row 16. Cycle 15 is where the
   driver-defect fixes in step 1 become assertions rather than a hand-run probe.
6. **The two reported-not-asserted tables** (`brain.mnc` at the default protocol, iteration
   counts), which PLAN §7 places after the cycles. Still open.
7. **Style and robustness 16-25, and driver defects 5-6** (still open). Take 16 (the
   `ParseArgv` table) together with cycle 15, so the table and the test that pins it land
   against each other; 20 (`exit()` in library code) is its own commit and must not ride
   along with a behavioural change.

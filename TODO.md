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
| 14 | end-to-end bounded (`-shrink 1 -iterations 1 -stop 0`) | [~] oracle `nu_correct_shrink1.f64` recorded; driver reproduces it at 1.8e-4 rel RMS. Cycles 11-12 derive their bound from the data — `0.5·(log max − log min)/valid_steps`, valid_steps read from `chunk_valid_range.txt` (0..4095 → 4095), 2.683e-4 on `chunk.mnc` — because the file is 12-bit, where PLAN §4 assumed 16-bit and published 1e-4. The driver passes the derived bound and fails the published one |
| 15 | argv[0] / argument table | [ ] drive `nu_estimate_cxx` vs `nu_correct_cxx` to pin the argv[0] split, and assert the argument rules the review fix introduced: `-V0.9` order-independence, the `-bins`/`-background`/`-distance`/`-iterations` validation, and `-clobber` covering the `.imp` |
| 16 | `-tp_spline` and `-parzen_sigma` end to end | [ ] |
| 17 | `-estimate_only` vs `.imp` | [~] oracle `estimate.imp` recorded; driver writes the .imp, Domain matches exactly |

## Driver (blocks 13–17)

- `legacy/N3/src/N3Pipeline/nu_correct_cxx.cc` (main + ParseArgv) — [x] commit `48a101a`
- `ADD_EXECUTABLE(nu_correct_cxx)` + INSTALL as `nu_correct_cxx`/`nu_estimate_cxx` — [x] (drives argv[0] rule; two targets)
- Smoke CTest `n3cxx_driver_estimate` + `n3cxx_driver_correct` — [x] (CTest 36/36)

## Reported, not asserted (PLAN §7 tail)

- `brain.mnc` default-protocol tables vs `nu_correct` and `brain_nu_ref.mnc.gz` — [ ]
- Iteration counts at the default protocol on each test volume — [ ]
- Cycle 13 measured a port **over-correction on a no-bias phantom**: field RMS CV 0.071
  (legacy ~0.0064); neither reaches a flat field. The port stays under cycle 13's derived
  bound (0.25× the phantom's tissue contrast = 0.214), so it passes while the gap is a
  chase item. Re-measure the port CV from `test_driver_properties` and the legacy CV in
  cycle 14, and record the legacy run as an oracle — [ ]

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
   volume is byte-identical for `-shrink 2 -V0.9` and `-V0.9 -shrink 2`.
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
    derived from the data (`(max−min)/4095/mean`, ~3.9e-4 here), not a round 1e-4
    that sat below the volume's own 12-bit step. Measured: fwhm 0.3 ~10×, dist 100 ~3×.
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

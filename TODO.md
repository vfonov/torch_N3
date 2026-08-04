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
| 13 | end-to-end properties | [ ] |
| 14 | end-to-end bounded (`-shrink 1 -iterations 1 -stop 0`) | [~] oracle `nu_correct_shrink1.f64` recorded; driver reproduces it at 1.8e-4 rel RMS. Cycles 11-12 derive their bound from the data — `0.5·(log max − log min)/4095`, 2.683e-4 on `chunk.mnc` (`test_evaluate.cc:24-28`) — because the file is 12-bit, where PLAN §4 assumed 16-bit and published 1e-4. The driver passes the derived bound and fails the published one; see item 5 below |
| 15 | argv[0] / argument table | [ ] must also assert items 1 and 2 below |
| 16 | `-tp_spline` and `-parzen_sigma` end to end | [ ] |
| 17 | `-estimate_only` vs `.imp` | [~] oracle `estimate.imp` recorded; driver writes the .imp, Domain matches exactly |

## Driver (blocks 13–17)

- `legacy/N3/src/N3Pipeline/nu_correct_cxx.cc` (main + ParseArgv) — [x] commit `48a101a`
- `ADD_EXECUTABLE(nu_correct_cxx)` + INSTALL as `nu_correct_cxx`/`nu_estimate_cxx` — [x] (drives argv[0] rule; two targets)
- Smoke CTest `n3cxx_driver_estimate` + `n3cxx_driver_correct` — [x] (36/36 ctest green)

## Reported, not asserted (PLAN §7 tail)

- `brain.mnc` default-protocol tables vs `nu_correct` and `brain_nu_ref.mnc.gz` — [ ]
- Iteration counts at the default protocol on each test volume — [ ]

## Fixed since the review

- `-fwhm` aliased `-distance` (`nu_correct_cxx.cc:219`), corrected in `40f94d1`. `-fwhm` now
  sets the deconvolution width and `-distance` alone sets the knots, which is what
  `nu_estimate.in:205` maps it to and what `:498` passes on as `-sharpen <width> 0.01`.
  Item 1 above adds an asserting test so it cannot silently regress.

## Found in review (2026-08-04)

All eight items below are fixed; each has its own commit (plus the two PLAN edits are
separate commits per AGENTS.md). CTest is 35/35; a fresh out-of-source configure with the
LIBMINC/EBTKS dirs succeeds after the in-source artifacts were removed.

| # | Resolved in | Item |
|---|---|---|
| 1 | `e6b46b3` | The two `40f94d1` exit-status-only smoke tests replaced by `test_driver_fwhm`, which drives the built binary and compares outputs: `-fwhm 0.15` byte-identical to the default, `-fwhm 0.3` and `-distance 100` each move the correction through their own, different channels. A re-aliasing of `-fwhm` fails an assertion now, not by exhausting memory |
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

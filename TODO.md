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

## Found in review (2026-08-04), not yet fixed

| # | Where | Item |
|---|---|---|
| 1 | `testing/CMakeLists.txt:142-157` | The two tests added with `40f94d1` assert only the exit status. Their comment states that changing the sharpening width must move the correction and that `-distance` must move it through the other channel, but neither output is compared to anything: a re-aliasing of `-fwhm` would fail these only by exhausting memory on a 0.3 mm knot spacing, not by an assertion. Cycle 15 should compare outputs — `-fwhm 0.15` byte-identical to the default, `-fwhm 0.3` and `-distance 100` each differing from it — which is the verification `40f94d1`'s message reports having done by hand |
| 2 | `/app/legacy/_install/bin/nu_correct_cxx` | Stale: 08:11, predating the fix, against the build tree's 13:21. The installed copy still aliases `-fwhm` to `-distance`, and `regenerate_reference.sh` resolves `N3_LOCAL_BIN` there. Re-run `make install` |
| 3 | `nu_correct_cxx.cc:318` | `-auto_mask`'s Talairach branch missing. PLAN §3 requires the `model_data/N3` average brain mask, label-resampled, on a Talairach-tagged volume; `-auto_mask` collapses to the bimodal threshold. Implement, or record the restriction in PLAN's scope list |
| 4 | `nu_correct_cxx.cc:341`, comment `:336` | A correct run writes no `.imp` unless `-mapping_dir` is given, while the comment says every correct run does. The Perl always writes one (`nu_estimate.in:57-58`, `replace_ext`, relocated by `replace_dir` only when `-mapping_dir` is given) |
| 5 | `PLAN.md` §4 and §7 cycle 14 | The published 1e-4 end-to-end bound assumes 16-bit intermediates; `chunk.mnc`'s `valid_range` is 0..4095, so cycles 11-12 derive 2.683e-4 from the data and measure 2.30e-4 and 1.83e-4 against it. The derived bound is a physical quantity, not a fitted one, and is the correct one. Correct PLAN in its own commit with the derivation |
| 6 | `nu_correct_cxx.cc:136-139` | `imp_path`'s `.mnc.gz`/`.mnc.Z` branches are dead: `find_last_of('.')` lands on `.gz`, so `out.mnc.gz` yields `out.mnc.gz.imp` |
| 7 | `legacy/N3`, commit `e01d441` | 17 in-source CMake artifacts committed into the tree: `CMakeCache.txt` (holding `LIBMINC_DIR-NOTFOUND`), `CMakeFiles/` including two `a.out`, `CPackConfig.cmake`, `CPackSourceConfig.cmake`, `DartConfiguration.tcl`, `Testing/`. Verified not to block a fresh out-of-source configure. `git rm --cached` them and add a `.gitignore` |
| 8 | `NuEstimate.cc:48-50`; `nu_correct_cxx.cc:275` | Dead fragments, no effect on any measurement. A ternary whose two arms are the same `resample_label(user_mask, work)` call, correct because that call is the identity at `shrink == 1` (asserted by `test_resample_label`); and `if(!A.sharpen && A.parzen_sigma > 0)`, unreachable since `A.sharpen` is initialised true, only ever set true, and no `-nosharpen` spelling exists — omitting `-sharpen` is what selects the Perl's EM branch, which PLAN excludes |

Verified alongside the gaps: CTest 36/36 including the 18 pre-existing legacy cases;
`legacy/EBTKS` carries no changes of its own; no pre-existing `legacy/N3` source or Perl driver
modified by the C++ task; `torch_n3/_legacy/n3/` still byte-identical to `legacy/N3/src`.

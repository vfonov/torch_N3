# TODO: `nu_correct` in C++, entirely in memory

Open work for `/app/PLAN.md` (authoritative). The implementation lives in the nested git
repository `legacy/N3/`, on branch `aislop` (per-cycle commits); `/app/TODO.md` and
`/app/PLAN.md` live in the outer repository.

**This file carries only what still needs doing.** Cycles 0-14 are closed, and the four
review passes of 2026-08-04/05/06/07 are in `/app/REVIEWS.md`, which is where the source
comments' `(review, 2026-08-06, item 11)`-style citations resolve. Status: `[ ]` pending,
`[~]` code written but not wired or tested.

State at 2026-08-07: `make && ctest` in `/app/legacy/_build/n3` is **37/37**.

## Red-green cycles (PLAN §7)

| # | Test | Status |
|---|---|---|
| 15 | argv[0] / argument table | [ ] drive `nu_estimate_cxx` vs `nu_correct_cxx` to pin the argv[0] split, and assert the argument rules: `-V0.9` order-independence, the `-bins`/`-background`/`-distance`/`-iterations` validation, `-clobber` covering the `.imp`, every out-of-scope option exiting non-zero, the four degenerate-argument cases of `7016c41` (`-distance 0`, `-fwhm 0`, `-parzen_sigma <= 0`, `-V0.9 -iterations 5`), and **the protocol-version resolution** against PLAN §7's "Protocol versions" table — a bare invocation gives `-V1.1`'s four values, `-V1.0`/`-V0.9` override them, each fills only what the user did not give, and the last `-V` wins |
| 16 | `-tp_spline` and `-parzen_sigma` end to end | [ ] the `-parzen_sigma` oracle is `/app/legacy/_install/bin/nu_correct` with `/app/legacy/_install/bin` first on `PATH` (PLAN §7 row 16): `MNI::Spawn` resolves `volume_hist` through `PATH` and the stock one has no `-gaussian_window` |
| 17 | `-estimate_only` vs `.imp` | [~] oracle `estimate.imp` recorded; the driver writes the `.imp` and its Domain matches exactly. `estimate.imp` is read by `test_evaluate.cc:62,75` as an *input*; no test evaluates the driver's own `.imp` and the Perl's and diffs the two fields, which is what the cycle specifies |

**Cycle 15 also closes the default-protocol gap.** `-V1.1` is what a bare invocation selects
(`9154eac`), and no test asserts a value produced under it: every value-asserting driver test
pins `-V1.0`, and the two that run the default (`n3cxx_driver_estimate`,
`n3cxx_driver_correct`) assert exit status only, with `-iterations 1 -stop 0.0` overriding two
of its four parameters. The default also selects `-parzen_sigma 4.0`, whose end-to-end oracle
is cycle 16, so until both close the shipped protocol rests on the one path with no oracle
comparison behind it.

## Reported, not asserted (PLAN §7 tail)

- `brain.mnc` default-protocol tables vs `nu_correct` and `brain_nu_ref.mnc.gz` — [ ]. "The
  default protocol" means the Perl's, which is `nu_correct_cxx -V1.0`.
- Iteration counts at the default protocol on each test volume — [ ]
- **The port over-corrects on a no-bias phantom** — [ ]. Cycle 13 measures field RMS CV 0.071
  against the legacy's 0.035; neither reaches a flat field. The port stays under cycle 13's
  derived bound (0.25× the phantom's tissue contrast = 0.214), so it passes while the gap is a
  chase item. Not an oracle in any cycle: cycle 14 closed against `nu_correct_shrink1.f64`
  alone.
- **The same property does not constrain a uniform gain** — [ ]. Its criterion is a CV about
  the field's own mean, and on a bias-free volume the ideal field is 1.0 where the measured
  means are 1.078 (port) and 1.064 (legacy) — a larger departure than the dispersion the bound
  does constrain. Jensen's inequality accounts for about 0.0025 of the observed 6-8%. No bound
  is asserted: one derived from the only measurement in hand would be fitted, which the
  tolerance rule excludes. This is an algorithmic question about what cycle 13 should assert.

## Driver defects, reachable from the command line

| # | Where | What needs fixing |
|---|---|---|
| 5 | `nu_correct_cxx.cc:344` | `-version` prints `Program nu_correct_cxx` only. The Perl prints `Program nu_correct, built from:` then `N3 1.12.00`. `config.h` carries the version |
| 6 | `nu_correct_cxx.cc:488`, `Buffers.cc:296-303` | The appended history line has no `<date>>>>` prefix. Every legacy program writes one through `time_stamp()` (`brain_nu_ref.mnc.gz` shows it), so an output's history mixes timestamped inherited lines with one bare line. The `.imp` is unaffected — `fieldIO` stamps it itself |

## Style and robustness

None of these changes a corrected voxel; they are ordered last for that reason, with the
exception noted in item 16.

| # | Where | What needs fixing |
|---|---|---|
| 16 | `nu_correct_cxx.cc:272-346` | The driver hand-rolls argument parsing where five sibling programs use `ParseArgv`: `src/{SharpenHist,SplineSmooth,VolumeStats,VolumeHist,EvaluateField}/*Args.{cc,h}` each declare an `ArgvInfo` table. The chain is 31 `else if(tok == …)` branches and 11 copy-pasted `if(i+1>=argc)` guards. **This was the root cause of the four `7016c41` defects**, and while `parse_double`/`parse_int` closed the error channel, the shape that produced them is unchanged |
| 17 | `nu_correct_cxx.cc:91`, `:103`, `:279`, `:414` | Four spellings of the program's own name, all reachable from `nu_estimate_cxx`: `die()` hardcodes `"nu_correct_cxx: "` at `:91` (12 call sites), `usage()` hardcodes it at `:103`, `:414` uses raw `argv[0]` (the full path), and only `out_of_scope()` (`:279`) uses `program_name` |
| 18 | `nu_correct_cxx.cc:194` | `struct Parsed { Arguments args; };` is declared and never used. Eleven lines exceed 100 columns, all in the parse chain, against ~80 elsewhere in `src/N3Pipeline/` |
| 19 | `nu_correct_cxx.cc:308`, `:457`, `NuEstimate.h` | `A.blur` / `EstimateOptions::blur` is set by `-nodeblur` and means *skip* the deconvolution, so `if(options.blur)` reads as the opposite of what it does. The Perl calls it `$nodeblur_flag` |
| 20 | 17 sites in `src/N3Pipeline/` | The library `exit(1)`s from inside `namespace n3` (`Buffers.cc:12`, `FitField.cc` ×7, `MincTools.cc` ×3, `Sharpen.cc:25`, `SmoothField.cc:149`, `NuEstimate.cc:86`, `Stopping.cc:13`). All of it links into 16 test executables, so no failure path can be asserted without forking — the empty-composite-mask rejection (`NuEstimate.cc:83-87`) is untestable by construction. The legacy programs each *are* a process; this is a library |
| 24 | `nu_correct_cxx.cc:479` | `e.bimodal_bins = (int) ceil(hi - lo + 1)` is unbounded. Faithful to `volumeStats.cc:286`, but the port accepts any file: a wide stored `valid_range` sizes the histogram to that many bins with no cap and no diagnostic |
| 25 | `nu_correct_cxx.cc:437-519` | Every `VIO_Volume` is raw with a hand-placed `delete_volume` and no RAII, so each new early return inherits four cleanup obligations. `-help` also prints to stderr while `-version` prints to stdout, both returning 0 |

Numbering follows the 2026-08-06 pass, so it matches `REVIEWS.md` and the source comments; the
line numbers are `legacy/N3` HEAD's (`4537742`), re-checked 2026-08-07, not that pass's.
Items 21-23 of that pass (discarded `snprintf` returns, the triplicated `.imp` path rule, the
driver tests bypassing `fixture.h`) closed in `7967ffb`.

Not a defect, and the model the rest should follow: `Buffers.cc:63-107` verifies the
contiguity and no-voxel-scaling assumptions every bulk loop rests on instead of trusting them,
and fails loudly. The per-decision citations to the Perl line (`:1557`, `:499-505`, `:316-325`,
`fieldIO.cc:120-133`) are what make these reviews checkable and must not be relaxed.

## External BLAS/LAPACK for `legacy/EBTKS`/`legacy/N3`'s own build (PLAN §10)

Separate from the `nu_correct_cxx` work above and orthogonal to it — a build-system item, not
a change to a corrected voxel. `/app/PLAN.md` §10 is the design, already reframed to the
four-value switch below (its own task 1 is done, in effect: the retractions and the
"Default posture — resolved" section it called for are both present at `PLAN.md:776`,
`:864`, `:933`). Closed: the four-value `EBTKS_BLAS_BACKEND` switch (`bundled` default,
`lapack`, `lapacke`, `cblas`) in `legacy/EBTKS/CMakeLists.txt`, configure-time validation, no
autodetection, `bundled` verified bit-identical to the pre-change archive, and propagation to
`legacy/N3/CMakeLists.txt` on both the standalone (`FIND_PACKAGE(EBTKS)`) and superbuild
(`MINC_TOOLKIT_BUILD`) paths — except the one gap below. All three non-`bundled` backends
built, linked and measured against `bundled` on this machine: `lapack`/`cblas` (`openblas`)
one-iteration `spline_smooth` relative RMS 9.77e-08, `cblas` (`gslcblas`) exactly 0 (both
reference triple-loop implementations retracing the same summation order — explained, not
just observed), `ctest` 37/37 under each (2026-08-07). `lapacke` followed on 2026-08-11 once
`liblapacke-dev` was installed (user action; a standalone Debian package, unrelated to
OpenBLAS's own `NO_LAPACKE` flag — the original assumption that only a superbuild-built
OpenBLAS could supply it was wrong): the only backend that reaches real Cholesky
(`N3_HAVE_LAPACK_CHOLESKY`, `TBSpline.cc:101-105`) in `nu_correct_cxx`/`nu_estimate_cxx`'s
equilibration+Cholesky solve (`7d84753`), built in an isolated tree under `scratchpad/`, not
`/app/legacy/_install`; no `dsysv` fallback through 30 iterations on `chunk.mnc`, field
agreement with the `bundled` build 1.83e-09 to 4.32e-09 relative RMS across 1-20 iterations,
`spline_smooth`'s own gate (which cannot reach the modern solve) the same 9.77e-08 as
`lapack`/`cblas`, `ctest` 37/38.

Open:

- [ ] **`EBTKS_BLAS_BACKEND` does not propagate to a standalone N3 configure**, found building
  `lapacke`. `EBTKSConfig.cmake` exports `EBTKS_LAPACK_LIBRARIES`/`_LIBRARY_DIRS` but not
  `EBTKS_BLAS_BACKEND` itself, so `N3/CMakeLists.txt:77`'s `EBTKS_DSYSV_LAPACKE_SHIM` switch
  (needed for `N3_HAVE_LAPACK_CHOLESKY`) never fires under `FIND_PACKAGE(EBTKS)` unless
  `EBTKS_BLAS_BACKEND=lapacke` is *also* passed to N3's own `cmake`, redundantly with EBTKS's
  — silent, not a build failure. Unresolved in `legacy/N3/CMakeLists.txt`. (A related but
  non-code pitfall from the same build, not tracked as an item: `-L/opt/minc/.../lib`
  precedes an isolated EBTKS install in the link search order and shadows it, same as the
  `lapack` backend's own gotcha — worked around with `CMAKE_EXE_LINKER_FLAGS`, nothing to fix
  in the repository.)
- [ ] `nu_reference_1` observation under a non-`bundled` backend with `PATH` pointed at the
  build tree — not run, any backend. It shells out to the Perl driver via `$ENV{PATH}`
  regardless of backend, so `ctest`'s own 37/37 (or 37/38) never exercises this; the one
  `lapacke` `ctest` failure is this same `PATH` shadow, not a regression.
- [ ] `ExternalProject_Add` argument lists for a real superbuild — not written; no superbuild
  driver exists in this tree yet to write them against.
- `/app/legacy/_install` is unaffected throughout: default stays `bundled`, confirmed
  byte-identical before and after this work.

## Standing risk

**The three round-trip comparisons share one bound and are tight.** Cycles 11, 12 and 14 sit
at 0.86, 0.68 and 0.69 of `0.5·(log max − log min)/valid_steps` = 2.683e-4 on `chunk.mnc`, so
cycle 11's 2.299e-4 has 14% headroom and a change to the histogram or the spline can put it
red. The remedy when that happens is PLAN §9's: find the defect, or the confound. Not a wider
bound.

## Order of work

**Algorithm first.** A change that alters a corrected voxel, a field, or a recorded
measurement outranks one that alters a message, a name or a file layout, and the two must not
share a commit. PLAN §9 states the conventions each step is held to.

1. **Cycle 15, then 17, then 16.** 15 and 17 have their inputs recorded already; 16 needs the
   `PATH` arrangement of PLAN §7 row 16. Cycle 15 is where the `7016c41` driver-defect fixes
   and the protocol-version resolution become assertions rather than hand-run probes. Take
   style item 16 (the `ParseArgv` table) together with it, so the table and the test that pins
   it land against each other.
2. **The reported-not-asserted tables** (`brain.mnc` at the default protocol, iteration
   counts), which PLAN §7 places after the cycles.
3. **The two phantom chase items** — the 2.0× over-correction and the unconstrained uniform
   gain. Both are algorithmic and belong before the remaining ergonomics.
4. **Driver defects 5-6, then style items 17-20, 24, 25.** Item 20 (`exit()` in library code)
   is its own commit and must not ride along with a behavioural change.

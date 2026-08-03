# Known problems

Weaknesses in this repository's test suite, recorded so that they are visible
rather than discovered. Audited 2026-07-31; §2, §5, §7 and the worst entries of
§1 and §3 were fixed the same day. They are retained here rather than deleted,
since a bound that had ceased to constrain anything is worth recording even
after it constrains again.

Every number below comes from `python3 -m tests.margins`, which prints the table
at the bottom. Re-run it after changing a bound or a block.

The rule these are measured against is in [CLAUDE.md](CLAUDE.md#test-tolerances):
a threshold states what the code is *required* to do, and is not the measured
difference plus a margin. Several entries here fall short of that. None is a
failing test; they are cases in which a passing test establishes less than it
appears to.

---

## 1. Thresholds chosen to fit the measurement

These were set with knowledge of what the code already produced. They pass, but
they record behaviour rather than state a requirement, and would not detect a
regression that remained within the envelope they were drawn around.

| Where | Assertion | Now at |
|---|---|---|
| `test_pipeline.py` | `early < 1e-6`, `late > 100 * early` | 14%, 1800× |
| `test_histogram.py`, `test_sharpen.py`, `test_spline.py`, `test_field.py` | the `atol` constants | 0–50% |

**Two entries removed 2026-07-31.** `residual < non_uniformity(planted) / 4` and
`after < before / 3` were both fitted at the default 200 mm knot spacing, and
sweeping `-distance` established that neither holds elsewhere: `/4` fails at
100 mm and finer, `/3` fails at 100 mm and finer, and at 50 mm the correction
leaves the volume further from the truth than the planted field did
(`after/before` = 1.10 at 20%). Neither bound was widened. `residual` is now
held to `planted / 2`, stated as a minimum requirement for usefulness rather
than as a description of the measurement: a corrector that leaves more than half
a known field behind has not performed its function. `after < before` is a bare
ordering, asserted only at the shipped 200 mm, where it holds. The 50 mm
behaviour is asserted as its own ordering rather than averaged over.

The block `atol` values are round numbers a decade or two above what was
measured, which is defensible, but none is derived from a physical quantity.
Where a principled bound exists it is already used and is to be preferred: the
16-bit quantum of the file a legacy program wrote (`span(reference) / 65535`),
the 12-bit one (`/ 4095`), the six decimals `%lf` prints (`1e-6`).

**The denoising filter's intensity scale was set by `max()` — fixed
2026-08-02.** `blocks/denoise.py`'s noise floor is absolute (one intensity unit
in 256), so something must bring it onto the volume's own scale. The source code
uses `values.max()`, which was carried over unchanged at first. It is wrong for
two reasons.

*Not robust.* A maximum over 902,629 voxels is decided by one of them, and an
outlier can only raise it: a raised maximum raises the floor, and a raised floor
*excludes* voxels from filtering. Planting a single bright voxel in `brain.mnc`
and changing nothing else:

| planted voxel | floor | voxels filtered | volume moved (rel RMS) |
|---|---|---|---|
| none | 4214 | 92.7% | 1.14e-1 |
| 2× the maximum | 8428 | 64.3% | — |
| 10× | 42142 | 28.1% | 9.89e-2 |
| 100× | 421416 | **0.0%** | **3.63e-4** |

At 100× the filter was the identity to three digits: `--denoise` still cost its
eight seconds and still reported nothing wrong. Spikes of this kind are not
hypothetical in MRI — reconstruction artefacts, metal, a mis-set `valid_range` —
and none is visible in the output.

*Not translation-invariant.* Everything else in the filter is built from
differences, so adding a constant to a volume changes nothing. A floor taken
from a *level* does change, so the same anatomy with a DC offset would be
filtered as though its noise were smaller than it is.

**The fix** is to measure the floor against the volume's dynamic range, the
1st-to-99th centile separation (`SCALE_QUANTILES`), since noise is a spread and
must be judged against a spread. It removes both faults and makes the whole
filter affine-equivariant, `denoise(a*v + b) == a*denoise(v) + b`, which
`tests/test_denoise.py` now asserts in both arguments; the spike table above is
flat at every magnitude under it. Nearest-rank centiles by `kthvalue` rather
than `torch.quantile`, which refuses tensors above 2²⁴ elements — a 512³ volume
exceeds that eightfold — and which would also give up the exactness, since a
*selected* element rescales exactly where an interpolated one does not.

Two consequences. The floor drops by whatever `max/range` is (1.55 on
`brain.mnc`, so 92.7% of voxels filtered becomes 97.6%), which changes the
filter's behaviour on every volume rather than only on spiked ones; the
`README.md` table was re-measured under it rather than carried over. And a
volume that is *almost* all background now has no dynamic range to measure
against, where a maximum always had something: that is diagnosed rather than
silently admitted, and a volume with no spread anywhere is returned unchanged
before the question arises, its noise level being provably zero.

## 2. A bound that was far too loose — fixed 2026-07-31

`test_histogram.py::test_counts_match_the_volume_hist_binary` compared against
`volume_hist` with `atol=2.0`, justified in a comment by "a voxel near a bin
edge can land on the other side of it, and the counts either side then differ by
that voxel".

That does not occur: the measured maximum difference is `4.9e-7`. The bound was
four million times looser than required, so the assertion constrained almost
nothing; the histogram could have been substantially wrong and still passed.

Now `1e-6`, which is not a fitted number: `volume_hist` writes both of its
columns with `%lf`, so six decimals is the last digit it reports and there is
nothing finer to agree to. The same bound already applied to the bin centres in
the line above. Both now sit at 49% of it.

Retained here rather than deleted, because an over-loose bound is a defect in
the same way a moved one is — a test that has ceased to constrain anything — and
the record that one was shipped is part of the fix.

## 3. Bounds tight enough to fail intermittently

| Where | At | Why |
|---|---|---|
| `test_pipeline.py`, legacy backend vs `brain_nu_ref` | **74%** | MINC storage precision together with iteration amplification |
| `test_field_recovery.py`, residual at 20% / 50 mm | 73% | N3's own limit at that spacing, against a stated floor |

**The worst entry here was fixed 2026-07-31 by removing the confound, as this
section prescribed.** `test_field_recovery.py` compared implementations that had
run different numbers of iterations: N3 stops at `change < 0.001`, and once the
test swept `-distance` as well as amplitude, four of its six cells had the two
backends falling on opposite sides of that threshold, each stopping between
`0.000886` and `0.000998`. An entire additional field update separated them,
which is several times any block-level difference.

Every run in that file now uses a fixed iteration count with the early stop
disabled, so all three implementations do the same work. Agreement improved by
up to 8×, and the bound that had been at 87% is now at 36%, with nothing above
58% anywhere in the sweep. The bound itself never moved.

The remaining 73% entry is of a different kind: the bound is a stated minimum
(`planted / 2`), and 50 mm at the default `-lambda` asks N3 to do something it
is not configured for. Raising the penalty by two decades reduces that cell from
73% to 17% of the bound. The entry is a measurement rather than a fitted margin.

## 4. Assertions dropped rather than made to pass

- **Largest single difference, `test_field_recovery.py`.** The original test
  bounded `max |a - b|` as well as the RMS. At the uniform `1e-3` it would fail
  (measured `4.5e-3`), so it was dropped, on the argument that an extreme-value
  statistic over 240k voxels is dominated by a handful of mask-edge voxels and
  does not support a fixed bound. That argument is sound, but the effect is that
  an assertion which would have failed was removed rather than satisfied. If
  per-voxel behaviour is to be constrained, a percentile is the appropriate
  replacement.

- **Bending energy by quadrature, `test_spline.py`.** A check of
  `bending_energy()` against numerically integrated second derivatives of the
  basis was written and then replaced, before it ever ran, by symmetry,
  positive-semidefiniteness and bandedness, because the legacy basis is
  unnormalised (its four cubics sum to 6, not 1) and the quadrature version
  risked being either wrong or circular.

  **`J`'s value is constrained only by the parity tests against the shim.**
  Mutation testing on 2026-07-31 confirms this. Dropping the factor of two on
  the first-derivative cross terms, a plausible porting error, leaves every
  property test in `test_spline.py` passing, including the monotonicity test
  from §5, since that measures the energy with the same `J` the fit penalises
  with and therefore holds for any non-degenerate matrix. Only
  `test_fit_matches_the_legacy_spline` detects it. Degenerate mutations (zero
  `J`, or the second-derivative terms removed) are caught by both.

  If the legacy backend is ever removed, `J` loses its only substantive test. A
  correct quadrature check would be a worthwhile addition.

## 5. Inputs that were changed

Two comparisons had their inputs replaced. Both replacements were necessary,
since a recorded answer is usable only if the question can be posed again
exactly, but they narrowed what is tested:

- `test_pipeline.py::test_smooth_matches_spline_smooth` used seeded noise; it
  now uses an analytic field. The noise-rejection property it incidentally
  covered is tested directly in `test_spline.py::test_bspline_smooths_away_noise`.
- `test_legacy_backend.py`'s `sharpen_hist` comparison built its histogram from
  a seeded Gaussian mixture; it now uses the recorded histogram, which also
  isolates the deconvolution from the histogram code.

Separately, `test_spline.py::test_more_regularization_means_a_flatter_field` had
its **input** changed after a failure: `lam` was moved from `1e-3` to `1e-1`
when the "10× flatter" assertion did not hold. Both the λ pair and the factor
were free parameters, so moving one was equivalent to moving the other, and the
resulting pass carried no information that had not been constructed into it.

**Fixed 2026-07-31** by replacing the assertion itself rather than either
number. `test_regularization_trades_bending_energy_for_fit` now checks the exact
property of a penalised least-squares fit: as λ rises the bending energy `c'Jc`
can only fall and the residual can only rise. This follows from comparing the
objective at two weights, holds for every pair, and requires no factor, so it is
checked at every step of a decade sweep in both directions. The tightest step is
a 0.45% fall in energy against a 1e-9 round-off allowance. It also catches a bug
the old test could not: λ not being scaled by the sample count.

## 6. Principled bounds with substantial headroom

This is not the fault described in §1: these bounds are derived from a physical
quantity and would be correct if the code were as inaccurate as they permit.
They are, however, so far above what is measured that they would not detect a
large regression.

| Where | Bound is | At |
|---|---|---|
| `test_volume.py::test_shrink_matches_the_legacy_estimation_grid` | one part in 4095 of the span, the 12-bit quantum | 0.02% |
| `test_reproducibility.py` | one part in 65535, the 16-bit quantum, as relative RMS | 0.4% |

Both are defensible: a value returned through a file of that kind is not defined
more finely than its own quantum, whichever implementation computed it. Neither
has a tighter principled replacement, the alternative in each case being a
number drawn around the measurement, which is the subject of §1. They are
therefore retained, and the correct use of them is to monitor the measured
column: the reproducibility rows sit at 5.52e-08 relative RMS, and a change
there indicates that something moved whether or not the assertion fires.

## 7. A relative bound where an absolute one was meant — fixed 2026-07-31

`test_minc_tools.py::test_bimodal_threshold_matches_mincstats` compared this
port's Otsu threshold against `mincstats -biModalT` with
`abs(threshold - recorded) < 1e-3 * max(1.0, abs(recorded))`.

The comment above it read "mincstats prints four decimals of a value in the
hundreds of thousands", which implies an absolute `1e-4`. The code asserted a
relative `1e-3`, and on a threshold of 238347 that permits a difference of 238:
two million times looser than the printed precision, and sufficient for the
automatic mask to fall on a different tissue boundary. The comment and the
assertion had diverged, and the comment was correct.

The bound is now `1e-4`, the last digit `mincstats` reports, at 36% of it. This
is the same class of defect as §2, identified in the same way: by asking what
each bound was derived from rather than whether it passed.

---

## 8. Two bounds moved when the shim stopped linking EBTKS's LAPACK — 2026-07-31

`torch_n3/_legacy/` was made self-contained: it now compiles vendored N3 and
EBTKS sources and links the *system* LAPACK/BLAS, where before it linked
`libEBTKS.a`, which bundles its own f2c'd `dsysv`. Both solve the same
near-singular normal equations (condition number ~`1e13`); neither is wrong. At
block level the effect is negligible — the fitted spline field moves by
`3.1e-11` relative — but the iteration amplifies it, so two bounds had to move.
They were moved deliberately and are recorded here rather than widened without
record:

| Where | Was | Now | Why |
|---|---|---|---|
| `test_pipeline.py::test_matches_the_legacy_reference_volume` | `5e-3` | `1e-2` | the legacy backend's distance from N3's shipped `brain_nu_ref.mnc` went 0.3701% → 0.5211%, past the old bound. One percent relative RMS is the new bound; the port sits at 0.3007%. |
| `tests/inputs.py::PLATFORM_PROTOCOL` | 2 iterations | 1 iteration | the divergence threshold at the histogram bin boundary moved from the sixth iteration to the second, so at two the backends differ by 1.17e-3 relative RMS, against 5.5e-8 at one. |

The second is the confound removal `CLAUDE.md` prescribes rather than a loosened
bound: at two iterations that test measured which side of a rounding boundary
one voxel fell on. It costs the coverage of a second pass around the loop, which
is the price of removing the confound. `test_pipeline.py` still runs the
converged protocol.

The measured effect of the swap, end to end, is in
[README.md](README.md#which-lapack-the-legacy-backend-links).

---

## 9. The QR solver is not the default, so most of the suite still measures the ill-conditioned one — 2026-07-31

§8 is a symptom rather than the cause. The cause is that N3's spline fit is
posed as normal equations conditioned around `1e13` at the shipped 200 mm knot
spacing, which makes the last digits of the answer a property of the BLAS rather
than of the code, and which is why `PLATFORM_PROTOCOL` sits at 1 with no margin.

`blocks.spline` now has a second solver, `solver="qr"`, which fits the same
penalised least squares through `[A; sqrt(lambda N) D] c ~ [f; 0]` instead. Its
condition number is the square root of the other's, by construction rather than
incidentally, since the normal equations are that matrix's Gram matrix. The
measured consequences are large:

| Measured on `brain.mnc` | `normal` | `qr` |
|---|---|---|
| condition number of the system solved, 200 mm | `5.3e12` | `2.3e6` |
| spline field, CPU vs GPU, relative RMS | `2.5e-9` | `3.0e-13` |
| end-to-end divergence threshold, torch CPU vs CUDA | 3 iterations | 7 |
| end-to-end divergence threshold, legacy vs torch | 2 iterations | 4 |

**`normal` is still the default**, so every recorded reference, the whole
`nu_correct` half of the table above, `test_reproducibility.py`'s checked-in
volume and the `--lambda` × `--distance` tables are all still measurements of
the ill-conditioned solve. What the suite currently asserts about `qr` is only
block-level: that it reproduces the C++ oracle to the same bound, that both
solvers reach the same objective, that `D'D == J`, and the CPU/GPU bound above.

Changing the default is a deliberate and separate decision with a substantial
cost: it moves every end-to-end number in this file and requires
`tests/data/brain_nu_ref_legacy.mnc` to be regenerated. It would also allow
`PLATFORM_PROTOCOL` to rise above its floor for the first time, which is the
argument in favour. Until then, the two solvers' end-to-end outputs differ from
each other by `1.2e-3`, the same order as changing LAPACK, so results are
comparable only within one choice.

**One more item on that bill, measured 2026-08-01.** The planted-field recovery
sweep was run under all four direct solvers, and `qr` and `dr` breach
`test_field_recovery.py`'s `AGREEMENT` bound of `1e-3` against `nu_correct` in
exactly one cell of six:

| disagreement vs `nu_correct` | `normal` | `qr` | `blocked` | `dr` | `legacy` |
|---|---|---|---|---|---|
| 20% planted, 50 mm | 4.29e-4 | **1.04e-3** | 5.54e-4 | **1.04e-3** | 5.90e-4 |
| worst of the other five cells | 5.08e-4 | 5.58e-4 | 3.69e-4 | 5.58e-4 | 3.37e-4 |

Making `qr` or `dr` the default therefore fails
`test_it_recovers_as_much_as_nu_correct_did[20%-50mm-torch]`. Nothing else in
that file moves: every other pairing remains under `7.9e-4`, and every ordering
the sweep asserts holds under all four solvers.

**This does not indicate that the better-conditioned solvers are less
accurate.** In that same cell they recover more of the planted field than
`normal` does: 1.5070% remaining against `normal`'s 1.5131%, where `nu_correct`
itself leaves 1.5057%. The bound measures agreement with the original C++ about
the shape of the field, and `nu_correct` solves the same ill-conditioned normal
equations that `normal` does, so part of that agreement is shared formulation
rather than shared correctness. 50 mm is where `A` is rank deficient (§12) and
the two formulations have the greatest scope to differ.

The correct interpretation is that `AGREEMENT` is calibrated on implementations
that share a formulation, and would require restating — as a bound on recovery
rather than on similarity to one implementation — before a better-conditioned
solver could be held to it. This is recorded rather than fixed, and the bound
was not altered.

**One item is off that bill.** The 24 `--lambda` × `--distance` cells were
re-measured under **every direct solver** on 2026-07-31 and would not need
revising. `normal` reproduces both published copies exactly, all 24 cells at the
two decimals they print. The others move only slightly: `qr` and `dr` differ in
2 cells of 24, `blocked` in 3, and the largest relative change in any cell is
3.2% (`qr`, `dr`) or 4.7% (`blocked`). Every conclusion drawn in the prose — the
interior minimum in each column and its location, the decade-per-halving rule,
the asymmetry at 50 mm — is identical under all four. Those runs are 30
iterations deep, well past the divergence threshold that makes end-to-end
volumes incomparable, so this establishes that whatever the tables measure, it
is not rounding.

`python3 -m tests.tables` is now the way to re-measure them, and it diffs
against the published copies, so this claim is re-checkable rather than
recorded. It does not make them asserted, since nothing fails if a cell moves,
but it removes the manual transcription step. See CLAUDE.md, "Published numbers
no test checks".

---

## 10. A solver that ships without converging — 2026-07-31

`solver="sparse"` holds §9's stacked system in `scipy.sparse` and solves it with
`lsqr`. It does not converge, and is present in the tree as a recorded negative
result rather than as an option to be selected.

The constraint that produces this: the stacked system is rectangular, and
`scipy.sparse.linalg` provides no direct rectangular solver and no sparse QR, so
only an iterative method is available, and its convergence is governed by the
condition number the stacked form was chosen to reduce. At 200 mm LSQR
terminates after about 2,950 iterations with `istop=3`, "condition number
exceeds `conlim`", 18 s in and **1.8e-3** from the direct answer. That is a
larger error than the difference between this port and the original C++. Raising
`iter_lim` from 20,000 to 200,000 changes neither the iteration count nor the
answer.

What this costs the suite:

- It is excluded from every assertion of an exact fit. `DIRECT_SOLVERS`
  (`normal`, `qr`, `blocked`) is what those are parametrised over; `SOLVERS`
  includes `sparse` and is only for the CLI's choices.
- **The exclusion is not a widened bound.** `sparse` was run against the
  existing `1e-6 * span` parity bound and failed it by three orders. Nothing was
  loosened to accommodate it; it is held to a different assertion because it is
  a different kind of solver.
- What *is* asserted about it is that it reports non-convergence
  (`test_the_sparse_solver_reports_that_it_cannot_converge`). If a
  preconditioner ever makes it converge, that test reports it.

`solver="blocked"` solves the problem `sparse` was intended to address —
avoiding the dense design matrix — and solves it exactly, by ordering rather
than by iteration.

---

## 11. The converged reproducibility test currently fails — 2026-07-31

`tests/data/brain_nu_ref_legacy_30.mnc` was added so that
`test_reproducibility.py` covers a converged run and not only the single
iteration it has always pinned. One iteration exercises every stage but not the
loop: no convergence, no stopping rule, and no opportunity for a difference to
be fed back and amplified.

It is held to `1/65535`, the same bound as its one-iteration counterpart. **Two
of the three runs do not meet it**, and
`test_reproduces_the_recorded_converged_volume` fails for them:

| run | 1 iteration | 30 iterations | bound |
|---|---|---|---|
| `legacy`, cpu | 0 | 0 | 1.53e-5 |
| `torch`, cpu | 5.52e-08 | **1.855e-03** | 1.53e-5 — over by 121× |
| `torch`, cuda | 5.52e-08 | **2.540e-03** | 1.53e-5 — over by 166× |

The same code is within the bound by a factor of 277 at one iteration. The
difference between the columns is the divergence threshold described in §8: past
it, an end-to-end volume records which side of a rounding boundary one voxel
fell on. `legacy/cpu` passes because it wrote the file.

This is recorded rather than resolved. The bound was set deliberately rather
than fitted, and the failure is the measurement it produces. Three remedies are
available, none of them adopted here:

- accept a looser bound for the converged run (`1e-2`, the whole-pipeline bound
  `test_pipeline.py` already uses) and label it a coarse regression net;
- record one volume per `(backend, device)` so each is compared against its own
  build rather than against the legacy's;
- drop the converged volume comparison and keep only
  `test_running_it_twice_gives_the_same_bits` at thirty, which passes.

The two volumes differ by 45% in relative RMS, and
`test_the_two_recorded_runs_are_not_the_same_volume` asserts that they differ,
so an incorrectly configured protocol cannot produce two identical files and two
tests agreeing for the wrong reason.

`test_running_it_twice_gives_the_same_bits` runs at both counts and passes at
both. It is the one comparison here that retains full strength at thirty: bit
equality requires nothing of the platform, and thirty passes around a sensitive
loop is where genuine non-determinism would appear rather than remain in the
last bits.

---

## 12. The Demmler–Reinsch solver, and the formulation of it that does not work — 2026-07-31

`solver="dr"` reparameterizes §9's stacked system into the Demmler–Reinsch
basis, where the penalty is diagonal and a weight costs an elementwise division.
It is a **fourth direct solver**, in `DIRECT_SOLVERS`, and it meets every bound
`qr` meets at exactly the same margins (7.49e-08, 1.52e-08, 2.24e-12 against the
shim) — necessarily, since at `lambda == anchor` the divisor is 1 and the two
are the same back-substitution.

What it adds is `BSplineField.refit(lam)`, which reuses the factorization:

| `chunk.mnc`, per `lambda` | `qr` (fresh fit) | `dr` (`refit`) |
|---|---|---|
| 200 mm | 103 ms | 0.040 ms |
| 50 mm | 453 ms | 0.130 ms |

**The textbook formulation was attempted first and does not work here**, which
is recorded because it fails without any diagnostic. Demmler–Reinsch is normally
written on the QR factorization of the design `A` alone. Here `A` is the
*masked* design, and at fine knot spacings the mask leaves basis functions with
no data beneath them:

| `-distance` | size | rank(A) | cond(A) = cond(R) | cond(stacked) | cond(normal) |
|---|---|---|---|---|---|
| 200 mm | 64 | 64 | 8.4e7 | 3.5e6 | 1.2e13 |
| 100 mm | 100 | 100 | 7.1e6 | 8.6e5 | 7.5e11 |
| 50 mm | 245 | **243** | **7.5e12** | 2.9e5 | 8.6e10 |

`cond(R)` is therefore not better than the stacked system; it is worse at every
spacing, and at 50 mm `A` is rank deficient and `R` singular to working
precision, worse there than the normal equations. `D R^-1` then overflows into a
`gamma` with 83 non-positive entries reaching `-5.7e6`, and
`1 + lambda N gamma` passes through zero at `-7.5e4`. Clipping `gamma` at zero,
the standard recommendation, does not recover this: the eigenvectors are as
damaged as the eigenvalues, so the answer is incorrect rather than merely
imprecise.

Anchoring the QR on `[A; sqrt(lambda_0 N) D]` instead costs nothing and removes
the failure, because the penalty rows span exactly the directions the data
leaves empty. The cost is an explicit constraint rather than a hidden one: the
basis is valid only at or above its anchor, and `refit()` raises an exception
below it rather than returning a value.

Two questions remain open:

- **`dr` is not the default and changes nothing that ships.** Everything §9
  states about `qr` not being the default applies unchanged; `dr` is a further
  method of computing the same fit, so the `normal`-versus-stacked question is
  unaffected by it.
- **The `--lambda` × `--distance` tables were re-measured under `dr`** (and
  under every other direct solver) and did not need revising; see §9. They are
  still swept as separate whole-pipeline runs rather than through `refit`,
  because each cell is 30 iterations and the spline's right-hand side changes at
  every iteration; reusing one basis across the `lambda` grid would require the
  factorization to be carried across iterations as well, which is a further
  change to `_solve_dr` and was not made. `python3 -m tests.tables` is the sweep.

One incidental finding, unrelated to the solver but surfaced by it:
`bending_energy_factor` returns a `D` of numerical rank `size - 2`, not
`size - 4`. It clamps only *negative* eigenvalues of `J`, and two of `J`'s four
near-zero eigenvalues come back positive-tiny (7.7e-14, 2.9e-13 at 200 mm), so
they survive as rows of norm ~1e-7. This is harmless, since they contribute
about 1e-14 to `D'D` and
`test_the_bending_energy_factor_squares_back_to_the_tensor` still passes, but
`D` is not a minimal factor and a test asserting `rank(D) == size - 4` will
fail. The null space is recovered correctly in the transformed basis, where
`gamma` shows exactly four zeros separated from the remainder by more than ten
decades (1.7e-9 against 91 at 50 mm).

---

## 13. The denoising filter's effect on N3 is asserted by nothing — 2026-08-02

`tests/test_denoise.py` contributes **no rows to the table below**, by
construction: every bound in it is either exact (`torch.equal`) or an ordering,
because the filter is a modification with no oracle and a numeric bound could
only have been justified by first running it. That is the right choice for the
filter's own properties, but it means the suite pins only that `--denoise`
changes the estimated field, never *how*. The numbers that say whether it is
worth using — `README.md`'s "Non-local means" table — are re-measured by
`python3 -m tests.denoise` and asserted by no test, which is the status
`CLAUDE.md` records for the `--parzen-sigma` tables. The control cell is the one
guard: `off`/`linear (N3)`/`snr inf` is `tables.py`'s published cell, so drift
elsewhere surfaces there.

Two further limits on that measurement, both stated in the script:

- It is **one analytic field, one volume, one seed**, and it reached the wrong
  answer. `tests/parzen.py` carries the same caveat, and `experiments/`
  addresses it: over 450 random fields per configuration, `--denoise` improves
  95% of trials at SNR 20 and 88% at SNR 40, where the single sweep had
  suggested it was not worth using at all (`experiments/README.md`,
  "Prefiltering the volume", 2026-08-02). The single-field number was not
  *wrong* — it is a different volume, spacing and metric — but it was quoted as
  a verdict, and it does not support one. It was also wrong on the second axis:
  it concluded that `--denoise` and `--parzen-sigma` are substitutes, and the
  same 450 trials crossed over both show them to be complements, the pair
  beating the denoiser alone in all nine cells on the median paired difference
  and the window alone in all six noisy cells. This is the second recorded case
  of a single analytic field misleading about a modification, after the
  histogram window. The general rule: **`tests/*.py` measurement scripts orient,
  `experiments/` decides.**
- The filter holds no more than about 119 bytes per voxel, all of it live at
  once: a 512³ volume would need ~15 GB. Above that the loop would have to be
  tiled with a `search + patch` halo, which is **not implemented** and would
  fail as an allocation error rather than as a diagnosed one. Reaching for
  float32 instead is not the remedy, and CLAUDE.md forbids it.

`optimize.py`'s opening (`:207-215`) also duplicates `pipeline.py`'s and is now
one line longer, since both call `_denoised`. It was left duplicated
deliberately: every recorded reference runs through those lines, and factoring
them out in the same change as a behavioural addition would confound the two.

---

## Where every comparison currently sits

`python3 -m tests.margins` prints everything above the recovery sweep; the
values below are from 2026-07-31. All differences are absolute except the
`nu_correct` and `platform reference` rows, which are relative RMS: RMS
difference over mean signal, `compare_nu_result.pl`'s own measure, shared as
`tests.conftest.relative_rms`.

```
comparison                                 measured    bound       of bound
histogram parzen=True vs shim              4.52e-11    1e-09          4.5%
histogram parzen=False vs shim             0           1e-09          0.0%
histogram vs volume_hist (counts)          4.95e-07    1e-06         49.5%   (was 2.0, §2)
bin centres vs volume_hist                 4.99e-07    1e-06         49.9%
sharpen_lut deblur=False vs shim           3.2e-14     1e-11          0.3%
sharpen_lut deblur=True vs shim            2.31e-14    1e-11          0.2%
sharpen_lut vs shim (real histogram)       2.29e-13    1e-09          0.0%
sharpen_lut vs sharpen_hist                4.99e-07    1e-06         49.9%
apply_lut vs minclookup                    7.11e-15    1e-09          0.0%
bimodal_threshold vs mincstats             3.56e-05    0.0001        35.6%   (was 238, §7)
spline[normal] d=200 sub=1 vs shim         9.64e-08    5.94e-07      16.2%
spline[qr] d=200 sub=1 vs shim             7.49e-08    5.94e-07      12.6%
spline[blocked] d=200 sub=1 vs shim        7.49e-08    5.94e-07      12.6%
spline[dr] d=200 sub=1 vs shim             7.49e-08    5.94e-07      12.6%
spline[normal] d=200 sub=2 vs shim         1.69e-10    6.21e-07       0.0%
spline[qr] d=200 sub=2 vs shim             1.52e-08    6.21e-07       2.5%
spline[blocked] d=200 sub=2 vs shim        1.52e-08    6.21e-07       2.5%
spline[dr] d=200 sub=2 vs shim             1.52e-08    6.21e-07       2.5%
spline[normal] d=50  sub=1 vs shim         1.28e-11    2.9e-07        0.0%
spline[qr] d=50  sub=1 vs shim             2.24e-12    2.9e-07        0.0%
spline[blocked] d=50  sub=1 vs shim        2.24e-12    2.9e-07        0.0%
spline[dr] d=50  sub=1 vs shim             2.24e-12    2.9e-07        0.0%
spline[qr] d=200 cpu vs cuda               1.18e-12    5.94e-12      19.8%   <- §9
correct_field vs shim                      4.27e-06    7.96e-05       5.4%
correct_field vs binary                    4.27e-06    7.96e-05       5.4%
shrink vs mincresample                     0.0312      204            0.0%   <- §6
_sharpen[torch] vs sharpen_volume          2.47e-06    3.21e-05       7.7%
_sharpen[legacy] vs sharpen_volume         2.47e-06    3.21e-05       7.7%
_smooth[torch] vs spline_smooth            2.32e-08    5.4e-06         0.4%
_smooth[legacy] vs spline_smooth           6.38e-09    5.4e-06         0.1%
nu_correct[torch] chunk i1 s3              3.34e-05    0.001          3.3%
nu_correct[torch] chunk i3 s4              7.82e-05    0.001          7.8%
nu_correct[legacy] chunk i1 s3             3.34e-05    0.001          3.3%
nu_correct[legacy] chunk i3 s4             8.47e-05    0.001          8.5%
nu_correct[torch] vs brain_nu_ref          0.00301     0.01          30.1%
nu_correct[legacy] vs brain_nu_ref         0.00521     0.01          52.1%   <- §8
amplification, early (bound is a maximum)  1.39e-07    1e-06         13.9%   <- §1
amplification, late / early (a minimum)    100         6.91e+03       1.4%   <- §1
platform reference, torch/cpu              5.52e-08    1.53e-05       0.4%   <- §6
platform reference, legacy/cpu             0           1.53e-05       0.0%
platform reference, torch/cuda             5.52e-08    1.53e-05       0.4%
converged reference, torch/cpu             0.00185     1.53e-05   12153.7%   <- §11, FAILS
converged reference, legacy/cpu            0           1.53e-05       0.0%
converged reference, torch/cuda            0.00254     1.53e-05   16642.3%   <- §11, FAILS

recovery sweep, as % of bound (fixed 30 iterations, no early stop).  The two
columns involving `legacy` moved when the shim changed LAPACK (§8); `t-bin`,
which involves neither, did not.
  amp  dist | residual vs planted/2    | agreement vs 1e-3
             torch  legacy  nu_correct | t-l    t-bin  l-bin | after/before
  20%  200mm   15%    15%     15%      |  32%    19%    14%  |  0.245
  20%  100mm   29%    29%     29%      |  27%    48%    26%  |  0.695
  20%   50mm   73%    73%     73%      |  20%    43%    59%  |  1.101  <- §1, §3
  40%  200mm   14%    14%     14%      |  45%    37%    10%  |  0.124
  40%  100mm   24%    24%     24%      |  11%    22%    13%  |  0.354
  40%   50mm   47%    47%     47%      |  33%    51%    34%  |  0.575
```

---

## Not problems with the tests

The following are properties of N3 or of the legacy implementation, measured and
documented elsewhere. No tolerance can be tightened beyond them:

- The B-spline normal equations are ill-conditioned (~1e13 at the default 200 mm
  spacing), so coefficients are undetermined to ~1e-4 by any solver. Tests
  compare fitted fields, never coefficients. See `PLAN.md`.
- `correct_field` relaxes in raster order, which is inherently sequential; the
  port sweeps checkerboard colours instead. Both approximate the same Laplace
  solution, to ~5e-6.
- N3's loop amplifies differences: implementations agreeing on the field to
  1.4e-7 after one iteration disagree by 2.5e-4 after ten, and by 1.1e-3 end to
  end under the shipped protocol. No end-to-end N3 comparison is informative
  past three digits; running the same backend on a GPU moves the result by
  1.3e-3, marginally more than changing backend does.
- The *installed* N3 passes every intermediate between its programs as a
  slice-scaled MINC file, so neither backend here can reach the legacy suite's
  own 1e-4: the legacy backend calls the same C++ within one process, on
  float64, and is rounded nowhere. (It ends up 5.21e-3 from `brain_nu_ref.mnc`,
  further out than the port's 3.01e-3.) See `CLAUDE.md` and
  `torch_n3/backends/legacy.py`.
- `test_reproducibility.py` runs one iteration rather than the shipped fifty.
  This is not a weakened requirement but the strongest one that is well posed:
  past one iteration the answer is moved by a single histogram count crossing a
  bin boundary (see `CLAUDE.md`), and the backends finish 1.17e-3 relative RMS
  apart, four orders of magnitude above where they sit at one iteration. The
  converged protocol is covered, three digits at a time, by
  `test_pipeline.py::test_matches_the_legacy_reference_volume`.

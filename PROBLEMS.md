# Known problems

Weak spots in this repository's test suite, written down so they are visible
rather than discovered. Audited 2026-07-31; §2, §5, §7 and the worst entries of
§1 and §3 were fixed the same day, and are kept here rather than deleted — a
bound that had stopped asking a question is worth recording even once it does
again.

Every number below comes from `python3 -m tests.margins`, which prints the
table at the bottom. Re-run it after changing a bound or a block.

The rule these are measured against is in [CLAUDE.md](CLAUDE.md#test-tolerances):
a threshold states what the code is *required* to do, and is not the measured
difference plus a margin. Several here fall short of that. None of them is a
failing test; they are places where a passing test says less than it appears to.

---

## 1. Thresholds chosen to fit the measurement

These were set knowing what the code already produced. They pass, but they
record behaviour rather than state a requirement, so they would not catch a
regression that stayed inside the envelope they were drawn around.

| Where | Assertion | Now at |
|---|---|---|
| `test_pipeline.py` | `early < 1e-6`, `late > 100 * early` | 14%, 1800× |
| `test_histogram.py`, `test_sharpen.py`, `test_spline.py`, `test_field.py` | the `atol` constants | 0–50% |

**Two entries removed 2026-07-31.** `residual < non_uniformity(planted) / 4`
and `after < before / 3` were both fitted at the default 200 mm knot spacing,
and sweeping `-distance` showed they did not survive it: `/4` fails at 100 mm
and finer, `/3` fails at 100 mm and finer, and at 50 mm the correction leaves
the volume *further* from the truth than the planted field did (`after/before`
= 1.10 at 20%). Neither was widened. `residual` is now held to `planted / 2`,
stated as a floor on being useful rather than as a description — a corrector
that leaves more than half a known field behind is not doing its job — and
`after < before` is a bare ordering, asserted only at the shipped 200 mm where
it is true. The 50 mm behaviour is asserted as its own ordering instead of
being averaged over.

The block `atol`s are round numbers a decade or two above what was measured,
which is defensible, but none of them is derived from anything. Where a
principled bound exists it is already used and should be preferred: the 16-bit
quantum of the file a legacy program wrote (`span(reference) / 65535`), the
12-bit one (`/ 4095`), the six decimals `%lf` prints (`1e-6`).

## 2. A bound that was far too loose — fixed 2026-07-31

`test_histogram.py::test_counts_match_the_volume_hist_binary` compared against
`volume_hist` with `atol=2.0`, justified in a comment by "a voxel near a bin
edge can land on the other side of it, and the counts either side then differ
by that voxel".

That does not happen: the measured maximum difference is `4.9e-7`. The bound
was four million times looser than it needed to be, so the assertion tested
almost nothing — the histogram could have been badly wrong and still passed.

Now `1e-6`, which is not a fitted number: `volume_hist` writes both of its
columns with `%lf`, so six decimals is the last digit it reports and there is
nothing finer to agree to. The same bound already applied to the bin centres
in the line above. Both now sit at 49% of it.

Kept here rather than deleted, because an over-loose bound is a defect in the
same way a moved one is — a test that has stopped asking a question — and the
record of one having been shipped is worth as much as the fix.

## 3. Bounds that are tight enough to flake

| Where | At | Why |
|---|---|---|
| `test_pipeline.py`, legacy backend vs `brain_nu_ref` | **74%** | MINC storage precision plus iteration amplification |
| `test_field_recovery.py`, residual at 20% / 50 mm | 73% | N3's own limit at that spacing, against a stated floor |

**The worst entry here was fixed 2026-07-31 by removing the confound, as this
section said it should be.** `test_field_recovery.py` compared implementations
that had run *different numbers of iterations*: N3 stops at `change < 0.001`,
and once the test swept `-distance` as well as amplitude, four of its six
cells had the two backends landing on opposite sides of that threshold —
every one of them stopping between `0.000886` and `0.000998`. An entire extra
field update separated them, several times more than any block-level
difference.

Every run in that file now uses a fixed iteration count with the early stop
disabled, so all three implementations do the same work. Agreement improved by
up to 8×, and the bound that had been at 87% is now at 36% with nothing above
58% anywhere in the sweep. The bound itself never moved.

The remaining 73% entry is different in kind: the bound is a stated floor
(`planted / 2`), and 50 mm at the default `-lambda` is N3 asked to do
something it is not set up for — raising the penalty a couple of decades drops
that cell from 73% to 17% of the bound. It is information, not a fitted
margin.

## 4. Assertions dropped rather than made to pass

- **Largest single difference, `test_field_recovery.py`.** The original test
  bounded `max |a - b|` as well as the RMS. At the uniform `1e-3` it would
  fail (measured `4.5e-3`), so it was dropped, on the argument that an
  extreme-value statistic over 240k voxels is dominated by a handful of
  mask-edge voxels and does not deserve a fixed bound. That argument is sound,
  but the effect is that an assertion which would have failed was removed
  rather than satisfied. If per-voxel behaviour matters, a percentile would be
  the honest replacement.

- **Bending energy by quadrature, `test_spline.py`.** A check of
  `bending_energy()` against numerically integrated second derivatives of the
  basis was written and then replaced, before it ever ran, by symmetry,
  positive-semidefiniteness and bandedness — because the legacy basis is
  unnormalised (its four cubics sum to 6, not 1) and the quadrature version
  risked being either wrong or circular.

  **`J`'s value is pinned only by the parity tests against the shim.** Mutation
  testing on 2026-07-31 confirms it. Dropping the factor of two on the
  first-derivative cross terms — a plausible porting slip — leaves every
  property test in `test_spline.py` green, including the monotonicity one from
  §5, because that measures the energy with the same `J` the fit penalises
  with and so holds for any non-degenerate matrix. Only
  `test_fit_matches_the_legacy_spline` catches it. Degenerate mutations (zero
  `J`, or the second-derivative terms removed) are caught by both.

  So if the legacy backend ever goes away, `J` loses its only real test. A
  correct quadrature check would be worth having.

## 5. Inputs that were changed

Two comparisons had their inputs replaced. Both were necessary — a recorded
answer is only worth having if the question can be asked again exactly — but
they narrowed what is tested:

- `test_pipeline.py::test_smooth_matches_spline_smooth` used seeded noise;
  it now uses an analytic field. The noise-rejection property it incidentally
  covered is tested directly in `test_spline.py::test_bspline_smooths_away_noise`.
- `test_legacy_backend.py`'s `sharpen_hist` comparison built its histogram
  from a seeded Gaussian mixture; it now uses the recorded histogram, which
  also has the effect of isolating the deconvolution from the histogram code.

Separately, `test_spline.py::test_more_regularization_means_a_flatter_field`
had its **input** changed after a failure — `lam` went from `1e-3` to `1e-1`
when the "10× flatter" assertion did not hold. Both the λ pair and the factor
were free parameters, so moving one of them was the same act as moving the
other; the pass carried no information that had not been engineered into it.

**Fixed 2026-07-31**, by replacing the whole assertion rather than either
number. `test_regularization_trades_bending_energy_for_fit` now checks the
exact property of a penalised least-squares fit: as λ rises the bending energy
`c'Jc` can only fall and the residual can only rise. That follows from
comparing the objective at two weights, holds for every pair, and needs no
factor at all — so it is checked at every step of a decade sweep, in both
directions. The tightest step is a 0.45% fall in energy against a 1e-9
round-off allowance. It also catches a bug the old test could not: λ not being
scaled by the sample count.

## 6. Principled bounds with a great deal of headroom

Not the same fault as §1 — these bounds are derived from something real, and
would be the right answer if the code were as bad as they allow. But they are
so far above what is measured that they would not notice a large regression.

| Where | Bound is | At |
|---|---|---|
| `test_volume.py::test_shrink_matches_the_legacy_estimation_grid` | one part in 4095 of the span, the 12-bit quantum | 0.02% |
| `test_reproducibility.py` | one part in 65535, the 16-bit quantum, as relative RMS | 0.4% |

Both are defensible: a value that came back through a file of that kind is not
defined more finely than its own quantum, whoever computed it. Neither has a
tighter principled replacement — the alternative in each case is a number drawn
around the measurement, which is what §1 is about. So they stay, and the honest
use of them is to watch the *measured* column: the reproducibility rows sit at
5.52e-08 relative RMS, and a change there means something moved whether or not
the assertion fires.

## 7. A relative bound where an absolute one was meant — fixed 2026-07-31

`test_minc_tools.py::test_bimodal_threshold_matches_mincstats` compared our
Otsu threshold against `mincstats -biModalT` with
`abs(threshold - recorded) < 1e-3 * max(1.0, abs(recorded))`.

The comment above it read "mincstats prints four decimals of a value in the
hundreds of thousands" — which argues for an absolute `1e-4`. The code asserted
a *relative* `1e-3`, and on a threshold of 238347 that permits a difference of
238: two million times looser than the printed precision, and easily enough for
the automatic mask to land on a different tissue boundary. The comment and the
assertion had drifted apart, and the comment was the one that was right.

Now `1e-4`, the last digit `mincstats` reports, at 36% of it. The same species
of defect as §2, found the same way: by asking what each bound was derived
from, rather than whether it passed.

---

## 8. Two bounds moved when the shim stopped linking EBTKS's LAPACK — 2026-07-31

`torch_n3/_legacy/` was made self-contained: it now compiles vendored N3 and
EBTKS sources and links the *system* LAPACK/BLAS, where before it linked
`libEBTKS.a`, which bundles its own f2c'd `dsysv`. Both solve the same
near-singular normal equations (condition number ~`1e13`); neither is wrong.
The blocks barely notice — the fitted spline field moves by `3.1e-11` relative
— but the iteration amplifies it, so two bounds had to move. Deliberately, and
recorded here rather than quietly widened:

| Where | Was | Now | Why |
|---|---|---|---|
| `test_pipeline.py::test_matches_the_legacy_reference_volume` | `5e-3` | `1e-2` | the legacy backend's distance from N3's shipped `brain_nu_ref.mnc` went 0.3701% → 0.5211%, past the old bound. One percent relative RMS is the new bound; the port sits at 0.3007%. |
| `tests/inputs.py::PLATFORM_PROTOCOL` | 2 iterations | 1 iteration | the histogram-bin knife-edge moved from the sixth iteration to the second, so at two the backends land 1.17e-3 relative RMS apart, against 5.5e-8 at one. |

The second is the confound-removal `CLAUDE.md` asks for, not a loosened bound:
at two iterations that test was measuring which side of a rounding boundary one
voxel fell on. It costs the coverage of a second trip round the loop, which is
the honest price. `test_pipeline.py` still runs the converged protocol.

The measured effect of the swap, end to end, is in
[README.md](README.md#which-lapack-the-legacy-backend-links).

---

## 9. The QR solver is not the default, so most of the suite still measures the ill-conditioned one — 2026-07-31

§8 is a symptom, not the disease. The disease is that N3's spline fit is posed
as normal equations conditioned around `1e13` at the shipped 200 mm knot
spacing, which is what makes the last digits of the answer a property of the
BLAS rather than of the code, and it is why `PLATFORM_PROTOCOL` sits at 1 with
no margin.

`blocks.spline` now has a second solver, `solver="qr"`, which fits the same
penalised least squares through `[A; sqrt(lambda N) D] c ~ [f; 0]` instead.
Its condition number is the square root of the other's — by construction, not
by luck; the normal equations are that matrix's Gram matrix — and the measured
consequences are large:

| Measured on `brain.mnc` | `normal` | `qr` |
|---|---|---|
| condition number of the system solved, 200 mm | `5.3e12` | `2.3e6` |
| spline field, CPU vs GPU, relative RMS | `2.5e-9` | `3.0e-13` |
| end-to-end cliff, torch CPU vs CUDA | 3 iterations | 7 |
| end-to-end cliff, legacy vs torch | 2 iterations | 4 |

**`normal` is still the default**, so every recorded reference, the whole
`nu_correct` half of the table above, `test_reproducibility.py`'s checked-in
volume and the `--lambda` × `--distance` tables are all still measurements of
the ill-conditioned solve. What the suite currently asserts about `qr` is only
block-level: that it reproduces the C++ oracle to the same bound, that both
solvers reach the same objective, that `D'D == J`, and the CPU/GPU bound above.

Changing the default is a deliberate, separate decision with a real cost — it
moves every end-to-end number in this file and needs
`tests/data/brain_nu_ref_legacy.mnc` regenerated. It would also let
`PLATFORM_PROTOCOL` rise off its floor for the first time, which is the
argument for doing it. Until then, the two solvers' end-to-end outputs differ
from each other at `1.2e-3` — the same order as switching LAPACK — so results
are only comparable within one choice.

**One more item on that bill, measured 2026-08-01.** The planted-field recovery
sweep was run under all four direct solvers, and `qr` and `dr` breach
`test_field_recovery.py`'s `AGREEMENT` bound of `1e-3` against `nu_correct` in
exactly one cell of six:

| disagreement vs `nu_correct` | `normal` | `qr` | `blocked` | `dr` | `legacy` |
|---|---|---|---|---|---|
| 20% planted, 50 mm | 4.29e-4 | **1.04e-3** | 5.54e-4 | **1.04e-3** | 5.90e-4 |
| worst of the other five cells | 5.08e-4 | 5.58e-4 | 3.69e-4 | 5.58e-4 | 3.37e-4 |

So making `qr` or `dr` the default fails
`test_it_recovers_as_much_as_nu_correct_did[20%-50mm-torch]`. Nothing else in
that file moves: every other pairing stays under `7.9e-4`, and every ordering
the sweep asserts holds under all four solvers.

**This is not the better-conditioned solvers being worse.** In that same cell
they recover *more* of the planted field than `normal` does — 1.5070% left
against `normal`'s 1.5131%, where `nu_correct` itself leaves 1.5057%. What the
bound measures is agreement about the field's *shape* with the original C++,
and `nu_correct` solves the same ill-conditioned normal equations `normal`
does, so part of that agreement is shared formulation rather than shared
correctness. 50 mm is where `A` is rank deficient (§12) and the two
formulations have the most room to differ.

The honest reading is that `AGREEMENT` is calibrated on implementations that
share a formulation, and would need restating — as a bound on recovery rather
than on likeness to one implementation — before a better-conditioned solver
could be held to it. Recorded, not fixed, and the bound was not touched.

**One item is off that bill.** The 24 `--lambda` × `--distance` cells were
re-measured under **every direct solver** on 2026-07-31 and would not need
revising. `normal` reproduces both published copies exactly, all 24 cells at
the two decimals they print. The others move barely: `qr` and `dr` differ in 2
cells of 24, `blocked` in 3, and the largest relative move in any cell is 3.2%
(`qr`, `dr`) or 4.7% (`blocked`). Every conclusion the prose draws — the
interior minimum in each column and where it sits, the decade-per-halving rule,
the asymmetry at 50 mm — is identical under all four. That is a mild surprise
worth recording on its own, since those runs are 30 iterations deep, well past
the knife-edge that makes end-to-end volumes incomparable; whatever the tables
are measuring, it is not rounding.

`python3 -m tests.tables` is now the way to re-measure them, and it diffs
against the published copies, so this claim is re-checkable rather than
recorded. It does not make them *asserted* — nothing fails if a cell moves —
but it removes the hand-transcription step. See CLAUDE.md, "Published numbers
no test checks".

---

## 10. A solver that ships without converging — 2026-07-31

`solver="sparse"` holds §9's stacked system in `scipy.sparse` and solves it with
`lsqr`. It does not reach an answer, and it is in the tree as a recorded
negative result rather than as an option anyone should select.

The constraint that produces it: the stacked system is rectangular, and
`scipy.sparse.linalg` has no direct rectangular solver — no sparse QR — so the
only thing available is an iterative method, whose convergence is governed by
the condition number the stacked form was chosen to *reduce*. At 200 mm LSQR
stops after ~2,950 iterations with `istop=3`, "condition number exceeds
`conlim`", 18 s in and **1.8e-3** from the direct answer. That is a larger error
than the gap between this port and the original C++. Raising `iter_lim` from
20,000 to 200,000 changes neither the iteration count nor the answer.

What this costs the suite:

- It is excluded from every assertion of an exact fit. `DIRECT_SOLVERS`
  (`normal`, `qr`, `blocked`) is what those are parametrised over;
  `SOLVERS` includes `sparse` and is only for the CLI's choices.
- **The exclusion is not a widened bound.** `sparse` was run against the
  existing `1e-6 * span` parity bound and failed it by three orders. Nothing
  was loosened to accommodate it; it is held to a different assertion because
  it is a different kind of solver.
- What *is* asserted about it is that it reports non-convergence
  (`test_the_sparse_solver_reports_that_it_cannot_converge`). If a
  preconditioner ever makes it converge, that test is what will say so.

`solver="blocked"` is the answer to the problem `sparse` was reaching for —
avoiding the dense design matrix — and it reaches it exactly, by ordering rather
than by iteration. Use that.

---

## 11. The converged reproducibility test currently fails — 2026-07-31

`tests/data/brain_nu_ref_legacy_30.mnc` was added so that
`test_reproducibility.py` covers a converged run and not only the single
iteration it has always pinned. One iteration exercises every stage but never
the loop: no convergence, no stopping rule, and no opportunity for a difference
to be fed back in and grown.

It is held to `1/65535`, the same bound as its one-iteration twin. **Two of the
three runs do not meet it**, and `test_reproduces_the_recorded_converged_volume`
fails for them:

| run | 1 iteration | 30 iterations | bound |
|---|---|---|---|
| `legacy`, cpu | 0 | 0 | 1.53e-5 |
| `torch`, cpu | 5.52e-08 | **1.855e-03** | 1.53e-5 — over by 121× |
| `torch`, cuda | 5.52e-08 | **2.540e-03** | 1.53e-5 — over by 166× |

The same code is inside the bound by a factor of 277 at one iteration. The
difference between the columns is the histogram knife-edge described in §8: past
it, an end-to-end volume records which side of a rounding boundary one voxel
fell on. `legacy/cpu` passes only because it wrote the file.

This is recorded rather than resolved. The bound was set deliberately, not
fitted, and the failure is the measurement it produces. Three ways it could go,
none taken here:

- accept a looser bound for the converged run (`1e-2`, the whole-pipeline bound
  `test_pipeline.py` already uses) and label it a coarse regression net;
- record one volume per `(backend, device)` so each is compared against its own
  build rather than against the legacy's;
- drop the converged volume comparison and keep only
  `test_running_it_twice_gives_the_same_bits` at thirty, which passes.

The two volumes are 45% apart in relative RMS, and
`test_the_two_recorded_runs_are_not_the_same_volume` asserts they differ, so a
mis-wired protocol cannot leave two identical files and two tests agreeing for
the wrong reason.

`test_running_it_twice_gives_the_same_bits` runs at both counts and passes at
both. It is the one comparison here that keeps full strength at thirty: bit
equality asks nothing of the platform, and thirty trips round a sensitive loop
is where real non-determinism would surface rather than hide in the last bits.

---

## 12. The Demmler–Reinsch solver, and the reading of it that does not work — 2026-07-31

`solver="dr"` reparameterizes §9's stacked system into the Demmler–Reinsch
basis, where the penalty is diagonal and a weight costs an elementwise
division. It is a **fourth direct solver**, in `DIRECT_SOLVERS`, and it meets
every bound `qr` meets at exactly the same margins (7.49e-08, 1.52e-08,
2.24e-12 against the shim) — necessarily, because at `lambda == anchor` the
divisor is 1 and the two are the same back-substitution.

What it adds is `BSplineField.refit(lam)`, which reuses the factorization:

| `chunk.mnc`, per `lambda` | `qr` (fresh fit) | `dr` (`refit`) |
|---|---|---|
| 200 mm | 103 ms | 0.040 ms |
| 50 mm | 453 ms | 0.130 ms |

**The textbook recipe was tried first and does not work here**, which is worth
recording because it fails quietly rather than loudly. Demmler–Reinsch is
normally written on the QR of the design `A` alone. But `A` here is the *masked*
design, and at fine knot spacings the mask leaves basis functions with no data
under them:

| `-distance` | size | rank(A) | cond(A) = cond(R) | cond(stacked) | cond(normal) |
|---|---|---|---|---|---|
| 200 mm | 64 | 64 | 8.4e7 | 3.5e6 | 1.2e13 |
| 100 mm | 100 | 100 | 7.1e6 | 8.6e5 | 7.5e11 |
| 50 mm | 245 | **243** | **7.5e12** | 2.9e5 | 8.6e10 |

So `cond(R)` is not "much better than the stacked system" — it is *worse at
every spacing*, and at 50 mm `A` is rank deficient and `R` singular to working
precision, worse even than the normal equations there. `D R^-1` then overflows
into a `gamma` with 83 non-positive entries reaching `-5.7e6`, and
`1 + lambda N gamma` passes through zero at `-7.5e4`. Clipping `gamma` at zero,
which is the usual advice, does not rescue this: the eigenvectors are as
damaged as the eigenvalues, so the answer is wrong rather than merely
imprecise.

Anchoring the QR on `[A; sqrt(lambda_0 N) D]` instead costs nothing and removes
the failure outright, because the penalty rows span exactly the directions the
data leaves empty. The cost is a real constraint, not a hidden one: the basis is
valid only at or above its anchor, and `refit()` raises below it rather than
returning a number.

Two things this leaves open:

- **`dr` is not the default and changes nothing that ships.** Everything §9
  says about `qr` not being the default applies unchanged; `dr` is another way
  to compute the same fit, so the whole `normal`-vs-stacked question is
  untouched by it.
- **The `--lambda` × `--distance` tables were re-measured under `dr`** (and
  under every other direct solver) and did not need revising — see §9. They are
  still swept as separate whole-pipeline runs rather than through `refit`,
  because each cell is 30 iterations and the spline's right-hand side changes
  every iteration; reusing one basis across the `lambda` grid would need the
  factorization carried across iterations too, which is a further change to
  `_solve_dr` and was not made. `python3 -m tests.tables` is the sweep.

One incidental finding, unrelated to the solver but surfaced by it:
`bending_energy_factor` returns a `D` of numerical rank `size - 2`, not
`size - 4`. It clamps only *negative* eigenvalues of `J`, and two of `J`'s four
near-zero eigenvalues come back positive-tiny (7.7e-14, 2.9e-13 at 200 mm), so
they survive as rows of norm ~1e-7. This is harmless — they contribute ~1e-14
to `D'D`, and `test_the_bending_energy_factor_squares_back_to_the_tensor` still
passes — but `D` is not a minimal factor, and a test asserting `rank(D) ==
size - 4` will fail. The null space *is* recovered correctly in the transformed
basis, where `gamma` shows exactly four zeros separated from the rest by more
than ten decades (1.7e-9 against 91 at 50 mm).

---

## Where every comparison currently sits

`python3 -m tests.margins` prints everything above the recovery sweep; these
are from 2026-07-31. All differences are absolute except the `nu_correct` and
`platform reference` rows, which are relative RMS -- RMS difference over mean
signal, `compare_nu_result.pl`'s own measure, shared as
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

These are properties of N3 or of the legacy implementation, measured and
documented elsewhere, and no tolerance can be tightened past them:

- The B-spline normal equations are ill-conditioned (~1e13 at the default
  200 mm spacing), so coefficients are undetermined to ~1e-4 by any solver.
  Tests compare fitted fields, never coefficients. See `PLAN.md`.
- `correct_field` relaxes in raster order, which is inherently sequential; the
  port sweeps checkerboard colours instead. Both approximate the same Laplace
  solution, to ~5e-6.
- N3's loop amplifies: implementations agreeing on the field to 1.4e-7 after
  one iteration disagree by 2.5e-4 after ten, and by 1.1e-3 end to end under
  the shipped protocol. No end-to-end N3 comparison means much past three
  digits — running the same backend on a GPU moves it by 1.3e-3, slightly
  more than changing backend does.
- The *installed* N3 passes every intermediate between its programs as a
  slice-scaled MINC file, so neither backend here can reach the legacy suite's
  own 1e-4 — the legacy backend calls the same C++ in one process, on float64,
  and is rounded nowhere. (It ends up 5.21e-3 from `brain_nu_ref.mnc`, further
  out than the port's 3.01e-3.) See `CLAUDE.md` and
  `torch_n3/backends/legacy.py`.
- `test_reproducibility.py` pins one iteration rather than the shipped fifty. That is not
  a weakened requirement but the largest one that is well posed: past one,
  what moves the answer is a single histogram count crossing a bin boundary
  (see `CLAUDE.md`), and the backends finish 1.17e-3 relative RMS apart — four
  orders of magnitude above where they sit at one iteration. The converged
  protocol is covered, three digits at a time, by
  `test_pipeline.py::test_matches_the_legacy_reference_volume`.

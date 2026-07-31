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
| `test_volume.py::test_shrink_matches_the_legacy_estimation_grid` | one 12-bit storage level (`span / 4095`) | 0.02% |
| `test_reproducibility.py` | one 16-bit storage level (`span / 65535`) | 1.2% |

Both are defensible: a value that came back through a file of that kind is not
defined more finely than one level, whoever computed it. Neither has a tighter
principled replacement — the alternative in each case is a number drawn around
the measurement, which is what §1 is about. So they stay, and the honest use of
them is to watch the *measured* column: the reproducibility rows have sat at
0.208 since the file was recorded, and a change there means something moved
whether or not the assertion fires.

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

## Where every comparison currently sits

`python3 -m tests.margins` prints everything above the recovery sweep; these
are from 2026-07-31. All differences are absolute except the `nu_correct` rows,
which are relative RMS.

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
spline d=200 sub=1 vs shim                 9.63e-08    5.94e-07      16.2%
spline d=200 sub=2 vs shim                 4.28e-10    6.21e-07       0.1%
spline d=50  sub=1 vs shim                 1.27e-11    2.9e-07        0.0%
correct_field vs shim                      4.27e-06    7.96e-05       5.4%
correct_field vs binary                    4.27e-06    7.96e-05       5.4%
shrink vs mincresample                     0.0312      204            0.0%   <- §6
_sharpen[torch] vs sharpen_volume          2.47e-06    3.21e-05       7.7%
_sharpen[legacy] vs sharpen_volume         2.47e-06    3.21e-05       7.7%
_smooth[torch] vs spline_smooth            2.32e-08    5.4e-06         0.4%
_smooth[legacy] vs spline_smooth           6.36e-09    5.4e-06         0.1%
nu_correct[torch] chunk i1 s3              3.34e-05    0.001          3.3%
nu_correct[torch] chunk i3 s4              7.82e-05    0.001          7.8%
nu_correct[legacy] chunk i1 s3             3.34e-05    0.001          3.3%
nu_correct[legacy] chunk i3 s4             0.000153    0.001         15.3%
nu_correct[torch] vs brain_nu_ref          0.00301     0.005         60.1%
nu_correct[legacy] vs brain_nu_ref         0.0037      0.005         74.0%   <- §3
amplification, early (bound is a maximum)  1.41e-07    1e-06         14.1%   <- §1
amplification, late / early (a minimum)    100         1.78e+03       5.6%   <- §1
platform reference, torch/cpu              0.208       18             1.2%   <- §6
platform reference, legacy/cpu             0           18             0.0%
platform reference, torch/cuda             0.208       18             1.2%

recovery sweep, as % of bound (fixed 30 iterations, no early stop)
  amp  dist | residual vs planted/2    | agreement vs 1e-3
             torch  legacy  nu_correct | t-l    t-bin  l-bin | after/before
  20%  200mm   15%    15%     15%      |  19%    19%     7%  |  0.245
  20%  100mm   29%    29%     29%      |  36%    48%    22%  |  0.695
  20%   50mm   73%    73%     73%      |  18%    43%    58%  |  1.101  <- §1, §3
  40%  200mm   14%    14%     14%      |  13%    37%    26%  |  0.124
  40%  100mm   24%    24%     24%      |  11%    22%    33%  |  0.354
  40%   50mm   47%    47%     47%      |  30%    51%    35%  |  0.575
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
  and is rounded nowhere. (It ends up 3.7e-3 from `brain_nu_ref.mnc`, slightly
  further out than the port.) See `CLAUDE.md` and `torch_n3/backends/legacy.py`.
- `test_reproducibility.py` pins two iterations rather than the shipped fifty. That is not
  a weakened requirement but the largest one that is well posed: past two,
  what moves the answer is a single histogram count crossing a bin boundary
  (see `CLAUDE.md`), and CPU and GPU finish 444 storage levels apart. The
  converged protocol is covered, three digits at a time, by
  `test_pipeline.py::test_matches_the_legacy_reference_volume`.

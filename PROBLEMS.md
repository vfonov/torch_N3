# Known problems

Weak spots in this repository's test suite, written down so they are visible
rather than discovered. Audited 2026-07-31, at commit `dde3099`; §2 fixed the
same day.

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
| `test_field_recovery.py` | `residual < non_uniformity(planted) / 4` | **85% of bound** |
| `test_field_recovery.py` | `after < before / 3` | 76% |
| `test_pipeline.py` | `early < 1e-6`, `late > 100 * early` | ~14%, 1800× |
| `test_histogram.py`, `test_sharpen.py`, `test_spline.py`, `test_field.py` | the `atol` constants | 0.1–70% |

`residual < planted / 4` is the one to fix first: `/5` fails the 20% case, so
the bound was picked to clear a number rather than to state what a bias
corrector must achieve. Deciding that requirement independently — and then
seeing whether N3 meets it — would be a better test than the current one.

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

Three comparisons sit above 70% of their bound. Two of them have understood
causes, which is the only reason they are being left alone.

| Where | At | Why |
|---|---|---|
| `test_field_recovery.py`, torch vs legacy at 40% | **87%** | the two stop at *different iterations*; see below |
| `test_pipeline.py`, legacy backend vs `brain_nu_ref` | **74%** | MINC storage precision plus iteration amplification |
| `test_field_recovery.py`, residual at 20% | **85%** | the fitted bound from §1 |

The first is the one to check before touching anything: N3 stops at
`change < 0.001`, and at 40% one backend reaches `0.000975` at iteration 20
where the other is still at `0.001011` and runs a 21st. A whole extra field
update separates them, which is far more than any block-level difference. The
fix consistent with CLAUDE.md is to remove the confound — run both for a fixed
iteration count in that one test, so it measures block agreement rather than
which side of the stopping rule they landed on — **not** to widen the bound.
It has not been done, because it changes what the test measures and that is a
decision worth making deliberately.

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
  risked being either wrong or circular. `J` is therefore only pinned
  indirectly, through the spline fits that use it. A correct quadrature check
  would be worth having.

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
when the "10× flatter" assertion did not hold. The current form (`1e-9` vs
`1e-1`, measured 69×) is a reasonable property test, but it was arrived at by
turning a dial until the test went green, which is the same practice as moving
a threshold.

---

## Where every comparison currently sits

Regenerate with the script in the session log, or by hand; these are from
2026-07-31.

```
comparison                                 measured    bound       of bound
histogram parzen=True vs shim              4.5e-11     1.0e-09       4.5%
histogram parzen=False vs shim             0.0e+00     1.0e-09       0.0%
histogram vs volume_hist (counts)          4.9e-07     1.0e-06      49.5%   (was 2.0)
bin centres vs volume_hist                 5.0e-07     1.0e-06      49.9%
sharpen_lut vs shim                        3.2e-14     1.0e-11       0.3%
sharpen_lut vs sharpen_hist                5.0e-07     1.0e-06      49.9%
shim vs sharpen_hist (chunk)               7.0e-07     1.0e-06      69.9%
apply_lut vs minclookup                    7.1e-15     1.0e-09       0.0%
spline d=200 sub=1 vs shim                 9.6e-08     5.9e-07      16.2%
spline d=200 sub=2 vs shim                 4.3e-10     6.2e-07       0.1%
spline d=50  sub=1 vs shim                 1.3e-11     2.9e-07       0.0%
correct_field vs shim                      4.3e-06     8.0e-05       5.4%
correct_field vs binary                    4.3e-06     8.0e-05       5.4%
_sharpen[torch]  vs sharpen_volume         2.5e-06     3.2e-05       7.7%
_sharpen[legacy] vs sharpen_volume         2.5e-06     3.2e-05       7.7%
_smooth[torch]   vs spline_smooth          2.3e-08     5.4e-06       0.4%
_smooth[legacy]  vs spline_smooth          6.4e-09     5.4e-06       0.1%
shrink vs mincresample                     3.1e-02     2.0e+02       0.0%
nu_correct[torch]  chunk i1 s3             3.3e-05     1.0e-03       3.3%
nu_correct[torch]  chunk i3 s4             7.8e-05     1.0e-03       7.8%
nu_correct[legacy] chunk i1 s3             3.3e-05     1.0e-03       3.3%
nu_correct[legacy] chunk i3 s4             1.5e-04     1.0e-03      15.3%
nu_correct[torch]  vs brain_nu_ref         3.0e-03     5.0e-03      60.1%
nu_correct[legacy] vs brain_nu_ref         3.7e-03     5.0e-03      74.0%   <- §3
recovery residual, 20% RF                  8.8e-03     1.0e-02      85.0%   <- §1, §3
recovery residual, 40% RF                  1.0e-02     2.1e-02      48.0%
recovery torch/legacy, 20%                 1.3e-04     1.0e-03      13.0%
recovery torch/legacy, 40%                 8.7e-04     1.0e-03      87.0%   <- §3
recovery torch/nu_correct, 20%             1.5e-04     1.0e-03      15.0%
recovery torch/nu_correct, 40%             2.1e-04     1.0e-03      21.0%
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
- N3's loop amplifies: implementations agreeing to 1.4e-7 after one iteration
  disagree by 5e-4 after ten. No end-to-end N3 comparison means much past
  three digits — including running the same code on a GPU.
- The legacy passes every intermediate through a slice-scaled MINC file, so
  the port cannot reach the legacy suite's own 1e-4. See `CLAUDE.md`.

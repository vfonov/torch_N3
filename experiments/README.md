# Simulated bias-field recovery

A **random** smooth field is planted on colin27, Gaussian noise is added at a
stated SNR, `nu_estimate` is run, and the proportion of the field recovered is
measured. This is repeated over many seeds, so every configuration yields a
distribution rather than a single value.

```bash
python3 -m experiments.recovery --seeds 50 --verbose      # hours; resumable
python3 -m experiments.summarize --group solver snr       # read the rows
```

Nothing here is a test. `pytest.ini` collects `tests/` only, and the only part
of this directory covered automatically is the generators, from
`tests/test_simulation.py`.

## Motivation

`tests/test_field_recovery.py` and `tests/tables.py` plant one analytic field on
one volume with no noise. Every value they publish is a single draw, and
CLAUDE.md states the consequence: if a result comes to depend on a particular
cell, the correct remedy is to widen `LAMBDAS` and assert the property being
claimed. This is the statistical form of the same experiment: random fields,
added noise, and an interquartile range against which a difference can be read.

The score is deliberately the one `tests/tables.py` publishes, so the two can be
read together.

## The data

`experiments/data/` holds the three volumes, none of them checked in
(`.gitignore:3` excludes `*.mnc`):

| file | what |
|---|---|
| `colin27_t1_tal_lin.mnc` | the MRI, 181x217x181 at 1 mm, uint16 |
| `colin27_t1_tal_lin_headmask.mnc` | the head mask, 4,006,446 voxels — the default |
| `colin27_t1_tal_lin_mask.mnc` | the brain mask, for `--mask` |

From <https://packages.bic.mni.mcgill.ca/mni-models/colin27/mni_colin27_1998_minc2.zip>;
unzip it and move the three files here. They are already MINC2, so `load_volume`
reads them without `mincconvert`.

colin27 is an average of 27 scans of one subject and therefore carries very
little noise of its own, which makes it a suitable base for noise of a known
magnitude.

## Measured quantities

**The planted field** (`simulation.random_bias_field`) is a sum of
`--field-terms` (8) cosine waves in world coordinates, with directions uniform
on the sphere, wavelengths no shorter than `--field-scale` mm, random phases,
and amplitudes falling off with frequency; then rescaled to exactly
`--amplitude` log peak-to-peak inside the mask and normalised to mean 1. The
wavelengths are in mm rather than in normalised axes, so a trial's difficulty is
a property of the field rather than of the volume's size. The field is
deliberately not a B-spline: N3's basis must approximate it rather than
reproduce it.

`--field-scale` defaults to **400 mm**, the scale a receive coil's sensitivity
varies on, and about the smoothness of `tests.inputs.synthetic_bias_field`
(`cos(0.9u)`, ~620 mm across this volume). It was calibrated against the `floor`
column below: at 400 mm a planted field costs the basis 0.002-0.005% at 75 mm
knots, which is negligible, so the sweep measures the estimation rather than the
basis. At shorter scales this ceases to hold: an early probe at about 300 mm in
normalised axes left 2.3% that no estimator could have recovered.

**The noise** is `clamp(clean * planted + sigma*N(0,1), min=0)` — field first,
then noise. `sigma = mean(clean[mask]) / SNR`, taken from the clean, *unbiased*
volume so SNR 20 means the same thing at every amplitude (sigma = 12,135 here;
6,068 at SNR 40). `--snr inf` is the noiseless control, in which the field
remains random per seed. The field and the noise of one trial come from two
independent generators, both derived from the seed, so either can be reproduced
alone.

**The score** is `tests/tables.py`'s:

```
ratio = recovered / baseline / planted        # masked, flattened
ratio = ratio / ratio.mean()
unexplained_pct = 100 * ratio.std(unbiased=False)
```

`baseline` is the same configuration's answer on the clean, unbiased volume,
divided out because colin27 is not perfectly uniform and N3 removes that too. It
is computed once per `(solver, protocol, distance, lam)`, which fixes the loop
order, and not per noise realisation, so noise appears as error. That is the
purpose of the experiment.

**Two regions are scored, and the difference between them is substantial.** The
estimation runs over the head mask, as does the primary score, but a head mask
includes scalp, skull and neck, where the field is least well determined and
where the correction is not applied in practice. Every trial is therefore scored
twice, over the estimation mask and over the brain (`--brain-mask`, the `_brain`
columns). At 75 mm on colin27 one field scores **2.11 % over the head and
0.96 % over the brain**, and `tests.inputs.synthetic_bias_field`'s analytic
field divides in the same proportion (1.94 % / 0.80 %), so the effect is
attributable to the region rather than to the random generator. The brain
columns measure how well a brain is corrected; the mask columns measure how well
the estimation problem was solved as posed.

## The columns

| column | what |
|---|---|
| `seed`, `amplitude`, `snr`, `method`, `backend`, `solver`, `protocol`, `distance`, `lam`, `penalty`, `sample_size`, `max_iterations`, `parzen_sigma`, `shrink`, `device` | the key: every parameter that changes the answer. A parameter omitted from it causes a sweep over that parameter to be skipped without any diagnostic, which occurred four times before the list was complete |
| `parzen_sigma` | the histogram kernel: a Gaussian Parzen window this many bin widths wide, or **empty for N3's own linear split**, which is what every row written before the column existed ran. `--method n3` only; see "Gaussian Parzen window" in the top-level README |
| `denoise` | `1` if the volume was prefiltered by the non-local-means pass of `blocks/denoise.py` before the field was estimated, **empty if it was not** — which is what every row written before the column existed ran. Applies to every method but `oracle`, which reads no intensities and so cannot be affected by filtering them. See "Non-local means" in the top-level README |
| `loss` | the loss the descent reached; empty for `n3` |
| `unexplained_pct` | the score above, over the estimation mask. Lower is better |
| `rms_log` | the same residual as RMS of `log(ratio)` about its mean |
| `planted_cv_pct` | the non-uniformity that was planted: what there was to remove |
| `floor_pct` | the best score this basis admits: `unexplained_pct` of fitting the planted field with the spline itself, at the same grid, spacing, weight and solver. This is the `oracle` method's score, carried on every row so that any trial can be read against its own ceiling |
| `unexplained_brain_pct`, `rms_log_brain`, `planted_cv_brain_pct`, `floor_brain_pct` | the same four over the brain mask instead |
| `iterations` | the number the run required (counted from `nu_estimate(verbose=True)`) |
| `seconds`, `baseline_seconds` | wall time of the estimate |
| `backend`, `field_scale`, `field_terms`, `timestamp`, `git_commit`, `torch_version`, `host` | provenance, on every row, since rows from several runs and machines accumulate in one file |

`unexplained_pct / planted_cv_pct` is the fraction not removed; `floor_pct`
gives the part of it the basis could not have represented.

## Methods

`--method` selects the estimator. `method` is a key column, so results from
different estimators are never pooled.

- **`n3`** (default) — `pipeline.nu_estimate`, the shipped alternating
  iteration. `--protocol`, `--lambda` and the solver apply.
- **`hoyer`**, **`tightness`** — `optimize.nu_optimize`, gradient descent on a
  stated sharpness objective over the *same* B-spline field. `--protocol` does
  not apply (there is no stopping rule to choose); `--penalty` replaces
  `--lambda` and is on an unrelated scale; `--sample-size` and
  `--max-iterations` control the descent.
- **`oracle`** — not an estimator: it is given the field that was planted and
  only fits it with the same penalized spline the other two arrive at, on the
  same grid, at the same `--distance`, `--lambda` and solver. It reads no voxel
  intensity, so no method that must infer the field from the data can score
  better. See "The ceiling" below.

The descent's voxel subset is drawn once per cell with a fixed seed
(`recovery.OPTIMIZE_SEED`) rather than from the trial's seed. The baseline is
shared by every trial in a cell, so the two must see the same voxels; otherwise
the ratio measures the subsets rather than the fields. The trial's seed still
governs the planted field and the noise, which is what is being swept.

### Comparison on colin27, 50 seeds

`experiments/results/recovery.csv` holds 4,050 trials: 3,600 `n3` (4 solvers ×
2 protocols) and 450 `hoyer` at `--penalty 1e-3 --distance 75`, on **identical
planted fields and noise realisations** — those come from the trial's seed, so
the two methods answer the same 450 questions.

Matched configuration (`normal`, `fixed30`), median unexplained non-uniformity
over the brain, n = 50 per cell:

| planted | SNR | `n3` | `hoyer` |
|---|---|---|---|
| 20 % | ∞ | 1.03 % | **0.28 %** |
| 20 % | 40 | 1.52 % | **0.34 %** |
| 20 % | 20 | 3.43 % | **1.38 %** |
| 40 % | ∞ | 2.31 % | **0.50 %** |
| 40 % | 40 | 2.83 % | **0.54 %** |
| 40 % | 20 | 4.05 % | **1.44 %** |
| 80 % | ∞ | 5.09 % | **1.18 %** |
| 80 % | 40 | 5.38 % | **1.19 %** |
| 80 % | 20 | 5.83 % | **1.76 %** |

`hoyer` is better in every cell by a factor of 2.5–5, and no interquartile
ranges overlap. The difference increases with amplitude: N3 degrades steeply as
the planted field grows (1.03 → 5.09 % at SNR ∞) where `hoyer` degrades only
slightly (0.28 → 1.18 %).

Two qualifications apply to that table:

- **`hoyer` has a failure tail that N3 does not.** It is worse on 17 of 450
  matched trials, all at 40–80 % amplitude, and its worst trial (seed 9, 80 %)
  reaches 9.3 % against N3's 6.9 %. N3's spread is narrow and its worst case
  bounded, whereas `hoyer` is better typically and occasionally worse. A mean
  well above the median in the 80 % rows is that tail.
- **79 of 450 trials reached the 400-iteration cap**, all in the harder cells,
  so those values are lower bounds on what the objective would reach rather than
  its optimum.

### The ceiling: the best score the basis admits

`--method oracle` is given the field that was planted and only fits it with the
same penalized spline, on the same grid, at the same `--distance`, `--lambda`,
solver and `--shrink`. It reads no voxel intensity and is therefore not an
estimator: it is the **best score attainable at that configuration**, and the
difference between it and a real method is estimation error with the
representation error removed.

1,800 trials (4 solvers × 450), median over the brain, beside the same cells as
above:

| planted | ceiling | `hoyer` | `n3` | uncorrected |
|---|---|---|---|---|
| 20 % | **0.0023 %** | 0.28–1.38 % | 1.03–3.43 % | 3.50 % |
| 40 % | **0.0049 %** | 0.50–1.44 % | 2.31–4.05 % | 6.98 % |
| 80 % | **0.0143 %** | 1.18–1.76 % | 5.09–5.83 % | 14.03 % |

**The basis is not the limiting factor.** Per matched trial, N3 is a median
**550×** above the ceiling and `hoyer` **136×**; the closest either comes is 51×
and 20× respectively, and neither reaches it on any of the 450 trials.
Essentially none of the residual anywhere in this experiment is the spline's
inability to represent the field; it is all estimation error. This also confirms
the `--field-scale 400` calibration: the sweep measures the quantity it was
designed to measure.

The oracle has two properties by construction, and both are reproduced by the
data, which checks the harness rather than establishing a result:

- **It does not depend on the SNR**: identical to five decimal places across
  `inf`/40/20 in every cell. It never reads a voxel, so noise reaches it only
  through the estimation mask (`data > background`), which at these values of
  sigma moves a handful of voxels and changes the score by about 1e-5 relative.
- **It does not depend on the solver**: `normal`, `qr`, `dr` and `blocked` give
  the same median and the same maximum to five decimals. Representation error is
  a property of the basis, and four solvers of the same objective agree on it.

It also costs **0.021 s**, against 0.69 s for the least expensive N3
configuration: one spline fit is about 3 % of a 30-iteration run.

An oracle row's `unexplained_pct` reproduces its own `floor_pct` column, which
is the same quantity computed on the clean volume: exactly at SNR ∞, and within
2.3e-4 relative at worst over all 1,800 rows. The residual difference is the
mask effect described above, and this confirms that the baseline division is an
identity for this method (`tests/test_simulation.py` asserts the reason: a unit
field fits back to unity, because cubic B-splines are a partition of unity and a
constant has zero bending energy, so the penalty cannot pull the fit off it at
any `lam`).

### Cases in which correction increases non-uniformity

Reading the `n3` column against `planted_cv`, which the figure draws as the
dashed line: at **20 % planted, SNR 20, over the head mask, N3's median residual
is 5.35 % against the 4.23 % initially present, or 1.26× the uncorrected
volume.** Over the brain the same cell is neutral (0.98×). This is the only cell
of the nine in which it occurs, and it arises from the combination of a weak
field and heavy noise: there is little signal to recover and a large source of
error. In every other cell N3 removes 40–70 % of what was planted (head) or
60–70 % (brain). The whole distribution lies above the line, not only the
median, which a table of medians states and a violin plot displays directly.

### Run time

Per estimate on the GPU, median, same volume and settings:

| | seconds | iterations | per iteration |
|---|---|---|---|
| `n3` `normal`/`fixed30` | 0.69 | 30 | 22.9 ms |
| `n3` `normal`/`default` | 1.19 | ~50 | — |
| `n3` `blocked` | 1.74 | 30 | — |
| `n3` `qr` / `dr` | 2.58 / 2.62 | 30 | — |
| `hoyer` (cap 400) | 2.48 | 197 | **12.6 ms** |

`hoyer` is 3.6× slower than the least expensive N3 configuration but comparable
to `qr` and `dr`, which are the configurations used for cross-platform
reproducibility. Per iteration it costs **half of N3**: a gather over 16 k
samples, a soft histogram and a backward pass, against a full-volume histogram,
a deconvolution and a complete spline solve. It requires more iterations.

**Those iterations are not necessary.** Capping the descent (10 seeds ×
3 amplitudes × 3 SNRs = 90 trials per budget, brain, median):

| budget | seconds | median | IQR | mean |
|---|---|---|---|---|
| 20 | **0.26** | 1.43 % | 0.92–2.56 | 2.89 % |
| 30 | 0.36 | 1.53 % | 0.97–2.90 | 3.01 % |
| 50 | 0.56 | 1.69 % | 1.20–2.33 | 2.67 % |
| 100 | 1.26 | 1.06 % | 0.49–1.72 | 2.00 % |
| 200 | 2.59 | 0.87 % | 0.38–1.56 | 1.68 % |
| 400 | 2.49 | 0.87 % | 0.37–1.56 | 1.45 % |
| `n3`, 30 iterations | 0.68 | 3.57 % | 2.06–5.08 | 3.58 % |

At **20 iterations `hoyer` is both 2.6× faster than N3 and 2.5× more accurate**
(0.26 s, 1.43 % against 0.68 s, 3.57 %). The 4× cost quoted above is therefore a
choice rather than a requirement: it obtains the final factor of about 1.6 in
accuracy (1.43 % → 0.87 %). Between 20 and 50 iterations the curve is flat to
slightly worse and the interquartile ranges overlap; the substantive improvement
occurs after 100. Beyond 200 only the tail changes, the mean continuing to fall
from 1.68 to 1.45 % as the slow trials complete.

**Measured against N3 on `tests/tables.py`'s experiment** — `brain_nu_ref.mnc`,
the analytic planted field, the same score N3's published table reports — each
method at its own best weight, 20 % planted:

| knots | `n3` | `hoyer` | `tightness` |
|---|---|---|---|
| 200 mm | **0.13 %** | 0.28 % | diverges |
| 100 mm | **0.17 %** | 0.21 % | diverges |
| 50 mm | 0.25 % | **0.17 %** | diverges |

N3's column reproduces its published table exactly, which establishes that the
harness is faithful. `hoyer` is worse where N3 is strongest and better at 50 mm,
and the two trends are opposed: N3 degrades as the field gains freedom while
`hoyer` improves. Its usable weights are `1e-4`–`1e-3`; below about `1e-5` it
becomes unstable at fine spacings (at 50 mm and `1e-6` the field diverged to
1316 % non-uniformity).

`tightness` **does not work**, and not for lack of tuning: its loss falls while
its estimate deteriorates. See `torch_n3/optimize.py`'s docstring and
`tests/test_optimize.py::test_tightness_makes_its_own_estimate_worse_as_it_converges`.

### Prefiltering the volume (`--denoise`)

`blocks/denoise.py` filters the volume with one non-local-means pass before the
field is estimated. On a *single* analytic field it appeared barely worthwhile:
`tests/denoise.py` found it helping less than the histogram window it competes
with, and the top-level README said so. **Over 450 random fields per
configuration the conclusion reverses, and the earlier verdict was wrong.**

Median `unexplained_brain_pct`, 50 seeds per cell, `--distance 75 --lambda 1e-7
--solver normal --protocol fixed30` on the GPU, N3's own linear histogram:

| planted | SNR ∞ | SNR 40 | SNR 20 |
|---|---|---|---|
| **`n3`** 20 % | 1.030 → 1.055 (+2 %) | 1.522 → 1.073 (−29 %) | 3.428 → 1.092 (**−68 %**) |
| 40 % | 2.307 → 2.435 (+6 %) | 2.832 → 2.419 (−15 %) | 4.050 → 2.496 (−38 %) |
| 80 % | 5.094 → 5.206 (+2 %) | 5.384 → 5.194 (−4 %) | 5.835 → 5.236 (−10 %) |
| **`hoyer`** 20 % | 0.519 → 0.731 (+41 %) | 0.563 → 0.754 (+34 %) | 2.193 → 0.753 (**−66 %**) |
| 40 % | 0.513 → 0.723 (+41 %) | 0.652 → 0.725 (+11 %) | 2.179 → 0.741 (−66 %) |
| 80 % | 1.121 → 1.046 (−7 %) | 1.178 → 1.046 (−11 %) | 2.247 → 0.911 (−59 %) |

Paired per trial, which is the stronger statement — the same seed, the same
planted field, the same noise draw, filtered and not:

| | SNR ∞ | SNR 40 | SNR 20 |
|---|---|---|---|
| `n3` improved on | 15 % of trials | **88 %** | **95 %** |
| `hoyer` improved on | 37 % | 41 % | **85 %** |

**The result is not "denoising helps" but something sharper: it makes both
estimators almost indifferent to noise.** N3 at 20 % planted goes 1.055 / 1.073
/ 1.092 as the SNR falls from infinite to 20 — flat — where unfiltered it goes
1.030 / 1.522 / 3.428, a factor of 3.3. `hoyer` likewise holds 0.73 / 0.75 /
0.75 against 0.52 / 0.56 / 2.19. The filter converts a noise-sensitive estimator
into one whose error is set by the field and the basis alone.

The cost is paid where the control predicted it. With no noise to remove a
spatial filter can only take away structure the estimate was using, and both
methods are slightly worse at SNR ∞ — N3 by 2–6 %, `hoyer` by up to 41 %,
`hoyer` being the better estimator there and so having more to lose. The
break-even is between SNR 40 and ∞ for N3, and between 20 and 40 for `hoyer`.

Two further readings. Denoising helps **most at low field amplitude**: at 80 %
planted the error is dominated by what the basis can represent, and no amount of
noise removal reaches it (−10 % for N3 at SNR 20, against −68 % at 20 %
planted). And it slightly *reduces* the tail — trials scoring above 5 % fall
from 112 to 97 of 450 for N3 and from 56 to 48 for `hoyer` — so it does not buy
its median by making the hard draws worse.

`hoyer` beats N3 at every cell on the median, filtered or not, but its mean sits
far above its median (2.60 against 0.91 at 80 %/SNR 20) and its worst trial
reaches 16 %, against N3's 7.5 %: the divergence tail documented above is
unaffected by prefiltering.

Cost on the GPU, per estimate on colin27's 7.1 M voxels: N3 0.7 s → 2.5 s,
`hoyer` 2.5 s → 4.4 s. The filter itself is the ~1.8 s difference, and it runs
once per estimate at full resolution.

### Crossed with the histogram window

Both modifications suppress noise-driven variance in the same data term, so the
question neither measures alone is whether they are **substitutes or
complements**. `tests/denoise.py` asked it on one analytic field and answered
*substitutes*; over the same 50 seeds the answer is the opposite. Median
`unexplained_brain_pct`, `n3`, paired:

| planted | SNR | linear | `sigma 4` | `--denoise` | both |
|---|---|---|---|---|---|
| 20 % | ∞ | 1.030 | 1.036 | 1.055 | 1.059 |
| | 40 | 1.522 | 1.309 | 1.073 | **1.051** |
| | 20 | 3.428 | 1.912 | **1.092** | 1.136 |
| 40 % | ∞ | 2.307 | **2.148** | 2.435 | 2.183 |
| | 40 | 2.832 | 2.443 | 2.419 | **2.204** |
| | 20 | 4.050 | 3.257 | 2.496 | **2.293** |
| 80 % | ∞ | 5.094 | **4.923** | 5.206 | 4.965 |
| | 40 | 5.384 | 5.441 | 5.194 | **5.017** |
| | 20 | 5.835 | 6.815 | 5.236 | **5.095** |

The median *paired* difference of both against denoising alone is negative in
all nine cells (−0.032 to −0.333 points) and against the window alone in all six
noisy cells (−0.224 to −1.674), costing only +0.030 to +0.053 at SNR ∞. Win
rates for the pair: 96–100 % against the window alone at SNR 40 and 20, and
58–76 % against the denoiser alone at every SNR.

The median of the paired *differences* and the difference of the medians
disagree at 20 % planted / SNR 20: the pair's median (1.136) is above the
denoiser's (1.092) while its median paired difference is −0.055 with a 60 % win
rate. The paired statistic answers the question; the other compares two
different trials' medians.

Which of the two carries a cell depends on the field amplitude. At 20 % planted
and SNR 20 the denoiser does essentially all the work (3.428 → 1.092) and the
window adds little on top; at 40 % and 80 % the window is worth 0.2–0.3 points
*on a denoised volume* in every noisy cell.

At 80 % planted and SNR 20 the window alone is a net **harm** — 6.815 against
N3's own 5.835, the only such cell in the sweep — and prefiltering removes it,
the pair giving 5.095, the best in that row. The window's instability at large
field amplitude is therefore a noise effect, and the two together are more
robust than either alone.

## Protocols

`--protocol` sweeps both by default:

- **`fixed30`** — `iterations=(30,), stop=(0.0,)`. Every trial performs the same
  work, so a difference between two cells is attributable to the fit. This is
  the protocol `tests/tables.py` uses.
- **`default`** — `iterations=(50,), stop=(0.001,)`, as shipped by `nu_correct`.
  Noise moves the stopping point, so `iterations` and `seconds` become
  measurements rather than constants.

## Running the sweep

The full sweep is 50 seeds x 3 amplitudes x 3 SNRs x 4 solvers x 2 protocols =
3,600 trials. `--dry-run` counts them first.

It runs on the **GPU by default**, falling back to the CPU only on a machine
without one; `--device cpu` forces it. Measured at `--distance 75 --shrink 4` on
an RTX A6000, one 30-iteration trial:

| solver | GPU | CPU |
|---|---|---|
| `normal` | 0.7 s | 2.5 s |
| `blocked` | 1.8 s | — |
| `qr`, `dr` | 2.6 s | ~5-7 s |

The sweep takes about two hours on the GPU against three on the CPU. `device` is
one of the key columns, so rows from the two are never pooled in a summary, and
a sweep resumes only against rows from the same device. See CLAUDE.md before
comparing across them: the same backend on a GPU differs from the CPU by 1.3e-3
end to end, which is larger than most of the differences this experiment is
intended to resolve.

The sweep is **resumable**: rows are flushed as they are produced and a re-run
skips any trial already present in the file, so it can be interrupted and
restarted, or extended with further seeds, without losing or repeating work. A
configuration whose trials are all present does not recompute its baseline.

```bash
python3 -m experiments.recovery --dry-run
python3 -m experiments.recovery --seeds 50 --verbose      # or nohup ... &
python3 -m experiments.recovery --seeds 80 --verbose      # adds seeds 51-80

# the histogram-kernel sweep below: 'none' is N3's own split, whose 900 trials
# are already in the file, so this adds 2,700 and takes about 50 min
python3 -m experiments.recovery --method n3 --solver normal \
    --parzen-sigma none 1 2 4 --verbose
```

Run one sweep at a time: two concurrent sweeps share the same GPU, or the same
cores on the CPU, and neither completes sooner. A second job does not change the
scores, which are deterministic given the seed, but it does invalidate the
`seconds` column.

## Interpreting the output

```bash
python3 -m experiments.summarize --group solver snr
python3 -m experiments.summarize --group amplitude snr --metric rms_log
python3 -m experiments.summarize --group protocol solver --metric seconds
python3 -m experiments.summarize --group distance lam --csv > grid.csv

# one file now holds several experiments, so exclude those not under
# consideration; `x=` matches an empty column, which is how N3's own histogram
# is denoted
python3 -m experiments.summarize --where method=n3 solver=normal \
    device=cuda protocol=default --group amplitude snr parzen_sigma
```

Per group: `n`, mean, median, IQR, extremes, mean run time, and mean `floor`.
The interquartile range is reported rather than a standard deviation because the
spread across seeds is not symmetric; a mean well above the median indicates
that a few difficult draws determine it.

### Figures

```bash
python3 -m experiments.figures                    # all figures, into results/
python3 -m experiments.figures --figure recovery
```

Violin plots are used because every cell is 50 random fields and the quantity of
interest is the shape of those 50: whether a method's advantage is the whole
distribution moving or a few favourable draws. A bar chart of medians would
conceal both of the findings this sweep exists for, namely `hoyer`'s failure
tail and the small number of trials on which two backends differ by 18 %.

| figure | what |
|---|---|
| `results/summary.png` | the three estimators in one figure: legacy N3 (the original C++ blocks through the shim, CPU), the port at `--solver normal` on the GPU, and `hoyer`, over all nine cells, beside their wall times. The two N3 violins coincide at every cell, so the separation below them is attributable to the change of objective and not to the change of implementation |
| `results/recovery.png` | the principal figure: `n3`, `hoyer` and the `oracle` ceiling over all nine cells, head and brain, with the uncorrected level as a dashed line |
| `results/cells.png` | the same data disaggregated: **one panel per (amplitude, SNR)**, with N3's four solvers drawn separately and `hoyer` and `oracle` beside them. Brain only, one shared range across all nine |
| `results/runtime.png` | wall time per estimate, every configuration in the file |
| `results/implementation.png` | the three comparisons expected to show no effect: solver, backend/device, and the solver's effect on the ceiling |
| `results/solvers.png` | the solver comparison posed per trial: difference from `normal` (six decades, log) beside seconds (a factor of 3.8, linear) |
| `results/windows.png` | N3's linear split against the Gaussian Parzen window, one panel per cell, and the only figure here that draws **both protocols**, because the reduction the window yields depends on the length of the iteration |

`recovery.png` pools the four solvers into one `n3` violin per cell, which is
the appropriate summary but does not address two questions: whether the solvers
separate in any particular cell (they do not — nine cells, four violins each,
all indistinguishable), and how a cell's own spread compares with the difference
between methods. `cells.png` addresses both.

Two conventions apply. **Densities are estimated in log space** wherever the
axis is logarithmic: a kernel density estimate fitted in linear space and drawn
on a log axis depicts the wrong distribution, and these scores span four
decades. `implementation.png` uses **linear axes scaled to their own data**,
deliberately: those differences are parts per hundred, and on the decade axis
the other figures use they would appear as a single flat line, which would
depict the axis rather than the measurement. The spread within each violin is
read against the difference between them.

## The histogram kernel, over 450 trials each

`--parzen-sigma` replaces N3's `-parzen` — linear interpolation into two bins —
with a Gaussian Parzen window of a stated width in bin widths. The top-level
README describes it; `tests/parzen.py` measures it on **one** analytic field
with **no noise**, and this section poses the same question to 450 random fields
per window at three amplitudes and three SNRs.

Median unexplained non-uniformity over the brain, 50 seeds per cell, 75 mm
knots, `--lambda 1e-7`, `--solver normal`, on the GPU. Best in each row bold:

*30 iterations (`--protocol fixed30`):*

| planted | SNR | linear (N3) | σ 1 | σ 2 | σ 4 |
|---|---|---|---|---|---|
| 20 % | ∞ | 1.03 % | **1.02 %** | **1.02 %** | 1.04 % |
| 20 % | 40 | **1.52 %** | 1.71 % | 1.75 % | 1.31 % |
| 20 % | 20 | 3.43 % | 3.63 % | 2.92 % | **1.91 %** |
| 40 % | ∞ | 2.31 % | 2.32 % | 2.20 % | **2.15 %** |
| 40 % | 40 | 2.83 % | 2.92 % | 2.68 % | **2.44 %** |
| 40 % | 20 | 4.05 % | 4.15 % | 3.66 % | **3.26 %** |
| 80 % | ∞ | 5.09 % | 5.00 % | **4.70 %** | 4.92 % |
| 80 % | 40 | 5.38 % | 5.31 % | **5.01 %** | 5.44 % |
| 80 % | 20 | **5.83 %** | 5.86 % | 5.84 % | 6.81 % |

*The shipped protocol (`--protocol default`, 50 iterations and `-stop 0.001`):*

| planted | SNR | linear (N3) | σ 1 | σ 2 | σ 4 |
|---|---|---|---|---|---|
| 20 % | ∞ | 0.81 % | 0.80 % | 0.80 % | **0.64 %** |
| 20 % | 40 | 1.69 % | 2.12 % | 2.01 % | **1.03 %** |
| 20 % | 20 | 4.63 % | 4.84 % | 3.47 % | **1.71 %** |
| 40 % | ∞ | 1.75 % | 1.77 % | 1.57 % | **1.32 %** |
| 40 % | 40 | 2.75 % | 2.92 % | 2.40 % | **1.61 %** |
| 40 % | 20 | 4.92 % | 4.94 % | 3.58 % | **2.33 %** |
| 80 % | ∞ | 3.65 % | 3.64 % | 3.02 % | **2.88 %** |
| 80 % | 40 | 4.22 % | 4.20 % | 3.50 % | **3.18 %** |
| 80 % | 20 | 5.46 % | 5.28 % | 4.33 % | 4.34 % |

Trial by trial against the linear split on the same seed, field and noise. This
is the informative comparison, since a seed that is unfavourable for the basis
is difficult for every kernel:

| protocol | window | median ratio | better in |
|---|---|---|---|
| 30 iterations | σ 1 | 1.003 | 213/450 (47 %) |
| | σ 2 | 0.942 | 320/450 (71 %) |
| | σ 4 | 0.919 | 289/450 (64 %) |
| shipped | σ 1 | 1.014 | 180/450 (40 %) |
| | σ 2 | 0.822 | 356/450 (79 %) |
| | σ 4 | **0.664** | **425/450 (94 %)** |

Four results follow.

**The reduction tracks noise rather than amplitude.** At 20 % planted with no
noise every kernel scores 1.03 %; at SNR 20 the linear split degrades to 3.43 %
while σ 4 holds 1.91 %. This is the expected behaviour of a kernel density
estimate, being a variance reduction on the counts, and it is not observable in
`tests/parzen.py`, whose single field is noiseless. It is the reason for running
this sweep rather than relying on the analytic tables.

**σ 1 does not improve on N3's linear split.** It is worse more often than it is
better under both protocols. The linear split is already a triangular kernel of
standard deviation 0.41 bins, so a Gaussian of 1 bin is only marginally wider,
and the additional blur is incurred before the variance reduction is obtained.

**The window's benefit increases with the iteration count, which is the largest
effect measured here.** Under N3's histogram, increasing from 30 to 50
iterations makes the noisy cells worse — 20 % at SNR 20 moves from 3.43 % to
4.63 % — because the alternating iteration returns the high-variance histogram's
noise to the mapping at each pass. Under σ 4 the same cells continue to improve
(1.91 % → 1.71 %). The comparison at 30 iterations therefore understates the
effect, which is why `windows.png` draws both protocols; under the shipped
protocol the window is better on 94 % of trials.

**A wide window can also degrade the result.** At 80 % planted and 30
iterations, σ 4 is the worst kernel in two of the three cells (6.81 % against
the linear split's 5.83 % at SNR 20). A field of that amplitude moves the
histogram range sufficiently that the added blur — 4 bins, wider than the
σ = 0.064 log units that `--fwhm 0.15` instructs the deconvolution to remove —
costs more than the smoothing yields. Under the shipped protocol the ordering
reverses. **σ 2 is the only width that is not the worst of the four in any
cell** under either protocol, and is the appropriate default in the absence of a
measurement.

Iteration counts are uninformative here: essentially every trial under both
protocols runs to its cap (median 50 of 50 under the shipped `-stop`), for every
kernel. The window changes what the iteration converges to, not where it stops.
The cost is small: 1.21 s against 1.19 s per estimate at 50 iterations, about
1 %.

Two structural limitations apply. The measurement is at 75 mm knots with
`--lambda 1e-7`, which the top-level README's tables show to be
under-regularized for that spacing, and those tables also show the window
substituting for regularization, so part of this reduction is attributable to a
penalty set too low. The planted fields are also sums of three low-order
cosines, and are smoother than a real coil profile.

## The four solvers, over 450 trials

`--solver` selects how the B-spline normal equations are solved. All four are
swept at both protocols, on the same trials and the same GPU.

**On accuracy there is no difference between them.** The four score
distributions coincide, so the comparison must be made within a trial:

| against `normal` (brain, `fixed30`) | median | 90th | max | >1 % |
|---|---|---|---|---|
| `qr` | 2.0e-5 | 9.5e-5 | 3.0e-2 | 2 of 450 |
| `dr` | 2.0e-5 | 9.5e-5 | 2.9e-2 | 2 of 450 |
| `blocked` | 2.1e-5 | 9.2e-5 | 3.7e-2 | 2 of 450 |

This is the same magnitude and the same distribution as the backend and device
perturbations below, and has the same cause: both trials with a spread above 1 %
are at **20 % planted, SNR 20**, the cell nearest the histogram's rounding
boundary. No solver is systematically better either: against `normal` each is
better on 215–219 of 450 trials, about half of the 431 non-ties, which is
indistinguishable from chance.

**The four are not independent implementations, and the data shows the
grouping.** The frequency with which two solvers agree to every digit recorded:

| pair | `fixed30` | `default` |
|---|---|---|
| `qr` vs `dr` | 88 % | 68 % |
| `qr` vs `blocked` | 69 % | 44 % |
| `dr` vs `blocked` | 70 % | 43 % |
| any of those vs `normal` | **4 %** | **4–5 %** |

`qr`, `dr` and `blocked` all factorize the same stacked system
`[A; sqrt(lambda N) D]`, as CLAUDE.md states: `dr` is `qr` at its anchor weight
and `blocked` is `qr` one band at a time. `normal` forms `AtA` and squares the
condition number. The correct grouping is therefore `{normal}` against
`{qr, dr, blocked}`, and the 4 % figure is the cost of that squaring appearing
in the last digits.

**The difference in cost is substantial**, and is the only substantial
difference:

| solver | seconds (`fixed30`) | per iteration | vs `normal` |
|---|---|---|---|
| `normal` | **0.69** | 22.9 ms | — |
| `blocked` | 1.74 | 57.8 ms | 2.5× |
| `qr` | 2.58 | 85.8 ms | 3.7× |
| `dr` | 2.62 | 87.3 ms | 3.8× |

At the shipped 75 mm spacing `blocked` has one band and reduces to `qr` plus a
sort, and `dr` yields no benefit until a second `lambda` is requested, both as
CLAUDE.md predicts. In this experiment the ranking is therefore the ranking of
work performed, and `normal`'s 3.8× advantage over `dr` costs nothing measurable
in accuracy.

**What this does not measure.** CLAUDE.md's argument for `qr` is cross-platform
reproducibility: the fitted field moving 3.0e-13 between CPU and GPU instead of
2.5e-9. Only `normal` was run on the CPU here, so that claim is unaddressed by
these 3,600 trials, which establish that the solvers agree with each other on
one device rather than that they would drift equally across two. `python3 -m
tests.convergence --solver qr` tests that.

One further null result, weaker than it appears: under `default`, **no trial
stopped at a different iteration** under a different solver. However, 439 of 450
ran to the 50-iteration cap, so only 11 trials involved a stopping decision. The
result states that the stopping rule did not amplify the solver difference on
the 11 trials that exercised it, and is not a general statement.

## The two backends, over 450 trials

`--backend legacy` runs N3's original C++ blocks through the CFFI shim (CPU
only, `--solver normal` only). Run at the matched configuration with a
**torch-on-CPU control**, so the backend is isolated from the device:

| | median score, SNR ∞ / 40 / 20 | median seconds |
|---|---|---|
| `legacy` / cpu | 2.3066 / 2.8318 / 4.2358 % | 6.05 |
| `torch` / cpu | 2.3066 / 2.8317 / 4.2413 % | 2.55 |
| `torch` / cuda | — | 0.69 |

The aggregates agree to four or five significant figures. Per matched trial the
result is more informative, and quantifies what CLAUDE.md describes
qualitatively:

| relative difference in the score | median | 90th | max |
|---|---|---|---|
| backend (legacy/cpu vs torch/cpu) | 2.2e-5 | 9.8e-5 | **1.8e-1** |
| device (torch/cuda vs torch/cpu) | 2.0e-5 | 8.7e-5 | **1.7e-1** |

**Both perturbations are of the same magnitude and have the same tail.**
Typically the two implementations agree to one part in 50,000; on 4 trials of
450 they differ by more than 1 %, and on one by 18 %. This is the histogram's
rounding boundary rather than floating-point noise: a single count crossing a
bin boundary sends the iteration along a different path. The evidence that the
cause is the trial and not the perturbation is that the same trial (seed 22,
20 %, SNR 20) is the worst case for backend and device alike, and that all four
outliers are at SNR 20, where the noise places most voxels near a boundary.

The choice of backend and the choice of device are therefore the same question,
and neither is answerable trial by trial, only in distribution, which is the
purpose of this harness.

The torch backend is **2.4× faster than legacy on the same CPU**, and the GPU is
8.8× faster than legacy.

## torch_n3 against `nu_correct_cxx`, two protocols, 50 seeds each

`legacy/N3/src/N3Pipeline/nu_correct_cxx` (`PLAN.md`) is neither backend above:
it is the same C++ blocks as `legacy`, but linked into one file-based program
that runs the whole pipeline -- estimate and evaluate -- in a single process,
the in-memory equivalent of the installed `nu_correct`. `experiments.compare_cxx`
runs it as a subprocess, one MINC2 round trip per trial at `float64` storage
(no quantisation), and recovers the field it applied as `input / output`.
Matched against `torch_n3.pipeline.nu_estimate` at identical options, under
both of `nu_correct_cxx`'s protocols (`-V1.0`/`torch_n3.pipeline.V1_0` and
`-V1.1`/`torch_n3.pipeline.DEFAULTS`, `-nolegacy_rounding` on both sides --
see the docstring of `experiments/compare_cxx.py` for why): 50 seeds each,
one random field per seed, 40% planted, noiseless, scored over the head mask
and the brain within it (`results/compare_cxx.csv`).

**The summary score**, `unexplained_pct` (the same statistic `recovery.py`
uses, baseline-divided and renormalised):

| protocol | `unexplained_pct`, mask / brain (torch = cxx to 4 s.f.) | median seconds, torch / cxx |
|---|---|---|
| `v1.0` | 3.0770 / 1.6325 % | 0.52 / 7.44 |
| `v1.1` | 0.2360 / 0.1566 % | 7.39 / 72.0 |

`v1.1`'s tighter stop and Gaussian histogram leave **13× less** residual
non-uniformity than `v1.0` on this noiseless case, at the cost of running to a
median 570 iterations (range 408-776) against `v1.0`'s fixed 50 -- both
implementations picked the *same* iteration count on every one of the 50
seeds, staying within the staged cap.

**The direct comparison** (`results/compare_cxx_direct.csv`) is a pointwise
relative RMS of the *recovered field itself* -- `field / baseline`,
renormalised to mean 1 -- rather than a comparison of two separately-computed
summary statistics, so it is not diluted by the two implementations
happening to answer the same *question* similarly; it asks whether they
computed the same *field*:

| protocol | region | torch↔cxx | torch↔truth | cxx↔truth |
|---|---|---|---|---|
| `v1.0` | mask | 9.3e-6 (90th 2.3e-5, max 3.7e-5) | 3.18e-2 | 3.18e-2 |
| `v1.0` | brain | 5.4e-6 (90th 1.3e-5, max 1.9e-5) | -- | -- |
| `v1.1` | mask | **7.6e-11** (90th 1.4e-10, max 2.0e-10) | 2.42e-3 | 2.42e-3 |
| `v1.1` | brain | **4.0e-11** (90th 9.3e-11, max 1.4e-10) | -- | -- |

`torch↔truth` and `cxx↔truth` are shown only over the mask and agree with
each other to within noise at every seed, which is the direct-comparison
restatement of the summary table above: the two implementations are not
merely close to each other, they are close to each other *because* they are
both close to the same ground truth, not because they share a common bias.

**`v1.1` agreement is five orders of magnitude tighter than `v1.0`'s**, not
merely tighter. This is the opposite of what the `legacy`/`torch` backend
comparison above found, where more iterations *amplify* disagreement
(CLAUDE.md, "The iteration amplifies"): there the two backends take
different floating-point paths through the same histogram, and a difference
in the fifth decimal after one iteration is a part in a thousand after ten,
because the *stopping point itself* depends on which side of a rounding
boundary a count falls. `v1.0`'s `1e-3` stop threshold is reached quickly,
while noise from the histogram's linear-split/rounding boundaries is still
significant relative to the remaining field update. `v1.1`'s `1e-5` threshold
forces both implementations to keep iterating until the update is small on an
absolute scale, and the Gaussian Parzen window removes the bin-boundary
discontinuity that `v1.0`'s linear split is sensitive to (CLAUDE.md, "The
amplification starts at a discontinuity, not at float noise"). Both effects
push the same way: the fixed point the iteration converges to is shared and
strongly attracting, rather than a path-dependent stopping point, so running
it out further makes two independent implementations agree *more*, not less.

`cxx` is 9.7-14× slower than `torch` on the GPU, run for run: it is
single-threaded CPU C++ against a GPU tensor pipeline, not a measurement of
the algorithm. The gap narrows under `v1.1` because per-iteration GPU launch
overhead amortises better over more iterations while `cxx`'s cost is closer
to linear in iteration count throughout.

## Contents of `results/`

| file | what |
|---|---|
| `recovery.csv` | 9,900 trials: 3,600 `n3` on the GPU (4 solvers × 2 protocols × 50 seeds × 3 amplitudes × 3 SNRs), 2,700 more `n3` at `--solver normal` for the three Gaussian Parzen windows (3 × 2 protocols × 450), 900 `hoyer` (450 at `--penalty 1e-3 --max-iterations 400` plus the budget sweep), 900 on the CPU — 450 `torch` and 450 `legacy` — for the backend comparison above, and 1,800 `oracle` (4 solvers × 450) for the ceiling. Everything but those 2,700 ran under N3's own histogram, which is what an empty `parzen_sigma` means |
| `summary.png`, `recovery.png`, `cells.png`, `runtime.png`, `implementation.png`, `solvers.png`, `windows.png` | the figures above, from `python3 -m experiments.figures`. Checked in because they summarise a run of several hours, and regenerated from the CSV rather than maintained by hand |
| `pilot_grid.csv` | the 96-trial `--distance` × `--lambda` pilot for `n3` |
| `recovery_cpu_partial.csv` | 430 trials from an aborted CPU run, retained as the only CPU sample. Written before `method`/`penalty` existed, so it uses the older column set; `summarize` reads it, and `recovery` refuses to append to it |
| `compare_cxx.csv` | 200 trials (50 seeds × `torch`/`cxx` × `v1.0`/`v1.1`) for the `nu_correct_cxx` comparison above, from `python3 -m experiments.compare_cxx --seeds 50`. Its own column set; `summarize` does not read it |
| `compare_cxx_direct.csv` | 100 trials (50 seeds × `v1.0`/`v1.1`) of the same run's direct pointwise comparison, from the same command (`--out-direct`) |

## The grid search

Same driver, more values; `distance` and `lam` are already columns:

```bash
python3 -m experiments.recovery --out experiments/results/grid.csv \
    --seeds 10 --amplitude 0.2 0.4 --snr 40 20 --solver normal \
    --distance 200 100 75 50 --lambda 1e-7 1e-6 1e-5 1e-4
python3 -m experiments.summarize experiments/results/grid.csv \
    --group distance lam
```

The baseline is keyed on `(solver, protocol, distance, lam)`, so a grid pays one
extra run per cell, not per trial.

## Pilot results

`experiments/results/pilot_grid.csv` is checked in: 3 seeds × 20% planted ×
`{inf, 40}` SNR × 4 spacings × 4 weights, `normal`, `fixed30`, 96 trials in five
minutes. It is not conclusive, since 3 seeds is not a distribution, but it is
what the principal sweep is directed at, and it establishes three points.

**Noise dominates the weight.** Over the brain, pooled across spacings:

```
lam     snr      n      mean    median          IQR
1e-07   inf     12    0.9632    0.9802   0.8841-1.0125
1e-07   40      12    1.6248    1.5807   1.4965-1.7276
1e-04   inf     12    1.0996    1.0694   0.9481-1.2178
1e-04   40      12    1.2803    1.3037   1.1826-1.3307
```

Noise costs more than any `--lambda` in the grid recovers, and the best weight
depends on the noise: `1e-4` is the best of the four at SNR 40 and the worst at
SNR inf. The shipped `1e-7` is adequate in the noiseless case and is not the
appropriate choice at SNR 40. That trade-off is the reason to run the sweep
rather than cite `tests/tables.py`.

**Spacing separates the cells only weakly.** All 16 `(distance, lam)` cells lie
between 1.06% and 1.36% over the brain, and their interquartile ranges overlap
almost completely; at 3 seeds nothing here is distinguishable from seed-to-seed
variation. Any conclusion from the grid search requires the full set of seeds.

**The basis is not the limiting factor.** Every cell's `floor` is at most 0.26%
and usually below 0.05%, against scores above 1%. The sweep measures the
estimation.

## Limitations

- No conclusion should be drawn from a single cell's last digit. CLAUDE.md
  measures 30 iterations amplifying float64 rounding to 0.3% between
  algebraically identical solvers, and to 1.1e-3 end to end between backends.
  That is why an interquartile range is reported here.
- No conclusion should be drawn across devices without stating so. `device` is a
  key column: the same sweep on a GPU is a different sweep.
- No conclusion should be drawn about `sparse`. It is not offered here, because
  it does not converge (`PROBLEMS.md` §10).

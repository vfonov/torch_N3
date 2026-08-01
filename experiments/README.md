# Simulated bias-field recovery

Plant a **random** smooth field on colin27, add Gaussian noise at a stated
SNR, run `nu_estimate`, and measure how much of the field came back. Repeat
over many seeds, so that every configuration gets a *distribution* rather than
a number.

```bash
python3 -m experiments.recovery --seeds 50 --verbose      # hours; resumable
python3 -m experiments.summarize --group solver snr       # read the rows
```

Nothing here is a test. `pytest.ini` collects `tests/` only, and the one thing
in this directory that is checked automatically is the generators, from
`tests/test_simulation.py`.

## Why

`tests/test_field_recovery.py` and `tests/tables.py` plant *one* analytic
field on *one* volume with no noise. Every number they publish is a single
draw, and CLAUDE.md says what that is worth: "If you find yourself relying on
a cell, the honest fix is to widen `LAMBDAS` and assert the shape being
claimed". This is the statistical version of the same experiment — random
fields, real noise, and an IQR to read a difference against.

The score is deliberately the one `tests/tables.py` publishes, so the two can
be read together.

## The data

`experiments/data/` holds the three volumes, none of them checked in
(`.gitignore:3` excludes `*.mnc`):

| file | what |
|---|---|
| `colin27_t1_tal_lin.mnc` | the MRI, 181x217x181 at 1 mm, uint16 |
| `colin27_t1_tal_lin_headmask.mnc` | the head mask, 4,006,446 voxels — the default |
| `colin27_t1_tal_lin_mask.mnc` | the brain mask, for `--mask` |

From <https://packages.bic.mni.mcgill.ca/mni-models/colin27/mni_colin27_1998_minc2.zip>;
unzip it and move the three files here. They are MINC2 already, so
`load_volume` reads them without `mincconvert`.

colin27 is an average of 27 scans of one subject, so it starts out with very
little noise of its own — which is what makes it a good base for noise that
is *known*.

## What is measured

**The planted field** (`simulation.random_bias_field`) is a sum of
`--field-terms` (8) cosine waves in world coordinates, with directions uniform
on the sphere, wavelengths no shorter than `--field-scale` mm, random phases,
and amplitudes falling off with frequency; then rescaled to exactly
`--amplitude` log peak-to-peak inside the mask and normalised to mean 1. In mm
rather than in normalised axes, so a trial's difficulty is a property of the
field and not of the volume's size. Deliberately not a B-spline: N3's basis
should have to approximate it, not reproduce it.

`--field-scale` defaults to **400 mm**, the scale a receive coil's sensitivity
actually varies on, and about the smoothness of
`tests.inputs.synthetic_bias_field` (`cos(0.9u)`, ~620 mm across this volume).
It was calibrated against the `floor` column below: at 400 mm a planted field
costs the basis 0.002-0.005% at 75 mm knots, i.e. essentially nothing, so what
the sweep measures is the *estimation* rather than the basis. Shorten it and
that stops being true — an early probe at ~300 mm in normalised axes left 2.3%
that no estimator could have recovered.

**The noise** is `clamp(clean * planted + sigma*N(0,1), min=0)` — field first,
then noise. `sigma = mean(clean[mask]) / SNR`, taken from the clean, *unbiased*
volume so that SNR 20 means the same thing at every amplitude (sigma = 12,135
here; 6,068 at SNR 40). `--snr inf` is the noiseless control, and the field is
still random per seed. The field and the noise of one trial come from two
independent generators, both derived from the seed, so either can be
reproduced alone.

**The score** is `tests/tables.py`'s:

```
ratio = recovered / baseline / planted        # masked, flattened
ratio = ratio / ratio.mean()
unexplained_pct = 100 * ratio.std(unbiased=False)
```

`baseline` is the same configuration's answer on the clean, unbiased volume,
divided out because colin27 is not perfectly uniform to begin with and N3
removes that too. It is computed once per `(solver, protocol, distance, lam)`
— which is what fixes the loop order — and *not* per noise realisation, so
noise shows up as error. That is the point of the experiment.

**Two regions, and the difference matters.** The estimation runs over the head
mask, and so does the headline score — but a head mask takes in scalp, skull
and neck, where the field is worst determined and where nobody is going to use
the correction. Every trial is therefore scored twice, over the estimation
mask and over the brain (`--brain-mask`, the `_brain` columns). At 75 mm on
colin27 one field scores **2.11 % over the head and 0.96 % over the brain**,
and `tests.inputs.synthetic_bias_field`'s analytic field splits the same way
(1.94 % / 0.80 %) — so it is the region, not the random generator. Read the
brain columns for "how well does this correct a brain", the mask columns for
"how well was the estimation problem solved as posed".

## The columns

| column | what |
|---|---|
| `seed`, `amplitude`, `snr`, `method`, `backend`, `solver`, `protocol`, `distance`, `lam`, `penalty`, `sample_size`, `max_iterations`, `shrink`, `device` | the key: every parameter that changes the answer. Anything left out of it makes a sweep over that parameter silently skip — which happened four times before the list was complete |
| `loss` | the loss the descent reached — empty for `n3` |
| `unexplained_pct` | the score above, over the estimation mask. Lower is better |
| `rms_log` | the same residual as RMS of `log(ratio)` about its mean |
| `planted_cv_pct` | the non-uniformity that was planted — what there was to remove |
| `floor_pct` | the best this basis could have done: `unexplained_pct` of fitting the planted field with the spline itself, same grid, spacing, weight and solver |
| `unexplained_brain_pct`, `rms_log_brain`, `planted_cv_brain_pct`, `floor_brain_pct` | the same four over the brain mask instead |
| `iterations` | how many the run took (counted off `nu_estimate(verbose=True)`) |
| `seconds`, `baseline_seconds` | wall time of the estimate |
| `backend`, `field_scale`, `field_terms`, `timestamp`, `git_commit`, `torch_version`, `host` | provenance, on every row, because rows from several runs and machines end up in one file |

`unexplained_pct / planted_cv_pct` is the fraction *not* removed;
`floor_pct` says how much of it the basis was never going to get.

## Methods

`--method` chooses the estimator, and `method` is a key column so the results
never mix:

- **`n3`** (default) — `pipeline.nu_estimate`, the shipped alternating
  iteration. `--protocol`, `--lambda` and the solver apply.
- **`hoyer`**, **`tightness`** — `optimize.nu_optimize`, gradient descent on a
  stated sharpness objective over the *same* B-spline field. `--protocol` does
  not apply (there is no stopping rule to choose); `--penalty` replaces
  `--lambda` and is on a different scale entirely; `--sample-size` and
  `--max-iterations` control the descent.

The descent's voxel subset is drawn once per cell with a fixed seed
(`recovery.OPTIMIZE_SEED`), not from the trial's seed: the baseline is shared
by every trial in a cell, so the two must see the same voxels or the ratio
measures the subsets rather than the fields. The trial's seed still governs
the planted field and the noise, which is what is being swept.

### The colin27 head-to-head, 50 seeds

`experiments/results/recovery.csv` holds 4,050 trials: 3,600 `n3` (4 solvers ×
2 protocols) and 450 `hoyer` at `--penalty 1e-3 --distance 75`, on **identical
planted fields and noise realisations** — those come from the trial's seed, so
the two methods answer the same 450 questions.

Matched configuration (`normal`, `fixed30`), median unexplained
non-uniformity over the brain, n = 50 per cell:

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

`hoyer` wins every cell by 2.5–5×, and no IQR overlaps. The gap widens with
amplitude: N3 degrades steeply as the planted field grows (1.03 → 5.09 % at
SNR ∞) where `hoyer` barely does (0.28 → 1.18 %).

Three things that belong next to that table:

- **`hoyer` has a failure tail N3 does not.** It loses on 17 of 450 matched
  trials, all at 40–80 % amplitude, and its worst trial (seed 9, 80 %) reaches
  9.3 % against N3's 6.9 %. N3's spread is narrow and its worst case bounded;
  `hoyer` is better *typically* and occasionally worse. A mean well above its
  median in the 80 % rows is that tail showing.
- **79 of 450 trials hit the 400-iteration cap**, all in the harder cells, so
  those are lower bounds on what the objective would reach, not its optimum.
### Run time

Per estimate on the GPU, median, same volume and settings:

| | seconds | iterations | per iteration |
|---|---|---|---|
| `n3` `normal`/`fixed30` | 0.69 | 30 | 22.9 ms |
| `n3` `normal`/`default` | 1.19 | ~50 | — |
| `n3` `blocked` | 1.74 | 30 | — |
| `n3` `qr` / `dr` | 2.58 / 2.62 | 30 | — |
| `hoyer` (cap 400) | 2.48 | 197 | **12.6 ms** |

`hoyer` is 3.6× slower than the *cheapest* N3 configuration but level with
`qr` and `dr`, which are what you would run for cross-platform
reproducibility. Per iteration it is **half N3's cost** — a gather over 16 k
samples, a soft histogram and a backward pass, against a full-volume histogram,
a deconvolution and a whole spline solve. It simply takes more iterations.

**And it does not need them.** Capping the descent (10 seeds × 3 amplitudes ×
3 SNRs = 90 trials per budget, brain, median):

| budget | seconds | median | IQR | mean |
|---|---|---|---|---|
| 20 | **0.26** | 1.43 % | 0.92–2.56 | 2.89 % |
| 30 | 0.36 | 1.53 % | 0.97–2.90 | 3.01 % |
| 50 | 0.56 | 1.69 % | 1.20–2.33 | 2.67 % |
| 100 | 1.26 | 1.06 % | 0.49–1.72 | 2.00 % |
| 200 | 2.59 | 0.87 % | 0.38–1.56 | 1.68 % |
| 400 | 2.49 | 0.87 % | 0.37–1.56 | 1.45 % |
| `n3`, 30 iterations | 0.68 | 3.57 % | 2.06–5.08 | 3.58 % |

At **20 iterations `hoyer` is both 2.6× faster than N3 and 2.5× more
accurate** (0.26 s, 1.43 % against 0.68 s, 3.57 %). So the 4× cost quoted above
is a choice, not a requirement: it buys the last factor of ~1.6 in accuracy
(1.43 % → 0.87 %). Between 20 and 50 the curve is flat-to-slightly-worse and
the IQRs overlap; the real gains arrive after 100. Beyond 200 nothing changes
except the tail (the mean keeps falling, 1.68 → 1.45 %, as slow trials finish).

**Measured against N3 on `tests/tables.py`'s experiment** — `brain_nu_ref.mnc`,
the analytic planted field, the same score N3's published table reports — each
method at its own best weight, 20 % planted:

| knots | `n3` | `hoyer` | `tightness` |
|---|---|---|---|
| 200 mm | **0.13 %** | 0.28 % | diverges |
| 100 mm | **0.17 %** | 0.21 % | diverges |
| 50 mm | 0.25 % | **0.17 %** | diverges |

N3's column reproduces its published table exactly, so the harness is
faithful. `hoyer` loses where N3 is strong and wins at 50 mm, and the trends
run opposite: N3 degrades as the field gains freedom, `hoyer` improves. Its
useful weights are `1e-4`–`1e-3`; below ~`1e-5` it becomes unstable at fine
spacings (at 50 mm and `1e-6` the field ran away to 1316 % non-uniformity).

`tightness` **does not work**, and not for want of tuning — its loss falls
while its estimate gets worse. See `torch_n3/optimize.py`'s docstring and
`tests/test_optimize.py::test_tightness_makes_its_own_estimate_worse_as_it_converges`.

## Protocols

`--protocol` sweeps both by default:

- **`fixed30`** — `iterations=(30,), stop=(0.0,)`. Every trial does the same
  work, so a difference between two cells is the fit. This is what
  `tests/tables.py` uses.
- **`default`** — `iterations=(50,), stop=(0.001,)`, what `nu_correct` ships.
  Noise moves the stopping point, so `iterations` and `seconds` become real
  measurements rather than constants.

## Running it

The full sweep is 50 seeds x 3 amplitudes x 3 SNRs x 4 solvers x 2 protocols =
3,600 trials. `--dry-run` counts them first.

It runs on the **GPU by default**, falling back to the CPU only on a machine
without one; `--device cpu` forces it. Measured at `--distance 75 --shrink 4`
on an RTX A6000, one 30-iteration trial:

| solver | GPU | CPU |
|---|---|---|
| `normal` | 0.7 s | 2.5 s |
| `blocked` | 1.8 s | — |
| `qr`, `dr` | 2.6 s | ~5-7 s |

About two hours for the sweep on the GPU against three on the CPU. `device` is
one of the key columns, so rows from the two never mix in a summary — and a
sweep only resumes against rows from the same device. CLAUDE.md is worth
reading before comparing across them: the same backend on a GPU lands 1.3e-3
from the CPU end to end, which is larger than most differences this experiment
is looking for.

It is **resumable**: rows are flushed as they are produced and a re-run skips
any trial already in the file, so it can be killed and restarted, or extended
with more seeds, without losing or repeating work. A configuration whose
trials are all present does not even recompute its baseline.

```bash
python3 -m experiments.recovery --dry-run
python3 -m experiments.recovery --seeds 50 --verbose      # or nohup ... &
python3 -m experiments.recovery --seeds 80 --verbose      # adds seeds 51-80
```

Give one sweep the machine: two side by side share the same GPU (or, on the
CPU, the same cores) and neither goes faster. A second job does not change the
scores — they are deterministic given the seed — but it does make the
`seconds` column meaningless.

## Reading it

```bash
python3 -m experiments.summarize --group solver snr
python3 -m experiments.summarize --group amplitude snr --metric rms_log
python3 -m experiments.summarize --group protocol solver --metric seconds
python3 -m experiments.summarize --group distance lam --csv > grid.csv
```

Per group: `n`, mean, median, IQR, extremes, mean run time, and mean `floor`.
The IQR rather than a standard deviation because the spread across seeds is
not symmetric — a mean well above the median means a few hard draws are
setting it.

## The two backends, over 450 trials

`--backend legacy` runs N3's original C++ blocks through the CFFI shim
(CPU only, `--solver normal` only). Run at the matched configuration with a
**torch-on-CPU control**, so the backend is isolated from the device:

| | median score, SNR ∞ / 40 / 20 | median seconds |
|---|---|---|
| `legacy` / cpu | 2.3066 / 2.8318 / 4.2358 % | 6.05 |
| `torch` / cpu | 2.3066 / 2.8317 / 4.2413 % | 2.55 |
| `torch` / cuda | — | 0.69 |

The aggregates agree to four or five significant figures. Per *matched
trial* the picture is more interesting, and it quantifies what CLAUDE.md
describes qualitatively:

| relative difference in the score | median | 90th | max |
|---|---|---|---|
| backend (legacy/cpu vs torch/cpu) | 2.2e-5 | 9.8e-5 | **1.8e-1** |
| device (torch/cuda vs torch/cpu) | 2.0e-5 | 8.7e-5 | **1.7e-1** |

**Both perturbations are the same size, and both have the same tail.**
Typically the two implementations agree to one part in 50,000; on 4 trials
of 450 they differ by more than 1 %, and on one by 18 %. That is the
histogram knife-edge, not float noise: a single count crossing a bin
boundary sends the iteration down a different path. The evidence that it is
the trial and not the perturbation is that the *same* trial (seed 22, 20 %,
SNR 20) is the worst case for backend and device alike, and all four
outliers are at SNR 20, where the noise puts most voxels near a boundary.

So "which backend" and "which device" are the same question, and neither is
answerable trial by trial — only in distribution, which is what this
harness is for.

The torch backend is **2.4× faster than legacy on the same CPU**, and the
GPU is 8.8× faster than legacy.

## What is in `results/`

| file | what |
|---|---|
| `recovery.csv` | 5,400 trials: 3,600 `n3` on the GPU (4 solvers × 2 protocols × 50 seeds × 3 amplitudes × 3 SNRs), 900 `hoyer` (450 at `--penalty 1e-3 --max-iterations 400` plus the budget sweep), and 900 on the CPU — 450 `torch` and 450 `legacy` — for the backend comparison above |
| `pilot_grid.csv` | the 96-trial `--distance` × `--lambda` pilot for `n3` |
| `recovery_cpu_partial.csv` | 430 trials from an aborted CPU run, kept as the only CPU sample. Written before `method`/`penalty` existed, so it is in the older column set — `summarize` reads it, `recovery` will refuse to append to it |

## The grid search

Same driver, more values; `distance` and `lam` are already columns:

```bash
python3 -m experiments.recovery --out experiments/results/grid.csv \
    --seeds 10 --amplitude 0.2 0.4 --snr 40 20 --solver normal \
    --distance 200 100 75 50 --lambda 1e-7 1e-6 1e-5 1e-4
python3 -m experiments.summarize experiments/results/grid.csv \
    --group distance lam
```

The baseline is keyed on `(solver, protocol, distance, lam)`, so a grid pays
one extra run per cell, not per trial.

## What the pilot said

`experiments/results/pilot_grid.csv` is checked in: 3 seeds x 20% planted x
`{inf, 40}` SNR x 4 spacings x 4 weights, `normal`, `fixed30`, 96 trials in
five minutes. It is not an answer — 3 seeds is not a distribution — but it is
what the headline sweep is aimed at, and it already says three things.

**Noise dominates the weight.** Over the brain, pooled across spacings:

```
lam     snr      n      mean    median          IQR
1e-07   inf     12    0.9632    0.9802   0.8841-1.0125
1e-07   40      12    1.6248    1.5807   1.4965-1.7276
1e-04   inf     12    1.0996    1.0694   0.9481-1.2178
1e-04   40      12    1.2803    1.3037   1.1826-1.3307
```

Noise costs more than any `--lambda` in the grid recovers, and the best weight
*depends on the noise*: `1e-4` is the best of the four at SNR 40 and the worst
at SNR inf. The shipped `1e-7` is fine noiseless and is not what you would
choose at SNR 40. That trade-off is the reason to run the sweep rather than
quote `tests/tables.py`.

**Spacing barely separates.** All 16 `(distance, lam)` cells sit between 1.06%
and 1.36% over the brain, and their IQRs overlap almost completely — at 3
seeds, nothing here is distinguishable from seed-to-seed variation. Whatever
the grid search concludes will need the seeds to say it.

**The basis is not the limit.** Every cell's `floor` is at most 0.26% and
usually under 0.05%, against scores above 1%. What the sweep measures is the
estimation.

## What not to conclude

- Not from one cell's last digit. CLAUDE.md measures 30 iterations amplifying
  float64 rounding to 0.3% between algebraically identical solvers, and to
  1.1e-3 end to end between backends. That is exactly why this reports an IQR.
- Not across devices without saying so. `device` is a key column for a reason:
  the same sweep on a GPU is a different sweep.
- Not about `sparse`. It is not offered here — it does not converge
  (`PROBLEMS.md` §10).

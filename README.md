# torch_n3

A reimplementation of **N3** — Non-parametric Non-uniform intensity Normalization
(Sled, Zijdenbos & Evans, 1998) — the standard first step for removing the smooth
multiplicative "bias field" that MRI scanners leave across a volume.

The original is a set of Perl scripts driving a dozen C++ programs
(`legacy/N3/`, reference only). This is the same algorithm as one readable PyTorch
package, block by block, with each block checked against the program it replaces.

---

## Quick start

There is no install step and nothing to compile: run everything from the repository
root. (The CFFI extension under `torch_n3/_legacy/` is only needed to run the tests
or `--backend legacy`; see [Building and testing](#building-and-testing).)

Correct a volume:

```bash
python3 -m torch_n3 brain.mnc corrected.mnc --mask brain_mask.mnc
```

That is the equivalent of `nu_correct brain.mnc corrected.mnc -mask brain_mask.mnc`,
and it uses the same defaults. On the 91×109×91 test brain it takes about 0.4 s.

To watch it converge, and to keep the field it found:

```bash
python3 -m torch_n3 brain.mnc corrected.mnc \
    --mask brain_mask.mnc --field field.mnc --verbose
```

```
iteration 0: field change 0.009601
iteration 1: field change 0.008428
...
iteration 30: field change 0.000971
```

The volumes in `legacy/N3/testing/` work as-is, gzipped:

```bash
python3 -m torch_n3 legacy/N3/testing/chunk.mnc.gz out.mnc \
    --mask legacy/N3/testing/chunk_mask.mnc.gz
```

### The mask matters

N3 works from the intensity histogram, so anything you leave in the mask shapes the
answer. Pass `--mask` whenever you have a brain mask — background voxels contribute
noise and nothing else. Without one, the estimation keeps everything above an
intensity of 1, which is rarely what you want.

`--evaluation-mask` is a separate thing: it says where the fitted field may be used
as-is before being smoothly extended outwards over the rest of the volume. Left
alone, one is derived from the data with an Otsu threshold, exactly as
`nu_evaluate` does.

### Options

All of the protocol options mirror `nu_correct`'s and default to its values:

| Option | Default | What it does |
|---|---|---|
| `--distance` | 200 (mm) | B-spline knot spacing. **The main knob**: smaller means a field that can wiggle more. |
| `--fwhm` | 0.15 | How much blur to assume the field put into the histogram, in log-intensity units. |
| `--noise` | 0.01 | Wiener constant of the deconvolution. Raise it if the sharpening looks unstable. |
| `--bins` | 200 | Histogram bins. |
| `--shrink` | 4 | Estimation runs on a grid this many times coarser. The field is a spline, so the output is still full resolution. |
| `--lambda` | 1e-7 | Bending-energy penalty on the spline fit. |
| `--iterations` | 50 | Iteration budget, one number per stopping stage. |
| `--stop` | 0.001 | Stop when the field moves less than this, one per stage. |
| `--field-floor` | 0.1 | Smallest field value allowed, so the division stays sane. |
| `--device` | cpu | Any torch device, e.g. `cuda`. See [How close is it?](#how-close-is-it) first. |
| `--backend` | torch | `legacy` runs the original C++ for every block instead. |

Staged stopping works as in N3 — `--iterations 10 20 --stop 0.001 0.005` means "stop
at 0.001, but after iteration 10 accept 0.005 as well".

---

## From Python

The command line is a thin wrapper; the library is the interesting part.

```python
from torch_n3.volume import load_volume, save_volume
from torch_n3.pipeline import nu_estimate, nu_evaluate, evaluate_field

volume = load_volume("brain.mnc")
mask = load_volume("brain_mask.mnc")

field = nu_estimate(volume, mask=mask)        # the bias field, as a spline
corrected = nu_evaluate(volume, field, mask=mask)

save_volume("corrected.mnc", corrected, like="brain.mnc")
```

`nu_correct(volume, mask=mask)` does both steps in one call.

`nu_estimate` returns a fitted `BSplineField` — the in-memory stand-in for N3's `.imp`
mapping file. It is compact -- a few dozen coefficients at the default knot spacing --
and can be evaluated on any grid in the same world space, which is how the estimation
gets away with running on a coarse grid:

```python
field.coefficients          # (80,) for this volume
field.grid.shape            # (24, 28, 24) -- the shrunken estimation grid
field.evaluate_on(volume)   # ...but sampled at full resolution
```

Use `evaluate_field(volume, field, mask=mask)` rather than `evaluate_on` when you want
the field *as the correction applies it*: masked, extended outwards, and floored.

### Volumes

`Volume` is a tensor plus the geometry N3 needs, always in standard order —
C-ordered, axis 0 slowest, positive steps:

```python
volume.data       # float64 tensor, shape (nz, ny, nx)
volume.step       # voxel size along each axis   (these two stay in numpy: they
volume.start      # world coordinate of (0,0,0)   describe the grid, and don't
                  #                               travel to a device with it)

volume.like(new_data)          # same grid, different values
volume.to("cuda")              # same volume, data on another device
volume.shrink(4)               # the coarse estimation grid
other.resample_like(volume)    # nearest-neighbour onto this grid
```

Everything downstream stays on whatever device `volume.data` is on.

`load_volume` handles gzipped and MINC1 files by converting them with `mincconvert`
first — `minc2_simple` itself reads MINC2 (HDF5) only.

---

## How it works

Everything happens in log-intensity space, where a multiplicative field becomes an
additive one. Each iteration:

1. **Sharpen.** Histogram the masked volume, deconvolve that histogram with a Gaussian
   using a Wiener filter, and map every voxel to `E[u | v]` under that kernel. This is
   the guess at what the volume would look like with no field blurring its histogram.
2. **Attribute the difference.** Whatever the sharpening changed is, by assumption,
   the field.
3. **Keep only the smooth part.** Fit a cubic tensor B-spline with a bending-energy
   penalty, so only variation on the scale of `--distance` survives.

Repeat until the field stops moving, exponentiate, and divide. Step 1 is
`blocks/histogram.py` and `blocks/sharpen.py`, step 3 is `blocks/spline.py`, and
`blocks/field.py` is what makes the fitted field usable outside the mask before the
division.

Read `torch_n3/pipeline.py` alongside `legacy/N3/src/NUcorrect/nu_estimate_np_and_em.in`
— it is laid out to follow the original step by step.

### Layout

| Module | Role |
|---|---|
| `torch_n3/pipeline.py` | N3 itself: `nu_estimate`, `nu_evaluate`, `nu_correct`. |
| `torch_n3/blocks/histogram.py` | The masked histogram, `volume_hist`. |
| `torch_n3/blocks/sharpen.py` | The sharpened mapping, `sharpen_hist` — the mathematical core. |
| `torch_n3/blocks/spline.py` | The smooth field fit, `spline_smooth`. |
| `torch_n3/blocks/field.py` | The field extension past the mask, `correct_field`. |
| `torch_n3/volume.py` | MINC I/O, geometry, shrinking and resampling. |
| `torch_n3/minc_tools.py` | The two MINC utilities on the critical path: continuous lookup, and the Otsu threshold. |
| `torch_n3/backends/legacy.py` | The same blocks, as the original C++. |
| `torch_n3/_legacy/` | The CFFI shim that compiles those blocks out of `legacy/N3/src`. |
| `torch_n3/cli.py` | The command line. |

The legacy backend is deliberate: it makes the original code a numerical oracle, so each
PyTorch block has something exact to be tested against. `torch_n3.backends.resolve` picks
between the two and the pipeline runs on either — which is all `--backend legacy` does.

---

## Building and testing

The CFFI extension is built in place and is not checked in:

```bash
python3 torch_n3/_legacy/build_legacy.py
```

It compiles `Spline.cc`, `TBSpline.cc`, `DHistogram.cc`, `WHistogram.cc`,
`sharpen_hist.cc` and `correctField.cc` straight out of `legacy/N3/src` — nothing is
copied or modified — and links against the EBTKS and LAPACK that ship with the
installed MINC toolkit. Only the tests and `--backend legacy` need it.

```bash
python3 -m pytest              # the whole suite, about 21 s
python3 -m pytest tests/test_spline.py         # one block
python3 -m pytest -k "legacy"                  # everything, on the C++ backend
```

The tests come in two kinds. The cases from `legacy/N3/testing/CMakeLists.txt` are
re-expressed as comparisons: run the installed N3 program, run the Python, require
agreement — those need the MINC toolkit on `PATH`. The per-block tests
(`test_histogram.py`, `test_sharpen.py`, `test_spline.py`, `test_field.py`) compare
the PyTorch block against the same code compiled into the CFFI shim.

### How close is it?

Block by block, against the original C++:

| Block | Agreement |
|---|---|
| histogram range, plain histogram, Otsu threshold, resampling | exact |
| Parzen histogram | 5e-11 over 130k samples — summation order |
| sharpened lookup table | 4e-13 |
| continuous lookup, vs `minclookup` | 5e-14 |
| B-spline fit, as a fitted field | 1e-6 relative |
| field extension, vs `correct_field` | 5e-6 relative |

The last two are loose for reasons that are not going away. The normal equations N3
solves have a condition number around 1e13 at the default 200 mm knot spacing — the
knots are further apart than the volume is wide — so the *coefficients* are barely
determined and any two solvers disagree about them; the fitted field, which is what
gets used, is fine. And `correct_field` relaxes in raster order in `float`, which is
inherently sequential; the port sweeps the two checkerboard colours in turn instead,
the same iteration reordered so that it vectorises.

### Does it actually remove a bias field?

`tests/test_field_recovery.py` plants one and asks for it back. A smooth
multiplicative field of a set amplitude goes onto `brain_nu_ref.mnc.gz` — which
has already been through `nu_correct`, so it is close to uniform to begin with —
the result is written out as `brain_nu_artificial.mnc`, and every implementation
available is asked to correct it: the PyTorch blocks, the same pipeline driving
the original C++ blocks, and the installed `nu_correct` itself if it is on
`PATH` (those cases skip if it is not).

| Planted field | Non-uniformity planted | left by `torch` | by `legacy` | by `nu_correct` |
|---|---|---|---|---|
| 20% (`exp(0.2)` peak-to-peak) | 4.14% | 0.876% | 0.876% | 0.878% |
| 40% | 8.28% | 0.999% | 1.068% | 1.000% |

Two things to read off that. N3 recovers most but not all of a field — about
0.9% of non-uniformity survives here — and that is a property of the algorithm,
not of this code: the original leaves the same amount. And all three agree about
*which* field is there to between 2e-5 and 9e-4 RMS, which is the comparison an
implementation can actually be held to.

End to end, correcting `brain.mnc.gz` lands **3.0e-3 relative RMS** from
`brain_nu_ref.mnc.gz`, where the legacy suite asks for 1e-4. Two things account for
that, and neither is an error in the port.

**Storage.** Legacy N3 passes every intermediate volume between programs as a MINC
file: 12-bit before the mask is applied and 16-bit after, scaled slice by slice on
write and rescaled onto a single global grid on read. `torch_n3` keeps float64
throughout, which is more accurate but not identical. Reproducing the reference voxel
for voxel would mean modelling MINC's storage rather than N3; see `PLAN.md`.

**Amplification.** N3's loop feeds its output back into itself, and it magnifies small
differences: the PyTorch and C++ backends agree to 1.4e-7 after one iteration and to
only 5e-4 after ten. So no end-to-end number here is meaningful past three digits —
running the same code on a GPU moves it by as much (3.4e-3), because the reductions
happen in a different order. That is worth knowing before reading too much into a
comparison of two N3 outputs, whoever produced them.

## Requirements

Python 3.12, `numpy`, `cffi`, `minc2_simple`, and the MINC toolkit on `PATH` (for
`mincconvert`, and for the legacy programs the tests compare against). All already
installed here — nothing needs fetching.

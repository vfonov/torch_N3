# torch_n3

A reimplementation of **N3** — Non-parametric Non-uniform intensity Normalization
(Sled, Zijdenbos & Evans, 1998) — the standard first step for removing the smooth
multiplicative "bias field" that MRI scanners leave across a volume.

The original is a set of Perl scripts driving a dozen C++ programs
(`legacy/N3/`, reference only). This is the same algorithm as one readable Python
package, block by block, with each block checked against the program it replaces.

---

## Quick start

There is no install step: run everything from the repository root, and build the
extension once first (see [Building and testing](#building-and-testing)).

Correct a volume:

```bash
python3 -m torch_n3 brain.mnc corrected.mnc --mask brain_mask.mnc
```

That is the equivalent of `nu_correct brain.mnc corrected.mnc -mask brain_mask.mnc`,
and it uses the same defaults. On the 91×109×91 test brain it takes about 1.5 s.

To watch it converge, and to keep the field it found:

```bash
python3 -m torch_n3 brain.mnc corrected.mnc \
    --mask brain_mask.mnc --field field.mnc --verbose
```

```
iteration 0: field change 0.009601
iteration 1: field change 0.008413
...
iteration 30: field change 0.000973
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

`Volume` is a numpy array plus the geometry N3 needs, always in standard order —
C-ordered, axis 0 slowest, positive steps:

```python
volume.data       # float64 ndarray, shape (nz, ny, nx)
volume.step       # voxel size along each axis
volume.start      # world coordinate of voxel (0, 0, 0)

volume.like(new_data)          # same grid, different values
volume.shrink(4)               # the coarse estimation grid
other.resample_like(volume)    # nearest-neighbour onto this grid
```

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

Repeat until the field stops moving, exponentiate, and divide.

Read `torch_n3/pipeline.py` alongside `legacy/N3/src/NUcorrect/nu_estimate_np_and_em.in`
— it is laid out to follow the original step by step.

### Layout

| Module | Role |
|---|---|
| `torch_n3/pipeline.py` | N3 itself: `nu_estimate`, `nu_evaluate`, `nu_correct`. |
| `torch_n3/volume.py` | MINC I/O, geometry, shrinking and resampling. |
| `torch_n3/minc_tools.py` | The two MINC utilities on the critical path: continuous lookup, and the Otsu threshold. |
| `torch_n3/backends/legacy.py` | The original C++ blocks — histogram, sharpened lookup table, B-spline fit, field extension. |
| `torch_n3/_legacy/` | The CFFI shim that compiles those blocks out of `legacy/N3/src`. |
| `torch_n3/cli.py` | The command line. |

The legacy backend is deliberate: it makes the original code a numerical oracle, so the
PyTorch replacements planned in `PLAN.md` can be checked against it one block at a time.

---

## Building and testing

The CFFI extension is built in place and is not checked in:

```bash
python3 torch_n3/_legacy/build_legacy.py
```

It compiles `Spline.cc`, `TBSpline.cc`, `DHistogram.cc`, `WHistogram.cc`,
`sharpen_hist.cc` and `correctField.cc` straight out of `legacy/N3/src` — nothing is
copied or modified — and links against the EBTKS and LAPACK that ship with the
installed MINC toolkit.

```bash
python3 -m pytest              # the whole suite, about 5 s
python3 -m pytest tests/test_pipeline.py -k sharpen    # one test
```

The tests are the cases from `legacy/N3/testing/CMakeLists.txt`, re-expressed as
comparisons: run the installed N3 program, run the Python, require agreement. They
need the MINC toolkit on `PATH`.

### How close is it?

Every block matches its legacy counterpart to the precision of the file the legacy
writes it into — histogram, sharpened lookup table, `minclookup`, `spline_smooth`,
`evaluate_field`, `correct_field`, `mincresample`, `resample_labels`,
`mincstats -biModalT`.

End to end, correcting `brain.mnc.gz` lands **2.9e-3 relative RMS** from
`brain_nu_ref.mnc.gz`, where the legacy suite asks for 1e-4. That gap is storage, not
arithmetic. Legacy N3 passes every intermediate volume between programs as a MINC file:
12-bit before the mask is applied and 16-bit after, scaled slice by slice on write and
rescaled onto a single global grid on read. The estimation is a feedback loop, so
thirty iterations amplify that rounding. `torch_n3` keeps float64 throughout, which is
more accurate but not identical. Reproducing the reference voxel for voxel would mean
modelling MINC's storage rather than N3; see `PLAN.md`.

## Requirements

Python 3.12, `numpy`, `cffi`, `minc2_simple`, and the MINC toolkit on `PATH` (for
`mincconvert`, and for the legacy programs the tests compare against). All already
installed here — nothing needs fetching.

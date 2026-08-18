# torch_n3

A PyTorch reimplementation of **N3** — non-parametric non-uniform intensity
normalization (Sled, Zijdenbos & Evans, 1998) — which removes the smooth
multiplicative intensity inhomogeneity ("bias field") introduced by MRI scanners.

The original implementation is a set of Perl drivers over a dozen C++ programs
(`legacy/N3/`, retained for reference only). This package implements the same
algorithm block by block, with each block validated against the program it
replaces.

![Non-uniformity remaining after correction, and wall time, for legacy N3, the PyTorch
port and a gradient-descent estimator](experiments/results/summary.png)

Non-uniformity remaining after correction on colin27, over 50 random fields at each
of three field amplitudes and three noise levels, beside the wall time of the same
trials. The original C++ blocks and the port coincide in every cell, and the GPU runs
the port 8.8× faster than the original on a CPU (0.69 s against 6.05 s per estimate).
`hoyer` is not N3: it is the second estimator in `torch_n3/optimize.py`, which retains
N3's B-spline field and bending-energy penalty and replaces the alternating iteration
with gradient descent on a stated sharpness objective. It is better in every cell by a
factor of 2.5–5, and worse on 17 of the 450 matched trials. Method, complete tables and
limitations are in [experiments/README.md](experiments/README.md).

---

## Quick start

No installation or compilation step is required; all commands are run from the
repository root. The CFFI extension under `torch_n3/_legacy/` is needed only by the
test suite and by `--backend legacy`; see [Building and testing](#building-and-testing).

Correct a volume:

```bash
python3 -m torch_n3 brain.mnc corrected.mnc --mask brain_mask.mnc
```

This is equivalent to `nu_correct brain.mnc corrected.mnc -mask brain_mask.mnc` and
uses the same defaults. Run time on the 91×109×91 test volume is about 0.4 s.

To report convergence and retain the estimated field:

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

The volumes in `legacy/N3/testing/` are read directly, gzipped:

```bash
python3 -m torch_n3 legacy/N3/testing/chunk.mnc.gz out.mnc \
    --mask legacy/N3/testing/chunk_mask.mnc.gz
```

### Mask selection

N3 estimates the field from the intensity histogram, so the contents of the mask
determine the result. Supply `--mask` whenever a brain mask is available; background
voxels contribute noise and no signal. Without one, the background is removed by an
Otsu threshold over the estimation grid, which separates tissue from air but not brain
from skull.

`--background` is the alternative rule, and N3's own: a fixed intensity, defaulting to
1, below which voxels are discarded. Being absolute it depends on the range the file
was written on. `brain.mnc` peaks at 1,078,824, where a threshold of 1 admits the air;
the same anatomy stored on [0, 1] leaves nothing above it. `--no-bimodal` selects that
rule, and supplying `--mask` selects it as well, so the automatic threshold does not
reach a masked run. Every recorded reference in `tests/` and every table below was
produced under the fixed rule.

The automatic rule is `mincstats -biModalT`, the same one `--evaluation-mask` falls
back to. N3's estimation side uses the other of the two bimodal rules,
`volume_stats -biModalT`: the same Otsu criterion over a histogram whose bin count
comes from the file's stored voxel range, which is not a quantity a float64 tensor
carries. `--bimodal` applies the automatic rule inside `--mask` too, which is what N3's
`-bimodalT` flag does.

`--evaluation-mask` serves a different purpose. It delimits the region in which the
fitted field is applied directly, outside of which the field is smoothly extrapolated.
If unspecified, it is derived from the data by an Otsu threshold, as `nu_evaluate` does.

### Options

All protocol options mirror those of `nu_correct` and default to its values:

| Option | Default | What it does |
|---|---|---|
| `--distance` | 200 (mm) | B-spline knot spacing. The dominant parameter: a smaller spacing admits a less smooth field. |
| `--fwhm` | 0.15 | Assumed width, in log-intensity units, of the blur the field introduces into the histogram. |
| `--noise` | 0.01 | Wiener constant of the deconvolution. Increase it if the sharpening is unstable. |
| `--bins` | 200 | Histogram bins. |
| `--shrink` | 4 | Estimation is performed on a grid coarser by this factor. The field is a spline, so the output remains at full resolution. |
| `--lambda` | 1e-7 | Bending-energy penalty on the spline fit. Must be set jointly with `--distance`, at approximately one decade per halving. See [the trade-off](#bias-field-recovery-on-simulated-data). |
| `--iterations` | 50 | Iteration budget, one number per stopping stage. |
| `--stop` | 0.001 | Terminate when the field changes by less than this, one per stage. |
| `--field-floor` | 0.1 | Smallest field value admitted, which bounds the division. |
| `--background` | 1 | Fixed intensity below which voxels are never estimated from. An absolute value; see [Mask selection](#mask-selection). |
| `--bimodal` / `--no-bimodal` | auto | Take that threshold from the data with Otsu's rule instead. Applied when no `--mask` is given and not otherwise. |
| `--denoise` | off | Filter the volume with one non-local-means pass before estimating the field, for the *estimation only*: the output is the original volume divided by the fitted field, and is never denoised. A modification to the algorithm rather than part of it, with no counterpart in the original; torch backend only. Expensive; see [Non-local means](#non-local-means---denoise). |
| `--denoise-search` | 3 | Search radius in voxels. The cost is cubic in this and in nothing else. |
| `--denoise-patch` | 1 | Patch half-width in voxels. Larger is a stricter similarity test, and so less smoothing. |
| `--denoise-strength` | 1.0 | Multiplies the estimated noise level. Zero is exactly the identity. |
| `--device` | cpu | Any torch device, for example `cuda`. See [Limits on end-to-end agreement](#limits-on-end-to-end-agreement) first. |
| `--backend` | torch | `legacy` substitutes the original C++ for every block. |
| `--solver` | normal | Formulation used for the spline fit. `qr` fits the same spline through a far better-conditioned system and is substantially more reproducible across machines; `blocked` returns the same answer without holding the design matrix, and is the appropriate choice at a fine `--distance`; `dr` reparameterizes `qr` so that the penalty is diagonal, which reduces an entire `--lambda` grid to one factorization. (`sparse` does not converge; see below.) See [Solver formulations](#solver-formulations). |

Staged termination follows N3: `--iterations 10 20 --stop 0.001 0.005` terminates at a
change below 0.001, or below 0.005 once iteration 10 has been passed.

---

## From Python

The command line is a thin wrapper over the library.

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

`nu_estimate` returns a fitted `BSplineField`, the in-memory equivalent of N3's `.imp`
mapping file. It is compact — a few dozen coefficients at the default knot spacing —
and can be evaluated on any grid in the same world space, which is what permits
estimation to run on a coarse grid:

```python
field.coefficients          # (80,) for this volume
field.grid.shape            # (24, 28, 24) -- the shrunken estimation grid
field.evaluate_on(volume)   # ...but sampled at full resolution
```

Use `evaluate_field(volume, field, mask=mask)` rather than `evaluate_on` to obtain the
field as the correction applies it: masked, extrapolated and floored.

### Volumes

`Volume` is a tensor together with the geometry N3 requires, always in standard order:
C-ordered, axis 0 slowest, positive steps.

```python
volume.data       # float64 tensor, shape (nz, ny, nx)
volume.step       # voxel size along each axis   (these two remain numpy arrays: they
volume.start      # world coordinate of (0,0,0)   describe the grid and do not follow
                  #                               the data onto a device)

volume.like(new_data)          # same grid, different values
volume.to("cuda")              # same volume, data on another device
volume.shrink(4)               # the coarse estimation grid
other.resample_like(volume)    # nearest-neighbour onto this grid
```

All downstream computation remains on the device holding `volume.data`.

`load_volume` reads gzipped and MINC1 files by converting them with `mincconvert`
first; `minc2_simple` itself reads MINC2 (HDF5) only.

---

## Algorithm

All computation is performed in log-intensity space, where a multiplicative field
becomes an additive one. Each iteration:

1. **Sharpen.** Histogram the masked volume, deconvolve that histogram with a Gaussian
   by Wiener filtering, and map every voxel to `E[u | v]` under that kernel. The result
   is the estimate of the volume with no field-induced blur in its histogram.
2. **Attribute the difference.** The change introduced by the sharpening is attributed,
   by assumption, to the field.
3. **Retain the smooth component.** Fit a cubic tensor B-spline with a bending-energy
   penalty, so that only variation on the scale of `--distance` is retained.

Iterate until the field ceases to change, exponentiate, and divide. Step 1 is
`blocks/histogram.py` and `blocks/sharpen.py`, step 3 is `blocks/spline.py`, and
`blocks/field.py` makes the fitted field usable outside the mask before the division.

`torch_n3/pipeline.py` follows `legacy/N3/src/NUcorrect/nu_estimate_np_and_em.in` step
by step, and is best read alongside it.

### Layout

| Module | Role |
|---|---|
| `torch_n3/pipeline.py` | N3 itself: `nu_estimate`, `nu_evaluate`, `nu_correct`. |
| `torch_n3/blocks/histogram.py` | The masked histogram, `volume_hist`. |
| `torch_n3/blocks/sharpen.py` | The sharpened mapping, `sharpen_hist` — the mathematical core. |
| `torch_n3/blocks/spline.py` | The smooth field fit, `spline_smooth`. |
| `torch_n3/blocks/field.py` | The field extension past the mask, `correct_field`. |
| `torch_n3/blocks/denoise.py` | Not a port: the optional non-local-means prefilter (`--denoise`). No oracle; the legacy backend refuses it. |
| `torch_n3/volume.py` | MINC I/O, geometry, shrinking and resampling. |
| `torch_n3/minc_tools.py` | The two MINC utilities on the critical path: continuous lookup, and the Otsu threshold. |
| `torch_n3/backends/legacy.py` | The same blocks, as the original C++. |
| `torch_n3/_legacy/` | The CFFI shim, plus the N3 (`n3/`) and EBTKS (`ebtks/`) sources it compiles. |
| `torch_n3/cli.py` | The command line. |

The legacy backend makes the original code a numerical **oracle**: a reference
implementation whose answer each PyTorch block is compared against.
`torch_n3.backends.resolve` selects between the two and the pipeline runs on either,
which is the whole of what `--backend legacy` does.

It is not the installed `nu_correct`, and the distinction affects every number reported
below. The original N3 is a Perl script driving a dozen separate executables, so every
intermediate volume it computes is written to a MINC file — 12-bit or 16-bit, rescaled
slice by slice — and read back rounded. The shim calls the same C++ routines directly,
on `float64` arrays, within one process. `--backend legacy` is therefore the original
arithmetic without the original's quantisation, and for that reason does not reproduce
`nu_correct` exactly either: it differs from `brain_nu_ref.mnc` by 5.2e-3 relative RMS,
further than the PyTorch blocks at 3.0e-3. Its purpose is block-against-block
comparison with no rounding between stages.

---

## Building and testing

The CFFI extension is built in place and is not checked in:

```bash
python3 torch_n3/_legacy/build_legacy.py
```

It compiles `Spline.cc`, `TBSpline.cc`, `DHistogram.cc`, `WHistogram.cc`,
`sharpen_hist.cc` and `correctField.cc` from `torch_n3/_legacy/n3/`, and the seven EBTKS
sources they need from `torch_n3/_legacy/ebtks/`. Both trees are vendored byte for byte
— out of `legacy/N3/src` and `legacy/EBTKS` — unmodified and checked in, so the
extension builds with no MINC toolkit and no N3 or EBTKS checkout.

The one external dependency it links is **LAPACK and BLAS**, and that choice has
measurable consequences: see [Which LAPACK the legacy backend
links](#which-lapack-the-legacy-backend-links). `volume_io`, `time_stamp` and
`ParseArgv` are supplied by small stand-ins in `torch_n3/_legacy/compat/`, which allows
`correctField.cc` and `args.cc` to remain byte-identical while requiring no libminc2.
Only the tests and `--backend legacy` require any of this.

```bash
python3 -m pytest              # the whole suite, about 26 s
python3 -m pytest tests/test_spline.py         # one block
python3 -m pytest -k "legacy"                  # everything, on the C++ backend
```

The tests are of four kinds:

- **Per block** (`test_histogram.py`, `test_sharpen.py`, `test_spline.py`,
  `test_field.py`) — the PyTorch block against the same code compiled into the CFFI
  shim. These carry the tightest bounds and are what constrain the port.
- **Against the programs** (`test_pipeline.py`, `test_minc_tools.py`, `test_volume.py`)
  — the cases from `legacy/N3/testing/CMakeLists.txt`, re-expressed as comparisons
  against the answers the installed N3 gave.
- **Against a planted field** (`test_field_recovery.py`) — recovery of a known
  non-uniformity, which is the question an application has, rather than agreement with
  the original.
- **Against a recorded result** (`test_reproducibility.py`) — a checked-in volume that
  every backend and device must reproduce. See
  [Cross-platform reproducibility](#cross-platform-reproducibility).

`python3 -m tests.margins` prints where each of those comparisons sits against its
bound; [PROBLEMS.md](PROBLEMS.md) is that table together with its known weaknesses.

No test runs an N3 program. Those programs are deterministic, so their answers were
recorded once into `tests/reference/` (9.4 MB) and are read from there. This keeps the
suite fast, distinguishes a failure of this port from a different build of the original,
and allows the suite to run on machines with no MINC toolkit. `tests/inputs.py` holds
the inputs they were given, so that both sides construct them identically. To re-record:

```bash
python3 -m tests.regenerate_reference     # needs the MINC toolkit on PATH
git diff --stat tests/reference           # empty if they still say the same thing
```

(`tests/data/brain_nu_ref_legacy.mnc` is rewritten as well, and always appears as
changed: MINC stamps each file with the user, host and time that wrote it. Its voxel
data is reproducible; those header bytes are not.)

**The suite requires no MINC program.** The volumes it runs on are checked in as MINC2
under `tests/data/` (10 MB, two thirds of it the reproducibility reference) —
byte-for-byte the same images as `legacy/N3/testing/`, converted once, because
`minc2_simple` opens MINC2 only. The single exception is
`test_gzipped_minc1_input_is_readable`, which tests the conversion path itself and skips
without `mincconvert`.

### Cross-platform reproducibility

`tests/test_reproducibility.py` establishes it. `tests/data/brain_nu_ref_legacy.mnc` is
a volume checked into the repository: this pipeline's output on `brain.mnc` with the
original C++ blocks driving it, stored as `float64` so that it records what was computed
rather than what a 16-bit file could hold. Both backends, on every available device,
must reproduce it to within one part in 65535 of relative RMS, which is N3's own working
precision — the original passes every intermediate volume to the next program through a
16-bit file.

All figures are relative RMS: RMS difference over mean signal, the measure
`compare_nu_result.pl` uses. The largest difference anywhere is not reported, because
over 900k voxels it is determined by a handful at the mask edge and characterises those
voxels rather than the volume.

| | relative RMS from the reference | of the bound |
|---|---|---|
| `legacy`, CPU | 0 — bit-identical, it wrote the file | 0.0% |
| `torch`, CPU | 5.52e-08 | 0.4% |
| `torch`, CUDA | 5.52e-08 | 0.4% |

Almost all of that difference is between the two block implementations rather than
between platforms: CPU and GPU differ from each other by far less than either differs
from the legacy backend.

The test runs one iteration rather than the fifty of the shipped protocol. N3's
histogram range is taken from the data and then rounded to the six decimals its text
interchange prints, so a voxel on a bin boundary can fall either side of it. After one
iteration the backends agree to 5.5e-08 relative RMS; at two, one whole count moves
between bins — out of the 3,724 samples the shrunken estimation grid contributes — and
they finish 1.17e-3 apart, four orders of magnitude worse. This is the **divergence
threshold**: the iteration count at which two implementations cease to agree, defined
and measured in [Limits on end-to-end agreement](#limits-on-end-to-end-agreement) below.
No implementation can be constrained past it, so the test stops where the answer is
still a continuous function of rounding error.

### Agreement with the reference implementation

Block by block, against the original C++. These are measured differences, not the bounds
the tests assert; the bounds are in the tests, and where each comparison sits against
its bound is tabulated in [PROBLEMS.md](PROBLEMS.md).

| Block | Agreement |
|---|---|
| histogram range, plain histogram, mask resampling | exact |
| Parzen histogram | 5e-11 over 132k samples — summation order |
| sharpened lookup table | 3e-14 |
| continuous lookup, vs `minclookup` | 7e-15 |
| Otsu threshold, vs `mincstats -biModalT` | 4e-5 on a value of 2.4e5 — all `mincstats` printed |
| `shrink`, vs `mincresample` | 3e-2 on values of 8e5 — the 12-bit file it came back through |
| B-spline fit, as a fitted field | 2e-7 relative |
| field extension, vs `correct_field` | 5e-6 relative |

The last two bounds are loose for structural reasons. The normal equations N3 solves
have a condition number of about 1e13 at the default 200 mm knot spacing, because the
knots are further apart than the volume is wide; the coefficients are therefore poorly
determined and any two solvers disagree about them, while the fitted field, which is
what is used, is well determined. `correct_field` relaxes in raster order in `float`,
which is inherently sequential; the port sweeps the two checkerboard colours in turn,
which is the same iteration reordered so that it vectorises.

### Limits on end-to-end agreement

The blocks above agree to between six and fifteen digits. Over the whole pipeline that
reduces to three:

| End to end, on `brain.mnc`, shipped protocol | relative RMS |
|---|---|
| `torch` vs N3's `brain_nu_ref.mnc` | 3.01e-3 |
| `legacy` vs N3's `brain_nu_ref.mnc` | 5.21e-3 |
| `torch` vs `legacy` | 3.1e-3 |
| `torch` on CPU vs the same code on a GPU | 1.3e-3 |

The legacy test suite requires 1e-4. No configuration here reaches it, and the last row
identifies why: it is one implementation disagreeing with itself over nothing but the
order of a few reductions. Two mechanisms are responsible, neither of them an error in
the port.

**Quantisation.** The legacy N3 is a Perl script driving a dozen separate programs, so
every intermediate volume is written to a MINC file and read back rounded: 12 bits
before the mask is applied, 16 after, scaled slice by slice on write and rescaled onto a
single global grid on read, so that the two operations are not mutually consistent.
Every iteration passes through this, and the reference volume was produced through all
of them. `torch_n3` retains float64 throughout, which is more accurate and therefore not
identical. Reproducing `brain_nu_ref.mnc` voxel for voxel would require modelling MINC's
storage rather than N3; see `PLAN.md`.

The same applies in reverse, and is why `--backend legacy` does not reach 1e-4 either:
it is the original arithmetic with the original's rounding removed.

**Amplification.** N3's loop feeds its output back into itself, so any difference that
survives is multiplied. The two backends agree on the field to 1.4e-7 after one
iteration and to 9.6e-4 after ten, a factor of 6,900.

The growth does not begin from floating-point noise. The histogram range is taken from
the data and then rounded to the six decimals N3's text interchange prints, so a voxel
on a bin boundary can fall either side of it and a whole count moves between bins. This
is a discontinuity within the loop: past it, an end-to-end comparison records which side
of a rounding boundary one voxel fell on. The iteration at which it occurs is the
divergence threshold defined above, and the next section measures it directly.

### Which LAPACK the legacy backend links

The following is a direct measurement of the mechanism above.

The spline fit solves normal equations with a condition number of about `1e13` — at the
default 200 mm spacing the knots are further apart than the volume is wide — by calling
LAPACK's `dsysv`. The shim formerly linked `libEBTKS.a`, which bundles its own f2c'd
LAPACK. It subsequently compiled vendored sources and linked the **system** LAPACK/BLAS
(OpenBLAS on this machine), so as to require nothing from the MINC toolkit; the
measurements below are from that build. It now links no LAPACK of its own by default,
and instead shares PyTorch's; see "Building against a different LAPACK" below for the
reason. The object files and the inputs are the same in each case, and only the
implementation of `dsysv` differs.

At block level the substitution has almost no effect. Fitting the default 200 mm spline
to a tilted plane on `chunk.mnc`, the two builds differ by:

| | largest | relative RMS |
|---|---|---|
| spline coefficients | 1.5e-07 | 2.9e-06 |
| the fitted field | 2.4e-10 | **3.1e-11** |

Three parts in `1e11`. Neither solver is in error: at that conditioning the coefficients
are not determined to better than `1e-4` by any solver, which is why this repository
compares fitted fields and never coefficients.

End to end on `brain.mnc`, under the same substitution: `legacy` against `torch`, one
row per iteration count, so that the growth over the loop is visible.

| | EBTKS's f2c'd LAPACK | system LAPACK |
|---|---|---|
| `legacy` vs N3's `brain_nu_ref.mnc`, shipped protocol | 0.3701% | **0.5211%** |
| `legacy` vs `torch`, 1 iteration | 5.49e-08 | 5.52e-08 |
| `legacy` vs `torch`, 2 iterations | 5.55e-08 | **1.17e-3** |
| `legacy` vs `torch`, 3 iterations | 5.65e-08 | 2.12e-3 |
| `legacy` vs `torch`, 4 iterations | 5.73e-08 | 1.69e-3 |
| `legacy` vs `torch`, 5 iterations | 5.82e-08 | 5.40e-4 |
| `legacy` vs `torch`, 6 iterations | **8.37e-04** | 5.07e-4 |

Each column is read downwards. Both builds hold agreement at `5.5e-08` and then lose it
entirely at one iteration: the bin-boundary transition described above, which is a step
rather than a drift. What the substitution changes is its location — the sixth iteration
under EBTKS's LAPACK, the second under the system one.

A perturbation of `3.1e-11` in the fitted field therefore moved the divergence threshold
four iterations earlier, and with it the end-to-end answer by four orders of magnitude.
Which LAPACK is linked is not an implementation detail of the oracle.

This also bounds how far the reproducibility test can be taken: agreement survives only
up to the divergence threshold, wherever it falls for a given build, which is why
`tests/test_reproducibility.py` runs one iteration and no more. Two bounds were moved as
a result, deliberately and with the measurements recorded, in
[PROBLEMS.md](PROBLEMS.md) §8.

The same effect appears with no change of LAPACK: identical `torch` code on a CPU and on
a GPU agrees to `6.3e-11` after one iteration and `1.0e-10` after two, then diverges to
`1.17e-3` at the third.

### Measuring the divergence threshold on another LAPACK

The measurements above are from one machine. For packaging, or for a move to a cluster,
the quantity required is the divergence threshold on the target platform, because
`tests/inputs.py::PLATFORM_PROTOCOL` must remain below the smallest such threshold on
any platform the code is expected to run on. It is currently 1, with no margin.

```bash
python3 -m tests.convergence                      # legacy vs torch, 1..6 iterations
python3 -m tests.convergence --device cuda        # and CPU vs GPU
python3 -m tests.convergence --volume chunk       # 20x smaller, just checks it runs
python3 -m tests.convergence --max-iterations 10
```

It reports the platform, the LAPACK the extension resolved against, and one row per
comparison:

```
platform:   Linux-6.17.0-35-generic-x86_64-with-glibc2.39, python 3.12.3, torch 2.13.0+cu130
shim links: liblapack.so.3, libblas.so.3, libopenblas.so.0

relative RMS                     1          2          3          4          5          6
legacy vs torch           5.52e-08    0.00117    0.00212    0.00169    0.00054   0.000507
                       ^ divergence at 2 iterations (5.52e-08 -> 0.00117)
torch cpu vs cuda          6.3e-11   9.97e-11    0.00117   0.000214    0.00107   8.29e-05
                       ^ divergence at 3 iterations (6.3e-11 -> 0.00117)
```

Report the divergence threshold rather than the individual values. Those before it are
informative only in being small; those after it are not comparable between machines,
since they record which side of a rounding boundary a single voxel fell on.

#### Building against a different LAPACK

By default the extension links no LAPACK/BLAS of its own: `dgemm_`, `dsysv_` and the
rest are left as undefined symbols and resolved at import time against whatever is
already loaded in the process. Every entry point imports `torch` before the shim, and
PyTorch loads its own dependency library with `RTLD_GLOBAL` (see its `__init__.py`,
"Note [Global dependencies]") for this form of sharing. The shim therefore uses
**PyTorch's own BLAS** rather than a second one selected by this package. This matters
on macOS in particular: a conda or Homebrew environment's `liblapack.dylib` is typically
a symlink to an OpenBLAS build carrying its own bundled `libomp.dylib`, while the
PyTorch wheel bundles a different `libomp.dylib`. Linking that OpenBLAS directly, as
`-llapack -lblas` formerly did, loads two copies of LLVM's OpenMP runtime into one
process, which aborts, or segfaults if forced past that with `KMP_DUPLICATE_LIB_OK`.
Sharing PyTorch's library avoids this on any platform without requiring the build to
know which BLAS vendor PyTorch selected.

Two environment variables restore an explicit LAPACK/BLAS dependency, overriding the
default, for comparison against a specific build as in the table above. Both are read at
build time:

```bash
rm -rf torch_n3/_legacy/build          # cffi will not relink without this

# Under whichever names the platform uses
N3_LAPACK_LIBS="mkl_rt" \
N3_LAPACK_LIB_DIRS="/opt/intel/oneapi/mkl/latest/lib" \
    python3 torch_n3/_legacy/build_legacy.py

# EBTKS's own bundled f2c'd LAPACK, where libEBTKS.a is available. This is the
# left-hand column of the table above. The archive resolves after the shim's own
# objects, so only its clapack members are taken.
N3_LAPACK_LIBS="EBTKS" N3_LAPACK_LIB_DIRS="$MINC_TOOLKIT/lib" \
    python3 torch_n3/_legacy/build_legacy.py
```

Without rebuilding, `LD_PRELOAD` is expected to substitute a LAPACK on Linux: preloaded
libraries take precedence in the process's global symbol scope, ahead of whatever
`torch` loads subsequently, which is what the default build resolves `dgemm_`/`dsysv_`
against in place of a dependency of its own. This has not been verified on a Linux build
beyond that reasoning; confirm it against the `shim links:` line described below before
relying on it.

```bash
LD_PRELOAD=/path/to/other/liblapack.so.3 python3 -m tests.convergence
```

On Debian and Ubuntu, `update-alternatives --config liblapack.so.3-$(uname -m)-linux-gnu`
switches the system-wide implementation, but only among those installed. This machine
carries only `libopenblas`; `apt install liblapack3` adds the reference build, which is
the more informative second data point, being the same Fortran from which EBTKS's f2c'd
copy was translated.

In either case, check the `shim links:` line the script prints: the library named in a
build is not always the one the loader finds, and a statically linked LAPACK does not
appear there at all. The script reports `no dynamic LAPACK (static?)` in that case,
which is the expected output for the EBTKS build above.

#### A divergence threshold of 1

A threshold of 1 means that no iteration count is reproducible on that platform and
`tests/test_reproducibility.py` cannot hold there, since the recorded volume describes a
run the build does not reproduce even once around the loop. Report it rather than
working around it: it would mean the bin-boundary transition is triggered on the first
pass, which has not been observed here and which the block-level tests should detect.


### Solver formulations

All of the above follows from one decision made in N3 in 1998: the spline is fitted by
forming the normal equations.

```
(AtA + lambda N J) c = At f
```

`A` holds the basis functions at every masked voxel. Squaring it squares the condition
number, and at the shipped 200 mm knot spacing — where the knots are further apart than
the head is wide — this reaches about `1e13`. A system that ill-conditioned does not
have an incorrect answer; it has an answer whose last digits are determined by whichever
BLAS computed it. This is the mechanism behind the LAPACK measurements above, and behind
the divergence of a CPU and a GPU running identical code.

The same fit can be posed without squaring `A`. Stacking the penalty beneath the data
gives one least-squares problem:

```
[      A       ]         [ f ]
[              ] c   ~   [   ] ,     where  D'D = J
[ sqrt(lam N) D]         [ 0 ]
```

Its normal equations are those above, term for term: the same minimiser and the same
objective, not an approximation. The matrix factorised is `A` itself, so its condition
number is the square root of the former, about `1e6`. `--solver qr` selects this
formulation.

Measured on `brain.mnc`, on the 24×28×24 grid produced by `--shrink 4` — 3,724 samples
and 80 coefficients, which is the fit N3 performs:

| | `--solver normal` | `--solver qr` |
|---|---|---|
| condition number of the system solved | 5.3e12 | 2.3e6 |
| fitted field, CPU vs GPU, relative RMS | 2.5e-9 | 3.0e-13 |
| iterations before CPU and GPU diverge | 3 | 7 |
| iterations before `legacy` and `torch` diverge | 2 | 4 |
| 10 iterations, CPU | 0.38 s | 0.41 s |
| peak RSS, `--shrink 1` | 1.0 GB | 1.3 GB |

The third row is the principal result: the divergence threshold is where an end-to-end
comparison ceases to measure the code and begins to measure which side of a rounding
boundary one voxel fell on. The fourth row is less expected — the QR formulation tracks
the C++ oracle longer than the port's own normal equations do, although the oracle
solves the normal equations itself. The more accurate solve is therefore worth more here
than matching the other implementation's formulation.

The cost is that `A` is held dense rather than only `AtA`, which the last two rows
quantify. `python3 -m tests.convergence --solver qr` measures the divergence thresholds
on a given machine.

#### Solving without holding `A`

`--solver blocked` is the same stacked system processed one band at a time. Sorting the
rows by the first-axis knot they belong to makes `A` banded: a sample whose corner is
`k` occupies only columns `[k·n₁n₂, (k+4)·n₁n₂)`. The rows can then be folded into a
running `R` one window at a time, and `A` is never formed. It is the same factorization
in a different order, so it returns `qr`'s answer — the two agree to 4e-15, which is
rounding — and it inherits the conditioning and the CPU/GPU reproducibility exactly
(1.23e-13 against 1.27e-13).

Its advantage is in resource cost, and only at fine spacings, where the dense stack no
longer fits. On `chunk.mnc`:

| `--distance` | coefficients | window | `qr` | `blocked` | `normal` |
|---|---|---|---|---|---|
| 200 mm | 64 | 64 | 0.12 s | 0.18 s | 0.05 s |
| 50 mm | 245 | 140 | 0.37 s | 0.29 s | 0.13 s |
| 12.5 mm | 3168 | 792 | 12.72 s | **3.45 s** | 8.84 s |
| 12.5 mm, peak RSS | | | 7.53 GB | **1.55 GB** | 1.09 GB |

At 200 mm there is a single group and `blocked` reduces to `qr` plus the cost of a sort,
so at the shipped spacing it offers no advantage. Below 50 mm the advantage is
substantial: at 12.5 mm it is 3.7× faster than `qr` on a fifth of the memory, and faster
than the normal equations as well.

#### Sweeping `--lambda`: `--solver dr`

`--solver dr` answers the same stacked system, reparameterized so that the penalty
becomes **diagonal** — the Demmler–Reinsch basis of the P-spline literature. Factorize
`[A; sqrt(λ₀N) D]` once at an *anchor* weight λ₀, so that `R₀ᵀR₀ = AᵀA + λ₀NJ`, and set
`D̃ = sqrt(N) D R₀⁻¹`. Then for every weight

```
AᵀA + λNJ  =  R₀ᵀ (I + (λ − λ₀) D̃ᵀD̃) R₀
```

and eigendecomposing `D̃ᵀD̃ = U diag(γ) Uᵀ` renders the middle factor diagonal. A fit is
then one elementwise division by `1 + (λ − λ₀)γ` and a triangular solve.

At `λ = λ₀` the divisor is 1 and the method reduces to `qr`, back-substitution included;
the two agree to rounding, and `dr` meets every bound `qr` meets at identical margins.
Its advantage is in sweeping `λ`. The QR factorization, the triangular solve and the
eigendecomposition are all independent of `λ`, so `BSplineField.refit(lam)` consists of
the division alone.

Measured on `brain.mnc`'s estimation grid: one spline fit per solver, against `dr`'s
marginal cost for an additional weight.

| `--distance` | coefficients | `normal` | `qr` | `blocked` | `dr` | `dr` refit |
|---|---|---|---|---|---|---|
| 200 mm | 80 | 4.9 ms | 5.6 ms | 8.6 ms | 7.3 ms | **0.033 ms** |
| 100 mm | 150 | 3.6 ms | 9.0 ms | 12.5 ms | 14.9 ms | **0.044 ms** |
| 50 mm | 392 | 8.3 ms | 27.7 ms | 26.5 ms | 43.6 ms | **0.140 ms** |

This is what makes GCV or REML smoothing-parameter selection affordable here, and it is
the appropriate solver for re-measuring the `--lambda` × `--distance` tables above.

For a single weight it is the slowest of the solvers and should not be selected. The
eigendecomposition is `O(k³)` in addition to everything `qr` performs, and yields no
benefit until a second weight is requested: a complete 30-iteration run takes 1.8 s at
50 mm, against 1.2 s for `qr` and 0.4 s for `normal`. It becomes economical at the second
weight and decisively so by the fourth. Below that, use `qr`.

#### Solver cost on a GPU

Peak CUDA memory *allocated* (not process RSS) for one spline fit on
`brain_nu_ref.mnc`, measured 2026-08-01 on an RTX A6000. `legacy` is CFFI and CPU-only;
`sparse` goes through `scipy` on the CPU; neither has a GPU footprint.

| estimation grid | `--distance` | coefficients | `normal` | `qr` | `blocked` | `dr` |
|---|---|---|---|---|---|---|
| shrink 4 (3.7k samples) | 200 mm | 80 | 15 MB | 28 MB | 30 MB | 23 MB |
| shrink 4 | 50 mm | 392 | 26 MB | 61 MB | 35 MB | 48 MB |
| shrink 4 | 12.5 mm | 7,581 | 1.77 GB | 2.87 GB | 3.63 GB | 4.62 GB |
| shrink 1 (238k samples) | 50 mm | 392 | 355 MB | 2.90 GB | 1.14 GB | 1.73 GB |
| shrink 1 | 25 mm | 1,452 | 370 MB | 8.75 GB | 1.36 GB | 5.65 GB |

Under the shipped protocol the complete 30-iteration pipeline peaks near 100 MB
whichever solver is used, so these differences do not arise there. Away from it, two of
them do.

`normal` is the most memory-frugal by an order of magnitude, because it holds `AᵀA` and
never `A`. This is the substantive counterweight to its conditioning.

**`blocked`'s advantage is governed by the sample-to-coefficient ratio rather than by
`--distance`.** It is decisively better where samples greatly outnumber coefficients
(1.36 GB against `qr`'s 8.75 GB at shrink 1 and 25 mm). Where coefficients exceed
samples — shrink 4 at 12.5 mm, 7,581 against 3,724 — it is the second-worst of the four,
because its running `R` is `O(size²)` and there is little of `A` left to avoid holding.

`γ` is non-negative, so the divisor never falls below 1 and no dynamic range in `γ` can
affect the answer. The single constraint is that the basis is valid only at or above its
anchor; below it the divisor can pass through zero. `anchor=` therefore belongs at the
bottom of the grid to be swept, and `refit()` raises an exception beneath it rather than
returning a value.

**The textbook formulation does not work here.** Demmler–Reinsch is normally written on
the QR factorization of the design `A` alone. Here `A` is the *masked* design, and at
fine knot spacings the mask leaves basis functions with no data beneath them:

| `--distance` | coefficients | rank(A) | cond(A) | cond(stacked) | cond(normal) |
|---|---|---|---|---|---|
| 200 mm | 64 | 64 | 8.4e7 | 3.5e6 | 1.2e13 |
| 100 mm | 100 | 100 | 7.1e6 | 8.6e5 | 7.5e11 |
| 50 mm | 245 | **243** | **7.5e12** | 2.9e5 | 8.6e10 |

`R` derived from `A` alone is therefore worse conditioned than the stacked system at
every spacing, and at 50 mm `A` is rank deficient and `R` singular to working precision
— worse there than the normal equations. `D R⁻¹` then overflows into a `γ` with 83
non-positive entries reaching `-5.7e6`, and the divisor passes through zero. Clipping
`γ` at zero, the standard recommendation, does not recover it: the eigenvectors are as
damaged as the eigenvalues. Anchoring on the stacked matrix costs nothing and removes
the failure, because the penalty rows span exactly the directions the data leaves empty.
See [PROBLEMS.md](PROBLEMS.md) §12.

#### Negative results

`--solver sparse` holds the same stacked system in `scipy.sparse` and passes it to
`lsqr`. **It does not converge and must not be used for results.** The cause is
structural rather than a matter of tuning: a rectangular system excludes every direct
sparse solver in `scipy.sparse.linalg`, which provides no sparse QR, leaving iterative
methods, and LSQR's convergence is governed by the same condition number the stacked
form exists to reduce. At 200 mm it terminates after about 2,950 iterations reporting
`istop=3` ("condition number exceeds `conlim`"), 18 s in and 1.8e-3 from the direct
answer — a larger error than the difference between this port and the original C++.
Raising the iteration limit has no effect; it is not terminating early.
`BSplineField.solve_info` reports what LSQR returned. The solver is retained because the
measurement is informative, and because `blocked` solves the problem it was intended to
address.

**`normal` is the default**, and every recorded reference in `tests/` was produced with
it. Changing it is a deliberate decision with a substantial cost, since it moves every
end-to-end number in this file; [PROBLEMS.md](PROBLEMS.md) §9 records the arguments for
and against.

### Bias-field recovery on simulated data

`tests/test_field_recovery.py` plants a field and measures its recovery. A smooth
multiplicative field of a given amplitude is applied to `brain_nu_ref.mnc`, which has
already been processed by `nu_correct` and is therefore close to uniform; the result is
written as `brain_nu_artificial.mnc`, and three implementations correct it at three knot
spacings: the PyTorch blocks, the same pipeline driving the original C++ blocks, and the
installed `nu_correct`, whose answers are recorded rather than recomputed.

| Planted field | Knot spacing | Non-uniformity planted | left by `torch` | by `legacy` | by `nu_correct` |
|---|---|---|---|---|---|
| 20% (`exp(0.2)` peak-to-peak) | 200 mm (default) | 4.14% | 0.31% | 0.31% | 0.31% |
| | 100 mm | | 0.61% | 0.61% | 0.61% |
| | 50 mm | | 1.51% | 1.52% | 1.51% |
| 40% | 200 mm (default) | 8.28% | 0.58% | 0.58% | 0.58% |
| | 100 mm | | 1.00% | 1.00% | 1.00% |
| | 50 mm | | 1.96% | 1.96% | 1.95% |

Three results follow. All three implementations agree on which field is present,
throughout the sweep, to between 1.0e-4 and 5.9e-4 relative RMS. That is the comparison
an implementation can be held to, and it is why this sweep is more informative than any
single end-to-end number.

N3 recovers most but not all of a field, and the proportion depends almost entirely on
`--distance`: at the shipped 200 mm about 7% of the planted non-uniformity remains, at
50 mm about a third of it.

**`--distance` and `--lambda` must be set jointly**, which is relevant before reducing
`--distance` alone. A bias field is smooth, so the default spacing already has
sufficient freedom to represent one; halving it without adjusting the penalty spends the
additional coefficients on tissue contrast, which is returned as field that was not
present. At 50 mm and the default `--lambda`, the corrected volume is further from the
truth than the uncorrected one, at 1.10× the original deviation.

Raising the penalty accordingly removes that effect. Residual non-uniformity over the
whole grid, with the best value in each column in bold:

*20% planted field (4.14% non-uniformity):*

| `--lambda` | `--distance` 200 mm | 100 mm | 50 mm |
|---|---|---|---|
| 1e-7 (default) | 0.31% | 0.61% | 1.51% |
| 1e-6 | **0.13%** | 0.22% | 1.01% |
| 1e-5 | 0.33% | **0.17%** | **0.25%** |
| 1e-4 | 0.85% | 0.54% | 0.35% |

*40% planted field (8.28% non-uniformity):*

| `--lambda` | `--distance` 200 mm | 100 mm | 50 mm |
|---|---|---|---|
| 1e-7 (default) | 0.58% | 1.00% | 1.96% |
| 1e-6 | **0.27%** | 0.44% | 1.30% |
| 1e-5 | 0.67% | **0.35%** | **0.48%** |
| 1e-4 | 1.71% | 1.10% | 0.71% |

The two amplitudes give the same result, approximately doubled: which penalty suits a
given spacing does not depend on the strength of the field, only on the freedom
available to the spline. Each column has an interior minimum — too small a penalty and
the fit follows tissue contrast, too large a penalty and it cannot follow the field. The
minima are at `1e-6` for 200 mm and `1e-5` for both 100 mm and 50 mm, identically at
both amplitudes.

The optimum therefore moves one decade for the first halving of the spacing and does not
move for the second. **Raise `--lambda` by approximately one decade for each halving of
`--distance`** remains the applicable rule, although it deliberately overshoots at
50 mm, which is the preferable direction of error here. The columns are not symmetric
about their minima: at 50 mm, `1e-4` costs 0.10 points relative to the best cell while
the default `1e-7` costs 1.26, so an error of an order of magnitude upward is
inexpensive and one downward is not.

The two parameters constitute a single setting: `--distance` determines how many
coefficients describe the field, `--lambda` how much bending is permitted between them.
Neither is meaningful without the other, which is why they are swept jointly rather than
individually.

The 20% table is also printed by `python3 -m torch_n3 --help`.

**These are `--solver normal`'s values, and they depend on that choice only
marginally.** Both tables were re-measured under every direct solver on 2026-07-31 with
`python3 -m tests.tables`, which is also the procedure for re-measuring them after any
block is modified. `normal` reproduces all 24 cells exactly; `qr` and `dr` differ in 2
cells of 24 and `blocked` in 3, none by more than 0.01 points, and the largest relative
change in any cell is 3.2% (`qr`, `dr`) or 4.7% (`blocked`). Every conclusion above —
the interior minimum in each column and its location, the decade-per-halving rule, the
asymmetry at 50 mm — holds identically under all four. These runs are 30 iterations
deep, well past the point at which two implementations' volumes cease to be comparable,
so what the tables measure is the trade-off rather than the arithmetic.

It also establishes the precision at which the tables should be read. `dr` is
algebraically `qr` at a fixed weight, and 30 iterations move it 0.3% from `qr` in the
most sensitive cell. The last digit of a cell is not meaningful.

The shipped `1e-7` is not the best cell in either table: `1e-6` at the default spacing
halves the residual at both amplitudes. This is not a recommendation. The measurement is
one synthetic field on one volume, constructed from three low-order harmonics and
smoother than a real coil profile; a field with more structure is the case in which the
additional penalty would begin to cost. The tables establish the shape of the trade-off,
not the values within it.

### Gaussian Parzen window (`--parzen-sigma`)

N3's `-parzen`/`-window` is not a Parzen window. `WHistogram::add` divides each sample
linearly between the two bin centres it falls between: a triangular kernel exactly one
bin wide, whose width is determined by `--bins` and by the range `-auto_range` selected
for that iteration, rather than by any property of the measurement. `--parzen-sigma s`
replaces it with the estimator the name denotes: a Gaussian of standard deviation `s`
**bin widths**, evaluated at the bin centres and normalised per sample, so that a voxel
is distributed over as many bins as the kernel reaches. Nothing downstream is changed.

This is a modification to the algorithm and not part of the port. It is **disabled by
default**, the legacy backend rejects it, and every other value in this file is measured
without it. `python3 -m tests.parzen` produces the measurements below, and should be
re-run after any change to the histogram or the sharpening.

Residual non-uniformity on the planted-field sweep at `--solver normal`, 20% planted.
This is the same experiment as the `--lambda` × `--distance` tables above, so the first
row is that table's first row:

*At the shipped `--lambda 1e-7`:*

| window | `--distance` 200 mm | 100 mm | 50 mm |
|---|---|---|---|
| linear (N3) | 0.31% | 0.61% | 1.51% |
| `--parzen-sigma 0.5` | 0.31% | 0.61% | 1.52% |
| `--parzen-sigma 1` | 0.28% | 0.53% | 1.46% |
| `--parzen-sigma 2` | **0.22%** | 0.29% | 0.77% |
| `--parzen-sigma 4` | **0.22%** | **0.24%** | **0.34%** |

*Best cell in each column, `--lambda` swept over `1e-7 … 1e-4` as well:*

| window | `--distance` 200 mm | 100 mm | 50 mm |
|---|---|---|---|
| linear (N3) | 0.13% | 0.17% | 0.25% |
| `--parzen-sigma 1` | **0.12%** | 0.17% | 0.23% |
| `--parzen-sigma 2` | **0.12%** | **0.15%** | **0.17%** |
| `--parzen-sigma 4` | 0.16% | 0.17% | 0.19% |

The two tables are read together. At a fixed weight the window reduces the residual, by
a third at the shipped defaults and by a factor of four at 50 mm. Against a weight tuned
for the spacing it yields no reduction at 200 mm and about a third at 50 mm. The window
therefore acts largely as a **substitute for regularization**: it improves the cells that
were under-penalised, which is the axis `--lambda` already operates on, and the two
effects are not additive. Where `--lambda` has been tuned for the spacing, the window
has little effect; under the default parameters, `--parzen-sigma 2` is the less
expensive of the two adjustments.

A wide window carries a cost. `sigma 4` is the best of the sweep at 50 mm and the worst
at 200 mm with `--lambda 1e-6` (1.21× the linear split) and at every `1e-4` cell. The
mechanism is evident in the units: one bin is 0.026 log units on this volume, so
`sigma 2` adds a blur of 0.052 against the 0.064 standard deviation that `--fwhm 0.15`
instructs `sharpen_hist` to remove, and `sigma 4` exceeds it. The window adds a Gaussian
of which the deconvolution has not been informed, so beyond some width it under-sharpens.
The corresponding correction — removing the added width from `--fwhm` in quadrature — is
untested; no part of the sweep covers it.

End to end on `brain.mnc` under the shipped protocol the effect is not cosmetic: the
corrected volume moves 6.2e-3 relative RMS at `sigma 0.5` and 8.8e-2 at `sigma 4`, and
correspondingly away from `brain_nu_ref.mnc` (3.0e-3 → 4.9e-3 → 8.6e-2). The final
column is not a measure of accuracy, since N3 produced that reference and any change to
N3 moves away from it; the planted-field sweep is where the question is answered.

#### Recovery under noise, over 450 random fields per window

The measurements above are one analytic field on one volume with **no noise**, which is
the condition under which the window is least effective. `experiments/` plants random
fields on colin27 and adds Gaussian noise at a stated SNR. The same comparison there —
50 seeds × 3 amplitudes × 3 SNRs per window at 75 mm knots, `--solver normal` —
establishes what the noiseless tables cannot:

| | linear (N3) | σ 1 | σ 2 | σ 4 |
|---|---|---|---|---|
| 20 % planted, SNR ∞ | 0.81 % | 0.80 % | 0.80 % | **0.64 %** |
| 20 % planted, SNR 20 | 4.63 % | 4.84 % | 3.47 % | **1.71 %** |
| 80 % planted, SNR 20 | 5.46 % | 5.28 % | **4.33 %** | 4.34 % |
| better than N3, per trial | — | 40 % | 79 % | **94 %** |

(Median unexplained non-uniformity over the brain, shipped protocol; the last row is the
paired per-trial comparison over all 450 trials.)

**The reduction tracks noise rather than field amplitude**, which is the expected
behaviour of a kernel density estimate and is not observable in a noiseless test. It
also increases with the iteration count: under N3's own histogram, 30 → 50 iterations
makes the noisy cells worse (3.43 % → 4.63 % at 20 %/SNR 20), because the alternating
iteration returns the high-variance histogram's noise to the mapping at each pass; under
σ 4 the same cells continue to improve. The cost is about 1 % of run time.

The complete tables, the per-cell rates, `windows.png` and the two accompanying
limitations are in [experiments/README.md](experiments/README.md), "The histogram
kernel". In summary: σ 2 is the only width that is not the worst of the four in any cell
under either protocol; σ 4 is better still under the shipped protocol but is the worst of
the four at 80 % planted with 30 iterations; σ 1 does not improve on N3's linear split.

### Non-local means (`--denoise`)

The port carries a second modification, which addresses the same problem from the other
side. `--parzen-sigma` smooths the *histogram*; `--denoise` filters the *volume*, with
one non-local-means pass (Manjón et al. 2010, `blocks/denoise.py`) before the field is
estimated. Only the estimate is affected: the volume written out is the caller's own
intensities divided by the fitted field, never the filtered copy. It runs at full
resolution, before `--shrink`, because `Volume.shrink` is nearest-neighbour subsampling
and would otherwise alias the noise it is meant to remove.

Since both modifications suppress noise-driven variance in N3's data term, the question
is not whether either helps in isolation but whether they are **substitutes or
complements**. Measured together (`python3 -m tests.denoise`): the recovery experiment at
the shipped `--distance 200 --lambda 1e-7`, with white noise added at a stated SNR.
Non-uniformity left in the recovered field, lower is better:

| | SNR ∞ | SNR 40 | SNR 20 |
|---|---|---|---|
| linear (N3) | 0.31% | 0.29% | 0.51% |
| `--parzen-sigma 2` | **0.22%** | **0.23%** | 0.40% |
| `--parzen-sigma 4` | **0.22%** | 0.24% | 0.38% |
| `--denoise` | 0.29% | 0.42% | 0.66% |
| `--denoise --parzen-sigma 2` | **0.22%** | 0.33% | 0.44% |
| `--denoise --parzen-sigma 4` | 0.28% | 0.30% | **0.30%** |

On this one field the window appears the better of the two, and denoising a substitute
for it that is not worth its cost: of the nine paired comparisons the window alone beats
the same window with denoising added in six, ties in one, and loses two, and against
N3's own linear split denoising makes matters worse at both noise levels (0.42% against
0.29% at SNR 40, 0.66% against 0.51% at SNR 20). The noiseless column behaves as the
control predicts — with no noise to remove a spatial filter can only take away structure
the estimate was using, and the `sigma 4` row degrades from 0.22% to 0.28%.

**That reading is wrong, and the table above must not be quoted as the verdict.** It is
one analytic field, on one volume, at one seed, at a knot spacing of 200 mm where the
field has very little freedom. Over 450 random fields per configuration the conclusion
reverses on both axes: denoising is the *stronger* of the two, and the two are
complements rather than substitutes. The measurement is below; the full tables are in
[`experiments/README.md`](experiments/README.md), "Prefiltering the volume".

It is expensive. The filter runs at full resolution while the estimation runs on a grid
coarser by `--shrink 4`, so on `brain.mnc` it costs forty to fifty times the whole
estimation it feeds — 8.3 s against 0.15–0.19 s on a CPU (the ratio moves between runs
because the estimation is short enough for its timing to be noisy), or 0.24 s with
`--device cuda`. On colin27's 7.1 M voxels it is about 1.8 s of GPU time per estimate,
taking N3 from 0.7 s to 2.5 s and `hoyer` from 2.5 s to 4.4 s.

**It is therefore off by default, and belongs on data that is genuinely noisy**, where
it is worth considerably more than its cost. `--denoise --parzen-sigma 4` is the
configuration to reach for.

The filter is also available on its own, outside the N3 pipeline, as
`python3 -m torch_n3.denoise_cli input.mnc output.mnc [--search N] [--patch N]
[--strength X] [--device cuda]`. Unlike `--denoise` above, which filters a copy that
feeds only the field estimate, this writes the filtered volume itself.

#### Recovery under noise, over 450 random fields

The same Monte Carlo as the window's, on colin27 at 75 mm knots, `--solver normal`,
50 seeds × 3 amplitudes × 3 SNRs. Median non-uniformity left in the brain, `off → on`,
N3's own linear histogram:

| planted | SNR ∞ | SNR 40 | SNR 20 |
|---|---|---|---|
| **`n3`** 20% | 1.030 → 1.055 | 1.522 → 1.073 | 3.428 → 1.092 (**−68%**) |
| 40% | 2.307 → 2.435 | 2.832 → 2.419 | 4.050 → 2.496 (−38%) |
| 80% | 5.094 → 5.206 | 5.384 → 5.194 | 5.835 → 5.236 (−10%) |
| **`hoyer`** 20% | 0.519 → 0.731 | 0.563 → 0.754 | 2.193 → 0.753 (**−66%**) |
| 40% | 0.513 → 0.723 | 0.652 → 0.725 | 2.179 → 0.741 (−66%) |
| 80% | 1.121 → 1.046 | 1.178 → 1.046 | 2.247 → 0.911 (−59%) |

Paired per trial — same seed, same planted field, same noise draw — denoising improves
N3 on 15% of trials at SNR ∞, 88% at SNR 40 and 95% at SNR 20, and `hoyer` on 37%, 41%
and 85%.

**The result is not "denoising helps" but something sharper: it makes both estimators
nearly indifferent to noise.** N3 at 20% planted goes 1.055 / 1.073 / 1.092 as the SNR
falls from infinite to 20 — flat — where unfiltered it goes 1.030 / 1.522 / 3.428, a
factor of 3.3. The cost is paid where the control predicted it, at SNR ∞ and nowhere
else.

#### Denoising against the window, crossed

Crossing the two over the same 50 seeds settles the substitutes question on Monte-Carlo
data. Median, paired, `n3`:

| planted | SNR | linear | `sigma 4` | `--denoise` | both |
|---|---|---|---|---|---|
| 20% | ∞ | 1.030 | 1.036 | 1.055 | 1.059 |
| | 40 | 1.522 | 1.309 | 1.073 | **1.051** |
| | 20 | 3.428 | 1.912 | **1.092** | 1.136 |
| 40% | ∞ | 2.307 | **2.148** | 2.435 | 2.183 |
| | 40 | 2.832 | 2.443 | 2.419 | **2.204** |
| | 20 | 4.050 | 3.257 | 2.496 | **2.293** |
| 80% | ∞ | 5.094 | **4.923** | 5.206 | 4.965 |
| | 40 | 5.384 | 5.441 | 5.194 | **5.017** |
| | 20 | 5.835 | 6.815 | 5.236 | **5.095** |

**They are complements.** The median *paired* difference of both against denoising alone
is negative in all nine cells (−0.03 to −0.33 points), and against the window alone in
all six noisy cells (−0.22 to −1.67), costing only +0.03 to +0.05 at SNR ∞. Which of the
two carries a cell depends on the field: at 20% planted and SNR 20 the denoiser does
essentially all the work and the window adds little on top (60% of trials), while at 40%
and 80% the window is worth 0.2–0.3 points *on a denoised volume* in every noisy cell.

At 80% planted and SNR 20 the window alone is a net **harm** — 6.815 against N3's own
5.835, the only such cell in the sweep — and prefiltering removes it: both together give
5.095, the best in the row. The window's instability at large fields is a noise effect,
so the pair is more robust than either alone.

End to end on `brain.mnc` under the shipped protocol, denoising stops the iteration five
steps earlier (26 against 31) and moves the corrected volume 1.10e-1 relative RMS. That
figure is not an accuracy verdict — N3 produced `brain_nu_ref.mnc`, so anything that
changes N3 moves away from it — but it establishes that the option is far from cosmetic.

## Requirements

Python 3.12, `torch`, `numpy`, `cffi` and `minc2_simple`.

The MINC toolkit is *not* needed to run `torch_n3` or its tests. It is needed to read
gzipped or MINC1 input (`load_volume` shells out to `mincconvert`), to re-record the
reference answers, and to run the legacy backend's comparisons. All are already
installed here.

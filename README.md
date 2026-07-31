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
| `--lambda` | 1e-7 | Bending-energy penalty on the spline fit. Moves *with* `--distance` — about a decade per halving. See [the trade-off](#does-it-actually-remove-a-bias-field). |
| `--iterations` | 50 | Iteration budget, one number per stopping stage. |
| `--stop` | 0.001 | Stop when the field moves less than this, one per stage. |
| `--field-floor` | 0.1 | Smallest field value allowed, so the division stays sane. |
| `--device` | cpu | Any torch device, e.g. `cuda`. See [Why end-to-end numbers stop at three digits](#why-end-to-end-numbers-stop-at-three-digits) first. |
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
| `torch_n3/_legacy/` | The CFFI shim, plus the N3 (`n3/`) and EBTKS (`ebtks/`) sources it compiles. |
| `torch_n3/cli.py` | The command line. |

The legacy backend is deliberate: it makes the original code a numerical oracle, so each
PyTorch block has something exact to be tested against. `torch_n3.backends.resolve` picks
between the two and the pipeline runs on either — which is all `--backend legacy` does.

It is *not* the installed `nu_correct`, and the difference matters when reading any
number below. Original N3 is a Perl script driving a dozen separate executables, so
every intermediate volume it computes goes out to a MINC file — 12-bit or 16-bit,
rescaled slice by slice — and comes back rounded. The shim calls the same C++ routines
directly, on `float64` arrays, in one process. So `--backend legacy` is the original
arithmetic *without* the original's quantisation, which is why it does not reproduce
`nu_correct` bit for bit either: it lands 5.2e-3 relative RMS from
`brain_nu_ref.mnc`, *further* out than the PyTorch blocks at 3.0e-3. What it is for
is comparing block against block with nothing rounded in between.

---

## Building and testing

The CFFI extension is built in place and is not checked in:

```bash
python3 torch_n3/_legacy/build_legacy.py
```

It compiles `Spline.cc`, `TBSpline.cc`, `DHistogram.cc`, `WHistogram.cc`,
`sharpen_hist.cc` and `correctField.cc` from `torch_n3/_legacy/n3/`, and the seven
EBTKS sources they need from `torch_n3/_legacy/ebtks/`. Both trees are vendored byte
for byte — out of `legacy/N3/src` and `legacy/EBTKS` — unmodified and checked in, so
the extension builds with no MINC toolkit and no N3 or EBTKS checkout.

The only thing it links from outside is **LAPACK and BLAS**, and that choice is not
free: see [Which LAPACK](#which-lapack-the-legacy-backend-links). `volume_io`,
`time_stamp` and `ParseArgv` are supplied by small stand-ins in
`torch_n3/_legacy/compat/`, which is what lets `correctField.cc` and `args.cc` stay
byte-identical while needing no libminc2. Only the tests and `--backend legacy` need
any of this.

```bash
python3 -m pytest              # the whole suite, about 26 s
python3 -m pytest tests/test_spline.py         # one block
python3 -m pytest -k "legacy"                  # everything, on the C++ backend
```

The tests come in four kinds:

- **Per block** (`test_histogram.py`, `test_sharpen.py`, `test_spline.py`,
  `test_field.py`) — the PyTorch block against the same code compiled into the CFFI
  shim. These are the tight ones, and the ones that pin the port.
- **Against the programs** (`test_pipeline.py`, `test_minc_tools.py`, `test_volume.py`)
  — the cases from `legacy/N3/testing/CMakeLists.txt`, re-expressed as comparisons with
  what the installed N3 answered.
- **Against a planted field** (`test_field_recovery.py`) — the question a user has,
  rather than whether the port matches: given a known non-uniformity, is it recovered?
- **Against ourselves** (`test_reproducibility.py`) — a checked-in volume every
  backend and device has to land on. See [below](#will-it-give-the-same-answer-on-your-machine).

`python3 -m tests.margins` prints where every one of those comparisons sits against its
bound; [PROBLEMS.md](PROBLEMS.md) is that table plus what is wrong with it.

No test runs an N3 program. Those programs are deterministic, so their answers were
recorded once into `tests/reference/` (9.4 MB) and are read from there — which keeps
the suite fast, keeps it honest about whether a failure is yours or a different build
of theirs, and lets it run wherever. `tests/inputs.py` holds the inputs they were
given, so both sides build them the same way. To re-record:

```bash
python3 -m tests.regenerate_reference     # needs the MINC toolkit on PATH
git diff --stat tests/reference           # empty if they still say the same thing
```

(`tests/data/brain_nu_ref_legacy.mnc` is rewritten too, and will always show as
changed: MINC stamps each file with the user, host and time that wrote it. Its voxel
data is reproducible; those header bytes are not.)

**The suite needs no MINC program at all.** The volumes it runs on are checked in
as MINC2 under `tests/data/` (10 MB, two thirds of it the reproducibility reference)
— byte-for-byte the same images as `legacy/N3/testing/`, converted once, because
`minc2_simple` opens MINC2 only. The one exception is
`test_gzipped_minc1_input_is_readable`, which is *about* the conversion path and
skips without `mincconvert`.

### Will it give the same answer on your machine?

`tests/test_reproducibility.py` is there to say so. `tests/data/brain_nu_ref_legacy.mnc`
is a volume checked into the repository — this pipeline's output on `brain.mnc`,
with the original C++ blocks driving it, stored `float64` so that it records what
was computed rather than what a 16-bit file could hold. Both backends, on every
device available, have to land on it to within one part in 65535 of relative RMS,
which is N3's own working precision: the legacy passes every intermediate volume
to the next program through a 16-bit file.

Relative RMS throughout — RMS difference over mean signal, which is the measure
`compare_nu_result.pl` uses. Not the largest difference anywhere: over 900k
voxels that is decided by a handful at the mask edge and says nothing about the
volume.

| | relative RMS from the reference | of the bound |
|---|---|---|
| `legacy`, CPU | 0 — bit-identical, it wrote the file | 0.0% |
| `torch`, CPU | 5.52e-08 | 0.4% |
| `torch`, CUDA | 5.52e-08 | 0.4% |

Almost all of that is the two block implementations disagreeing, not the
platform: CPU and GPU differ from each other by far less than either differs
from the legacy.

It runs one iteration rather than the shipped fifty, and that is not laziness.
N3's histogram range is taken from the data and then rounded to the six decimals
its text interchange prints, so a voxel on a bin boundary can fall either side
of it. After one iteration the backends agree to 5.5e-08 relative RMS; at two,
one whole count moves between bins — out of the 3,724 samples the shrunken
estimation grid contributes — and they finish 1.17e-3 apart, four orders of
magnitude worse. Nobody's implementation can be pinned past that — see
[Amplification](#why-end-to-end-numbers-stop-at-three-digits) below — so the test stops where the answer is
still a continuous function of rounding error.

### How close is it?

Block by block, against the original C++. These are *measured* differences, not
the bounds the tests assert — those are in the tests, and where each one sits
against its bound is tabulated in [PROBLEMS.md](PROBLEMS.md).

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

The last two are loose for reasons that are not going away. The normal equations N3
solves have a condition number around 1e13 at the default 200 mm knot spacing — the
knots are further apart than the volume is wide — so the *coefficients* are barely
determined and any two solvers disagree about them; the fitted field, which is what
gets used, is fine. And `correct_field` relaxes in raster order in `float`, which is
inherently sequential; the port sweeps the two checkerboard colours in turn instead,
the same iteration reordered so that it vectorises.

### Why end-to-end numbers stop at three digits

The blocks above agree to between six and fifteen digits. Run the whole pipeline
and that collapses to three:

| End to end, on `brain.mnc`, shipped protocol | relative RMS |
|---|---|
| `torch` vs N3's `brain_nu_ref.mnc` | 3.01e-3 |
| `legacy` vs N3's `brain_nu_ref.mnc` | 5.21e-3 |
| `torch` vs `legacy` | 3.1e-3 |
| `torch` on CPU vs the same code on a GPU | 1.3e-3 |

The legacy suite asks for 1e-4. Nothing here reaches it, and the last row is the
tell: that is one implementation disagreeing with *itself* over nothing but the
order of a few reductions. Two things are going on, and neither is an error in
the port.

**Quantisation.** Legacy N3 is a Perl script driving a dozen separate programs,
so every intermediate volume goes out to a MINC file and comes back rounded:
12 bits before the mask is applied, 16 after, scaled slice by slice on write and
rescaled onto a single global grid on read — the two do not even agree with each
other. Every iteration goes through that, and the reference volume was built
through all of them. `torch_n3` keeps float64 from end to end, which is more accurate
and therefore *not the same*. Reproducing `brain_nu_ref.mnc` voxel for voxel
would mean modelling MINC's storage rather than N3; see `PLAN.md`.

This cuts both ways, and it is why `--backend legacy` does not reach 1e-4
either: it is the original arithmetic with the original's rounding removed.

**Amplification.** N3's loop feeds its output back into itself, so whatever
survives gets multiplied. The two backends agree on the field to **1.4e-7** after
one iteration and to **9.6e-4** after ten — a factor of 6,900.

Worse, the growth does not start from float noise. The histogram range is taken
from the data and then rounded to the six decimals N3's text interchange prints,
so a voxel sitting on a bin boundary can fall either side of it and a whole count
moves between bins. That is a genuine discontinuity in the middle of the loop:
past it, an end-to-end comparison is recording which side of a rounding boundary
one voxel landed on. The next section is that effect caught in the act.

### Which LAPACK the legacy backend links

The clearest demonstration of all that, and a cautionary tale.

The spline fit solves normal equations with a condition number around `1e13` —
at the default 200 mm the knots are further apart than the volume is wide — and
it calls LAPACK's `dsysv` to do it. The shim used to link `libEBTKS.a`, which
bundles its own f2c'd LAPACK. It now compiles vendored sources and links the
**system** LAPACK/BLAS (OpenBLAS here), so that it needs nothing from the MINC
toolkit. Same object files, same inputs; the only difference is which `dsysv`.

At block level the swap is nothing. Fitting the default 200 mm spline to a
tilted plane on `chunk.mnc`, the two builds differ by:

| | largest | relative RMS |
|---|---|---|
| spline coefficients | 1.5e-07 | 2.9e-06 |
| the fitted field | 2.4e-10 | **3.1e-11** |

Three parts in `1e11`. Neither solver is wrong: at that conditioning the
coefficients are not determined to better than `1e-4` by *any* solver, which is
why this repository compares fitted fields and never coefficients.

End to end on `brain.mnc`, the same swap. `legacy` against `torch`, one row per
iteration count, so that the loop can be watched doing its work:

| | EBTKS's f2c'd LAPACK | system LAPACK |
|---|---|---|
| `legacy` vs N3's `brain_nu_ref.mnc`, shipped protocol | 0.3701% | **0.5211%** |
| `legacy` vs `torch`, 1 iteration | 5.49e-08 | 5.52e-08 |
| `legacy` vs `torch`, 2 iterations | 5.55e-08 | **1.17e-3** |
| `legacy` vs `torch`, 3 iterations | 5.65e-08 | 2.12e-3 |
| `legacy` vs `torch`, 4 iterations | 5.73e-08 | 1.69e-3 |
| `legacy` vs `torch`, 5 iterations | 5.82e-08 | 5.40e-4 |
| `legacy` vs `torch`, 6 iterations | **8.37e-04** | 5.07e-4 |

Read each column downwards. Both hold agreement at `5.5e-08` and then lose it
all at once — that is the bin-boundary flip, and it is a step, not a drift. What
the swap changed is *when*: the sixth iteration on EBTKS's LAPACK, the second on
the system one.

So a perturbation of `3.1e-11` in the fitted field moved a discontinuity four
iterations earlier, and with it the end-to-end answer by four orders of
magnitude. Which LAPACK you link is not an implementation detail of the oracle.

It also sets the ceiling on how far this test can be pushed: agreement survives
only up to the flip, wherever it happens to fall for a given build, which is why
`tests/test_reproducibility.py` pins one iteration and not more. Two bounds
moved as a result — deliberately, with the measurements written down — in
[PROBLEMS.md](PROBLEMS.md) §8.

The same knife-edge shows up without changing LAPACK at all: the same `torch`
code on a CPU and on a GPU agrees to `6.3e-11` after one iteration and `1.0e-10`
after two, then parts company at `1.17e-3` on the third.

### Running this experiment on your own LAPACK

Everything above is one machine's answer. If you are packaging this, or moving
it to a cluster, the number you want is **where your own cliff falls** — because
`tests/inputs.py::PLATFORM_PROTOCOL` has to stay below the smallest cliff of any
platform this is expected to run on, and it is currently at 1 with no margin.

```bash
python3 -m tests.convergence                      # legacy vs torch, 1..6 iterations
python3 -m tests.convergence --device cuda        # and CPU vs GPU
python3 -m tests.convergence --volume chunk       # 20x smaller, just checks it runs
python3 -m tests.convergence --max-iterations 10
```

It prints the platform, the LAPACK the extension actually resolved against, and
one row per comparison:

```
platform:   Linux-6.17.0-35-generic-x86_64-with-glibc2.39, python 3.12.3, torch 2.13.0+cu130
shim links: liblapack.so.3, libblas.so.3, libopenblas.so.0

relative RMS                     1          2          3          4          5          6
legacy vs torch           5.52e-08    0.00117    0.00212    0.00169    0.00054   0.000507
                       ^ cliff at 2 iterations (5.52e-08 -> 0.00117)
torch cpu vs cuda          6.3e-11   9.97e-11    0.00117   0.000214    0.00107   8.29e-05
                       ^ cliff at 3 iterations (6.3e-11 -> 0.00117)
```

**Report the cliff, not the numbers.** The values before it are only
interesting for being small; the ones after it are not comparable between
machines at all — they record which side of a rounding boundary a single voxel
fell on.

#### Building against a different LAPACK

The extension links `-llapack -lblas` by default. Two environment variables
override that, and both are read at build time:

```bash
rm -rf torch_n3/_legacy/build          # cffi will not relink without this

# Whatever your platform calls them
N3_LAPACK_LIBS="mkl_rt" \
N3_LAPACK_LIB_DIRS="/opt/intel/oneapi/mkl/latest/lib" \
    python3 torch_n3/_legacy/build_legacy.py

# EBTKS's own bundled f2c'd LAPACK, if you have libEBTKS.a. This is the
# left-hand column of the table above. The archive resolves after our own
# objects, so only its clapack members are taken.
N3_LAPACK_LIBS="EBTKS" N3_LAPACK_LIB_DIRS="$MINC_TOOLKIT/lib" \
    python3 torch_n3/_legacy/build_legacy.py
```

Without rebuilding, on Linux, `LD_PRELOAD` also works, because the default build
resolves `liblapack.so.3` dynamically:

```bash
LD_PRELOAD=/path/to/other/liblapack.so.3 python3 -m tests.convergence
```

On Debian and Ubuntu, `update-alternatives --config liblapack.so.3-$(uname -m)-linux-gnu`
switches the system-wide one — but only between what is installed. This machine
has only `libopenblas`; `apt install liblapack3` adds the reference build, which
is the interesting second data point, being the same Fortran that EBTKS's f2c'd
copy was translated from.

Whichever route, check the `shim links:` line the script prints — the library
named in a build is not always the one the loader finds, and a statically linked
LAPACK will not appear there at all. It reports `no dynamic LAPACK (static?)`
in that case, which is the expected output for the EBTKS build above.

#### If your cliff is at 1

Then no iteration count is safe on your platform, and
`tests/test_reproducibility.py` cannot hold there — the golden volume records a
run that your build does not reproduce even once round the loop. That is worth
reporting rather than working around: it would mean the flip is being triggered
by something at the *first* pass, which nothing here has seen and which the
block-level tests should have caught.


### Does it actually remove a bias field?

`tests/test_field_recovery.py` plants one and asks for it back. A smooth
multiplicative field of a set amplitude goes onto `brain_nu_ref.mnc` — which
has already been through `nu_correct`, so it is close to uniform to begin with —
the result is written out as `brain_nu_artificial.mnc`, and three
implementations are asked to correct it at three knot spacings: the PyTorch
blocks, the same pipeline driving the original C++ blocks, and the installed
`nu_correct`, whose answers are recorded rather than recomputed.

| Planted field | Knot spacing | Non-uniformity planted | left by `torch` | by `legacy` | by `nu_correct` |
|---|---|---|---|---|---|
| 20% (`exp(0.2)` peak-to-peak) | 200 mm (default) | 4.14% | 0.31% | 0.31% | 0.31% |
| | 100 mm | | 0.61% | 0.61% | 0.61% |
| | 50 mm | | 1.51% | 1.52% | 1.51% |
| 40% | 200 mm (default) | 8.28% | 0.58% | 0.58% | 0.58% |
| | 100 mm | | 1.00% | 1.00% | 1.00% |
| | 50 mm | | 1.96% | 1.96% | 1.95% |

Three things to read off that. All three implementations agree about *which*
field is there, everywhere in the sweep, to between 1.0e-4 and 5.9e-4 relative
RMS — the comparison an implementation can actually be held to, and the reason
this sweep is worth more than any single end-to-end number.

N3 recovers most but not all of a field, and how much depends almost entirely
on `--distance`: at the shipped 200 mm it leaves about 7% of what was planted,
at 50 mm about a third of it.

And **`--distance` and `--lambda` have to move together**, which is worth
knowing before reaching for a smaller `--distance` on its own. A bias field is
smooth, so the default spacing already has enough freedom to represent one;
halving it without touching the penalty means the extra coefficients get spent
following tissue contrast, and come back as field that was never there. At
50 mm and the default `--lambda` the "corrected" volume is *further* from the
truth than the uncorrected one — 1.10× the original deviation.

Raising the penalty to match undoes that. Residual non-uniformity left behind,
over the whole grid — **best in each column in bold**:

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

The two amplitudes give the same picture, roughly doubled: which penalty suits a
given spacing does not depend on how strong the field is, only on how much
freedom the spline has. Read down a column and each one has a minimum — too
little penalty and the fit chases tissue contrast, too much and it cannot follow
the field. The minima sit at `1e-6` for 200 mm and `1e-5` for both 100 mm and
50 mm, identically at both amplitudes.

So the optimum moves one decade for the first halving of the spacing and does
not move at all for the second. **Raise `--lambda` by about a decade each time
you halve `--distance`** is still the rule to carry away, but it is a rule that
deliberately overshoots at 50 mm — which is the right way to be wrong here. The
two columns are not symmetric about their minima: at 50 mm, `1e-4` costs 0.10
points against the best cell while the default `1e-7` costs 1.26, so erring an
order high is cheap and erring low is not.

The two knobs are one setting: `--distance` decides how many coefficients
describe the field, `--lambda` how much bending is allowed between them. Neither
means much without the other, which is the whole point of sweeping them
together rather than one at a time.

The 20% table is in `python3 -m torch_n3 --help` too, so it is to hand when the
question comes up.

Worth noting that the shipped `1e-7` is not the best cell in either table —
`1e-6` at the default spacing halves the residual at both amplitudes. Don't
read a recommendation into that. This is one synthetic field on one volume,
built from three low-order harmonics, and it is smoother than a real coil
profile; a field with more structure is exactly the case where the extra
penalty would start to cost. What the tables are good for is the *shape* of
the trade-off, not the numbers in them.

## Requirements

Python 3.12, `torch`, `numpy`, `cffi` and `minc2_simple`.

The MINC toolkit is *not* needed to run `torch_n3` or its tests. It is needed to read
gzipped or MINC1 input (`load_volume` shells out to `mincconvert`), to re-record the
reference answers, and to run the legacy backend's comparisons. All already installed
here — nothing needs fetching.

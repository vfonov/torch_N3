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
| `--solver` | normal | How the spline fit is solved. `qr` fits the same spline through a far better-conditioned system and is much more reproducible across machines; `blocked` gives the same answer without holding the design matrix, which is what to use at a fine `--distance`; `dr` is `qr` reparameterized so that the penalty is diagonal, which makes a whole `--lambda` grid cost one factorization. (`sparse` does not converge — see below.) See [Two ways to solve the same fit](#two-ways-to-solve-the-same-fit). |

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
bundles its own f2c'd LAPACK. It then compiled vendored sources and linked the
**system** LAPACK/BLAS (OpenBLAS here), so that it needs nothing from the MINC
toolkit; the measurements below are from that build. It now links no LAPACK of
its own by default at all, and instead shares PyTorch's -- see "Building
against a different LAPACK" below for why. Same object files, same inputs;
the only difference is which `dsysv`.

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

By default the extension links no LAPACK/BLAS of its own at all: `dgemm_`,
`dsysv_` and the rest are left as undefined symbols, resolved at import time
against whatever is already loaded in the process. Every entry point imports
`torch` before the shim, and PyTorch always loads its own dependency library
with `RTLD_GLOBAL` (see its `__init__.py`, "Note [Global dependencies]") for
exactly this kind of sharing -- so in practice this means the shim always
rides on **PyTorch's own BLAS**, not a second one this package chose. That
matters on macOS in particular: a conda or Homebrew environment's own
`liblapack.dylib` is typically a symlink to an OpenBLAS build with its own
bundled `libomp.dylib`, and PyTorch's wheel bundles a *different*
`libomp.dylib` -- linking that OpenBLAS directly, the way `-llapack -lblas`
used to, loads two copies of LLVM's OpenMP runtime into one process, which
aborts (or, forced past that with `KMP_DUPLICATE_LIB_OK`, segfaults). Sharing
PyTorch's avoids that on any platform, without this build needing to know
which BLAS vendor PyTorch chose.

Two environment variables give the shim an explicit LAPACK/BLAS dependency of
its own again, overriding the default -- for comparing against a specific
build, the way the table above does. Both are read at build time:

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

Without rebuilding, `LD_PRELOAD` should still let you substitute a LAPACK on
Linux: preloaded libraries are given first claim on the process's global
symbol scope, ahead of whatever `torch` loads afterwards, which is what the
default build now resolves `dgemm_`/`dsysv_` against instead of a dependency
of its own. Unverified on an actual Linux build past that reasoning --
confirm it with the `shim links:` line below before trusting it:

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


### Two ways to solve the same fit

Everything above is downstream of one decision N3 made in 1998: it fits the
spline by forming the normal equations.

```
(AtA + lambda N J) c = At f
```

`A` holds the basis functions at every masked voxel. Squaring it squares the
condition number, and at the shipped 200 mm knot spacing — where the knots are
further apart than the head is wide — that lands at ~`1e13`. A system that
ill-conditioned does not have a wrong answer; it has an answer whose last digits
belong to whichever BLAS computed it. That is the whole mechanism behind the
LAPACK story above, and behind a CPU and a GPU running identical code diverging.

The same fit can be posed without ever squaring `A`. Stack the penalty
underneath the data and solve one least-squares problem:

```
[      A       ]         [ f ]
[              ] c   ~   [   ] ,     where  D'D = J
[ sqrt(lam N) D]         [ 0 ]
```

Its normal equations are the ones above, term for term — same minimiser, same
objective, not an approximation — but the matrix being factorised is `A` itself,
so the condition number is the square root: ~`1e6`. `--solver qr` selects it.

Measured on `brain.mnc`, on the 24×28×24 grid `--shrink 4` leaves — 3,724
samples and 80 coefficients, which is the fit N3 actually performs:

| | `--solver normal` | `--solver qr` |
|---|---|---|
| condition number of the system solved | 5.3e12 | 2.3e6 |
| fitted field, CPU vs GPU, relative RMS | 2.5e-9 | 3.0e-13 |
| iterations before CPU and GPU diverge | 3 | 7 |
| iterations before `legacy` and `torch` diverge | 2 | 4 |
| 10 iterations, CPU | 0.38 s | 0.41 s |
| peak RSS, `--shrink 1` | 1.0 GB | 1.3 GB |

The third row is the one that matters: the cliff is where an end-to-end
comparison stops measuring the code and starts measuring which side of a
rounding boundary one voxel fell on. The fourth is the surprise — the QR tracks
the *C++ oracle* longer than the port's own normal equations do, even though the
oracle solves the normal equations itself. Being the more accurate solve turns
out to be worth more than matching the other implementation's formulation.

The cost is that `A` is held dense rather than only `AtA`, which is the last two
rows. `python3 -m tests.convergence --solver qr` measures the cliffs on your
machine.

#### Getting the same answer without holding `A`

`--solver blocked` is the same stacked system reached a band at a time. Sort the
rows by which first-axis knot they belong to and `A` becomes banded: a sample
whose corner is `k` touches only columns `[k·n₁n₂, (k+4)·n₁n₂)`. So the rows can
be folded into a running `R` one window at a time, and `A` never exists. It is
the same factorization in a different order, so it gives `qr`'s answer — they
agree to 4e-15, which is rounding — and it inherits the conditioning and the
CPU/GPU reproducibility exactly (1.23e-13 against 1.27e-13).

What it buys is the resource cost, and only at fine spacings, where the dense
stack stops fitting. On `chunk.mnc`:

| `--distance` | coefficients | window | `qr` | `blocked` | `normal` |
|---|---|---|---|---|---|
| 200 mm | 64 | 64 | 0.12 s | 0.18 s | 0.05 s |
| 50 mm | 245 | 140 | 0.37 s | 0.29 s | 0.13 s |
| 12.5 mm | 3168 | 792 | 12.72 s | **3.45 s** | 8.84 s |
| 12.5 mm, peak RSS | | | 7.53 GB | **1.55 GB** | 1.09 GB |

At 200 mm there is one group and `blocked` *is* `qr`, plus the cost of a sort —
so at the shipped spacing there is no reason to prefer it. Below 50 mm there is
every reason: at 12.5 mm it is 3.7× faster than `qr` on a fifth of the memory,
and faster than the normal equations too.

#### Sweeping `--lambda`: `--solver dr`

`--solver dr` answers the same stacked system, reparameterized so that the
penalty becomes **diagonal** — the Demmler–Reinsch basis of the P-spline
literature. Factorize `[A; sqrt(λ₀N) D]` once at an *anchor* weight λ₀, so that
`R₀ᵀR₀ = AᵀA + λ₀NJ`, and set `D̃ = sqrt(N) D R₀⁻¹`. Then for every weight

```
AᵀA + λNJ  =  R₀ᵀ (I + (λ − λ₀) D̃ᵀD̃) R₀
```

and eigendecomposing `D̃ᵀD̃ = U diag(γ) Uᵀ` makes the middle factor diagonal. A
fit becomes one elementwise division by `1 + (λ − λ₀)γ` and a triangular solve.

At `λ = λ₀` the divisor is 1 and this *is* `qr`, back-substitution and all — the
two agree to rounding, and `dr` meets every bound `qr` meets at identical
margins. What it buys is the sweep. The QR, the triangular solve and the
eigendecomposition do not depend on `λ`, so `BSplineField.refit(lam)` is the
division alone:

Measured on `brain.mnc`'s estimation grid — one spline fit per solver, against
`dr`'s marginal cost for another weight:

| `--distance` | coefficients | `normal` | `qr` | `blocked` | `dr` | `dr` refit |
|---|---|---|---|---|---|---|
| 200 mm | 80 | 4.9 ms | 5.6 ms | 8.6 ms | 7.3 ms | **0.033 ms** |
| 100 mm | 150 | 3.6 ms | 9.0 ms | 12.5 ms | 14.9 ms | **0.044 ms** |
| 50 mm | 392 | 8.3 ms | 27.7 ms | 26.5 ms | 43.6 ms | **0.140 ms** |

That is what makes GCV or REML smoothing-parameter selection affordable here,
and it is the tool to reach for when re-measuring the `--lambda` × `--distance`
tables above.

**For a single weight it is the slowest solver, and there is no reason to pick
it.** The eigendecomposition is `O(k³)` on top of everything `qr` does and buys
nothing until a second weight is asked for: a whole 30-iteration run takes
1.8 s at 50 mm where `qr` takes 1.2 s and `normal` 0.4 s. It pays from the
second weight and is dramatic by the fourth. Below that, use `qr`.

#### What each solver costs on a GPU

Peak CUDA memory *allocated* (not process RSS) for one spline fit on
`brain_nu_ref.mnc`, measured 2026-08-01 on an RTX A6000. `legacy` is CFFI and
CPU-only; `sparse` goes through `scipy` on the CPU; neither has a GPU
footprint.

| estimation grid | `--distance` | coefficients | `normal` | `qr` | `blocked` | `dr` |
|---|---|---|---|---|---|---|
| shrink 4 (3.7k samples) | 200 mm | 80 | 15 MB | 28 MB | 30 MB | 23 MB |
| shrink 4 | 50 mm | 392 | 26 MB | 61 MB | 35 MB | 48 MB |
| shrink 4 | 12.5 mm | 7,581 | 1.77 GB | 2.87 GB | 3.63 GB | 4.62 GB |
| shrink 1 (238k samples) | 50 mm | 392 | 355 MB | 2.90 GB | 1.14 GB | 1.73 GB |
| shrink 1 | 25 mm | 1,452 | 370 MB | 8.75 GB | 1.36 GB | 5.65 GB |

At the shipped protocol the whole 30-iteration pipeline peaks near 100 MB
whichever solver runs, so none of this matters there. Off it, two things do.

`normal` is the most frugal by an order of magnitude — it holds `AᵀA`, never
`A` — which is the real counterweight to its conditioning.

And **`blocked`'s advantage is about the sample-to-coefficient ratio, not about
`--distance`**. It wins decisively where samples greatly outnumber coefficients
(1.36 GB against `qr`'s 8.75 GB at shrink 1 and 25 mm). Where coefficients
*exceed* samples — shrink 4 at 12.5 mm, 7,581 against 3,724 — it is the
second-worst of the four, because its running `R` is `O(size²)` and there is no
longer much `A` to avoid holding.

`γ` is non-negative, so the divisor is never below 1 and no amount of dynamic
range in `γ` can reach the answer. The one constraint is that the basis is
valid only at or above its anchor — below it the divisor can pass through zero
— so `anchor=` belongs at the bottom of the grid you mean to sweep, and
`refit()` raises underneath it rather than returning a number.

**The textbook version of this does not work here**, which is worth knowing
before reaching for it. Demmler–Reinsch is normally written on the QR of the
design `A` alone. But `A` here is the *masked* design, and at fine knot spacings
the mask leaves basis functions with no data under them at all:

| `--distance` | coefficients | rank(A) | cond(A) | cond(stacked) | cond(normal) |
|---|---|---|---|---|---|
| 200 mm | 64 | 64 | 8.4e7 | 3.5e6 | 1.2e13 |
| 100 mm | 100 | 100 | 7.1e6 | 8.6e5 | 7.5e11 |
| 50 mm | 245 | **243** | **7.5e12** | 2.9e5 | 8.6e10 |

So `R` from `A` alone is *worse* conditioned than the stacked system at every
spacing, and at 50 mm `A` is rank deficient and `R` singular to working
precision — worse even than the normal equations there. `D R⁻¹` then overflows
into a `γ` with 83 non-positive entries reaching `-5.7e6`, and the divisor
passes through zero. Clipping `γ` at zero, the usual advice, does not rescue it:
the eigenvectors are as damaged as the eigenvalues. Anchoring on the stacked
matrix costs nothing and removes the failure, because the penalty rows span
exactly the directions the data leaves empty. See [PROBLEMS.md](PROBLEMS.md)
§12.

#### What did not work

`--solver sparse` holds the same stacked system in `scipy.sparse` and hands it
to `lsqr`. **It does not converge, and should not be used for results.** The
reason is structural rather than a matter of tuning: a rectangular system rules
out every direct sparse solver in `scipy.sparse.linalg` — there is no sparse QR
there — which leaves iterative methods, and LSQR's convergence is governed by
the very condition number the stacked form exists to reduce. At 200 mm it stops
after ~2,950 iterations reporting `istop=3` ("condition number exceeds
`conlim`"), 18 s in and 1.8e-3 away from the direct answer — a larger error than
the gap between this port and the original C++. Raising the iteration limit
changes nothing; it is not stopping early. `BSplineField.solve_info` reports
what LSQR said. It is kept because the measurement is worth having, and because
`blocked` is the answer to the problem it was reaching for.

**`normal` is the default**, and every recorded reference in `tests/` was
produced with it. Changing that is a deliberate decision with a real bill
attached — it moves every end-to-end number in this file — and
[PROBLEMS.md](PROBLEMS.md) §9 is where the case for and against is kept.

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

**These are `--solver normal`'s numbers, and they barely depend on that.** Both
tables were re-measured under every direct solver on 2026-07-31 with
`python3 -m tests.tables`, which is also how to re-measure them after touching
any block. `normal` reproduces all 24 cells exactly; `qr` and `dr` differ in 2
cells of 24 and `blocked` in 3, none by more than 0.01 points, and the largest
relative move in any cell is 3.2% (`qr`, `dr`) or 4.7% (`blocked`). Every
conclusion above — the interior minimum in each column and where it sits, the
decade-per-halving rule, the asymmetry at 50 mm — holds identically under all
four. That is worth stating because these runs are 30 iterations deep, well
past the point where two implementations' *volumes* stop being comparable: what
the tables measure is the trade-off, not the arithmetic.

It also sets the scale for reading them. `dr` is algebraically `qr` at a fixed
weight, and 30 iterations still move it 0.3% from `qr` in the most mobile cell.
The last digit of a cell is not meaningful.

Worth noting that the shipped `1e-7` is not the best cell in either table —
`1e-6` at the default spacing halves the residual at both amplitudes. Don't
read a recommendation into that. This is one synthetic field on one volume,
built from three low-order harmonics, and it is smoother than a real coil
profile; a field with more structure is exactly the case where the extra
penalty would start to cost. What the tables are good for is the *shape* of
the trade-off, not the numbers in them.

### A real Parzen window: `--parzen-sigma`

N3's `-parzen`/`-window` is not a Parzen window. `WHistogram::add` splits each
sample linearly between the two bin centres it falls between — a triangular
kernel exactly one bin wide, whose width is set by `--bins` and by wherever
`-auto_range` put the range this iteration rather than by anything about the
measurement. `--parzen-sigma s` replaces it with the estimator the name
promises: a Gaussian of standard deviation `s` **bin widths**, evaluated at the
bin centres and normalised per sample, so a voxel is spread over as many bins
as the kernel reaches. Nothing downstream changes.

This is an alternation to the algorithm, not part of the port. It is **off by
default**, the legacy backend rejects it, and every other number in this file
is measured without it. `python3 -m tests.parzen` is the measurement below,
re-run it after touching the histogram or the sharpening.

Residual non-uniformity left behind on the planted-field sweep, `--solver
normal`, 20% planted — the same experiment as the `--lambda` × `--distance`
tables above, so the first row *is* that table's first row:

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

Read the two together. At a *fixed* weight the window is worth having — a third
off the residual at the shipped defaults, a factor of four at 50 mm. Against a
weight that has been tuned for the spacing, it buys nothing at 200 mm and about
a third at 50 mm. So what the window mostly does is **substitute for
regularization**: it helps exactly the cells that were under-penalised, which is
the axis `--lambda` already moves along, and the two do not add up. If you have
tuned `--lambda` for your spacing, expect the window to do very little; if you
are running the defaults, `--parzen-sigma 2` is the cheaper of the two knobs to
reach for.

A wide window is not free. `sigma 4` is the best of the sweep at 50 mm and the
*worst* at 200 mm with `--lambda 1e-6` (1.21× the linear split) and at every
`1e-4` cell. The mechanism is visible in the units: one bin is 0.026 log units
on this volume, so `sigma 2` adds a blur of 0.052 against the 0.064 standard
deviation that `--fwhm 0.15` tells `sharpen_hist` to remove, and `sigma 4`
exceeds it. The window adds a Gaussian the deconvolution was never told about,
so past a point it simply under-sharpens. The obvious follow-up — take the added
width back out of `--fwhm` in quadrature — is untested; nothing in the sweep
covers it.

End to end on `brain.mnc` under the shipped protocol it is not a cosmetic
change: the corrected volume moves 6.2e-3 relative RMS at `sigma 0.5` and 8.8e-2
at `sigma 4`, and correspondingly away from `brain_nu_ref.mnc` (3.0e-3 → 4.9e-3
→ 8.6e-2). That last column is not an accuracy score — N3 produced that
reference, so anything that changes N3 moves away from it — which is why the
planted-field sweep is where the question gets answered.

#### And with noise, over 450 random fields per window

Everything above is one analytic field on one volume with **no noise**, which
turns out to be the case the window does least for. `experiments/` plants
*random* fields on colin27 and adds Gaussian noise at a stated SNR; the same
comparison there, 50 seeds × 3 amplitudes × 3 SNRs per window at 75 mm knots,
`--solver normal`, says something the noiseless tables cannot:

| | linear (N3) | σ 1 | σ 2 | σ 4 |
|---|---|---|---|---|
| 20 % planted, SNR ∞ | 0.81 % | 0.80 % | 0.80 % | **0.64 %** |
| 20 % planted, SNR 20 | 4.63 % | 4.84 % | 3.47 % | **1.71 %** |
| 80 % planted, SNR 20 | 5.46 % | 5.28 % | **4.33 %** | 4.34 % |
| better than N3, per trial | — | 40 % | 79 % | **94 %** |

(median unexplained non-uniformity over the brain, shipped protocol; the last
row is the paired per-trial comparison over all 450)

**The gain tracks noise, not field amplitude** — which is what a kernel density
estimate should do, and is invisible to a noiseless test. It also *grows with
the iteration count*: under N3's own histogram, 30 → 50 iterations makes the
noisy cells worse (3.43 % → 4.63 % at 20 %/SNR 20), because the alternating
iteration keeps feeding the ragged histogram's noise back into the mapping;
under σ 4 the same cells keep improving. Cost is about 1 % of run time.

Full tables, the per-cell win rates, `windows.png`, and the two caveats that go
with them are in [experiments/README.md](experiments/README.md), "The histogram
kernel". Short version: **σ 2 is the width that never loses badly**; σ 4 is
better still under the shipped protocol but is the worst kernel of the four at
80 % planted with only 30 iterations; σ 1 is not worth having.

## Requirements

Python 3.12, `torch`, `numpy`, `cffi` and `minc2_simple`.

The MINC toolkit is *not* needed to run `torch_n3` or its tests. It is needed to read
gzipped or MINC1 input (`load_volume` shells out to `mincconvert`), to re-record the
reference answers, and to run the legacy backend's comparisons. All already installed
here — nothing needs fetching.

# `torch_SR/` — Non-Local MRI upsampling (Manjon 2010), study notes

Notes on the subproject imported at `/app/torch_SR/`. It is **independent of `torch_n3/`**: no
shared code, no shared data, no imports in either direction. It is checked in as its own git
repository (10 commits, tip `63e757d`), and nothing here modifies it.

The one thing the two projects have in common is their shape: a PyTorch re-implementation of a
legacy MINC-era algorithm, kept honest against the original C as an oracle. Where that
comparison is handled differently from `torch_n3`'s, this note says so — §9.

Reference: Manjon JV, Coupe P, Buades A, Collins DL, Robles M. *MRI superresolution using
self-similarity and image priors.* Int J Biomed Imaging, 2010. The PDF is in the subproject as
`manjon_2010_upsample.pdf`. V. Fonov is an author of the paper the demo code accompanies.

---

## 1. What it does

Upsamples a 3-D MRI volume by integer factors per axis, without a training set. The method is
single-image super-resolution driven by self-similarity: a cubic-spline initial estimate is
refined by repeated Non-Local Means regularization, under a hard constraint that each output
block's mean stays equal to the low-resolution voxel it came from. That constraint is the
"image prior" — it is what makes the iteration a reconstruction rather than a smoothing.

Everything happens on the volume normalized to `[0, 256]`, and the result is rescaled at the
end. There is no tissue model and no learned component.

---

## 2. Layout

| Path | Role |
|---|---|
| `legacy/demo/` | The original Matlab + MEX distribution, read-only reference. `NLMUpsample/NLMUpsample2.m` is the driver; `NLMUpsample/linux-mac code/cMRegularizarNLM3D_V2.c` is the MEX kernel that everything else ports. `denoising/` (MBONLM3D, wavelet subband mixing) and `test3D.m` / `test3Dnoisy.m` / `testReal.m` are the paper's own experiments and are **not** ported. |
| `upsample/sr/` | The reference port. `nlm_core.c`/`.h` is the MEX-free C kernel, `_nlm_build.py` builds it through CFFI, `regularize.py` wraps it, `interpolate.py` is the initial spline, `core.py` is the pipeline, `metrics.py` is the paper's PSNR. |
| `upsample/sr_pytorch/` | The PyTorch port. `regularize.py` is the vectorized kernel, `core.py` the same pipeline on a device, `optimize.py` a different estimator (§6). |
| `upsample/sr/minc/` | `io.py` and `geo.py`, **vendored** from `minc2-simple/example/python/minc/`. |
| `minc2-simple/` | Declared as a submodule in `.gitmodules` but **not checked out** — the directory is empty. Nothing depends on it because of the vendoring above; `AGENTS.md` and the subproject's own `research.md` still point at paths inside it, and those paths do not exist. |
| `upsample/tests/` | pytest suite, five files. |
| `upsample/data/psnr_factor_{2,3,5,7}.npz` | Checked-in regression fixtures: paired HR/LR brain blocks. |
| `upsample/upsample.py`, `upsample/nlm_filter.py` | The two CLIs. |
| `upsample/scripts/` | `generate_test_blocks.py` (regenerates the fixtures), `benchmark_sr.py` (C vs PyTorch table), `nlm_upsample.py`. |
| `*.mnc` at the subproject root | Working data, **gitignored** — a 7T MP2RAGE volume and its derivatives (§8). |

`sr` and `sr_pytorch` are imported as top-level packages; `upsample/` is the working directory
for everything, and the scripts bootstrap `sys.path` to it themselves.

---

## 3. The algorithm as the legacy code implements it

Driver: `legacy/demo/NLMUpsample/NLMUpsample2.m`.

1. `ima = ima * 256 / max(ima)` — fixed intensity range for the whole run.
2. **Initial interpolation** (`InitialInterpolation`, same file): cubic spline, then boundary
   replication, then mean correction. See §4.
3. **Noise level**: `sigma = stdfilt(bima, ones(3,3,3))`, smoothed once more by a 3×3×3 box
   with symmetric padding, then halved. This is a *per-voxel* array, not a scalar, and it is
   the only thing that adapts the filter strength spatially.
4. **Iterate** `lima = cMRegularizarNLM3D_V2(F, v=3, f=1, level, lf)`:
   - `d(ii) = mean|F − lima|`.
   - When `d(ii−1)/d(ii) < tol` (1.2) and the level was not just halved, halve `level`, record
     `ds(iii) = mean|last − lima|`; if `ds(iii−1)/ds(iii) < tol` as well, **stop**.
   - Also stop when `d(ii) <= 0.001`.
   - `F = lima`.
5. `lima = lima * m / 256`.

The stopping rule is therefore two-level: an inner ratio test that triggers a *scale change*,
and an outer ratio test over the scale changes that ends the run. There is no iteration cap in
the Matlab; the ports add `max_iter` as an option, defaulting to off.

`tol` is hard-coded to 1.2 with the commented-out alternative `0.01*mean(sigma(:))` left in
place — the paper's adaptive threshold was abandoned in the shipped code.

---

## 4. Initial interpolation

The coordinate convention is the part worth getting right. LR index `i` sits at HR coordinate

```
hr = i * f + (f − 1) / 2
```

i.e. at the *centre* of its HR block, not at the block's first voxel. In the Matlab this is
`ori = (1 + lf) / 2` under 1-based indexing; `sr/interpolate.py:58` writes the 0-based form.
An off-by-half-a-block error here is invisible in shape and PSNR-visible only slightly, so it
is worth checking directly.

Consequences, in order:

- The first and last `floor(f/2)` slices along each axis fall outside the LR sample range.
  Matlab and the port both **replicate** the nearest interpolated slice rather than
  extrapolating (`interpolate.py:81-95`). Note that `RegularGridInterpolator` is constructed
  with `fill_value=None`, so it *would* extrapolate; the replication overwrites that.
- **Mean correction**: each `(fz, fy, fx)` block gets a constant offset so its mean equals the
  LR voxel. This is exact to float64 and is asserted at `1e-10` (`test_interpolate.py:57`).
- The same geometry has to be written into the output header, or the volume is off by half a
  block in world space. `upsample.py:_adjusted_v2w` does it: `new_step = step / f`,
  `new_start = start − (f − 1)/2 · new_step`, with the factor vector reversed from numpy
  `(fz,fy,fx)` order into MINC `(x,y,z)` order.

Both PyTorch paths call this scipy function on the CPU — only the NLM loop was ported. The
initial estimate is therefore bit-identical across all three backends.

---

## 5. The NLM kernel, three times over

### 5.1 The C reference (`sr/nlm_core.c`)

A faithful port of `cMRegularizarNLM3D_V2.c`: same loop structure, same indexing, same
pthread split. Documented changes are only the removal of `mex.h`/`matrix.h`, the plain-C
entry point, and dead-variable cleanup. Memory layout is C-contiguous `(nz, ny, nx)`, which is
what the MEX code already used, so no transposition is needed.

Structure:

1. 3×3×3 local means (`medias`), mirror BC.
2. Block means of the *input* (`tmp`) — kept for the mean-preservation step at the end.
3. `fima = ima`, `pesos = 1`.
4. `Nthreads = nz < 8 ? 1 : nz/8`, split over Z.
5. Per voxel `p` with `h[p] >= 1`, over the **upper triangle** of the `(2v+1)³` search
   neighbourhood:
   - reject the pair if `|medias[p] − medias[p1]| > 0.6 · h[p]`;
   - `d = Σ(patch diff²)/(2f+1)³` over the patch, mirror BC;
   - `w = exp(−max(d/(2h[p]²) − 1, 0))`;
   - accumulate **symmetrically**: `fima[p] += w·ima[p1]`, `fima[p1] += w·ima[p]`, and the
     same into `pesos`.
6. `fima /= pesos`.
7. Mean preservation: shift each `(fac_z, fac_y, fac_x)` block so its mean matches `tmp`.

Two properties of step 5 matter for what follows. The pair `(p, p1)` is visited **once**, and
the weight it contributes in **both** directions is computed from `h[p]` and `medias[p]` — the
first voxel's, not each voxel's own. And the accumulation is a cross-thread read-modify-write
with no synchronisation: the races are known and were kept, because the original has them.

### 5.2 The PyTorch kernel (`sr_pytorch/regularize.py`)

The same filter re-posed as a **gather**. The volume is padded once by `v+f` (reflect), and
the loop runs over all `(2v+1)³ − 1` search offsets; for each offset the patch distance for
*every* voxel at once is one `avg_pool3d` over the squared difference. Weight, mean-difference
rejection and `sigma >= 1` masking are elementwise. So each ordered pair is visited twice,
once from each end, and each visit uses the **centre voxel's own** `sigma` and local mean.

That is a different filter from §5.1 whenever `sigma` varies in space, which is always — the
noise level is estimated per voxel. Three concrete divergences, each measured on a
`16³` random volume, `v=3, f=1`, in a scratch copy of the package (my measurements, not the
subproject's):

| Divergence | Cause | Measured |
|---|---|---|
| Interior, spatially varying `sigma` | C uses `h[p]` for both directions of a pair; torch uses each end's own | **1.0% RMS** relative (factors 2,2,2); 0.73% at (2,1,1) |
| Interior, constant `sigma` | none — the two are algebraically the same filter here | 2.3e-13 max abs |
| Borders, constant `sigma` | C **skips** out-of-range search neighbours; torch **mirrors** them in from the reflect padding | 7.5e-05 RMS relative whole-volume, 0.31 max abs |
| Voxels with `sigma < 1` | C `continue`s at `p` but such a voxel still *receives* weight from a high-`sigma` neighbour's symmetric update; torch zeroes the weight and the voxel is exactly unchanged | C moves them by up to 0.092; torch by exactly 0 |

The constant-`sigma` interior agreement to 2.3e-13 is the useful result: it says the
vectorization is correct and the remaining ~1% is entirely the symmetry convention, not an
implementation error. The torch convention is the textbook asymmetric NLM; the C convention is
the paper's own approximation. Neither is a port bug, but **they are not the same estimator**,
and the suite does not distinguish them (§7).

Also note the reflect padding: `F.pad(..., mode="reflect")` requires the pad to be strictly
smaller than the dimension, so at the defaults every axis must be `>= v+f+1 = 5`. The C kernel
has no such floor.

### 5.3 What the PyTorch pipeline does differently

`sr_pytorch/core.py` reproduces `sr/core.py` step for step, including the two-level stopping
rule, with the noise-level helpers rewritten as `avg_pool3d` with reflect padding. The
differences are device and dtype: **float32 on CUDA, float64 on CPU**, defaulted at
`core.py:102`. Given that the iteration's stopping rule is a ratio test on a mean absolute
difference, a change of dtype can move the iteration count, and the iteration count moves the
output more than the dtype does — the same trap `torch_n3` documents for N3's `-stop`. Nothing
in the suite pins the iteration count, so the CPU/CUDA comparison in `test_psnr_close_to_c`
runs at a fixed `max_iter=3` and does not exercise it.

---

## 6. `optimize.py` — a second estimator, and a negative result

`sr_pytorch/optimize.py` replaces the fixed-point iteration with Adam on an additive HR
correction field `δ`, minimising

```
L(δ) = MSE( avg_pool3d( regularize(HR_init + δ) ), LR )
```

with `regularize(..., mean_preserve=False, detach_weights=True, no_grad=False)`. The NLM
weights are treated as constants, so the filter is a fixed spatially-varying linear operator
and gradients flow through the values only; the module's docstring puts the cost of the exact
alternative at ~39 GB for `v=3, f=1` on a 200×200×250 volume.

Two design notes are worth keeping, both recorded in the source:

- `mean_preserve` must be **off** inside the loop. With it on, `avg_pool3d(regularize(x)) ==
  avg_pool3d(x)` by construction and the loss is identically zero at `δ = 0` — no gradient
  signal at all (`optimize.py:124-126`).
- The target is the actual LR volume, not `avg_pool3d(HR_init)`, for the same reason
  (`optimize.py:105-108`).

**It does not work better.** Its own docstring says so ("slower than iterative NLM and does not
outperform it in PSNR"), the CLI help repeats it, and the measured PSNRs recorded in
`test_pytorch.py:64-70` (factor 2→25.658, 3→21.974, 5→18.857, 7→17.238 dB at `n_steps=100,
lr=0.02`) sit *below* the iterative method at every factor and barely above the plain spline
(25.615 / 21.919 / 18.823 / 17.227). All of its tests are `pytest.mark.skip`-ed
(`test_pytorch.py:183`). The right reading is a recorded negative result, kept for the record
— the same status `torch_n3` gives its `sparse` solver and `tightness` objective.

The gradient plumbing itself (`detach_weights` / `no_grad` on `regularize`) is the reusable
part: it makes the NLM step droppable into a training loop at inference-level memory.

---

## 7. Test suite and what it actually pins

`upsample/tests/`, all pytest, no test runner config checked in.

| File | Skips when | Pins |
|---|---|---|
| `test_interpolate.py` | never | shape, mean preservation at 1e-10, finiteness, uniform input, `factors=(1,1,1)` identity, validation |
| `test_regularize.py` | C ext absent | shape, finiteness, `sigma < 1` ⇒ identity, validation |
| `test_core.py` | C ext absent | shape, finiteness, a loose overshoot band, uniform input at 1e-6, validation |
| `test_psnr.py` | C ext absent | NLM beats spline at every factor; NLM and spline PSNR floors |
| `test_pytorch.py` | fixture absent / CUDA absent | shape, finiteness, uniform input, **PyTorch within 0.5 dB of C**, PyTorch PSNR floors; `optimize` tests all skipped |

The fixtures are brain blocks from the 7T MP2RAGE volume, centred at voxel `(151, 99, 95)`
(the median of voxels > 500), downsampled by block averaging — which is exactly the forward
model the method assumes, so the fixtures are favourable by construction.

| factor | LR side | HR side | spline PSNR | NLM PSNR | spline floor | NLM floor | torch floor |
|---|---|---|---|---|---|---|---|
| 2 | 12 | 24 | 25.615 | 27.158 | 23.6 | 26.1 | 25.6 |
| 3 | 12 | 36 | 21.919 | 22.462 | 19.9 | 21.4 | 20.9 |
| 5 | 10 | 50 | 18.823 | 19.034 | 16.8 | 18.0 | 17.5 |
| 7 |  8 | 56 | 17.227 | 17.316 | 15.2 | 16.3 | 15.8 |

Measured at `max_iter=3, v=3, f=1`.

**Every floor in that table is the measured value minus a margin** — 1.0 dB for NLM, 2.0 dB
for spline, 1.5 dB for PyTorch — and `generate_test_blocks.py` prints the numbers so a
developer can re-derive them. This is precisely the practice `torch_n3/CLAUDE.md` forbids: a
threshold set from the measurement records what the code does, not what it is required to do.
It is a deliberate trade here — the floors are described as catching "total regressions", and
with a 1–2 dB skirt that is all they can catch. Note also how little headroom the method
itself has at the larger factors: at factor 7, NLM beats spline by 0.089 dB, which is inside
every floor's margin. `test_nlm_beats_spline` is the only test that pins that gap, and it
allows 0.05 dB of slack against a 0.089 dB effect.

What no test covers:

- **The C and PyTorch kernels are never compared directly.** `test_psnr_close_to_c` compares
  the *pipelines* at 0.5 dB, on the favourable fixtures. The ~1% per-pass kernel difference of
  §5.2 is well inside that, so the symmetry-convention divergence is invisible to the suite.
- The convergence path: nothing runs to convergence, nothing asserts an iteration count, and
  every PSNR number is at `max_iter=3`.
- `optimize.py` entirely (skipped).
- `sr/minc/` — no I/O test, and `minc2_simple` is imported at module scope in `io.py`, so
  anything touching it fails at import if the binding is missing.
- The CLIs.

---

## 8. CLIs, and the working data at the subproject root

`upsample/upsample.py in.mnc out.mnc` — the pipeline. `--factors FZ FY FX` (default `2 1 1`,
i.e. slice-direction only), `--v/--f/--tol/--max-iter`, `--initial-only` for the spline alone,
`--method {c,pytorch,optimize}` (defaults to `c` when the extension is built, else `pytorch`),
`--device`, and `--n-steps/--lr` for the optimize path.

`upsample/nlm_filter.py in.mnc out.mnc` — a *single* NLM pass through the PyTorch kernel,
which makes it a denoiser rather than an upsampler. `--sigma` (constant) / `--sigma-file`
(per-voxel map) / auto-estimate; `--h` multiplies whatever sigma results, so it is the
smoothing-strength knob; `--no-mean-preserve` drops the block constraint, which is what you
want when the input is not block-structured; `--factors` defaults to `1 1 1` here.

The `.mnc` files at the subproject root are gitignored working data. Their MINC history
attributes record what they are, and they are the only evidence of the tool being run on a
real volume:

| File | Provenance |
|---|---|
| `VF_20190417_MP2RAGE.mnc` | 7T MP2RAGE, 260×228×192 (z,y,x). The fixture source. |
| `VF_20190417_MP2RAGE_lr.mnc` | `minc_downsample` → 130×228×192, Z halved |
| `VF_20190417_MP2RAGE_vlr.mnc` | `minc_downsample --3dfactor 4` → 65×57×48 |
| `..._lr_recon.mnc` | `upsample.py ..._lr.mnc --factors 2 1 1` |
| `..._lr_recon2.mnc` | `upsample.py ..._vlr.mnc --factors 4 4 4` |
| `..._vlr_bspline.mnc` | `itk_resample ..._vlr.mnc --like <HR>` — the B-spline baseline for the 4× case |
| `..._nlm.mnc` | `nlm_filter.py <HR> --no-mean-preserve --f 2 --v 5` — denoising, not upsampling |

So the two comparisons actually run on real data were 4× isotropic recovery against an
`itk_resample` B-spline baseline, and a single-pass denoise at a wider patch/search than the
defaults. Neither is scored anywhere in the repository.

---

## 9. Reading this next to `torch_n3/`

Both projects port a legacy C algorithm to PyTorch and keep the C as an oracle, so the
contrasts are informative rather than incidental:

- **`torch_n3` compares blocks; `torch_SR` compares pipelines.** `torch_n3` records per-block
  legacy answers in `tests/reference/legacy.npz` and constrains each stage, precisely because
  the iteration amplifies. `torch_SR` has an equivalent oracle available in-process — the CFFI
  kernel — and never calls it beside the PyTorch one. The 0.5 dB pipeline comparison is the
  only bridge, and §5.2 shows a real semantic difference sitting comfortably underneath it.
- **Both have an amplifying iteration with a ratio-based stopping rule**, and `torch_n3`'s
  hard-won lesson (the stopping rule quantises everything downstream; check the iteration
  count first) applies unchanged here. `torch_SR` never runs to convergence in a test.
- **Both carry a recorded negative result** — `optimize.py` here, `sparse`/`tightness` there —
  and both keep it in the tree with the reason attached. That practice is consistent.
- **Tolerance discipline differs.** `torch_n3` derives thresholds from a physical quantity and
  refuses to move them; `torch_SR`'s PSNR floors are measured-minus-margin throughout.
- `torch_SR` vendored `minc/io.py` and `geo.py` into `sr/minc/` and left the submodule
  unpopulated; `torch_n3` keeps `minc2-simple/` as a real checkout and imports the installed
  binding. The vendored copies here mean the subproject reads MINC2 with no submodule, at the
  cost of a fork that will not track upstream.

---

## 10. State of the checkout

- The C extension is **not built** — `sr/_nlm_core.*.so` is absent, so `have_c_ext()` is
  `False` and `test_core.py`, `test_regularize.py`, `test_psnr.py` and the C half of
  `test_pytorch.py` all skip. `python sr/_nlm_build.py` from `upsample/` builds it in a few
  seconds (verified in a scratch copy; the checkout was left untouched).
- `minc2-simple/` is an empty submodule directory; nothing needs it.
- `upsample/data/*.npz` are present, so the PSNR tests have their fixtures.
- The subproject's own `research.md` and `AGENTS.md` predate the last four commits: they
  describe only the C path, place the project at `/app/` rather than `/app/torch_SR/`, list
  `scripts/nlm_regularize.py` at a path where it no longer lives (it is `upsample/nlm_filter.py`
  since `6430dec`), and point at `minc2-simple/example/python/` for I/O that is now vendored in
  `sr/minc/`. `AGENTS.md` also prescribes mypy/flake8/black in CI; there is no CI configuration
  and no tool config in the tree.

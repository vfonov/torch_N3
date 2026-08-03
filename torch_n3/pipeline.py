"""N3 itself: ``nu_estimate``, ``nu_evaluate`` and ``nu_correct``.

A transcription of the Perl drivers in ``legacy/N3/src/NUcorrect``, with each
shell-out replaced by a call to a block in :mod:`torch_n3.blocks` or
:mod:`torch_n3.minc_tools`.  The structure follows
``nu_estimate_np_and_em.in`` step by step, so the two can be read side by side.

N3's premise: a multiplicative bias field becomes an *additive* one in
log-intensity space, and blurs the tissue-intensity histogram.  The algorithm
repeats three steps: (1) estimate the histogram without the blur and map every
voxel to that sharpened estimate, (2) attribute the remaining difference to the
field, and (3) retain only its smooth component by fitting a B-spline.  What
remains after several iterations is the field.

Every block is supplied either by the PyTorch port or by the original C++;
``backend`` selects between them and defaults to the port.  See
:func:`torch_n3.backends.resolve`.
"""

import torch

from torch_n3 import backends
from torch_n3.minc_tools import apply_lut, bimodal_threshold

#: What ``nu_correct`` passes down when given no options
#: (``nu_estimate.in:418-438``, ``nu_estimate_np_and_em.in:1537-1560``).
DEFAULTS = dict(
    distance=200.0,     # B-spline knot spacing, mm -- the dominant smoothness
                        # parameter, paired with `lam` below
    fwhm=0.15,          # width of the blur assumed in the histogram, log units
    noise=0.01,         # Wiener constant of the deconvolution
    bins=200,           # histogram bins
    iterations=(50,),   # per stopping stage
    stop=(0.001,),      # per stopping stage
    shrink=4,           # estimation runs on a grid this many times coarser
    lam=1e-7,           # B-spline bending-energy weight.  Together with
                        # `distance` this sets how much the field may bend, so
                        # the two move together: about a decade more `lam` per
                        # halving of `distance`.  Lowering `distance` alone
                        # permits the fit to follow tissue contrast rather
                        # than the field -- see tests/test_field_recovery.py.
    subsample=1,        # use every n-th voxel when fitting the spline
    solver="normal",    # how the spline fit is solved: "normal" is the
                        # legacy's own penalised normal equations, "qr"
                        # the better-conditioned stacked factorization of
                        # the same problem (torch backend only)
    background=1.0,     # voxels at or below this are never part of the mask
    parzen=True,
    parzen_sigma=None,  # None is N3's own `-parzen`: linear interpolation into
                        # the two neighbouring bins.  A number replaces it with
                        # a Gaussian Parzen window of that many bin widths,
                        # which is a modification to the algorithm rather than
                        # part of it (tests/parzen.py).
    deblur=False,       # True reproduces `-blur`: skip the deconvolution
    backend="torch",    # or "legacy", to run the original C++ instead
    denoise=False,      # Off, and not part of N3.  True filters the volume
                        # with one non-local-means pass (blocks/denoise.py)
                        # at full resolution, before `shrink`, for the
                        # *estimation* only: `nu_evaluate` still divides the
                        # original intensities, so the output is corrected and
                        # never denoised.  A modification rather than part of
                        # the algorithm, so there is no oracle for it and the
                        # legacy backend rejects it (tests/denoise.py).
    denoise_search=3,   # search radius in voxels; the cost is cubic in this
                        # and in nothing else
    denoise_patch=1,    # patch half-width; similarity is judged over
                        # (2*patch+1)^3 voxels, so larger is a stricter match
                        # and therefore less smoothing
    denoise_strength=1.0,  # multiplies the estimated noise level.  0 is
                        # exactly the identity, since no voxel then clears the
                        # filter's own noise floor, and larger smooths harder.
)


def nu_correct(volume, mask=None, evaluation_mask=None, field_floor=0.1,
               verbose=False, **options):
    """Estimate the bias field and divide it out.  Returns the corrected volume.

    ``mask`` restricts the *estimation* to a region of interest, and should be
    supplied wherever possible: N3 is a histogram method, and background voxels
    contribute only noise.  ``evaluation_mask`` restricts where the field is
    used directly before being extrapolated outwards; when omitted, one is
    derived from the data as ``nu_evaluate`` does.
    """
    field = nu_estimate(volume, mask=mask, verbose=verbose, **options)
    return nu_evaluate(volume, field, mask=evaluation_mask,
                       field_floor=field_floor,
                       backend=options.get("backend", DEFAULTS["backend"]))


def nu_estimate(volume, mask=None, verbose=False, **options):
    """Estimate the bias field.  Returns it as a fitted ``BSplineField``.

    The returned spline is the in-memory equivalent of N3's ``.imp`` mapping
    file: a compact description of a smooth field that can be evaluated on any
    grid, including the full-resolution one.

    With ``denoise`` the estimate is made on a filtered copy of the volume.
    Only the estimate: the returned spline describes a field on the same grid
    as before, and :func:`nu_evaluate` divides the caller's own intensities by
    it.
    """
    opts = dict(DEFAULTS, **options)
    _check_stages(opts)
    backend = backends.resolve(opts["backend"])

    # 0. Optionally, take the noise off first -- at full resolution, because
    #    the shrink below is nearest-neighbour and would alias it in.
    volume = _denoised(volume, opts)

    # 1. Estimation runs on a coarser grid; the answer is a spline, so nothing
    #    is lost by sampling the field sparsely (`WorkspaceSampling`).
    grid = volume if opts["shrink"] == 1 else volume.shrink(opts["shrink"])

    # 2 and 3. Log domain, and the mask N3 works inside.  The clamp keeps
    # log() away from zero, where it would consume the volume's dynamic range.
    # The mask is on whatever grid the caller had; it follows the estimation
    # onto the coarse one (`CheckSampling`).
    log_volume = torch.log(grid.data.clamp(min=1.0))
    inside = grid.data > opts["background"]
    if mask is not None:
        inside &= mask.resample_like(grid).data != 0
    if not inside.any():
        raise ValueError("the mask is empty: no voxel is above the background "
                         "threshold inside the region of interest")
    outside = torch.zeros_like(log_volume)
    log_volume = torch.where(inside, log_volume, outside)

    # 4. The field estimate, in log space, starts out flat.
    residue = torch.zeros_like(log_volume)

    # 5. Sharpen, attribute the residual to the field, smooth it, repeat.
    for iteration in range(max(opts["iterations"])):
        corrected = log_volume - residue

        estimate = _sharpen(corrected, inside, opts)
        working = log_volume - estimate

        previous, residue = residue, _smooth(working, inside, grid, opts)

        change = float(torch.std((previous - residue)[inside], unbiased=False))
        if verbose:
            print("iteration %d: field change %.6f" % (iteration, change))
        if _converged(iteration, change, opts):
            break

    # 6. Out of log space.
    field = torch.exp(residue)

    # 7. Refit as a compact spline, which is what gets carried to full
    #    resolution (`compact_spline_volume`).
    return backend.BSplineField(grid, opts["distance"], opts["lam"],
                                solver=opts["solver"]).fit(
        field, inside, opts["subsample"])


def nu_evaluate(volume, field, mask=None, field_floor=0.1, backend=None):
    """Divide ``volume`` by the estimated ``field`` (``nu_evaluate.in:46-80``)."""
    divisor = evaluate_field(volume, field, mask=mask, field_floor=field_floor,
                             backend=backend)
    return volume.like(volume.data / divisor.data)


def evaluate_field(volume, field, mask=None, field_floor=0.1, backend=None):
    """Sample ``field`` on ``volume``'s grid and make it safe to divide by.

    ``field`` is the spline returned by :func:`nu_estimate`.  Evaluation alone
    is not sufficient: the spline was fitted inside a mask, and outside it is
    meaningless and may be negative.  The values outside are therefore
    discarded, replaced by a smooth extension of those inside, and floored.
    """
    backend = backends.resolve(backend)

    if mask is None:
        # No mask supplied: keep everything above the automatic threshold.
        inside = volume.data >= bimodal_threshold(volume.data)
    else:
        inside = mask.resample_like(volume).data != 0

    values = field.evaluate_on(volume)
    values = torch.where(inside, values, torch.zeros_like(values))
    values = backend.correct_field(values, inside, volume.step)
    return volume.like(values.clamp(min=field_floor))


def _sharpen(values, inside, opts):
    """Map ``values`` through the sharpened histogram (``sharpen_volume.in``).

    The histogram is rebuilt every iteration (``-auto_range``), so the domain
    of the mapping moves as the correction improves.
    """
    backend = backends.resolve(opts["backend"])
    selected = values[inside]

    # volume_hist seeds its range scan with the whole volume's range and then
    # visits only the masked voxels.
    value_range = backend.histogram_range(
        selected, initial=(values.max(), values.min()))
    counts = backend.histogram(selected, opts["bins"], value_range,
                               opts["parzen"], sigma=opts["parzen_sigma"])

    # The drivers pass the histogram from volume_hist to sharpen_hist, and the
    # mapping from sharpen_hist to minclookup, through text files written with
    # "%lf" -- so six decimals is all that survives, and the domain the
    # mapping is defined over is the *rounded* one.
    value_range = _as_written(value_range)
    lut = _as_written(backend.sharpen_lut(_as_written(counts), value_range,
                                          opts["fwhm"], opts["noise"],
                                          opts["deblur"]))

    mapped = apply_lut(values, lut, value_range)
    return torch.where(inside, mapped, torch.zeros_like(mapped))


def _smooth(values, inside, grid, opts):
    """Keep only the smooth part of ``values`` (``spline_smooth -b_spline``).

    ``spline_smooth`` writes zeros outside the mask, and so does this.
    """
    backend = backends.resolve(opts["backend"])
    spline = backend.BSplineField(grid, opts["distance"], opts["lam"],
                                  solver=opts["solver"])
    spline.fit(values, inside, opts["subsample"])
    smoothed = spline.evaluate()
    return torch.where(inside, smoothed, torch.zeros_like(smoothed))


def _denoised(volume, opts):
    """The volume the *estimation* sees.  The caller's own, unless ``denoise``.

    Filtered at full resolution and before ``shrink``, because
    :meth:`torch_n3.volume.Volume.shrink` is nearest-neighbour subsampling: it
    aliases noise onto the estimation grid rather than averaging it away, so
    filtering afterwards would filter something the sampling had already
    corrupted.

    Returns a *new* ``Volume``, so :func:`nu_evaluate` divides the original
    intensities rather than the filtered ones.  The filter is reached through
    the backend, so ``backend="legacy"`` refuses here, where the option was
    set, rather than ignoring it.
    """
    if not opts["denoise"]:
        return volume

    backend = backends.resolve(opts.get("backend"))
    return volume.like(backend.denoise(volume.data,
                                       search=opts["denoise_search"],
                                       patch=opts["denoise_patch"],
                                       strength=opts["denoise_strength"]))


def _as_written(values):
    """Round as printf's "%lf" would: six digits after the decimal point."""
    return torch.round(torch.as_tensor(values, dtype=torch.float64),
                       decimals=6)


def _check_stages(opts):
    stages, thresholds = opts["iterations"], opts["stop"]
    if len(stages) != len(thresholds):
        raise ValueError("-iterations and -stop must name the same number of "
                         "stages (got %d and %d)" % (len(stages), len(thresholds)))


def _converged(iteration, change, opts):
    """N3's staged stopping rule (``nu_estimate_np_and_em.in:165-180``).

    ``-iterations a b -stop x y``: stop as soon as the field stops moving by
    more than ``x``; after iteration ``a``, the looser threshold ``y`` also
    applies.
    """
    stages, thresholds = opts["iterations"], opts["stop"]
    if change < thresholds[0]:
        return True
    return any(iteration >= stages[stage - 1] and change < thresholds[stage]
               for stage in range(1, len(stages)))

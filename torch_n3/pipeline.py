"""N3 itself: ``nu_estimate``, ``nu_evaluate`` and ``nu_correct``.

This is a transcription of the Perl drivers in ``legacy/N3/src/NUcorrect``,
with each shell-out replaced by a call to one of the blocks in
:mod:`torch_n3.backends.legacy` or :mod:`torch_n3.minc_tools`.  The structure
follows ``nu_estimate_np_and_em.in`` step by step so the two can be read side
by side.

The idea behind N3, in one paragraph: a multiplicative bias field becomes an
*additive* one in log-intensity space, and blurs the tissue-intensity
histogram.  So we repeatedly (1) guess what the histogram would look like
without the blur and map every voxel to that sharpened estimate, (2) attribute
the leftover difference to the field, and (3) keep only the smooth part of it
by fitting a B-spline.  What remains after a few rounds is the field.
"""

import numpy as np

from torch_n3.backends import legacy
from torch_n3.minc_tools import apply_lut, bimodal_threshold

#: What ``nu_correct`` passes down when given no options
#: (``nu_estimate.in:418-438``, ``nu_estimate_np_and_em.in:1537-1560``).
DEFAULTS = dict(
    distance=200.0,     # B-spline knot spacing, mm -- the main smoothness knob
    fwhm=0.15,          # width of the blur assumed in the histogram, log units
    noise=0.01,         # Wiener constant of the deconvolution
    bins=200,           # histogram bins
    iterations=(50,),   # per stopping stage
    stop=(0.001,),      # per stopping stage
    shrink=4,           # estimation runs on a grid this many times coarser
    lam=1e-7,           # B-spline bending-energy weight
    subsample=1,        # use every n-th voxel when fitting the spline
    background=1.0,     # voxels at or below this are never part of the mask
    parzen=True,
    deblur=False,       # True reproduces `-blur`: skip the deconvolution
)


def nu_correct(volume, mask=None, evaluation_mask=None, field_floor=0.1,
               verbose=False, **options):
    """Estimate the bias field and divide it out.  Returns the corrected volume.

    ``mask`` restricts the *estimation* to a region of interest (strongly
    recommended -- N3 is a histogram method, and background voxels only add
    noise).  ``evaluation_mask`` restricts where the field is taken at face
    value before being extended outwards; when omitted, one is derived from
    the data, exactly as ``nu_evaluate`` does.
    """
    field = nu_estimate(volume, mask=mask, verbose=verbose, **options)
    return nu_evaluate(volume, field, mask=evaluation_mask,
                       field_floor=field_floor)


def nu_estimate(volume, mask=None, verbose=False, **options):
    """Estimate the bias field.  Returns it as a fitted :class:`BSplineField`.

    The returned spline is the in-memory equivalent of N3's ``.imp`` mapping
    file: a compact description of a smooth field that can be evaluated on any
    grid, including the full-resolution one.
    """
    opts = dict(DEFAULTS, **options)
    _check_stages(opts)

    # 1. Estimation runs on a coarser grid; the answer is a spline, so nothing
    #    is lost by sampling the field sparsely (`WorkspaceSampling`).
    grid = volume if opts["shrink"] == 1 else volume.shrink(opts["shrink"])

    # 2 and 3. Log domain, and the mask N3 will work inside.  The clamp keeps
    # log() away from zero, where it would eat the volume's dynamic range.
    # The mask is on whatever grid the caller had; it follows the estimation
    # onto the coarse one (`CheckSampling`).
    log_volume = np.log(np.clip(grid.data, 1.0, None))
    inside = grid.data > opts["background"]
    if mask is not None:
        inside &= mask.resample_like(grid).data != 0
    if not inside.any():
        raise ValueError("the mask is empty: no voxel is above the background "
                         "threshold inside the region of interest")
    log_volume = np.where(inside, log_volume, 0.0)

    # 4. The field estimate, in log space, starts out flat.
    residue = np.zeros_like(log_volume)

    # 5. Sharpen, attribute the residual to the field, smooth it, repeat.
    for iteration in range(max(opts["iterations"])):
        corrected = log_volume - residue

        estimate = _sharpen(corrected, inside, opts)
        working = log_volume - estimate

        previous, residue = residue, _smooth(working, inside, grid, opts)

        change = float(np.std((previous - residue)[inside]))
        if verbose:
            print("iteration %d: field change %.6f" % (iteration, change))
        if _converged(iteration, change, opts):
            break

    # 6. Out of log space.
    field = np.exp(residue)

    # 7. Refit as a compact spline, which is what gets carried to full
    #    resolution (`compact_spline_volume`).
    return legacy.BSplineField(grid, opts["distance"], opts["lam"]).fit(
        field, inside, opts["subsample"])


def nu_evaluate(volume, field, mask=None, field_floor=0.1):
    """Divide ``volume`` by the estimated ``field`` (``nu_evaluate.in:46-80``)."""
    divisor = evaluate_field(volume, field, mask=mask, field_floor=field_floor)
    return volume.like(volume.data / divisor.data)


def evaluate_field(volume, field, mask=None, field_floor=0.1):
    """Sample ``field`` on ``volume``'s grid and make it safe to divide by.

    ``field`` is the spline returned by :func:`nu_estimate`.  Evaluating it is
    only half the job: it was fitted inside a mask and is meaningless (even
    negative) away from there, so the values outside are thrown away and
    replaced by a smooth extension of the ones inside, then floored.
    """
    if mask is None:
        # No mask supplied: keep everything above the automatic threshold.
        inside = volume.data >= bimodal_threshold(volume.data)
    else:
        inside = mask.resample_like(volume).data != 0

    values = np.where(inside, field.evaluate_on(volume), 0.0)
    values = legacy.correct_field(values, inside, volume.step)
    return volume.like(np.maximum(values, field_floor))


def _sharpen(values, inside, opts):
    """Map ``values`` through the sharpened histogram (``sharpen_volume.in``).

    The histogram is rebuilt every iteration (``-auto_range``), so the domain
    of the mapping moves as the correction improves.
    """
    selected = values[inside]

    # volume_hist seeds its range scan with the whole volume's range and then
    # visits only the masked voxels.
    value_range = legacy.histogram_range(
        selected, initial=(values.max(), values.min()))
    counts = legacy.histogram(selected, opts["bins"], value_range,
                              opts["parzen"])

    # The drivers pass the histogram from volume_hist to sharpen_hist, and the
    # mapping from sharpen_hist to minclookup, through text files written with
    # "%lf" -- so six decimals is all that survives, and the domain the
    # mapping is defined over is the *rounded* one.
    value_range = _as_written(value_range)
    lut = _as_written(legacy.sharpen_lut(_as_written(counts), value_range,
                                         opts["fwhm"], opts["noise"],
                                         opts["deblur"]))

    return np.where(inside, apply_lut(values, lut, value_range), 0.0)


def _smooth(values, inside, grid, opts):
    """Keep only the smooth part of ``values`` (``spline_smooth -b_spline``).

    ``spline_smooth`` writes zeros outside the mask, and so do we.
    """
    spline = legacy.BSplineField(grid, opts["distance"], opts["lam"])
    spline.fit(values, inside, opts["subsample"])
    return np.where(inside, spline.evaluate(), 0.0)


def _as_written(values):
    """Round as printf's "%lf" would: six digits after the decimal point."""
    return np.round(values, 6)


def _check_stages(opts):
    stages, thresholds = opts["iterations"], opts["stop"]
    if len(stages) != len(thresholds):
        raise ValueError("-iterations and -stop must name the same number of "
                         "stages (got %d and %d)" % (len(stages), len(thresholds)))


def _converged(iteration, change, opts):
    """N3's staged stopping rule (``nu_estimate_np_and_em.in:165-180``).

    ``-iterations a b -stop x y`` means: stop as soon as the field stops
    moving by more than ``x``; after iteration ``a``, the looser threshold
    ``y`` also counts.
    """
    stages, thresholds = opts["iterations"], opts["stop"]
    if change < thresholds[0]:
        return True
    return any(iteration >= stages[stage - 1] and change < thresholds[stage]
               for stage in range(1, len(stages)))

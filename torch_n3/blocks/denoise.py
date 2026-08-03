"""A non-local-means filter over the volume, run before the field is estimated.

This is **not part of N3** and has no oracle.  N3's data term is the intensity
histogram, so noise enters the field estimate directly; ``--parzen-sigma``
(:mod:`torch_n3.blocks.histogram`) suppresses that in the intensity domain, and
this suppresses the same variance spatially.  Both are off by default, both are
rejected by the legacy backend, and neither is compared against a recorded
answer.  ``tests/test_denoise.py`` states the properties this one is required to
have; ``python3 -m tests.denoise`` measures its effect on N3.

The denoised volume feeds the **estimation only**.
:func:`torch_n3.pipeline.nu_evaluate` divides the original intensities by the
fitted field, so the written output is bias-corrected and never denoised.  The
filter runs at full resolution, *before* ``-shrink``:
:meth:`torch_n3.volume.Volume.shrink` is nearest-neighbour subsampling, so it
aliases noise onto the estimation grid rather than averaging it away, and
filtering afterwards would filter something the sampling had already corrupted.

Every voxel is replaced by a weighted mean of its ``(2v+1)^3`` neighbourhood,
the weight falling with the squared distance between the two voxels'
``(2f+1)^3`` patches.  The premise is that a brain contains its own texture
several times over, so the average is taken over voxels that are alike rather
than merely adjacent, and edges survive it.

The one number that carries the intensity scale
-----------------------------------------------

Term by term the filter is scale-free: the patch distance and ``sigma^2`` both
scale as the square of the intensities, so their ratio is invariant, and the
:data:`MEAN_DIFFERENCE` rejection scales on both sides.  All of it is
translation-free as well, being built from differences.  :data:`SIGMA_FLOOR` is
the exception: an *absolute* floor of one intensity unit on
:data:`DEFAULT_SCALE`.  A MINC file's intensities are on whatever scale its
writer chose -- ``tests/data/brain.mnc`` peaks at 1,078,824 -- so the filter
establishes the scale itself rather than trusting its input.

Without that, the same anatomy would be filtered differently depending only on
how the file was written: ``brain.mnc`` would have 98.3% of its voxels clear the
floor, air included, while the same image stored on a [0, 1] scale would have
none clear it and the filter would reduce to the identity.  Both silently.

The evidence that the reconciliation works is not any particular fraction --
afterwards 97.6% of ``brain.mnc`` clears the floor, close to where it started,
because this volume's stored range happens to be near the assumed one.  It is
that the fraction no longer depends on the storage scale at all, which
``tests/test_denoise.py`` asserts exactly.

The reconciliation runs in the direction opposite to the source's: rather than
rescaling the volume onto :data:`DEFAULT_SCALE` and back, the floor is carried
onto the volume's own scale.  This is the same filter and it avoids a
multiply-and-divide round trip, so the output is exactly a combination of the
caller's own intensities rather than one to fifteen digits, which is what lets
``tests/test_denoise.py`` state the identity cases as exact.

The scale is the volume's **dynamic range**, taken as the 1st-to-99th centile
separation, rather than the maximum the source uses; :func:`_scale` gives the
reasoning and the measurements.  Noise is a spread and must be judged against a
spread, and a maximum is neither robust (one voxel decides it, and the resulting
failure is silent) nor translation-invariant (a DC offset would change what
counts as noise).  This is one of two deliberate departures from the source
code, the other being the variance clamp below.  Together they make the filter
affine-equivariant -- ``denoise(a*v + b) == a*denoise(v) + b`` -- which
``tests/test_denoise.py`` asserts in both arguments.

What was dropped from the source, and why
-----------------------------------------

Copied from ``torch_SR``'s ``upsample/sr_pytorch/regularize.py``, which
implements this for super-resolution rather than for denoising.  Three things
were removed, and none should be restored:

``factors`` / ``mean_preserve``
    The upsampler holds each output block's mean to the low-resolution voxel it
    came from.  At one voxel per block that correction is not merely inert, it
    returns the *input*: the block mean of the output is the output, so the
    offset added is exactly ``input - output``.  ``torch_SR``'s own denoising
    CLI passes ``--no-mean-preserve`` for this reason.

``detach_weights`` / ``no_grad``
    These exist so the filter can sit inside that project's Adam loop.  Here it
    runs once, upstream of everything differentiable:
    :mod:`torch_n3.optimize` descends on spline coefficients and the volume
    reaches it as a constant, so there is nothing for a gradient to flow back
    through and the loop is wrapped in ``no_grad``.

float32 on CUDA
    The source defaults to it for speed.  Everything N3 touches is float64
    (CLAUDE.md), and this is upstream of N3.

Not the same filter as the C kernel
-----------------------------------

``torch_SR`` also carries a C implementation, and it is **not an oracle for
this file**.  The C code visits each pair of voxels once and uses the *first*
voxel's ``sigma`` and local mean for both directions of the symmetric update;
this gathers, visiting each ordered pair from both ends and using each centre's
own.  The two agree to 2.3e-13 when ``sigma`` is spatially constant, so the
vectorization is faithful, and differ by 1.0% RMS under the per-voxel ``sigma``
used in practice.  That is a difference of convention rather than a defect in
either, and the two should not be reconciled.

Cost
----

``(2v+1)^3 - 1`` shifted passes over the whole volume: 342 at the default
``search=3``.  On ``tests/data/brain.mnc`` (903k voxels), measured here:

    =====================================  ========
    ``denoise``, CPU                        8.30 s
    ``denoise``, CUDA                       0.21 s
    a whole default ``nu_estimate``, CPU    0.19 s
    =====================================  ========

(Medians of repeated runs; the ratio moves between 35 and 55 across runs,
because the estimation is short enough for its timing to be noisy.)

The filter therefore costs **some forty times the entire estimation it feeds**.
This is not a defect of either: ``-shrink 4`` leaves the estimation grid tens of
times smaller than the input, while this runs at full resolution by design.  A
GPU removes the difference, and ``--device cuda`` is the intended way to use the
option.

Peak memory is 119 bytes per voxel, about a dozen volume-sized float64
temporaries, so a volume large enough to exhaust a device would need the loop
tiled with a ``search + patch`` halo.  That is not implemented; the arithmetic
is given so the ceiling can be calculated (``PROBLEMS.md``).

References
----------
Manjon, J. V., Coupe, P., Buades, A., Fonov, V., Collins, D. L., & Robles, M.
(2010). "Non-local MRI upsampling." *Medical Image Analysis* 14(6), 784-792.
The C ancestor of this filter, ``cMRegularizarNLM3D_V2.c``, is GPL-licensed;
see ``COPYING.NLM`` beside this file.
"""

import math

import torch
import torch.nn.functional as F

#: The intensity scale :data:`SIGMA_FLOOR` is calibrated on.  Every volume is
#: brought onto it before filtering and taken off it again afterwards.
DEFAULT_SCALE = 256.0

#: A voxel whose estimated noise is below this, on :data:`DEFAULT_SCALE`, is
#: returned untouched: there is nothing to remove, and averaging it with its
#: neighbours would cost resolution.
SIGMA_FLOOR = 1.0

#: The centiles whose separation represents the volume's dynamic range when
#: :data:`SIGMA_FLOOR` is brought onto its scale.  The source uses the maximum
#: alone; see :func:`_scale` for why neither the maximum nor any single level
#: is the right quantity.
SCALE_QUANTILES = (0.01, 0.99)

#: Two voxels whose local means differ by more than this many ``sigma`` are
#: never averaged, whatever their patches say.  The original's fast rejection
#: test, retained because it is part of the filter rather than an optimisation:
#: it prevents one tissue from borrowing from another.
MEAN_DIFFERENCE = 0.6


def denoise(values, search=3, patch=1, strength=1.0):
    """One non-local-means pass over ``values``.  Returns a new tensor.

    ``search`` is the radius of the neighbourhood averaged over and ``patch``
    the half-width of the window similarity is judged on, both in voxels and
    both as in the paper: ``v`` and ``f``, defaulting to 3 and 1.  The cost is
    cubic in ``search`` and in nothing else.

    ``strength`` multiplies the estimated noise level, and is the parameter to
    adjust first.  Zero is exactly the identity, since no voxel then clears
    :data:`SIGMA_FLOOR`; larger values widen the weights and loosen the
    :data:`MEAN_DIFFERENCE` rejection, so they smooth harder.

    Computed at float64 on the device holding ``values``.  The intensity scale
    is handled internally; see the module docstring.
    """
    values = torch.as_tensor(values, dtype=torch.float64)
    _check(values, search, patch, strength)

    # SIGMA_FLOOR is calibrated on DEFAULT_SCALE, so one of the two must be
    # brought to the other.  The source rescales the volume; carrying the floor
    # onto the volume's own scale is the same filter without a
    # multiply-and-divide round trip, so the output is exactly a combination of
    # the caller's intensities rather than one to fifteen digits.
    sigma = noise_level(values) * float(strength)
    if not float(sigma.max()) > 0.0:
        # No voxel has any local spread at all, so there is provably nothing to
        # remove: a constant volume, or `strength` of zero.  Answered here
        # because such a volume has no dynamic range either, and the scale
        # below would have nothing to measure.
        return values.clone()

    scale = _scale(values)
    if not scale > 0.0:
        raise ValueError(
            "denoise: this volume's %g-%g centile range is %g, so the noise "
            "floor has no dynamic range to be measured against, but its noise "
            "level is not zero either.  A volume that is almost entirely "
            "background reaches this; crop it to the region of interest."
            % (SCALE_QUANTILES[0], SCALE_QUANTILES[1], scale))

    floor = SIGMA_FLOOR * scale / DEFAULT_SCALE
    return _nlm(values, sigma, floor, int(search), int(patch))


def _scale(values):
    """The **dynamic range** :data:`SIGMA_FLOOR` is measured against.

    Noise is a spread, so the quantity it is judged against must be a spread.
    The source uses ``max()``, a *level*, which is the wrong kind of quantity
    for two reasons.

    It is not robust.  A maximum over a million voxels is decided by one of
    them, and an outlier can only raise it; a raised maximum raises the floor,
    and a raised floor *excludes* voxels from filtering.  A single bright
    artefact -- a reconstruction spike, metal, a mis-set ``valid_range`` --
    therefore degrades the filter towards doing nothing, with no diagnostic.
    Measured on ``brain.mnc`` under the maximum: one planted voxel at ten times
    it took the filtered fraction from 92.7% to 28.1%, and at a hundred times to
    zero, where ``denoise`` ran for its full eight seconds and returned its input
    unchanged.

    It is also not translation-invariant.  Everything else in the filter is: the
    patch distances are differences of intensities and the noise level is a
    local standard deviation, so adding a constant to a volume changes neither.
    A floor taken from a level does change, so the same anatomy with a DC offset
    -- intensities in [1000, 1200] rather than [0, 200] -- would be filtered as
    though its noise were six times smaller than it is.  The difference of two
    centiles removes both faults and makes the whole filter affine-equivariant:
    ``denoise(a*v + b) == a*denoise(v) + b``.

    Nearest-rank centiles, by selection rather than by interpolating between two
    order statistics, for two reasons: ``torch.quantile`` refuses tensors above
    2**24 elements, which a 512^3 volume exceeds by a factor of eight, and a
    selected element rescales *exactly* with the volume, so the equivariance
    above holds for any positive factor rather than only for powers of two.

    A deliberate departure from the source, and one that changes the filter's
    behaviour on every volume rather than only on spiked ones.
    """
    values = values.reshape(-1)
    count = values.numel()
    low, high = (values.kthvalue(max(1, math.ceil(q * count))).values
                 for q in SCALE_QUANTILES)
    return float(high - low)


def noise_level(values):
    """Manjon's per-voxel noise estimate: the local standard deviation,
    smoothed once more over the same neighbourhood, and halved.

    Per-voxel rather than one number for the volume, which is what adapts the
    filter spatially: MRI noise is not stationary once the reconstruction has
    scaled it, and a single ``sigma`` would either spare the noisy regions or
    flatten the quiet ones.  The estimate is smoothed before use because the
    local standard deviation is itself noisy and appears in a denominator.

    The halving is the paper's, and makes :data:`SIGMA_FLOOR` a threshold on
    half the local spread rather than on all of it.
    """
    return _local_mean(_local_std(values)) / 2.0


def _local_mean(values):
    """The 3x3x3 box mean, reflecting at the boundary.

    Used twice: inside :func:`noise_level`, and as the statistic the rejection
    test compares between two voxels.
    """
    padded = F.pad(values[None, None], [1] * 6, mode="reflect")
    return F.avg_pool3d(padded, kernel_size=3, stride=1)[0, 0]


def _local_std(values):
    """Population standard deviation over 3x3x3, reflecting at the boundary.

    ``E[x^2] - E[x]^2``, clamped at zero because cancellation can take it
    slightly below.  Population, not sample: ``unbiased=False`` as everywhere
    else in this port (CLAUDE.md).
    """
    mean = _local_mean(values)
    mean_of_squares = _local_mean(values.pow(2))
    return (mean_of_squares - mean.pow(2)).clamp(min=0.0).sqrt()


def _nlm(values, sigma, floor, search, patch):
    """The gather: one shifted pass per neighbour offset, accumulated.

    Written as ``(2v+1)^3 - 1`` whole-volume passes rather than as a loop over
    voxels, which makes it a tensor program: at each offset, every voxel's patch
    distance to its neighbour at that offset is one
    :func:`~torch.nn.functional.avg_pool3d` over the squared difference.  The
    volume is padded once, by ``search + patch``, and every shift is a view into
    that buffer.

    ``floor`` is :data:`SIGMA_FLOOR` expressed in the volume's own intensity
    units; a voxel whose ``sigma`` is below it keeps its value untouched.

    Each ordered pair is visited twice, once from each end, with each visit
    using the centre voxel's own ``sigma``.  The C original visits the pair once
    and uses the first voxel's for both; see the module docstring.  The two
    should not be reconciled.

    Run under ``no_grad``: the filter runs before anything differentiable, and
    retaining the graph across 342 passes would cost tens of gigabytes for a
    gradient no caller requires.
    """
    with torch.no_grad():
        depth, height, width = values.shape
        reach = search + patch
        window = 2 * patch + 1

        padded = F.pad(values[None, None], [reach] * 6, mode="reflect")[0, 0]

        # Every voxel's patch, before any shift: the centre block of `padded`
        # grown by `patch` on each side, so that avg_pool3d over it yields one
        # distance per voxel with no further padding.
        centre = padded[search:search + depth + 2 * patch,
                        search:search + height + 2 * patch,
                        search:search + width + 2 * patch]

        means = _local_mean(values)
        padded_means = F.pad(means[None, None], [search] * 6,
                             mode="reflect")[0, 0]

        # Clamping at `floor` prevents a silent neighbourhood from dividing by
        # zero and costs nothing: every voxel it touches has sigma below the
        # floor, so its weights are discarded a few lines below.  The source
        # clamps at a fixed 1e-10, which is safe only while the volume is on
        # DEFAULT_SCALE -- on a volume stored small it bites on voxels above
        # the floor, and the filter stops being scale-free.
        denominator = 2.0 * sigma.clamp(min=floor).pow(2)
        rejection = MEAN_DIFFERENCE * sigma
        audible = sigma >= floor

        # The voxel itself, at weight one, is the first term of both sums.
        weighted = values.new_zeros(values.shape)
        total = values.new_ones(values.shape)

        for dz in range(-search, search + 1):
            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    z, y, x = search + dz, search + dy, search + dx

                    shifted_patch = padded[z:z + depth + 2 * patch,
                                           y:y + height + 2 * patch,
                                           x:x + width + 2 * patch]
                    shifted = padded[z + patch:z + patch + depth,
                                     y + patch:y + patch + height,
                                     x + patch:x + patch + width]
                    shifted_means = padded_means[z:z + depth,
                                                 y:y + height,
                                                 x:x + width]

                    squared = (centre - shifted_patch).pow(2)
                    distance = F.avg_pool3d(squared[None, None],
                                            kernel_size=window, stride=1)[0, 0]

                    # exp(-max(d/2sigma^2 - 1, 0)): flat at one for anything
                    # closer than the noise, decaying beyond it.
                    weight = torch.exp(
                        -(distance / denominator - 1.0).clamp(min=0.0))
                    weight = weight.masked_fill(
                        (means - shifted_means).abs() > rejection, 0.0)
                    weight = weight.masked_fill(~audible, 0.0)

                    weighted += weight * shifted
                    total += weight

        return (values + weighted) / total


def _check(values, search, patch, strength):
    """The preconditions of a filtering pass."""
    if values.dim() != 3:
        raise ValueError("denoise: expected a 3-D volume, got shape %s"
                         % (tuple(values.shape),))
    if int(search) < 1:
        raise ValueError("denoise: search radius must be at least 1 (got %r)"
                         % (search,))
    if int(patch) < 1:
        raise ValueError("denoise: patch half-width must be at least 1 "
                         "(got %r)" % (patch,))
    if float(strength) < 0.0:
        raise ValueError("denoise: strength cannot be negative (got %r)"
                         % (strength,))

    reach = int(search) + int(patch)
    if min(values.shape) <= reach:
        raise ValueError("denoise: reflect padding needs every axis longer "
                         "than search + patch = %d, but the volume is %s.  A "
                         "volume this small cannot be filtered at this search "
                         "radius; reduce it, or leave denoising off."
                         % (reach, tuple(values.shape)))

"""How blurred an intensity distribution is, as something you can differentiate.

N3 answers that question implicitly: it deconvolves the histogram, maps every
voxel through ``E[u | v]``, and calls the difference the field's fingerprint.
The answer never appears as a number, so there is nothing to descend on.  This
module makes the question explicit -- two measures of *sharpness*, both smooth
functions of the voxel intensities, so that ``torch.autograd`` can carry a
gradient from them back to the spline coefficients that produced them
(:mod:`torch_n3.optimize`).

    :func:`hoyer_sparsity` of a :func:`soft_histogram`
        A blurred histogram is a spread-out one.  Model-free: it assumes
        nothing about how many tissues the volume holds.

    :func:`cluster_tightness`
        A blurred histogram is one whose voxels sit far from their nearest
        tissue mean.  This is the tissue model N3's own sources carry as dead
        code -- ``nu_estimate_np_and_em.in``'s EM branch ``die``s without
        ``-sharpen`` -- written differentiably.

**Both are degenerate on their own.**  Each is optimised by a *constant*
image: a single-bin histogram has Hoyer sparsity 1, and voxels sitting exactly
on their centroid have zero within-cluster variance.  A smooth multiplicative
field can produce exactly that -- it need only cancel the image.  What stops
it is :func:`standardize`, which pins the first two moments of the intensities
before either measure looks at them, so that flattening the volume buys
nothing.  Nothing else in this module or in :mod:`torch_n3.optimize` prevents
the collapse, which is why ``tests/test_sharpness.py`` demonstrates it
happening when the standardization is removed.
"""

import torch


def standardize(values):
    """``values`` with zero mean and unit standard deviation.

    Applied to the corrected intensities before every measure below, and the
    one thing keeping either of them honest: both are minimised by a constant
    image, and a bias field is free to produce one.  After this, an affine
    change in the intensities -- which is what flattening the volume amounts
    to -- leaves the measure exactly where it was, so there is nothing to gain
    from it.

    Population standard deviation, ``unbiased=False``, as everywhere else in
    this port (CLAUDE.md: ``torch.std`` defaults the other way).
    """
    return (values - values.mean()) / values.std(unbiased=False)


def soft_histogram(values, centers, sigma):
    """A histogram built from Gaussian kernels rather than from bins.

    ``values`` is a flat tensor of intensities and ``centers`` the bin
    positions.  Each sample contributes ``exp(-d^2 / 2 sigma^2)`` to every
    centre, so the result is a smooth -- infinitely differentiable -- function
    of the samples, where an ordinary histogram is a step function of them and
    the Parzen one in :mod:`torch_n3.blocks.histogram` is only piecewise
    linear.  That smoothness is what lets a quasi-Newton method work on it.

    The result is a mean over samples rather than a sum, and is deliberately
    *not* normalised to sum to one: :func:`hoyer_sparsity` is invariant to the
    scale of what it is handed, so normalising would change nothing but the
    reader's expectations.

    Cost is ``len(values) x len(centers)``, which is why
    :mod:`torch_n3.optimize` subsamples the volume rather than handing this
    every voxel.
    """
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    centers = torch.as_tensor(centers, dtype=torch.float64).reshape(-1)

    differences = values[:, None] - centers[None, :]
    return torch.exp(-differences.pow(2) / (2.0 * float(sigma) ** 2)).mean(0)


def hoyer_sparsity(histogram, eps=1e-8):
    """How concentrated ``histogram`` is, on a scale from 0 to 1.

    Hoyer's ratio of the L1 and L2 norms, normalised so that a flat histogram
    scores 0 and one with all its mass in a single bin scores 1.  A bias field
    spreads each tissue's peak out; removing it pulls the mass back together,
    so **larger is better** and :mod:`torch_n3.optimize` minimises the
    negative.

    Scale-invariant by construction -- only the *shape* of the distribution
    means anything, which is the same reason the pipeline compares fields only
    after dividing out their means.  Exactly so at ``eps=0``; the default
    ``eps`` costs a relative ``eps/|h|_2`` and exists to keep an empty
    histogram finite.

    One wart worth knowing, because it looks like a bug when it appears: if
    every sample fell outside the centres, ``h`` is zero, the guarded ratio is
    zero, and this returns ``sqrt(bins)/(sqrt(bins) - 1)``, which is *above*
    one and means nothing.  A value over 1 is therefore a signal that the
    centres do not cover the data -- which is why :mod:`torch_n3.optimize`
    standardizes first and puts them at +-4 standard deviations.

    References
    ----------
    Hoyer, P. O. (2004). "Non-negative matrix factorization with sparseness
    constraints." *Journal of Machine Learning Research* 5, 1457-1469.
    """
    histogram = torch.as_tensor(histogram, dtype=torch.float64).reshape(-1)
    bins = histogram.numel()
    if bins < 2:
        raise ValueError("hoyer_sparsity: need at least two bins")

    root = bins ** 0.5
    ratio = histogram.abs().sum() / (histogram.pow(2).sum().sqrt() + eps)
    return (root - ratio) / (root - 1.0)


def cluster_tightness(values, centroids, beta=200.0):
    """Within-cluster variance under soft assignment to ``centroids``.

    Every sample is assigned to the ``classes`` centroids in proportion to
    ``softmax(-beta * d^2)`` and pays the weighted squared distance to them.
    With the intensities standardized the total variance is one, so this is
    the *fraction* of the variance the tissue model fails to explain: **lower
    is better**, and 0 means every voxel sits on a centroid.

    ``beta`` sets how hard the assignment is.  Large values approach k-means,
    where a sample belongs to its nearest centroid alone and the gradient is
    nearly discontinuous where two clusters meet; small values blur the
    classes into each other until the measure stops distinguishing them.  In
    standardized units the assignment switches over a scale of
    ``1/sqrt(beta)``, so the default of 200 is about a fifteenth of a standard
    deviation.

    ``centroids`` is a tensor of positions, and in :mod:`torch_n3.optimize` it
    is *learned* alongside the field: the tissue means are not known in
    advance and are nuisance parameters, not inputs.
    """
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    centroids = torch.as_tensor(centroids, dtype=torch.float64).reshape(-1)
    if centroids.numel() < 2:
        raise ValueError("cluster_tightness: need at least two centroids")

    squared = (values[:, None] - centroids[None, :]).pow(2)
    weights = torch.softmax(-float(beta) * squared, dim=1)
    return (weights * squared).sum(dim=1).mean()


def quantile_centroids(values, classes):
    """``classes`` starting centroids, evenly spaced through the data.

    Quantiles rather than a random draw or k-means++, for two reasons: it is
    deterministic, so it adds no seed to an estimator that already has one;
    and every centroid starts with mass around it, where a centroid that
    starts empty stays empty -- its assignment weights are zero, so it
    receives no gradient and never moves.
    """
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    classes = int(classes)
    if classes < 2:
        raise ValueError("quantile_centroids: need at least two classes")

    fractions = (torch.arange(classes, dtype=torch.float64,
                              device=values.device) + 0.5) / classes
    return torch.quantile(values, fractions)


def em_centroids(values, centroids, beta=200.0):
    """The centroids that minimise :func:`cluster_tightness` at fixed weights.

    One EM step: with the assignment weights held, the exact minimiser over
    the centroids is the weighted mean of the samples assigned to each.  Used
    by ``centroid_update="em"``, which alternates this closed form with descent
    on the field instead of learning the centroids by gradient too.

    An empty cluster would divide by zero; its weight sum is clamped, which
    leaves such a centroid where it was rather than sending it to infinity.
    """
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    centroids = torch.as_tensor(centroids, dtype=torch.float64).reshape(-1)

    squared = (values[:, None] - centroids[None, :]).pow(2)
    weights = torch.softmax(-float(beta) * squared, dim=1)
    mass = weights.sum(0)
    return torch.where(mass > 0.0, (weights * values[:, None]).sum(0)
                       / mass.clamp(min=1e-30), centroids)


def cluster_occupancy(values, centroids, beta=200.0):
    """The share of the samples each centroid holds.

    A diagnostic, not part of any loss: a centroid that has lost its mass is
    no longer modelling anything, and the fit has quietly become one with
    fewer classes than it was asked for.  :func:`quantile_centroids` makes that
    unlikely at the start; this is how it gets noticed if it happens anyway.
    """
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    centroids = torch.as_tensor(centroids, dtype=torch.float64).reshape(-1)

    squared = (values[:, None] - centroids[None, :]).pow(2)
    return torch.softmax(-float(beta) * squared, dim=1).mean(0)

"""Extending the field past the mask (``correct_field``).

The spline N3 fits is exactly zero outside its domain and meaningless outside
the mask it was fitted in, but ``nu_evaluate`` divides the *whole* volume by it.
Before dividing, the masked field is therefore extended outwards by solving
Laplace's equation on everything the mask does not cover, with the fitted values
as boundary data: the smoothest continuation.

The solver follows ``legacy/N3/src/CorrectField/correctField.cc``: successive
over-relaxation, run on a coarse grid first and interpolated down, which is what
makes it converge in a workable number of sweeps.  The legacy sweeps in raster
order; sweeping the two checkerboard colours in turn is the same Gauss-Seidel
iteration in a different order, and vectorises.
"""

import torch

#: Over-relaxation factor (``correctField.cc:33``).
SOR = 1.9

#: Mean absolute change at which a level is considered converged.
THRESHOLD = 1.0e-10


def correct_field(field, mask, step):
    """Replace ``field`` outside ``mask`` by a smooth extension of the inside.

    ``step`` is the voxel size along each axis.  Returns a new tensor; the
    values inside the mask are untouched.
    """
    field = torch.as_tensor(field, dtype=torch.float64)
    mask = torch.as_tensor(mask).to(torch.bool)
    if field.shape != mask.shape:
        raise ValueError("field and mask must have the same shape")

    values = torch.where(mask, field, torch.zeros_like(field))
    free = ~mask
    weight = [1.0 / float(s) ** 2 for s in step]

    # Coarsen until the grid is at least 4 mm, as the legacy does, then work
    # back down.  The finest level solved is every second voxel; the voxels
    # between are filled by the last interpolation.
    stride = 2
    while stride < int(4.0 / min(abs(float(s)) for s in step)):
        stride *= 2

    sweeps = 4 * 200
    while stride >= 2:
        _relax(values, free, weight, stride, sweeps)
        sweeps = sweeps // 2 + 1
        _interpolate(values, free, stride)
        stride //= 2
    return values


def _relax(values, free, weight, stride, sweeps):
    """Solve Laplace's equation on every ``stride``-th voxel, in place."""
    coarse = values[::stride, ::stride, ::stride]
    movable = free[::stride, ::stride, ::stride]

    index = [torch.arange(n, device=values.device) for n in coarse.shape]
    checker = (index[0][:, None, None] + index[1][None, :, None]
               + index[2][None, None, :]) % 2
    colours = [movable & (checker == 0), movable & (checker == 1)]

    moved = int(movable.sum())
    if not moved:
        return
    nowhere = torch.zeros_like(coarse)

    for _ in range(sweeps):
        changed = 0.0
        for target in colours:
            neighbours, count = _neighbour_average(coarse, weight)
            step = SOR * (neighbours / count.clamp(min=1e-30) - coarse)
            step = torch.where(target & (count > 0), step, nowhere)
            coarse += step
            changed += float(step.abs().sum())

        if changed / moved < THRESHOLD:
            break


def _neighbour_average(values, weight):
    """The weighted sum of each voxel's neighbours, and the weight it carries.

    Voxels on the edge of the grid have fewer neighbours.  The legacy leaves the
    missing ones out of both sums rather than reflecting them, which is a
    zero-flux boundary.
    """
    total = torch.zeros_like(values)
    count = torch.zeros_like(values)

    for axis, coefficient in enumerate(weight):
        if values.shape[axis] < 2:
            continue
        upper = _along(axis, slice(1, None))    # voxels with a lower neighbour
        lower = _along(axis, slice(None, -1))   # voxels with an upper neighbour
        total[upper] += coefficient * values[lower]
        count[upper] += coefficient
        total[lower] += coefficient * values[upper]
        count[lower] += coefficient

    return total, count


def _interpolate(values, free, stride):
    """Fill in the voxels between one level's samples, axis by axis.

    The legacy's extension operator (``correctField.cc:126-179``): halve the
    spacing along the last axis, then the middle one, then the first, taking the
    mean of the two neighbours already known, and copying the last known value
    where the grid runs out before a midpoint does.
    """
    half = stride // 2
    coarse = slice(None, None, stride)
    fine = slice(None, None, half)

    for axis, before, after in [(2, coarse, coarse), (1, coarse, fine),
                                (0, fine, fine)]:
        length = values.shape[axis]
        middle = torch.arange(half, length - half, stride, device=values.device)
        index = [before] * axis + [middle] + [after] * (2 - axis)
        if middle.numel():
            _average_into(values, free, index, axis, half)

        # The C loop leaves its counter one step past the last midpoint; if
        # that lands inside the volume it is filled by copying, not averaging.
        edge = half + stride * int(middle.numel())
        if edge < length:
            index[axis] = edge
            _copy_into(values, free, index, axis, half)


def _average_into(values, free, index, axis, half):
    """Set the free voxels at ``index`` to the mean of their two neighbours."""
    index = tuple(index)
    lower, upper = list(index), list(index)
    lower[axis] = index[axis] - half
    upper[axis] = index[axis] + half
    mean = 0.5 * (values[tuple(lower)] + values[tuple(upper)])
    values[index] = torch.where(free[index], mean, values[index])


def _copy_into(values, free, index, axis, half):
    """Set the free voxels at ``index`` to the neighbour below them."""
    index = tuple(index)
    source = list(index)
    source[axis] = index[axis] - half
    values[index] = torch.where(free[index], values[tuple(source)],
                                values[index])


def _along(axis, span):
    """``span`` applied to ``axis`` alone, as an indexing tuple."""
    return tuple(span if i == axis else slice(None) for i in range(3))

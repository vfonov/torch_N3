"""Stage 2: :mod:`torch_n3.blocks.spline` against ``TBSplineVolume``.

The B-spline is the block with the most room to be subtly wrong -- knot
placement, the domain, the bending-energy penalty and the sample weighting all
have to line up -- so it is checked against the legacy on real data at several
knot spacings, and separately against things that must be true of any correct
fit.

A note on tolerances.  The normal equations N3 solves are close to singular
(condition number around ``1e13`` at the default 200 mm spacing, where the
knots are further apart than the volume is wide), so the *coefficients* are
not determined to anything like full precision by either implementation and
the two wander apart in the near-null space.  What is well determined, and
what the pipeline uses, is the fitted field: that is what these compare.
"""

import pytest
import torch

from tests.conftest import assert_close, span
from torch_n3 import blocks
from torch_n3.backends import legacy
from torch_n3.volume import Volume


def unit_grid(shape):
    """A 1 mm isotropic grid at the origin -- geometry only, no data."""
    return Volume(torch.zeros(shape), start=(0.0, 0.0, 0.0), step=(1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def bumpy(chunk, chunk_mask):
    """A masked field with structure at several scales, as the loop produces."""
    inside = chunk_mask.data != 0
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)
                               for n in chunk.shape], indexing="ij")
    torch.manual_seed(11)
    values = (0.05 * torch.cos(z / 7.0) - 0.03 * torch.sin(y / 5.0)
              + 0.02 * x / chunk.shape[2]
              + 0.01 * torch.randn(chunk.shape, dtype=torch.float64))
    return torch.where(inside, values, torch.zeros_like(values)), inside


@pytest.mark.parametrize("distance,subsample", [(200.0, 1), (200.0, 2),
                                                (50.0, 1)])
def test_fit_matches_the_legacy_spline(chunk, bumpy, distance, subsample):
    values, inside = bumpy

    ours = blocks.BSplineField(chunk, distance, 1e-7).fit(values, inside,
                                                          subsample)
    theirs = legacy.BSplineField(chunk, distance, 1e-7).fit(values, inside,
                                                            subsample)

    fitted = ours.evaluate()
    assert_close(fitted, theirs.evaluate(), atol=1e-6 * span(fitted))


def test_evaluating_on_a_finer_grid_matches_the_legacy(chunk, bumpy):
    """Fitted coarse, evaluated fine -- the round trip through the ``.imp`` file."""
    values, inside = bumpy
    coarse = chunk.shrink(4)
    coarse_values = chunk.like(values).resample_like(coarse).data
    coarse_inside = chunk.like(inside.to(torch.float64)).resample_like(coarse).data != 0

    ours = blocks.BSplineField(coarse, 200.0, 1e-7).fit(coarse_values,
                                                        coarse_inside)
    theirs = legacy.BSplineField(coarse, 200.0, 1e-7).fit(coarse_values,
                                                          coarse_inside)

    fine = ours.evaluate_on(chunk)
    assert_close(fine, theirs.evaluate_on(chunk), atol=1e-6 * span(fine))


@pytest.mark.parametrize("size", [4, 5, 6, 9])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_bending_energy_is_a_gram_matrix(size, order):
    """``J`` is built from integrals of products, so it has to look like one.

    Symmetric, positive semi-definite, and banded: two basis functions more
    than three knots apart never overlap, so their integral is zero.
    """
    energy = blocks.spline.bending_energy(size, order)

    assert_close(energy, energy.T, atol=0.0)
    assert float(torch.linalg.eigvalsh(energy).min()) > -1e-12
    index = torch.arange(size)
    far = (index[:, None] - index[None, :]).abs() > 3
    assert not bool(energy[far].any())


def test_more_regularization_means_a_flatter_field():
    """What ``-lambda`` is for: it buys smoothness at the cost of fit."""
    shape = (24, 24, 24)
    x, _, _ = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    wavy = torch.sin(x / 3.0)

    grid = unit_grid(shape)
    curvature = []
    for lam in (1e-9, 1e-1):
        fitted = blocks.BSplineField(grid, 6.0, lam).fit(wavy).evaluate()
        curvature.append(float(fitted.diff(n=2, dim=0).abs().mean()))

    assert curvature[1] < curvature[0] / 10


def test_a_plane_is_fitted_exactly():
    """A plane lies in the span of the B-spline basis, so it must fit exactly."""
    shape = (12, 10, 8)
    x, y, z = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    ramp = 1.0 + 0.5 * x - 0.25 * y + 0.125 * z

    fitted = blocks.BSplineField(unit_grid(shape), 4.0, 1e-7).fit(ramp).evaluate()

    assert_close(fitted, ramp, atol=1e-6)


def test_the_fit_ignores_masked_voxels():
    """Masked-out voxels must not influence the fit, however wild they are."""
    shape = (12, 12, 12)
    x, _, _ = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    ramp = 0.25 * x

    mask = torch.ones(shape, dtype=torch.bool)
    mask[8:] = False
    corrupted = torch.where(mask, ramp, torch.full_like(ramp, 1000.0))

    grid = unit_grid(shape)
    clean = blocks.BSplineField(grid, 4.0).fit(ramp, mask).evaluate()
    corrupt = blocks.BSplineField(grid, 4.0).fit(corrupted, mask).evaluate()

    assert_close(clean[mask], corrupt[mask], atol=1e-8)


def test_the_spline_is_zero_outside_its_domain():
    """N3 relies on this: it is why ``nu_evaluate`` has to extend the field."""
    shape = (10, 10, 10)
    grid = unit_grid(shape)
    spline = blocks.BSplineField(grid, 4.0).fit(torch.ones(shape))

    beyond = Volume(torch.zeros((4, 10, 10)), start=(20.0, 0.0, 0.0),
                    step=(1.0, 1.0, 1.0))

    assert float(spline.evaluate_on(beyond).abs().max()) == 0.0


def test_subsampling_barely_changes_a_smooth_fit():
    """``-subsample`` thins the fit; with a smooth field it costs little."""
    shape = (24, 24, 24)
    x, y, _ = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    smooth = 1.0 + 0.01 * x - 0.005 * y

    grid = unit_grid(shape)
    every = blocks.BSplineField(grid, 12.0).fit(smooth).evaluate()
    thinned = blocks.BSplineField(grid, 12.0).fit(smooth, None, 3).evaluate()

    assert_close(every, thinned, atol=1e-6)

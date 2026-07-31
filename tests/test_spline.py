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

import numpy as np
import pytest
import torch

from tests.conftest import assert_close, span
from torch_n3 import blocks
from torch_n3.backends import legacy
from torch_n3.blocks.spline import DIRECT_SOLVERS
from torch_n3.volume import Volume


#: Slack for comparisons that hold exactly in exact arithmetic.  Not a
#: tolerance on the result -- an allowance for the last bits of a float64
#: solve, several orders below the differences it is applied to.
ROUNDING = 1e-9


def unit_grid(shape):
    """A 1 mm isotropic grid at the origin -- geometry only, no data."""
    return Volume(torch.zeros(shape), start=(0.0, 0.0, 0.0), step=(1.0, 1.0, 1.0))


def design_matrix(fitted, inside):
    """``A`` itself, rebuilt from the outside: column ``j`` is the spline of ``e_j``.

    The two solvers assemble ``A`` differently -- one folds it straight into
    ``AtA``, the other writes it out -- so the tests that compare them need a
    third construction that neither shares.  Evaluating the fitted spline with
    a single coefficient set to one gives exactly that column, which makes
    this an independent check of the assembly as well.
    """
    size = int(np.prod(fitted.n))
    unit = torch.zeros(size, dtype=torch.float64)
    columns = []
    for j in range(size):
        unit.zero_()
        unit[j] = 1.0
        fitted._coefficients = unit
        columns.append(fitted.evaluate()[inside])
    return torch.stack(columns, dim=1)


def objective(fitted, coefficients, matrix, values, lam, nsamples):
    """``||Ac - f||^2 + lambda*N*c'Jc`` -- what both solvers minimise."""
    penalty = blocks.spline.bending_energy_tensor(fitted.n)
    residual = matrix @ coefficients - values
    return float(residual @ residual
                 + lam * nsamples * (coefficients @ penalty @ coefficients))


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


@pytest.mark.parametrize("solver", DIRECT_SOLVERS)
@pytest.mark.parametrize("distance,subsample", [(200.0, 1), (200.0, 2),
                                                (50.0, 1)])
def test_fit_matches_the_legacy_spline(chunk, bumpy, distance, subsample,
                                       solver):
    """Every direct solver has to land on the oracle's answer, to one bound.

    The legacy has only the normal equations, so it is built plainly; what is
    being asked is whether solving the *same* fit a better-conditioned way --
    and, for ``"blocked"``, a band at a time -- still reproduces the C++.  It
    does, at every spacing.
    """
    values, inside = bumpy

    ours = blocks.BSplineField(chunk, distance, 1e-7, solver=solver).fit(
        values, inside, subsample)
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


@pytest.mark.parametrize("distance", [200.0, 100.0])
def test_the_direct_solvers_minimise_the_same_objective(chunk, bumpy, distance):
    """The one property that makes them interchangeable.

    ``[A; sqrt(lambda N) D]`` has the penalised normal equations as its own
    normal equations, term for term, so all three solvers are answering the
    same minimisation and none may reach a *higher* value of it than the
    others.  This is what would catch the penalty being scaled wrongly --
    ``lambda*N`` where ``sqrt(lambda*N)`` belongs, say -- which no comparison
    of the fitted fields would, because it stays a perfectly good fit to
    something else.  The stacked solvers are the more accurate, so they are
    allowed to win, and at 200 mm they do.
    """
    values, inside = bumpy
    lam = 1e-7

    fitted, coefficients = {}, {}
    for solver in DIRECT_SOLVERS:
        fit = blocks.BSplineField(chunk, distance, lam, solver=solver).fit(
            values, inside)
        fitted[solver], coefficients[solver] = fit, fit.coefficients.clone()

    reference = fitted["normal"]
    matrix = design_matrix(reference, inside)
    nsamples = int(inside.sum())
    scores = {solver: objective(reference, coefficients[solver], matrix,
                                values[inside], lam, nsamples)
              for solver in DIRECT_SOLVERS}

    for stacked in ("qr", "blocked"):
        assert scores[stacked] <= scores["normal"] * (1 + ROUNDING)
        # ...and each is at the minimum, not merely ordered.
        assert_close(scores[stacked], scores["normal"], rtol=ROUNDING,
                     atol=0.0)


@pytest.mark.parametrize("n", [[4, 4, 4], [5, 4, 6]])
def test_the_bending_energy_factor_squares_back_to_the_tensor(n):
    """``D`` is only meaningful if ``D'D`` is ``J``: that is the whole contract.

    ``J`` is singular -- an affine field bends not at all, so four directions
    of the coefficient space sit in its null space -- which is why ``D`` comes
    from the eigendecomposition rather than a Cholesky factor.  The clamp that
    handles those eigenvalues is where a wrong ``D`` would hide.
    """
    energy = blocks.spline.bending_energy_tensor(n)
    factor = blocks.spline.bending_energy_factor(n)

    assert_close(factor.T @ factor, energy,
                 atol=ROUNDING * float(energy.abs().max()))
    # The null space is real, and the factor has to reproduce its size.
    assert int(torch.linalg.matrix_rank(energy)) == energy.shape[0] - 4


@pytest.mark.parametrize("distance", [200.0, 100.0])
def test_the_stacked_system_is_the_square_root_of_the_normal_equations(
        chunk, bumpy, distance):
    """Why the QR solver exists, as the exact statement rather than a measurement.

    ``AtA + lambda*N*J`` is the Gram matrix of ``[A; sqrt(lambda N) D]``, and
    a Gram matrix has the square of its factor's condition number.  So the
    stacked form does not merely happen to be better conditioned here -- it is
    better conditioned by construction, on any data, and the exponent is what
    is being checked.  At the shipped 200 mm spacing that turns ``1e13``, where
    the last digits of a solve belong to the BLAS, into ``1e6``, where they
    do not.
    """
    values, inside = bumpy
    lam = 1e-7

    fit = blocks.BSplineField(chunk, distance, lam).fit(values, inside)
    matrix = design_matrix(fit, inside)
    nsamples = int(inside.sum())
    factor = blocks.spline.bending_energy_factor(fit.n)

    normal = (matrix.T @ matrix
              + lam * nsamples * blocks.spline.bending_energy_tensor(fit.n))
    stacked = torch.cat([matrix, np.sqrt(lam * nsamples) * factor], dim=0)

    conditions = (float(torch.linalg.cond(normal)),
                  float(torch.linalg.cond(stacked)))
    # Not a fitted bound: the ratio is 1 in exact arithmetic.  The slack is
    # for computing a 1e13 condition number in float64 at all, which is the
    # very ill-conditioning being complained about.
    assert conditions[0] == pytest.approx(conditions[1] ** 2, rel=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("solver", ["qr", "blocked"])
@pytest.mark.parametrize("distance", [200.0, 100.0])
def test_the_stacked_fit_is_the_same_on_the_gpu(chunk, bumpy, distance, solver):
    """The reproducibility this solver was added for, stated as a requirement.

    A fit that is determined only to the machine's BLAS is a fit that moves
    when the device does, and under the normal equations at 200 mm it moves by
    around ``5e-9`` relative -- above the ``5.5e-8`` at which one histogram
    count flips and the whole pipeline diverges (see CLAUDE.md).  The bound
    here is float64 rounding accumulated over the fit, ``sqrt(N) * eps`` with
    ``N`` around ``1e5``, which is ``1e-13``; ``1e-11`` leaves two decades and
    is still two orders below where the normal equations sit.
    """
    values, inside = bumpy

    def fit(grid, data, mask):
        return blocks.BSplineField(grid, distance, 1e-7, solver=solver).fit(
            data, mask).evaluate()

    on_cpu = fit(chunk, values, inside)
    on_gpu = fit(chunk.like(chunk.data.cuda()), values.cuda(), inside.cuda())

    assert_close(on_gpu.cpu(), on_cpu, atol=1e-11 * span(on_cpu))


def test_the_blocked_solver_reproduces_the_dense_one(chunk, bumpy):
    """The banded sweep is an implementation of ``"qr"``, not an approximation.

    Sorting the rows and folding them in a window at a time is the same
    factorization in a different order, so the two must agree to rounding and
    not merely to the bound the legacy comparison uses.  Anything looser would
    mean the windowing had dropped something -- most likely a row of ``R``
    finalised before it was safe to.
    """
    values, inside = bumpy

    for distance in (200.0, 50.0):
        dense = blocks.BSplineField(chunk, distance, 1e-7, solver="qr").fit(
            values, inside).evaluate()
        banded = blocks.BSplineField(chunk, distance, 1e-7,
                                     solver="blocked").fit(
            values, inside).evaluate()

        assert_close(banded, dense, atol=ROUNDING * span(dense))


def test_the_sparse_solver_reports_that_it_cannot_converge(chunk, bumpy):
    """What ``"sparse"`` is, asserted rather than left as a footnote.

    It is not held to the other solvers' bound because it does not meet it,
    and the reason is not a tolerance that could be nudged: LSQR's convergence
    is governed by the condition number of the stacked system, and it gives up
    reporting ``istop=3``, "condition number exceeds ``conlim``".  That is a
    property of the method against this matrix, so it is what gets asserted --
    if it ever stops reporting it, something has genuinely improved and this
    test should be the thing that says so.

    ``istop`` 1 or 2 would mean it had reached ``atol``/``btol``.
    """
    values, inside = bumpy

    fitted = blocks.BSplineField(chunk, 200.0, 1e-7, solver="sparse").fit(
        values, inside)

    assert fitted.solve_info["istop"] not in (1, 2)
    assert torch.isfinite(fitted.coefficients).all()


def test_the_legacy_backend_refuses_a_solver_it_does_not_have(chunk):
    """The oracle has one solver; asking it for the other must not go quiet."""
    with pytest.raises(ValueError, match="legacy backend"):
        legacy.BSplineField(chunk, 200.0, 1e-7, solver="qr")


def test_an_unknown_solver_is_rejected(chunk):
    with pytest.raises(ValueError, match="unknown solver"):
        blocks.BSplineField(chunk, 200.0, 1e-7, solver="cholesky")


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


@pytest.mark.parametrize("solver", DIRECT_SOLVERS)
def test_regularization_trades_bending_energy_for_fit(solver):
    """What ``-lambda`` is for, as the exact statement rather than a rule of thumb.

    The fit minimises ``||Ac - f||^2 + lambda*N*c'Jc``.  Compare the objective
    at two weights, each evaluated at the other's minimiser, and the cross
    terms cancel to give: as ``lambda`` rises the bending energy ``c'Jc`` can
    only fall and the residual can only rise.  That holds for every pair, with
    no factor to choose and nothing to tune -- so it is checked at every step
    of a sweep, in both directions.

    ``c'Jc`` is the integrated squared curvature of the fitted field, which is
    what "smoother" means here.

    What this does *not* do is check that ``J`` is right.  The energy is
    measured with the same matrix the fit penalises with, so the property
    holds for any non-degenerate ``J``: mutating the first-derivative cross
    terms to drop their factor of two leaves this test green (the parity test
    against the shim catches it).  Zeroing ``J`` altogether, or dropping the
    second-derivative terms, does fail here -- those make the sweep flat.
    """
    shape = (24, 24, 24)
    x, _, _ = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    wavy = torch.sin(x / 3.0)
    grid = unit_grid(shape)

    energy, residual = [], []
    for lam in [10.0 ** -power for power in range(9, 0, -2)]:
        fitted = blocks.BSplineField(grid, 6.0, lam, solver=solver).fit(wavy)
        penalty = blocks.spline.bending_energy_tensor(fitted.n)
        coefficients = fitted.coefficients

        energy.append(float(coefficients @ penalty @ coefficients))
        # Every voxel is a sample here, so evaluating the spline on the grid
        # is exactly ``Ac`` and this is the least-squares residual itself.
        residual.append(float(((fitted.evaluate() - wavy) ** 2).sum()))

    # Only floating point may break the ordering; the mathematics may not.
    for weaker, stronger in zip(energy, energy[1:]):
        assert stronger <= weaker * (1 + ROUNDING)
    for weaker, stronger in zip(residual, residual[1:]):
        assert stronger >= weaker * (1 - ROUNDING)

    # ...and the sweep has to be wide enough for the trade to be visible.
    assert energy[-1] < energy[0] / 100
    assert residual[-1] > residual[0]


@pytest.mark.parametrize("solver", DIRECT_SOLVERS)
def test_a_plane_is_fitted_exactly(solver):
    """A plane lies in the span of the B-spline basis, so it must fit exactly."""
    shape = (12, 10, 8)
    x, y, z = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    ramp = 1.0 + 0.5 * x - 0.25 * y + 0.125 * z

    fitted = blocks.BSplineField(unit_grid(shape), 4.0, 1e-7,
                                 solver=solver).fit(ramp).evaluate()

    assert_close(fitted, ramp, atol=1e-6)


@pytest.mark.parametrize("solver", DIRECT_SOLVERS)
def test_the_fit_ignores_masked_voxels(solver):
    """Masked-out voxels must not influence the fit, however wild they are."""
    shape = (12, 12, 12)
    x, _, _ = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    ramp = 0.25 * x

    mask = torch.ones(shape, dtype=torch.bool)
    mask[8:] = False
    corrupted = torch.where(mask, ramp, torch.full_like(ramp, 1000.0))

    grid = unit_grid(shape)
    clean = blocks.BSplineField(grid, 4.0, solver=solver).fit(
        ramp, mask).evaluate()
    corrupt = blocks.BSplineField(grid, 4.0, solver=solver).fit(
        corrupted, mask).evaluate()

    assert_close(clean[mask], corrupt[mask], atol=1e-8)


@pytest.mark.parametrize("solver", DIRECT_SOLVERS)
def test_the_spline_is_zero_outside_its_domain(solver):
    """N3 relies on this: it is why ``nu_evaluate`` has to extend the field."""
    shape = (10, 10, 10)
    grid = unit_grid(shape)
    spline = blocks.BSplineField(grid, 4.0, solver=solver).fit(torch.ones(shape))

    beyond = Volume(torch.zeros((4, 10, 10)), start=(20.0, 0.0, 0.0),
                    step=(1.0, 1.0, 1.0))

    assert float(spline.evaluate_on(beyond).abs().max()) == 0.0


@pytest.mark.parametrize("solver", DIRECT_SOLVERS)
def test_subsampling_barely_changes_a_smooth_fit(solver):
    """``-subsample`` thins the fit; with a smooth field it costs little."""
    shape = (24, 24, 24)
    x, y, _ = torch.meshgrid(*[torch.arange(s, dtype=torch.float64)
                               for s in shape], indexing="ij")
    smooth = 1.0 + 0.01 * x - 0.005 * y

    grid = unit_grid(shape)
    every = blocks.BSplineField(grid, 12.0, solver=solver).fit(
        smooth).evaluate()
    thinned = blocks.BSplineField(grid, 12.0, solver=solver).fit(
        smooth, None, 3).evaluate()

    assert_close(every, thinned, atol=1e-6)

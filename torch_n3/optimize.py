"""Bias correction posed as an optimization rather than a fixed-point iteration.

``pipeline.nu_estimate`` is N3: sharpen the histogram, attribute the residual
to the field, smooth it, repeat, and terminate when the field ceases to change.
That loop descends on no stated objective, and CLAUDE.md records the
consequences: the stopping rule quantises everything downstream, one histogram
count moving between bins changes the answer by four orders of magnitude, and
no end-to-end comparison is informative past three digits.

:func:`nu_optimize` keeps N3's *model* exactly -- a tensor cubic B-spline
field with a bending-energy penalty, in the log domain -- and replaces the
loop with gradient descent on a stated objective:

    minimise   measure(standardize(v - F_c))  +  penalty * c' J c

over the spline coefficients ``c``, where ``measure`` is one of the two in
:mod:`torch_n3.blocks.sharpness`.  There is then one number that goes down,
and it can be watched.

It returns what ``nu_estimate`` returns, a fitted
:class:`~torch_n3.blocks.spline.BSplineField` describing the *multiplicative*
field, so ``nu_evaluate``, ``evaluate_field`` and everything in
``experiments/`` accept it without modification.

**What is not shared with N3.**  ``penalty`` is not ``lambda``.  N3's weighs
the bending energy against a least-squares residual in log-intensity squared;
here the data term is a dimensionless measure of order one, so the two are on
unrelated scales and the default below was measured rather than inherited.

**What makes this well posed.**  ``standardize``.  Both measures are optimal on
a constant image, and a smooth field is able to produce one by cancelling the
volume.  Fixing the first two moments of the corrected intensities before
measuring them removes that solution, and nothing else here does.  See :mod:`torch_n3.blocks.sharpness` and the collapse tests.

**Where the two measures stand, measured.**  On ``tests/tables.py``'s
experiment -- ``brain_nu_ref.mnc``, the analytic planted field, the score N3's
own published table reports -- at each side's best weight, 20% planted:

    knots     N3      hoyer     tightness
    200 mm   0.13%    0.28%     diverges
    100 mm   0.17%    0.21%     diverges
     50 mm   0.25%    0.17%     diverges

``hoyer`` is a working estimator: worse than N3 where N3 is strongest, better
at 50 mm, and, unlike N3, improving as the field gains freedom, which is the
opposite trend.

``tightness`` **does not work, and not for lack of tuning.**  Its loss falls
monotonically while the estimate gets worse and the field's own
non-uniformity grows without bound (at 200 mm: loss 0.234 -> 0.185 while the
field goes from 35% to 55% non-uniform, and at a lighter penalty to 257%).
The cause is a second degeneracy that ``standardize`` does not address:
within-cluster variance is also minimised by a distribution concentrated at a
few widely separated modes, and a smooth field can approach that by
*amplifying* contrast as well as by flattening it.  Standardizing fixes the
first two moments; nothing fixes the third.  It is retained here because it is
implemented, tested and instructive, not because it should be used.
"""

import math

import torch

from torch_n3.blocks.sharpness import (cluster_occupancy, cluster_tightness,
                                       em_centroids, hoyer_sparsity,
                                       quantile_centroids, soft_histogram,
                                       standardize)
from torch_n3.blocks.spline import BSplineField, bending_energy_tensor
from torch_n3.pipeline import nu_estimate

#: The measures :func:`nu_optimize` can descend on.
OBJECTIVES = ("hoyer", "tightness")

#: Full-width-at-half-maximum to standard deviation.
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

#: Bending-energy weight for the final refit, which only has to *represent*
#: the optimized field rather than smooth anything -- so it is N3's default,
#: and it is not :data:`DEFAULTS`'s ``penalty``.
REFIT_LAMBDA = 1e-7

#: Weight the preconditioner is anchored at, and **deliberately not**
#: ``penalty``.
#:
#: Anchoring it on the weight being used makes ``penalty`` inert: with
#: ``R0'R0 = A'A/N + w J`` and ``c = R0^-1 z``, the penalty term in the
#: reparameterized objective is ``w z'R0^-T J R0^-1 z``, and once ``w J``
#: dominates the data block that is ``~z'z`` whatever ``w`` is.  Measured on
#: ``chunk.mnc``: sweeping ``penalty`` over six decades moved the loss by
#: nothing at all -- identical to six decimals, under two different
#: optimizers -- because the parameterization was undoing exactly what the
#: loss was asking for.  A *fixed* anchor is a fixed metric, and leaves
#: ``penalty`` free to mean something.
#:
#: It cannot be zero either: anchoring on ``A`` alone is what CLAUDE.md warns
#: about, since the masked design leaves basis functions with no data under
#: them and no metric to scale them by.
PRECONDITION_ANCHOR = 1.0

#: Default bending-energy weight, per objective, used when ``penalty`` is not
#: given.  Two entries rather than one because the measures are not on a
#: common scale -- Hoyer sparsity is a number in [0, 1] whose gradient is
#: gentle, within-cluster variance is a fraction of the total with quite
#: different curvature -- and a single weight cannot serve both.
#:
#: ``hoyer``'s value is the best of ten decades on ``tests/tables.py``'s
#: experiment -- ``brain_nu_ref.mnc``, the analytic planted field, all three
#: knot spacings -- where 1e-4 was best at 200 mm (0.28%) and 100 mm (0.21%)
#: and within a hair of the best at 50 mm.  Below about 1e-5 the fit becomes
#: unstable at fine spacings: at 50 mm and 1e-6 the field ran away to a
#: non-uniformity of 1316%.
#:
#: ``tightness`` has **no good value, at any spacing**, and the entry below is
#: only the least bad of those tried.  See the module docstring: minimising
#: that measure makes the estimate worse, so this is not a tuning problem.
PENALTY = {"hoyer": 1e-4, "tightness": 100.0}

#: Default step scale, per optimizer, used when ``learning_rate`` is not given.
#:
#: L-BFGS at ``lr=1`` fails on the first line search on ``brain_nu_ref.mnc``:
#: the loss never leaves its initial value and the gradient stays at 0.289,
#: while ``lr=0.1`` runs 66 iterations and reaches a gradient of 2e-5.  A
#: failed line search leaves the parameters exactly where they were, which
#: looks identical to convergence from the outside -- see :func:`_stopped`
#: for how the two are told apart now.
LEARNING_RATE = {"lbfgs": 0.1, "adam": 0.01}

#: How far the gradient must fall, **relative to its value at the start**,
#: for a settled loss to count as convergence rather than as a stall.
#:
#: Relative because the absolute floor is a property of the problem, not of
#: the run: on ``brain_nu_ref.mnc`` both L-BFGS and Adam plateau at
#: ``|grad|inf ~ 4e-5`` having reached the *same* loss to six decimals, so an
#: absolute 1e-6 would call a converged run stalled.  What a genuine stall
#: looks like is different in kind -- the failed line search at ``lr=1``
#: leaves the gradient exactly where it started -- and a relative threshold
#: separates the two cleanly.
GRADIENT_TOLERANCE = 1e-3

#: How many times a stalled L-BFGS is restarted at a tenth of the step.
#: Restarting from the current parameters with a fresh curvature history is
#: the standard remedy: it is the history, built from a bad step, that the
#: line search is choking on.
RESTARTS = 3

DEFAULTS = dict(
    objective="hoyer",

    # The field model, named as ``pipeline.DEFAULTS`` names it.
    distance=200.0,
    shrink=4,
    background=1.0,
    subsample=1,
    solver="normal",

    # The bending-energy weight.  `None` takes it from `PENALTY` below, which
    # is per objective because the two measures are not on the same scale.
    penalty=None,

    # `hoyer`: the soft histogram.  `sigma` defaults to N3's assumed blur
    # width, carried into standardized units at run time.
    bins=128,
    span=4.0,
    sigma=None,
    fwhm=0.15,

    # `tightness`: the tissue model.
    classes=3,
    beta=200.0,
    centroid_update="joint",

    # Which voxels the measure sees.
    sample_size=16384,
    resample="once",
    seed=0,

    # The descent.
    optimizer="lbfgs",
    max_iterations=200,
    learning_rate=None,
    tolerance=1e-9,
    gradient_tolerance=GRADIENT_TOLERANCE,
    restarts=RESTARTS,
    precondition="stacked",
    anchor=PRECONDITION_ANCHOR,
    init="zero",
    init_iterations=1,
)


def nu_optimize(volume, mask=None, verbose=False, **options):
    """Estimate the bias field by gradient descent.  Returns a ``BSplineField``.

    A drop-in alternative to :func:`torch_n3.pipeline.nu_estimate`: same
    arguments where they mean the same thing, same return type.  Diagnostics
    -- the loss history, the iteration count, why it stopped, and for
    ``objective="tightness"`` the learned centroids and their occupancy --
    are attached to the returned spline as ``optimize_info``.
    """
    opts = dict(DEFAULTS, **options)
    _check(opts)
    if opts["penalty"] is None:
        opts["penalty"] = PENALTY[opts["objective"]]
    if opts["learning_rate"] is None:
        opts["learning_rate"] = LEARNING_RATE[opts["optimizer"]]

    grid = volume if opts["shrink"] == 1 else volume.shrink(opts["shrink"])
    log_volume = torch.log(grid.data.clamp(min=1.0))
    inside = grid.data > opts["background"]
    if mask is not None:
        inside &= mask.resample_like(grid).data != 0
    if not inside.any():
        raise ValueError("the mask is empty: no voxel is above the background "
                         "threshold inside the region of interest")

    spline = BSplineField(grid, opts["distance"], REFIT_LAMBDA,
                          solver=opts["solver"])
    columns, weights, where = spline.design(inside, opts["subsample"])
    observed = log_volume[where[:, 0], where[:, 1], where[:, 2]]

    state = _Problem(spline, columns, weights, observed, opts)
    info = state.run(verbose)

    # The optimized field is a log field, defined only up to a constant --
    # both measures are blind to one.  Fixing that constant at zero over the
    # mask is the gauge, and makes the returned field comparable with N3's.
    spline.coefficients = state.coefficients().detach()
    log_field = spline.evaluate()
    log_field = log_field - log_field[inside].mean()

    field = BSplineField(grid, opts["distance"], REFIT_LAMBDA,
                         solver=opts["solver"]).fit(
        torch.exp(log_field), inside, opts["subsample"])
    field.optimize_info = info
    return field


class _Problem:
    """The objective, its parameters, and the descent that moves them.

    Holds everything that does not change between evaluations -- the design
    rows, the observed intensities, the penalty matrix, the preconditioner --
    so that a closure is a gather, a measure and a quadratic form.
    """

    def __init__(self, spline, columns, weights, observed, opts):
        self.spline = spline
        self.columns = columns
        self.weights = weights
        self.observed = observed
        self.opts = opts
        self.device = spline.device
        self.samples = observed.numel()

        self.generator = torch.Generator().manual_seed(int(opts["seed"]))
        self.batch = self._draw()

        self.penalty = bending_energy_tensor(spline.n, self.device)
        self.centers = torch.linspace(-opts["span"], opts["span"],
                                      int(opts["bins"]), dtype=torch.float64,
                                      device=self.device)
        self.sigma = self._sigma()
        self.triangle = self._preconditioner()

        self.parameter = self._initial()
        self.centroids = self._initial_centroids()

    # ------------------------------------------------------------- the model

    def coefficients(self, parameter=None):
        """The spline coefficients the current parameters describe.

        Identity without preconditioning; a triangular solve with it, which is
        differentiable, so the descent sees the whitened problem while the
        spline sees the coefficients it expects.
        """
        parameter = self.parameter if parameter is None else parameter
        if self.triangle is None:
            return parameter
        return torch.linalg.solve_triangular(
            self.triangle, parameter[:, None], upper=True)[:, 0]

    def corrected(self, coefficients, batch):
        """The standardized corrected intensities over ``batch``.

        ``(weights * c[columns]).sum(1)`` is the field at those samples: 64
        multiply-adds each, no dense design matrix, differentiable in ``c``.
        """
        rows = self.columns[batch]
        field = (self.weights[batch] * coefficients[rows]).sum(1)
        return standardize(self.observed[batch] - field)

    def loss(self, batch=None):
        """The objective at the current parameters."""
        batch = self.batch if batch is None else batch
        coefficients = self.coefficients()
        values = self.corrected(coefficients, batch)

        if self.opts["objective"] == "hoyer":
            measure = -hoyer_sparsity(
                soft_histogram(values, self.centers, self.sigma))
        else:
            measure = cluster_tightness(values, self.centroids,
                                        self.opts["beta"])

        bending = coefficients @ (self.penalty @ coefficients)
        return measure + self.opts["penalty"] * bending

    # ---------------------------------------------------------- the optimizer

    def run(self, verbose=False):
        """Descend, restarting a stalled line search at a smaller step.

        "The loss stopped moving" and "the optimizer converged" are different
        events, and with a line search the first happens without the second: a
        failed search leaves the parameters exactly where they were, so the
        change is zero while the gradient is whatever it was.  Treating that
        as convergence is how a run reports success having done nothing --
        measured on ``brain_nu_ref.mnc``, where the shipped ``lr=1`` moved the
        loss not at all and stopped with a gradient of 0.289.

        So a stall is diagnosed by the gradient, and answered by restarting
        with a fresh curvature history at a tenth of the step.  If the
        restarts run out, ``stopped`` says ``"stalled"`` -- an honest label
        the caller can see, and one the experiment records.
        """
        history, gradient = [], 0.0
        rate = float(self.opts["learning_rate"])
        budget = int(self.opts["max_iterations"])
        stopped = "max_iterations"

        # The gradient *before* any step, which is what "how far has it
        # fallen" is measured against.  Taking it after the first step would
        # credit that step's progress to the starting point.
        first = self._gradient()

        for _ in range(int(self.opts["restarts"]) + 1):
            optimizer = self._optimizer(rate)
            while len(history) < budget:
                value, gradient = self._step(optimizer)
                history.append(value)
                if verbose:
                    print("iteration %d: loss %.8f  |grad|inf %.3g"
                          % (len(history) - 1, value, gradient))
                if not _settled(history, self.opts["tolerance"]):
                    continue
                floor = self.opts["gradient_tolerance"] * max(first, 1e-300)
                stopped = "tolerance" if gradient <= floor else "stalled"
                break
            else:
                stopped = "max_iterations"

            if stopped != "stalled" or len(history) >= budget:
                break
            rate /= 10.0
            if verbose:
                print("  line search stalled at |grad|inf %.3g; "
                      "restarting at lr=%g" % (gradient, rate))

        info = {"loss": history, "iterations": len(history),
                "stopped": stopped, "gradient": gradient,
                "learning_rate": rate,
                "objective": self.opts["objective"],
                "samples": int(self.batch.numel())}
        if self.opts["objective"] == "tightness":
            with torch.no_grad():
                values = self.corrected(self.coefficients(), self.batch)
                info["centroids"] = self.centroids.detach().tolist()
                info["occupancy"] = cluster_occupancy(
                    values, self.centroids, self.opts["beta"]).tolist()
        return info

    def _optimizer(self, rate):
        parameters = [self.parameter]
        if self._learns_centroids():
            parameters.append(self.centroids)

        if self.opts["optimizer"] == "lbfgs":
            return torch.optim.LBFGS(parameters, lr=rate, max_iter=1,
                                     history_size=50,
                                     line_search_fn="strong_wolfe")
        return torch.optim.Adam(parameters, lr=rate)

    def _gradient(self):
        """``|grad|inf`` at the current parameters, without taking a step."""
        if self.parameter.grad is not None:
            self.parameter.grad = None
        self.loss().backward()
        gradient = float(self.parameter.grad.abs().max())
        self.parameter.grad = None
        return gradient

    def _step(self, optimizer):
        """One iteration.  Returns ``(loss, |grad|inf)``."""
        if self.opts["resample"] == "always":
            self.batch = self._draw()

        def closure():
            optimizer.zero_grad()
            value = self.loss()
            value.backward()
            return value

        value = float(optimizer.step(closure).detach())
        if self._updates_centroids_by_em():
            self._em_step()

        gradient = (0.0 if self.parameter.grad is None
                    else float(self.parameter.grad.abs().max()))
        return value, gradient

    # -------------------------------------------------------------- private

    def _draw(self):
        """A random subset of the masked samples, or all of them.

        Drawn on the CPU whatever device the volume is on, so ``seed`` names
        the same voxels everywhere.
        """
        wanted = int(self.opts["sample_size"])
        if wanted <= 0 or wanted >= self.samples:
            return torch.arange(self.samples, device=self.device)
        chosen = torch.randperm(self.samples, generator=self.generator)[:wanted]
        return chosen.to(self.device)

    def _sigma(self):
        """The soft histogram's width, in standardized units.

        ``fwhm`` is N3's assumed blur in log intensity; dividing by the
        volume's own spread carries it into the units the measure works in, so
        that the width this smooths with is the width N3 deconvolves.
        """
        if self.opts["sigma"] is not None:
            sigma = float(self.opts["sigma"])
        else:
            spread = float(self.observed.std(unbiased=False))
            sigma = float(self.opts["fwhm"]) * FWHM_TO_SIGMA / max(spread, 1e-12)

        spacing = 2.0 * float(self.opts["span"]) / (int(self.opts["bins"]) - 1)
        if sigma < spacing:
            raise ValueError(
                "sigma %.4g is below the bin spacing %.4g (span %g over %d "
                "bins): the soft histogram would be a comb of separate spikes "
                "and its sparsity an artefact of the grid.  Raise --bins or "
                "set --sigma explicitly."
                % (sigma, spacing, self.opts["span"], self.opts["bins"]))
        return sigma

    def _preconditioner(self):
        """``R0`` from the stacked system at a fixed anchor, or ``None``.

        CLAUDE.md measures the normal equations at ``cond ~ 5e12``; the
        stacked matrix's condition number is its square root by construction,
        and ``R0`` inherits that.  It is only a metric -- it does not have to
        be the Hessian of this objective, which is not a least-squares one --
        so it is built once, from the first batch, and left alone.

        The weight is :data:`PRECONDITION_ANCHOR` and *not* ``penalty``; see
        that constant for the measurement that says why.
        """
        if self.opts["precondition"] != "stacked":
            return None

        from torch_n3.blocks.spline import bending_energy_factor

        size = self.penalty.shape[0]
        batch = self.batch
        design = torch.zeros((batch.numel(), size), dtype=torch.float64,
                             device=self.device)
        design.scatter_(1, self.columns[batch], self.weights[batch])

        factor = bending_energy_factor(self.spline.n, self.device)
        stacked = torch.cat([design / math.sqrt(batch.numel()),
                             math.sqrt(self.opts["anchor"]) * factor], dim=0)
        return torch.linalg.qr(stacked, mode="r")[1]

    def _initial(self):
        """The starting parameters: a flat field, or N3's answer.

        ``init="n3"`` asks the more interesting question -- whether descent
        improves on N3's fixed point when it starts from it.
        """
        size = self.penalty.shape[0]
        start = torch.zeros(size, dtype=torch.float64, device=self.device)

        if self.opts["init"] == "n3":
            field = nu_estimate(
                self.spline.grid, distance=self.opts["distance"],
                lam=REFIT_LAMBDA, shrink=1, solver=self.opts["solver"],
                iterations=(int(self.opts["init_iterations"]),), stop=(0.0,))
            start = torch.log(field.coefficients.clamp(min=1e-12))
            if self.triangle is not None:
                start = self.triangle @ start

        return start.requires_grad_(True)

    def _initial_centroids(self):
        if self.opts["objective"] != "tightness":
            return None
        with torch.no_grad():
            values = self.corrected(self.coefficients(), self.batch)
            centroids = quantile_centroids(values, self.opts["classes"])
        return centroids.requires_grad_(self._learns_centroids())

    def _learns_centroids(self):
        return (self.opts["objective"] == "tightness"
                and self.opts["centroid_update"] == "joint")

    def _updates_centroids_by_em(self):
        return (self.opts["objective"] == "tightness"
                and self.opts["centroid_update"] == "em")

    def _em_step(self):
        with torch.no_grad():
            values = self.corrected(self.coefficients(), self.batch)
            self.centroids = em_centroids(values, self.centroids,
                                          self.opts["beta"])


def _settled(history, tolerance):
    """Whether the loss has stopped moving, relative to its own size.

    Necessary for convergence and not sufficient for it: a stalled line
    search settles too.  :meth:`_Problem.run` asks the gradient which it was.
    """
    if len(history) < 2:
        return False
    change = abs(history[-2] - history[-1])
    return change <= float(tolerance) * max(1.0, abs(history[-1]))


def _check(opts):
    if opts["objective"] not in OBJECTIVES:
        raise ValueError("unknown objective %r: expected one of %s"
                         % (opts["objective"], ", ".join(OBJECTIVES)))
    if opts["optimizer"] not in ("lbfgs", "adam"):
        raise ValueError("unknown optimizer %r: expected lbfgs or adam"
                         % (opts["optimizer"],))
    if opts["resample"] not in ("once", "always"):
        raise ValueError("unknown resample policy %r: expected once or always"
                         % (opts["resample"],))
    if opts["centroid_update"] not in ("joint", "em"):
        raise ValueError("unknown centroid_update %r: expected joint or em"
                         % (opts["centroid_update"],))
    if opts["precondition"] not in ("stacked", "none"):
        raise ValueError("unknown precondition %r: expected stacked or none"
                         % (opts["precondition"],))
    if opts["init"] not in ("zero", "n3"):
        raise ValueError("unknown init %r: expected zero or n3"
                         % (opts["init"],))

    if opts["resample"] == "always" and opts["optimizer"] == "lbfgs":
        raise ValueError(
            "resample=\"always\" makes the objective stochastic, and L-BFGS's "
            "line search and curvature pairs both assume it is not; the fit "
            "would be incorrect without raising an error.  Use "
            "optimizer=\"adam\" with it, or resample=\"once\" with L-BFGS.")

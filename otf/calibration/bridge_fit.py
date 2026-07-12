"""Calibrate the Schrödinger-bridge betas to a TARGET terminal law.

Solves the inverse problem: given a target sample of horizon log-returns
(e.g. realized non-overlapping k-day returns, or a market-implied terminal
sample), find beta = (b1..b5) whose bridged terminal law is closest to it.

Method
------
* **Common random numbers**: the :class:`otf.models.sfv.PathEngine` freezes
  all shocks at construction; an objective evaluation replays them under the
  trial drift, so the objective is deterministic in beta and the shared
  pure-stdlib Nelder-Mead applies.
* **Distance**: in one dimension the 2-Wasserstein distance is exact and
  closed-form -- match sorted quantiles -- so ``w2`` is the default objective.
  The entropic (debiased) **Sinkhorn divergence** of the paper's Sinkhorn/MOT
  program is also available and converges to W2^2 as epsilon -> 0.
* **Regularisation**: the map beta -> law is many-to-one (a 1-D marginal
  cannot identify five coefficients). An optional L2 penalty selects the
  minimal-norm correction, in the entropic spirit of the bridge itself; tests
  assert recovery of the LAW (distance, moments), not of the raw betas.

Reachability
------------
The restricted bridge acts on the variance channel only, so the price forward
is preserved by construction (the correction cannot manufacture drift). Targets
should differ in SHAPE -- variance level, tails, x-v feedback skew -- which is
exactly the regime the theory restricts to.

Ported from the Stocks repository (sim/bridge_calibration.py).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from otf.models.sfv import (PathEngine, sample_moments, sinkhorn_divergence,
                            w2_distance)
from otf.optim import nelder_mead

__all__ = ["BridgeFit", "calibrate_bridge"]


@dataclass(slots=True)
class BridgeFit:
    beta: Tuple[float, float, float, float, float, float]
    distance_before: float
    distance_after: float
    moments_target: Tuple[float, float, float, float]
    moments_prior: Tuple[float, float, float, float]
    moments_fit: Tuple[float, float, float, float]
    iterations: int
    converged: bool


def calibrate_bridge(
    target: Sequence[float],
    engine: PathEngine,
    objective: str = "w2",
    l2: float = 0.05,
    scale: float = 20.0,               # optimiser step scale for the betas
    max_iter: int = 300,
    restarts: int = 2,                 # simplex re-runs from the best point
    free: Sequence[int] = (1, 2, 3, 4, 5),   # which betas to optimise
    sinkhorn_eps: Optional[float] = None,
) -> BridgeFit:
    """Fit bridge betas so the engine's terminal law matches ``target``.

    ``objective`` is ``"w2"`` (exact 1-D Wasserstein-2, default) or
    ``"sinkhorn"`` (debiased entropic divergence, the paper's program).
    ``l2`` penalises ||beta||^2 to select the minimal-norm correction among
    the many that reproduce a 1-D marginal. ``free`` restricts the search to a
    subset of coefficients -- e.g. ``(2,)`` calibrates only the b2 diagonal
    (a pure variance-level correction, cleanly identified by a 1-D marginal).
    """
    tgt = list(target)
    if len(tgt) < 16:
        raise ValueError("need at least 16 target observations")
    if objective not in ("w2", "sinkhorn"):
        raise ValueError(f"unknown objective {objective!r}")
    free = tuple(int(i) for i in free)
    if not free or any(i < 1 or i > 5 for i in free) or len(set(free)) != len(free):
        raise ValueError("free must be distinct indices from 1..5 (b0 is inert)")

    def dist(sample: Sequence[float]) -> float:
        if objective == "w2":
            return w2_distance(sample, tgt)
        return sinkhorn_divergence(sample, tgt, eps=sinkhorn_eps)

    prior_sample = engine.terminal_logret((0.0,) * 6)
    d0 = dist(prior_sample)
    # Normalise the penalty by the prior distance so l2 is unitless-ish.
    pen_scale = l2 * max(d0, 1e-9) ** 2

    def to_beta(z: Sequence[float]) -> Tuple[float, ...]:
        b = [0.0] * 6
        for zi, idx in zip(z, free):
            b[idx] = zi * scale
        return tuple(b)

    def obj(z: Sequence[float]) -> float:
        d = dist(engine.terminal_logret(to_beta(z)))
        val = d * d if objective == "w2" else d
        return val + pen_scale * sum(c * c for c in z)

    z_best, f_best, iters, conv = nelder_mead(obj, [0.0] * len(free), step=0.5,
                                              tol_x=1e-4, tol_f=1e-10,
                                              max_iter=max_iter)
    for _ in range(max(0, restarts - 1)):
        z2, f2, it2, c2 = nelder_mead(obj, z_best, step=0.2,
                                      tol_x=1e-4, tol_f=1e-10, max_iter=max_iter)
        iters += it2
        if f2 < f_best:
            z_best, f_best, conv = z2, f2, c2
    beta = to_beta(z_best)
    fit_sample = engine.terminal_logret(beta)
    return BridgeFit(
        beta=beta,
        distance_before=d0,
        distance_after=dist(fit_sample),
        moments_target=sample_moments(tgt),
        moments_prior=sample_moments(prior_sample),
        moments_fit=sample_moments(fit_sample),
        iterations=iters,
        converged=conv,
    )

"""Nested hat-function (linear B-spline) constraint family in forward
log-moneyness k = log(S_T / F_{0,T}) (arch doc §6).

Level n has d_n interior nodes on a uniform grid over [k_min, k_max], with
the dyadic refinement d_{n+1} = 2 d_n + 1: the fine knot grid contains the
coarse one, so every coarse hat is *exactly* a linear combination of fine
hats — span(Psi_n) ⊆ span(Psi_{n+1}) holds by construction, not
approximately. Hats are bounded in [0, 1], compactly supported, measurable,
convergence determining in the limit, and stable under numerical
integration: the full §6.1 requirement list.

Normalization (§6.2): psi_tilde = (psi - E^{Q0}[psi]) / sqrt(Var + eps) on
a fixed prior sample declared before calibration. Gauge removal drops
sample-degenerate directions (hats with no mass in support) and exact
linear dependencies via pivoted QR. All metadata is stored in the level's
NormalizationMap so coefficients stay interpretable across levels.

Unbounded raw calls are deliberately excluded (§6.1): full call prices are
recovered through the projective limit plus first-moment control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otf.ssfv.types import ConstraintLevel, NormalizationMap

__all__ = ["NestedHatFamily", "LambdaPotential"]


@dataclass(frozen=True)
class NestedHatFamily:
    """ConstraintFamily implementation: nested hats on dyadic uniform grids."""

    k_min: float = -1.0
    k_max: float = 1.0
    base_dim: int = 4
    eps: float = 1.0e-12
    # Raw-variance floor under which a hat is declared sample-degenerate
    # and removed as a gauge direction. Also a statistical guard: a hat
    # with (almost) no prior mass has (almost) no dual curvature, so a
    # noisy target moment in that direction drives the multiplier toward
    # the relative-interior boundary (§6.3, diverging multipliers). The
    # principled cure is the paper's Sobolev regularization (M2); until
    # then directions below this floor are ungauged out.
    var_floor: float = 1.0e-3

    def dim_at(self, n: int) -> int:
        d = self.base_dim
        for _ in range(n):
            d = 2 * d + 1
        return d

    def level(self, n: int) -> ConstraintLevel:
        d = self.dim_at(n)
        knots = np.linspace(self.k_min, self.k_max, d + 2)
        return ConstraintLevel(
            n=n, family="nested_hat", knots=knots,
            k_min=self.k_min, k_max=self.k_max, normalization=None,
        )

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, level: ConstraintLevel, k: np.ndarray) -> np.ndarray:
        """Raw hat matrix, shape (n_points, dim_raw); values in [0, 1]."""
        knots = level.knots
        h = knots[1] - knots[0]
        centers = knots[1:-1]  # interior nodes
        return np.clip(1.0 - np.abs(k[:, None] - centers[None, :]) / h, 0.0, 1.0)

    def evaluate_normalized(self, level: ConstraintLevel, k: np.ndarray) -> np.ndarray:
        nm = level.normalization
        if nm is None:
            raise ValueError("level is not normalized; call normalize() first")
        raw = self.evaluate(level, k)
        tilde = (raw - nm.means[None, :]) / nm.stds[None, :]
        return tilde[:, list(nm.kept_indices)]

    # -- normalization and gauge removal (§6.2) --------------------------------

    def normalize(self, level: ConstraintLevel, k_prior_sample: np.ndarray,
                  normalization_seed: int = 0) -> ConstraintLevel:
        raw = self.evaluate(level, k_prior_sample)
        means = raw.mean(axis=0)
        variances = raw.var(axis=0)
        stds = np.sqrt(variances + self.eps)

        # Gauge 1: sample-degenerate hats (no prior mass in support).
        alive = np.flatnonzero(variances > self.var_floor)

        # Gauge 2: exact linear dependencies among standardized columns,
        # detected by pivoted QR on the sample matrix.
        tilde = (raw[:, alive] - means[alive]) / stds[alive]
        kept = alive
        if tilde.shape[1] > 0:
            q, r, piv = _qr_pivot(tilde)
            diag = np.abs(np.diag(r))
            tol = diag.max() * max(tilde.shape) * np.finfo(float).eps if diag.size else 0.0
            rank = int((diag > tol).sum())
            kept = np.sort(alive[piv[:rank]])

        nm = NormalizationMap(
            means=means, stds=stds, eps=self.eps,
            kept_indices=tuple(int(i) for i in kept),
            normalization_seed=normalization_seed,
        )
        return ConstraintLevel(
            n=level.n, family=level.family, knots=level.knots,
            k_min=level.k_min, k_max=level.k_max, normalization=nm,
        )

    # -- targets ---------------------------------------------------------------

    def targets_from_sample(self, level: ConstraintLevel, k_market_sample: np.ndarray) -> np.ndarray:
        """Target moments a_n = E^market[psi_tilde] from a terminal sample
        of the data-generating law (synthetic experiments; real surfaces
        will supply these through the MarketSurface adapter, D7)."""
        return self.evaluate_normalized(level, k_market_sample).mean(axis=0)

    # -- exact embedding (warm starts) ------------------------------------------

    def embed(self, coefficients: np.ndarray, level_from: ConstraintLevel,
              level_to: ConstraintLevel) -> np.ndarray:
        """Coefficients at level n -> level n+m representing the same
        potential up to an additive constant (a gauge direction).

        Exactness rests on the dyadic nesting: a coarse hat equals the fine
        piecewise-linear interpolant of itself, i.e. psi_j = sum_i
        psi_j(t'_i) psi'_i. Normalization means shift Phi by a constant
        only, which does not change the posterior (§16.2 property test).
        """
        nm_f, nm_t = level_from.normalization, level_to.normalization
        if nm_f is None or nm_t is None:
            raise ValueError("both levels must be normalized")
        centers_to = level_to.knots[1:-1]
        # Raw coarse basis evaluated at fine interior nodes: (d_to_raw, d_from_raw)
        E = self.evaluate(level_from, centers_to)
        lam_raw_from = np.zeros(level_from.dim_raw)
        lam_raw_from[list(nm_f.kept_indices)] = coefficients / nm_f.stds[list(nm_f.kept_indices)]
        c_raw_to = E @ lam_raw_from  # fine raw coefficients
        kept_to = list(nm_t.kept_indices)
        dropped = np.setdiff1d(np.arange(level_to.dim_raw), kept_to)
        if dropped.size:
            # Gauge-dropped directions have (near) zero prior-sample
            # variance, so judge the un-representable part by its
            # L^2(Q^0-sample) mass, not by raw coefficients: dropping a
            # component supported where the prior has no mass does not
            # change the law.
            l2_dropped = float(np.sum((c_raw_to[dropped] * nm_t.stds[dropped]) ** 2))
            l2_total = float(np.sum((c_raw_to * nm_t.stds) ** 2)) + 1e-300
            # 1% relative L2 mass: the embedding only seeds a warm start,
            # so mass lost in prior-null regions is immaterial there.
            if l2_dropped > 1e-2 * l2_total:
                raise ValueError(
                    "embedding loses non-negligible L2(Q0) mass to gauge-removed "
                    f"fine directions ({l2_dropped / l2_total:.2e} relative); "
                    "refine the normalization sample or lower var_floor"
                )
        return c_raw_to[kept_to] * nm_t.stds[kept_to]


@dataclass(frozen=True)
class LambdaPotential:
    """CylindricalPotential: Phi_n = lambda^T psi_tilde_n (one maturity).

    Bounded because the family is bounded and lambda is finite.
    """

    family: NestedHatFamily
    level: ConstraintLevel
    lam: np.ndarray

    def terminal_value(self, k: np.ndarray) -> np.ndarray:
        return self.family.evaluate_normalized(self.level, k) @ self.lam

    def sup_norm_bound(self) -> float:
        """Conservative bound: |psi_tilde_j| <= (1 + |mu_j|) / sigma_j."""
        nm = self.level.normalization
        if nm is None:
            raise ValueError("level is not normalized")
        kept = list(nm.kept_indices)
        col_bound = (1.0 + np.abs(nm.means[kept])) / nm.stds[kept]
        return float(np.abs(self.lam) @ col_bound)

    def lipschitz_bound(self) -> float:
        """Global Lipschitz constant of Phi in x — *exact*, not the
        triangle-inequality bound: Phi is piecewise linear with breakpoints
        at the knots, so the maximal slope is attained on a knot interval.
        Bounds |U_x| globally by translation invariance
        (paper prop:tangential-lipschitz)."""
        vals = self._knot_values()
        h_knot = float(self.level.knots[1] - self.level.knots[0])
        return float(np.max(np.abs(np.diff(vals))) / h_knot)

    def sup_norm_exact(self) -> float:
        """Exact sup|Phi|: a piecewise-linear function attains its extrema
        at the breakpoints (plus the constant extrapolation value outside
        the domain, where all hats vanish). Use this — not the
        triangle-inequality sup_norm_bound, which overestimates by orders
        of magnitude for overlapping normalized hats — wherever the bound
        feeds a maximum principle or a trust-region guard."""
        return float(np.max(np.abs(self._knot_values())))

    def _knot_values(self) -> np.ndarray:
        knots = self.level.knots
        h = float(knots[1] - knots[0])
        outside = np.array([knots[0] - h, knots[-1] + h])
        pts = np.concatenate([outside[:1], knots, outside[1:]])
        return self.terminal_value(pts)


def _qr_pivot(a: np.ndarray):
    """Pivoted QR; SciPy if available, otherwise a small Gram-Schmidt
    fallback adequate for the modest level dimensions used here."""
    try:
        from scipy.linalg import qr

        q, r, piv = qr(a, mode="economic", pivoting=True)
        return q, r, piv
    except ImportError:  # pragma: no cover - scipy is in the numerical extra
        n = a.shape[1]
        piv = list(range(n))
        work = a.copy()
        rs = np.zeros((n, n))
        for i in range(n):
            norms = np.linalg.norm(work[:, i:], axis=0)
            j = int(np.argmax(norms)) + i
            work[:, [i, j]] = work[:, [j, i]]
            piv[i], piv[j] = piv[j], piv[i]
            rs[i, i] = np.linalg.norm(work[:, i])
            if rs[i, i] > 0:
                qi = work[:, i] / rs[i, i]
                for jj in range(i + 1, n):
                    rs[i, jj] = qi @ work[:, jj]
                    work[:, jj] -= rs[i, jj] * qi
        return work, rs, np.array(piv)

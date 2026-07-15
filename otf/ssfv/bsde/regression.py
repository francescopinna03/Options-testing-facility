"""Regression Monte-Carlo solver for the martingale-projected quadratic BSDE
(paper eq:ssfv-quadratic-bsde; arch doc §9.1).

Discrete backward scheme on a frozen prior PathBatch, one maturity:

    Y_N = Phi(X_T)
    Z_j = E_j[ Y_{j+1} dW_j ] / dt          (both channels, cross-fitted)
    Y_j = E_j[ Y_{j+1} ] + 1/2 |Z_j^perp|^2 dt

Conditional expectations E_j[.|X_j, V_j] are tensor-polynomial regressions
in the standardized coordinates (x, sqrt(v)) — the square-root variance
coordinate improves behavior near the CIR boundary (§9.1) — with
cross-fitted folds to reduce in-sample bias.

The solution reconstructs the posterior density both ways, giving the
entropy double-entry for free (thm:bsde-semista-verification):

    LR route:  log L = Phi(X_T) - N^S_T - Y_0      (semistatic identity)
    EN route:  log L = N^perp_T - 1/2 <N^perp>_T   (stochastic exponential)

Their pathwise RMS difference is the energy-identity residual; with an
exactly solved discrete BSDE the two coincide identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otf.ssfv.projection.diffusion import DiffusionMartingaleProjector
from otf.ssfv.types import BSDESolution, PathBatch

__all__ = ["RegressionBSDESolver", "RegressionContext"]


def _poly_features(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Total-degree tensor monomials of standardized (x, y=sqrt(v))."""
    sx = x.std()
    sy = y.std()
    xs = (x - x.mean()) / (sx if sx > 1e-12 else 1.0)
    ys = (y - y.mean()) / (sy if sy > 1e-12 else 1.0)
    cols = []
    for total in range(degree + 1):
        for i in range(total + 1):
            cols.append(xs ** (total - i) * ys**i)
    return np.column_stack(cols)


def _clamped_hats(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Clamped linear B-spline basis on arbitrary knots.

    Partition of unity on the whole line with *constant* extrapolation
    beyond the terminal knots — no polynomial tail blowup. Boundary hats
    stay at 1 outside the knot range.
    """
    m = knots.shape[0]
    out = np.empty((x.shape[0], m))
    for i in range(m):
        left = knots[i - 1] if i > 0 else None
        right = knots[i + 1] if i < m - 1 else None
        up = np.ones_like(x) if left is None else np.clip((x - left) / (knots[i] - left), 0.0, 1.0)
        dn = np.ones_like(x) if right is None else np.clip((right - x) / (right - knots[i]), 0.0, 1.0)
        out[:, i] = np.minimum(up, dn)
    return out


def _clamped_hats_dx(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """d/dx of the clamped hat basis (piecewise constant, 0 outside)."""
    m = knots.shape[0]
    out = np.zeros((x.shape[0], m))
    for i in range(m):
        if i > 0:
            left, c = knots[i - 1], knots[i]
            rising = (x > left) & (x < c)
            out[rising, i] = 1.0 / (c - left)
        if i < m - 1:
            c, right = knots[i], knots[i + 1]
            falling = (x >= c) & (x < right)
            out[falling, i] = -1.0 / (right - c)
    return out


def _hat_tensor_features(x: np.ndarray, y: np.ndarray, n_x_knots: int, y_degree: int,
                         with_derivatives: bool = False,
                         extra_knots: np.ndarray | None = None):
    """Hats-in-x (quantile knots) tensor low-order-polynomial-in-y basis.

    Adapted to the value function's structure: the terminal condition is a
    hat combination in x (kinked — a global polynomial in x is badly
    biased), while the y = sqrt(v) dependence is smooth and low-order.

    ``extra_knots`` (e.g. the constraint-family knots) are merged into the
    quantile grid: the terminal condition has kinks exactly there, and a
    quantile-only grid cannot resolve potential features supported in
    low-density state regions.

    With ``with_derivatives`` also returns the analytic dF/dx and dF/dy
    matrices, used by the Picard solver to extract the score fields from
    the fitted value.
    """
    qs = np.linspace(0.005, 0.995, n_x_knots)
    knots = np.unique(np.quantile(x, qs))
    if extra_knots is not None:
        lo, hi = x.min(), x.max()
        inside = extra_knots[(extra_knots > lo) & (extra_knots < hi)]
        knots = np.unique(np.concatenate([knots, inside]))
    if knots.shape[0] < 2:  # degenerate early-step state spread
        knots = np.array([x.min() - 1e-8, x.max() + 1e-8])
    hats = _clamped_hats(x, knots)
    sy = y.std()
    sy = sy if sy > 1e-12 else 1.0
    my = y.mean()
    ys = (y - my) / sy
    y_pows = [ys**p for p in range(y_degree + 1)]
    F = np.concatenate([hats * yp[:, None] for yp in y_pows], axis=1)
    if not with_derivatives:
        return F
    hats_dx = _clamped_hats_dx(x, knots)
    Fx = np.concatenate([hats_dx * yp[:, None] for yp in y_pows], axis=1)
    dy_pows = [np.zeros_like(ys)] + [p * ys ** (p - 1) / sy for p in range(1, y_degree + 1)]
    Fy = np.concatenate([hats * dyp[:, None] for dyp in dy_pows], axis=1)
    return F, Fx, Fy


@dataclass(frozen=True)
class RegressionContext:
    """Per-batch cache: features and per-fold Gram pseudo-inverses.

    Independent of the potential, so the dual optimizer builds it once and
    every D_n(lambda) evaluation reuses it.
    """

    features: list  # [ (n_paths, n_feat) per step j = 0..n_steps-1 ]
    fold_masks: list  # [ boolean (n_paths,) per fold ]
    gram_pinv: list  # [ [ (n_feat, n_feat) per fold ] per step ]
    batch_hash: str


@dataclass(frozen=True)
class RegressionBSDESolver:
    """ProjectedBSDESolver backend: cross-fitted regression Monte Carlo.

    ``state_basis``: "hat_tensor" (default — hats in x on quantile knots,
    tensor low-order polynomial in y = sqrt(v)) or "tensor_poly" (global
    total-degree monomials; smooth-potential diagnostics only).
    """

    state_basis: str = "hat_tensor"
    degree: int = 3  # tensor_poly total degree
    n_x_knots: int = 8  # hat_tensor knots in x
    y_degree: int = 2  # hat_tensor polynomial order in y
    n_folds: int = 2
    projector: DiffusionMartingaleProjector = DiffusionMartingaleProjector()

    def _features(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.state_basis == "hat_tensor":
            return _hat_tensor_features(x, y, self.n_x_knots, self.y_degree)
        if self.state_basis == "tensor_poly":
            return _poly_features(x, y, self.degree)
        raise ValueError(f"unknown state basis {self.state_basis!r}")

    def build_context(self, paths: PathBatch) -> RegressionContext:
        if not paths.has_innovations:
            raise ValueError(
                f"scheme {paths.scheme!r} exposes no pathwise innovations; "
                "BSDE/likelihood require them (PathBatch.d_w is None)"
            )
        n_paths, n_steps = paths.n_paths, paths.n_steps
        idx = np.arange(n_paths)
        fold_masks = [idx % self.n_folds == f for f in range(self.n_folds)]
        features, gram_pinv = [], []
        for j in range(n_steps):
            F = self._features(paths.x[:, j], np.sqrt(np.maximum(paths.v[:, j], 0.0)))
            per_fold = []
            for mask in fold_masks:
                train = ~mask
                A = F[train]
                per_fold.append((np.linalg.pinv(A.T @ A, rcond=1e-10), train))
            features.append(F)
            gram_pinv.append(per_fold)
        return RegressionContext(features, fold_masks, gram_pinv, paths.batch_hash)

    def solve(self, paths: PathBatch, potential, context: RegressionContext | None = None) -> BSDESolution:
        if context is None:
            context = self.build_context(paths)
        if context.batch_hash != paths.batch_hash:
            raise ValueError("regression context was built for a different path batch")

        n_paths, n_steps = paths.n_paths, paths.n_steps
        dt = float(paths.times[1] - paths.times[0])

        # Maximum principle: the driver 1/2 |z^perp|^2 is dominated by the
        # full quadratic driver, whose solution is the Hopf-Cole value
        # log E_t[e^Phi] <= ||Phi||_inf; comparison gives |Y_t| <= ||Phi||_inf.
        # Capping the *fitted conditional mean* at this provable bound is a
        # projection onto the known solution set (it controls polynomial
        # extrapolation in the state tails), not silent truncation: the
        # capped fraction is reported as a solver residual.
        y_bound = float(potential.sup_norm_bound()) + 1e-12

        y = np.empty((n_paths, n_steps + 1))
        z = np.zeros((n_paths, n_steps, 2))
        y[:, -1] = potential.terminal_value(paths.x[:, -1])

        cv_num = cv_den = 0.0
        n_capped = 0
        for j in range(n_steps - 1, -1, -1):
            y_next = y[:, j + 1]
            if j == 0:
                # Deterministic initial state: conditional mean = plain mean.
                ey = np.full(n_paths, y_next.mean())
                resid = y_next - ey
                zs = np.full(n_paths, (resid * paths.d_w[:, 0, 0]).mean() / dt)
                zp = np.full(n_paths, (resid * paths.d_w[:, 0, 1]).mean() / dt)
            else:
                F = context.features[j]
                ey = np.empty(n_paths)
                zs = np.empty(n_paths)
                zp = np.empty(n_paths)
                for f, mask in enumerate(context.fold_masks):
                    gram_inv, train = context.gram_pinv[j][f]
                    At = F[train].T
                    Fm = F[mask]
                    coef_y = gram_inv @ (At @ y_next[train])
                    ey[mask] = Fm @ coef_y
                    # Variance-controlled Z estimator: regress the *centered*
                    # increment (Y - E[Y]) dW / dt, so the regression noise is
                    # proportional to the one-step innovation |Z| sqrt(dt),
                    # not to std(Y)/sqrt(dt). Without this control variate the
                    # quadratic driver amplifies estimator noise into a
                    # backward explosion.
                    resid_train = y_next[train] - F[train] @ coef_y
                    zs[mask] = Fm @ (gram_inv @ (At @ (resid_train * paths.d_w[train, j, 0]))) / dt
                    zp[mask] = Fm @ (gram_inv @ (At @ (resid_train * paths.d_w[train, j, 1]))) / dt
                cv_num += float(((y_next - ey) ** 2).mean())
                cv_den += float(y_next.var()) + 1e-30
            over = np.abs(ey) > y_bound
            if np.any(over):
                n_capped += int(over.sum())
                ey = np.clip(ey, -y_bound, y_bound)
            z[:, j, 0] = zs
            z[:, j, 1] = zp
            y[:, j] = ey + 0.5 * zp**2 * dt

        z_orth, proj_diag = self.projector.project(
            z[:, :, 0], z[:, :, 1], np.maximum(paths.v[:, :-1], 0.0)
        )

        y0 = float(y[:, 0].mean())
        n_s = (z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1)
        n_perp = (z_orth * paths.d_w[:, :, 1]).sum(axis=1)
        energy = 0.5 * (z_orth**2).sum(axis=1) * dt

        log_l_lr = y[:, -1] - n_s - y0  # Phi(X_T) - N^S - Y_0
        log_l_en = n_perp - energy

        # Residuals (arch doc §9.5).
        lse = _log_mean_exp(log_l_lr)
        residuals = {
            "likelihood_normalization": abs(float(lse)),
            "energy_identity_rms": float(np.sqrt(np.mean((log_l_lr - log_l_en) ** 2))),
            "terminal_residual": 0.0,  # Y_N = Phi by construction
            "regression_cv_error": float(cv_num / cv_den) if cv_den > 0 else 0.0,
            "maximum_principle_capped_fraction": n_capped / (n_paths * n_steps),
        }

        return BSDESolution(
            y0=y0, y=y, z=z, z_orth=z_orth, u_jump=None,
            log_density=log_l_lr - lse,  # exactly normalized in-sample
            projector=proj_diag, residuals=residuals,
            y0_sample=y0 + float(lse),
        )


def _log_mean_exp(a: np.ndarray) -> float:
    m = float(a.max())
    return m + float(np.log(np.mean(np.exp(a - m))))

"""Cumulative nested constraint family in forward log-moneyness
k = log(S_T / F_{0,T}) (arch doc §6, review fixes R1-R2).

Columns are bounded piecewise-linear functions of two kinds:

* **hats** — compactly supported, 1 at their node, 0 at the neighbours;
* **capped tail ramps** — 0 on one side of an edge, rising linearly to 1
  over one grid step and constant beyond: bounded tests that distinguish
  left- and right-tail mass, which hats on a fixed interval cannot do.

Levels are *cumulative*: level n+1 contains every level-n column
unchanged and appends (a) midpoint hats halving every current gap,
(b) one outward-extension hat per side (the observed support K_n grows
without bound), and (c) a fresh ramp pair at the new edges. Nesting is
therefore an identity of column lists — span(Psi_n) ⊆ span(Psi_{n+1})
needs no argument — and the *limit* family separates tails and is dense
on every compact, hence convergence determining on R as n -> infinity.
No fixed finite level is convergence determining; the identification
of the projective limit with the full-marginal problem uses the
cumulative limit, and the docstring of a fixed level must never claim
more (review fix R1).

Normalization (§6.2) builds a *nested plan*: inherited columns keep
their transform verbatim; columns new at this level are standardized,
then residualized (modified Gram-Schmidt on the pilot prior sample)
against everything already accepted, and rescaled to unit sample
variance. The resulting transform matrix is block-triangular in
creation order, so C_{n+1} ⊆ C_n is structural and upward embedding of
multipliers is zero-padding, exact to machine precision (review fix
R2). New columns whose raw or residual variance falls below
``var_floor`` are rejected and recorded — the statistical gauge (§6.3,
relative-interior guard) acts on new directions only; inherited
directions can never be removed.

**Triangular-scheme caveat (D11.8).** The variance gauge is sample
selection: on a *fixed* pilot batch of N paths, columns far enough in
the tails or narrow enough eventually all fall below ``var_floor``, so
the ACTIVE family stabilizes in a finite space and is not convergence
determining, even though the raw cumulative family is. What the code
realizes at fixed N is therefore one row Psi_{n(N),N} of a triangular
scheme; the theorem's full limit needs the double limit N -> infinity,
n(N) -> infinity, var_floor_N -> 0 (or, equivalently, replacing hard
rejection with a regularization that never permanently removes
directions — the paper's Sobolev route, M2). Rejections being recorded
per level is what makes the realized row of the scheme auditable.

Unbounded raw calls remain deliberately excluded (§6.1): full call
prices are recovered through the projective limit plus first-moment
control.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from otf.ssfv.types import ConstraintLevel, NormalizationMap

__all__ = ["NestedHatFamily", "LambdaPotential"]

_HAT, _RAMP_L, _RAMP_R = 0, 1, 2


@dataclass(frozen=True)
class NestedHatFamily:
    """ConstraintFamily implementation: cumulative hats + tail ramps."""

    k_min: float = -1.0
    k_max: float = 1.0
    base_dim: int = 4
    eps: float = 1.0e-12
    # Variance floor under which a *new* column is rejected (raw variance:
    # no prior mass in support; residual variance: statistically
    # indistinguishable from the span of the accepted columns). Both are
    # relative-interior guards (§6.3): a direction with (almost) no dual
    # curvature drives its multiplier to the boundary. The principled cure
    # is the paper's Sobolev regularization (M2); until then rejected
    # directions are recorded in the NormalizationMap, never silent.
    var_floor: float = 1.0e-3

    # -- column construction ----------------------------------------------------

    def _step(self) -> float:
        return (self.k_max - self.k_min) / (self.base_dim + 1)

    def _columns(self, n: int):
        """Cumulative column list (kind, loc, wl, wr) and final knot grid."""
        h0 = self._step()
        grid = list(np.linspace(self.k_min, self.k_max, self.base_dim + 2))
        cols: list[tuple[int, float, float, float]] = []
        cols.append((_RAMP_L, self.k_min, h0, 0.0))
        for g in grid:
            cols.append((_HAT, g, h0, h0))
        cols.append((_RAMP_R, self.k_max, h0, 0.0))
        for _ in range(n):
            mids = [(grid[i] + grid[i + 1]) / 2.0 for i in range(len(grid) - 1)]
            for i, m in enumerate(mids):
                half = (grid[i + 1] - grid[i]) / 2.0
                cols.append((_HAT, m, half, half))
            a, b = grid[0] - h0, grid[-1] + h0
            cols.append((_HAT, a, h0, h0))
            cols.append((_HAT, b, h0, h0))
            cols.append((_RAMP_L, a, h0, 0.0))
            cols.append((_RAMP_R, b, h0, 0.0))
            grid = sorted(grid + mids + [a, b])
        return cols, grid

    def dim_at(self, n: int) -> int:
        return len(self._columns(n)[0])

    def level(self, n: int) -> ConstraintLevel:
        cols, _ = self._columns(n)
        kind = np.array([c[0] for c in cols], dtype=int)
        loc = np.array([c[1] for c in cols])
        wl = np.array([c[2] for c in cols])
        wr = np.array([c[3] for c in cols])
        # Breakpoints: hat feet and peak; ramp foot and saturation point.
        bp = [loc]
        bp.append(np.where(kind == _RAMP_R, loc + wl, loc - wl))
        bp.append(np.where(kind == _HAT, loc + wr, loc))
        knots = np.unique(np.concatenate(bp))
        return ConstraintLevel(
            n=n, family="cumulative_hat_ramp", knots=knots,
            col_kind=kind, col_loc=loc, col_wl=wl, col_wr=wr,
            k_min=self.k_min, k_max=self.k_max, normalization=None,
        )

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, level: ConstraintLevel, k: np.ndarray) -> np.ndarray:
        """Raw column matrix, shape (n_points, dim_raw); values in [0, 1]."""
        k = np.asarray(k, dtype=float)
        kind, loc = level.col_kind, level.col_loc
        wl, wr = level.col_wl, level.col_wr
        K = k[:, None]
        out = np.empty((k.shape[0], kind.shape[0]))
        hat = kind == _HAT
        if hat.any():
            rise = (K - (loc[hat] - wl[hat])) / wl[hat]
            fall = ((loc[hat] + wr[hat]) - K) / wr[hat]
            out[:, hat] = np.clip(np.minimum(rise, fall), 0.0, 1.0)
        lr = kind == _RAMP_L
        if lr.any():
            out[:, lr] = np.clip((loc[lr] - K) / wl[lr], 0.0, 1.0)
        rr = kind == _RAMP_R
        if rr.any():
            out[:, rr] = np.clip((K - loc[rr]) / wl[rr], 0.0, 1.0)
        return out

    def evaluate_normalized(self, level: ConstraintLevel, k: np.ndarray) -> np.ndarray:
        nm = level.normalization
        if nm is None:
            raise ValueError("level is not normalized; call normalize() first")
        raw = self.evaluate(level, k)
        return (raw - nm.means[None, :]) @ nm.transform

    # -- nested normalization plan (§6.2, review fix R2) -------------------------

    def normalize(self, level: ConstraintLevel, k_prior_sample: np.ndarray,
                  normalization_seed: int = 0,
                  previous: ConstraintLevel | None = None) -> ConstraintLevel:
        """Build the level's normalization on the pilot prior sample.

        With ``previous`` (a normalized coarser level of this family),
        its transform is inherited verbatim and only the new columns are
        processed. Without it the whole plan is rebuilt from scratch in
        creation order — on the *same* pilot sample the two routes give
        identical transforms, so independently normalized levels still
        nest exactly.
        """
        raw = self.evaluate(level, k_prior_sample)
        n_s, d_raw = raw.shape
        means = raw.mean(axis=0)
        variances = raw.var(axis=0)
        stds = np.sqrt(variances + self.eps)
        centered = raw - means

        kept: list[int] = []
        rejected: list[int] = []
        w_cols: list[np.ndarray] = []
        t_cols: list[np.ndarray] = []
        inherited_dim = 0
        start = 0
        if previous is not None:
            nm_p = previous.normalization
            if nm_p is None:
                raise ValueError("previous level is not normalized")
            d_prev = previous.dim_raw
            if d_prev > d_raw or not (
                np.array_equal(previous.col_kind, level.col_kind[:d_prev])
                and np.allclose(previous.col_loc, level.col_loc[:d_prev])
            ):
                raise ValueError("previous level is not a column-list prefix of this level")
            for col in nm_p.transform.T:
                w = np.zeros(d_raw)
                w[:d_prev] = col
                w_cols.append(w)
                t_cols.append(centered @ w)
            kept = list(nm_p.kept_indices)
            rejected = list(nm_p.rejected_indices)
            inherited_dim = len(kept)
            start = d_prev

        for j in range(start, d_raw):
            if variances[j] <= self.var_floor:
                rejected.append(j)
                continue
            w = np.zeros(d_raw)
            w[j] = 1.0 / stds[j]
            c = centered[:, j] / stds[j]
            # Modified Gram-Schmidt against every accepted column: the
            # transform stays block-triangular in creation order, which is
            # exactly the nesting guarantee.
            for t, wv in zip(t_cols, w_cols):
                b = float(c @ t) / float(t @ t)
                c = c - b * t
                w = w - b * wv
            s2 = float(c @ c) / n_s
            if s2 <= self.var_floor:
                rejected.append(j)
                continue
            s = float(np.sqrt(s2))
            w_cols.append(w / s)
            t_cols.append(c / s)
            kept.append(j)

        transform = np.column_stack(w_cols) if w_cols else np.zeros((d_raw, 0))
        nm = NormalizationMap(
            means=means, stds=stds, transform=transform,
            kept_indices=tuple(int(i) for i in kept),
            rejected_indices=tuple(int(i) for i in rejected),
            inherited_dim=inherited_dim, eps=self.eps,
            normalization_seed=normalization_seed,
        )
        return replace(level, normalization=nm)

    # -- targets ---------------------------------------------------------------

    def targets_from_sample(self, level: ConstraintLevel, k_market_sample: np.ndarray) -> np.ndarray:
        """Target moments a_n = E^market[psi_tilde] from a terminal sample
        of the data-generating law (synthetic experiments; real surfaces
        will supply these through the MarketSurface adapter, D7)."""
        return self.evaluate_normalized(level, k_market_sample).mean(axis=0)

    # -- exact embedding --------------------------------------------------------

    def embed(self, coefficients: np.ndarray, level_from: ConstraintLevel,
              level_to: ConstraintLevel) -> np.ndarray:
        """Coefficients at a coarse level -> the same potential at a finer
        level, up to an additive constant (a gauge direction).

        Because the fine level's leading normalized columns ARE the
        coarse level's columns, the embedding is zero-padding. The prefix
        property is *verified* at machine precision (1e-12), never
        assumed: a mismatch means the two levels were normalized on
        different pilot samples, which is a hard error, not something to
        approximate through (review fix R2)."""
        nm_f, nm_t = level_from.normalization, level_to.normalization
        if nm_f is None or nm_t is None:
            raise ValueError("both levels must be normalized")
        coefficients = np.asarray(coefficients, dtype=float)
        d_f = len(nm_f.kept_indices)
        if coefficients.shape != (d_f,):
            raise ValueError(f"coefficients shape {coefficients.shape} != ({d_f},)")
        if nm_t.kept_indices[:d_f] != nm_f.kept_indices:
            raise ValueError(
                "fine level's accepted columns do not extend the coarse level's: "
                "levels were normalized on different pilot samples"
            )
        d_raw_f = level_from.dim_raw
        block = nm_t.transform[:d_raw_f, :d_f]
        tail = nm_t.transform[d_raw_f:, :d_f]
        err = max(
            float(np.max(np.abs(block - nm_f.transform))) if block.size else 0.0,
            float(np.max(np.abs(tail))) if tail.size else 0.0,
        )
        if err > 1.0e-12:
            raise ValueError(
                f"nested-normalization prefix violated (max deviation {err:.3e} "
                "> 1e-12): levels were normalized on different pilot samples"
            )
        out = np.zeros(len(nm_t.kept_indices))
        out[:d_f] = coefficients
        return out


@dataclass(frozen=True)
class LambdaPotential:
    """CylindricalPotential: Phi_n = lambda^T psi_tilde_n (one maturity).

    Bounded because the family is bounded and lambda is finite. Phi is
    piecewise linear with breakpoints at ``level.knots`` and *constant*
    outside them (hats vanish, ramps saturate), so its sup and Lipschitz
    constants are exact finite maxima, not bounds.
    """

    family: NestedHatFamily
    level: ConstraintLevel
    lam: np.ndarray

    def terminal_value(self, k: np.ndarray) -> np.ndarray:
        return self.family.evaluate_normalized(self.level, k) @ self.lam

    def _raw_coefficients(self) -> np.ndarray:
        nm = self.level.normalization
        if nm is None:
            raise ValueError("level is not normalized")
        return nm.transform @ np.asarray(self.lam, dtype=float)

    def sup_norm_bound(self) -> float:
        """Conservative triangle bound: raw columns take values in [0, 1],
        so |psi_j - mu_j| <= max(mu_j, 1 - mu_j)."""
        nm = self.level.normalization
        c = self._raw_coefficients()
        return float(np.abs(c) @ np.maximum(nm.means, 1.0 - nm.means))

    def _breakpoint_values(self) -> tuple[np.ndarray, np.ndarray]:
        knots = np.asarray(self.level.knots, dtype=float)
        pts = np.concatenate([[knots[0] - 1.0], knots, [knots[-1] + 1.0]])
        return pts, self.terminal_value(pts)

    def sup_norm_exact(self) -> float:
        """Exact sup|Phi|: a piecewise-linear function attains its extrema
        at its breakpoints (plus the constant values outside). Use this —
        not the triangle-inequality sup_norm_bound, which overestimates by
        orders of magnitude for overlapping normalized columns — wherever
        the value feeds a maximum principle or a trust-region guard."""
        return float(np.max(np.abs(self._breakpoint_values()[1])))

    def lipschitz_bound(self) -> float:
        """Global Lipschitz constant of Phi in x — exact: the maximal
        slope is attained on a breakpoint interval. Bounds |U_x| globally
        by translation invariance (paper prop:tangential-lipschitz)."""
        pts, vals = self._breakpoint_values()
        return float(np.max(np.abs(np.diff(vals) / np.diff(pts))))

"""Calibrate the bridge betas to a REAL option surface (SFV_M2 Level 2).

The two-layer program, run in order:

1. the **affine prior** (Heston block) is fitted to the surface first with
   the characteristic-function calibrator (:mod:`otf.calibration.heston_fit`)
   -- the prior stays the computational backbone;
2. the **bridge layer** deforms it minimally: the betas of the restricted,
   standardized, gated variance correction are fitted to what the prior
   could NOT price, under the full SFV_M2 calibration objective

       L(beta) = w_iv  * mean_j ((iv_j(beta)  - iv_j^mkt) / s_sigma)^2
               + w_px  * mean_j ((C_j(beta)   - C_j^mkt) / (vega_j + eps))^2
               + lam_mart   * ((E[e^{X_T - rT}] - 1))^2
               + lam_energy * E[int nu^2 / (xi^2 v + eps) dt]
               + lam_beta   * ||beta||^2
               [+ lam_sk    * Sinkhorn(model X_T, implied X_T)]

   with model prices from the CRN Monte Carlo pricer (no CF exists once
   beta != 0). Frozen shocks make L deterministic in beta, so the same
   pure-stdlib Nelder-Mead used everywhere else applies.

The price term is vega-weighted (a price error is worth its vol-point
equivalent, so deep options do not dominate), the martingale and energy
penalties keep the correction risk-neutral-consistent and entropically
minimal, and the L2 term selects the smallest beta among observationally
equivalent ones. One engine per expiry, all sharing the standardization and
gate constants that will be reported next to the betas.

Ported from the Stocks repository (sim/surface_calibration.py).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from otf.calibration.heston_fit import MarketQuote
from otf.models.black_scholes import bs_greeks, bs_price, implied_vol
from otf.models.sfv import (BridgeDiagnostics, PathEngine,
                            sinkhorn_divergence, standardization_for)
from otf.optim import nelder_mead

__all__ = ["SurfaceFit", "calibrate_bridge_to_surface", "surface_rmse",
           "implied_terminal_sample"]


def implied_terminal_sample(quotes: Sequence[MarketQuote], spot: float,
                            tte: float, r: float = 0.0,
                            n: int = 512) -> List[float]:
    """Market-implied sample of X_T = log(S_T/spot) at one expiry, by the
    discrete Breeden-Litzenberger construction (SFV_M2 'Market-implied
    terminal constraints').

    Puts are mapped to calls by parity, one call price per strike, the
    density is the second divided difference of the call curve (butterfly),
    negative butterflies -- bid/ask noise, our MC error -- are clamped to
    zero, and the atoms x_i = log(K_i/spot) are replicated proportionally to
    their mass (largest remainder) into a size-``n`` sample ready for the
    Sinkhorn divergence."""
    disc = math.exp(-r * tte)
    calls: Dict[float, float] = {}
    for q in quotes:
        px = bs_price(spot, q.strike, tte, q.iv, r, q.is_call)
        c = px if q.is_call else px + spot - q.strike * disc   # parity
        calls[q.strike] = c if q.strike not in calls else 0.5 * (calls[q.strike] + c)
    ks = sorted(calls)
    if len(ks) < 5:
        raise ValueError(f"need >= 5 strikes at one expiry, got {len(ks)}")
    masses, atoms = [], []
    for i in range(1, len(ks) - 1):
        k0, k1, k2 = ks[i - 1], ks[i], ks[i + 1]
        left = (calls[k0] - calls[k1]) / (k1 - k0)
        right = (calls[k1] - calls[k2]) / (k2 - k1)
        dens = 2.0 * (left - right) / (k2 - k0) / disc         # f_{S_T}(k1)
        mass = max(dens, 0.0) * 0.5 * (k2 - k0)
        if mass > 0.0:
            masses.append(mass)
            atoms.append(math.log(k1 / spot))
    tot = sum(masses)
    if tot <= 0.0:
        raise ValueError("implied density degenerate (all butterflies <= 0)")
    counts = [int(n * m / tot) for m in masses]
    rem = sorted(range(len(masses)),
                 key=lambda i: -(n * masses[i] / tot - counts[i]))
    for i in rem[:n - sum(counts)]:
        counts[i] += 1
    out: List[float] = []
    for a, c in zip(atoms, counts):
        out.extend([a] * c)
    return out


@dataclass(slots=True)
class SurfaceFit:
    beta: Tuple[float, float, float, float, float, float]
    loss_before: float
    loss_after: float
    iv_rmse_before: float           # vol points, prior vs market
    iv_rmse_after: float            # vol points, bridged vs market
    terms: Dict[str, float]         # loss decomposition at the optimum
    diagnostics: BridgeDiagnostics  # longest-expiry engine at the optimum
    std: Dict[str, float]           # standardization constants used
    gate: Tuple[float, float]       # (m, c)
    iterations: int
    converged: bool


def _model_rows(engines, groups, beta, spot, r):
    """Per quote: (quote, model_iv, model_px, market greeks)."""
    rows = []
    for tte, quotes in groups:
        eng = engines[tte]
        terminal = eng.terminal_logret(beta)
        disc = math.exp(-r * tte)
        for q in quotes:
            pays = 0.0
            if q.is_call:
                for x in terminal:
                    st = spot * math.exp(x)
                    if st > q.strike:
                        pays += st - q.strike
            else:
                for x in terminal:
                    st = spot * math.exp(x)
                    if st < q.strike:
                        pays += q.strike - st
            px = disc * pays / len(terminal)
            miv = implied_vol(px, spot, q.strike, tte, r, q.is_call)
            g = bs_greeks(spot, q.strike, tte, q.iv, r, q.is_call)
            rows.append((q, miv, px, g))
    return rows


def surface_rmse(rows) -> float:
    """IV RMSE (vol points) over the quotes with a recoverable model IV."""
    errs = [(miv - q.iv) ** 2 for q, miv, _, _ in rows if miv is not None]
    return math.sqrt(sum(errs) / len(errs)) if errs else float("nan")


def calibrate_bridge_to_surface(
    quotes: Sequence[MarketQuote],
    spot: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    lambda_S: float = 0.0, p_up: float = 0.5,
    eta_up: float = 50.0, eta_dn: float = 50.0,
    r: float = 0.0,
    free: Sequence[int] = (2, 5),
    w_iv: float = 1.0,
    w_px: float = 1.0,
    lambda_mart: float = 100.0,
    lambda_energy: float = 0.05,
    lambda_beta: float = 0.01,
    lambda_sk: float = 0.0,
    gate: Tuple[float, float] = (4.0, 1.0),
    n_paths: int = 800,
    seed: int = 99,
    max_iter: int = 250,
    restarts: int = 1,
) -> SurfaceFit:
    """Fit the bridge betas to the surface under the SFV_M2 Level-2 objective.

    The Heston/Kou arguments are the ALREADY-FITTED affine prior; the bridge
    corrects its residuals. ``free`` defaults to (2, 5): the diagonal variance
    terms a 1-D-per-expiry surface can actually identify.
    """
    qs = [q for q in quotes if q.tte > 1e-3 and q.iv > 1e-3]
    if len(qs) < 4:
        raise ValueError(f"need >= 4 usable quotes, got {len(qs)}")
    free = tuple(int(i) for i in free)
    if not free or any(i < 1 or i > 5 for i in free):
        raise ValueError("free must be indices from 1..5 (b0 is inert)")

    by_tte: Dict[float, List[MarketQuote]] = {}
    for q in qs:
        by_tte.setdefault(round(q.tte, 6), []).append(q)
    groups = sorted(by_tte.items())
    t_max = groups[-1][0]

    std = standardization_for(v0, kappa, theta, xi, t_max)
    gate_m, gate_c = float(gate[0]), float(gate[1])
    engines: Dict[float, PathEngine] = {}
    for tte, _ in groups:
        engines[tte] = PathEngine(
            n_paths=n_paths, n_steps=max(40, min(120, int(tte * 500))),
            horizon=tte, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
            r=r, lambda_S=lambda_S, p_up=p_up, eta_up=eta_up, eta_dn=eta_dn,
            seed=seed, sx=std["sx"], v_ref=std["v_ref"], sv=std["sv"],
            gate_m=gate_m, gate_c=gate_c)

    n_q = len(qs)
    eng_long = engines[t_max]

    # Optional Stage-5 marginal Sinkhorn on X_T: target from the densest
    # expiry with enough strikes for a Breeden-Litzenberger density.
    sk_target, sk_tte = None, None
    if lambda_sk > 0.0:
        for tte, group in sorted(groups, key=lambda g: -len(g[1])):
            try:
                sk_target = implied_terminal_sample(group, spot, tte, r, n=256)
                sk_tte = tte
                break
            except ValueError:
                continue
        if sk_target is None:
            raise ValueError("lambda_sk > 0 but no expiry has >= 5 strikes "
                             "for the implied terminal density")

    def loss_terms(beta) -> Dict[str, float]:
        rows = _model_rows(engines, groups, beta, spot, r)
        iv_sq = px_sq = 0.0
        for q, miv, px, g in rows:
            if miv is None:
                iv_sq += q.weight * (10.0 ** 2)   # unrecoverable IV: huge
                continue
            iv_sq += q.weight * ((miv - q.iv) / 0.01) ** 2   # per vol point
            mkt_px = bs_price(spot, q.strike, q.tte, q.iv, r, q.is_call)
            px_sq += q.weight * ((px - mkt_px) / (g.vega * 0.01 + 1e-6)) ** 2
        d = eng_long.diagnostics(beta)
        out = {
            "iv": w_iv * iv_sq / n_q,
            "price": w_px * px_sq / n_q,
            "martingale": lambda_mart * d.martingale_error ** 2,
            "energy": lambda_energy * d.control_energy,
            "l2": lambda_beta * sum(b * b for b in beta),
        }
        if sk_target is not None:
            model = engines[sk_tte].terminal_logret(beta)
            out["sinkhorn"] = lambda_sk * sinkhorn_divergence(
                model, sk_target, atoms=64, iters=40)
        return out

    def objective(vec) -> float:
        beta = [0.0] * 6
        for i, j in enumerate(free):
            beta[j] = vec[i]
        return sum(loss_terms(beta).values())

    beta0 = (0.0,) * 6
    rows0 = _model_rows(engines, groups, beta0, spot, r)
    rmse0 = surface_rmse(rows0)
    loss0 = sum(loss_terms(beta0).values())

    x = [0.0] * len(free)
    best, fbest, it_total, conv = x, loss0, 0, False
    for _ in range(max(1, restarts + 1)):
        x_opt, f_opt, its, ok = nelder_mead(objective, best, step=1.0,
                                            max_iter=max_iter)
        it_total += its
        if f_opt < fbest:
            best, fbest, conv = x_opt, f_opt, ok

    beta = [0.0] * 6
    for i, j in enumerate(free):
        beta[j] = best[i]
    beta_t = tuple(beta)
    rows1 = _model_rows(engines, groups, beta_t, spot, r)
    return SurfaceFit(
        beta=beta_t, loss_before=loss0, loss_after=fbest,
        iv_rmse_before=rmse0, iv_rmse_after=surface_rmse(rows1),
        terms=loss_terms(beta_t), diagnostics=eng_long.diagnostics(beta_t),
        std=std, gate=(gate_m, gate_c), iterations=it_total, converged=conv)

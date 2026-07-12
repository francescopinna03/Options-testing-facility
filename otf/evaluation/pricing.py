"""Score a frozen model against a set of market quotes (IV RMSE, vol points).

One function per model family, all with the same contract: take a cleaned
quote list (``otf.calibration.MarketQuote``) and a FROZEN model fitted on some
other day, return the RMSE between model and market implied vols. This is the
common yardstick of the out-of-sample study: every arm -- from the flat-vol
straw man to the bridged SFV -- is judged in the same units on the same quotes.

Convention inherited from the Stocks study: a quote whose model price cannot
be inverted to a sane implied vol (outside (1e-3, 4.9)) scores a flat
10-vol-point penalty rather than poisoning the average with the bisection
bracket edge.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple

from otf.calibration.heston_fit import HestonFit, MarketQuote
from otf.models.black_scholes import implied_vol
from otf.models.heston import heston_price
from otf.models.sfv import PathEngine
from otf.models.svi import SVISurface

__all__ = ["fit_flat_vol", "iv_rmse_flat", "iv_rmse_svi", "iv_rmse_heston",
           "iv_rmse_sfv"]

_PENALTY = 0.10          # vol points charged for an unrecoverable model IV
_IV_BAND = (1e-3, 4.9)


def _finish(se: float, n: float) -> Optional[float]:
    return math.sqrt(se / n) if n else None


def fit_flat_vol(quotes: Sequence[MarketQuote]) -> float:
    """Weight-averaged market IV: the single-parameter Black-Scholes 'fit'."""
    tot_w = sum(max(q.weight, 0.0) for q in quotes) or 1.0
    return sum(max(q.weight, 0.0) * q.iv for q in quotes) / tot_w


def iv_rmse_flat(quotes: Sequence[MarketQuote], sigma: float) -> Optional[float]:
    """RMSE of a frozen flat vol against the quotes' market IVs."""
    se = n = 0.0
    for q in quotes:
        se += (sigma - q.iv) ** 2
        n += 1
    return _finish(se, n)


def iv_rmse_svi(quotes: Sequence[MarketQuote], surface: SVISurface,
                spot: Optional[float] = None) -> Optional[float]:
    """RMSE of a frozen SVI surface, read sticky-moneyness at ``spot``
    (the evaluation day's underlying close; defaults to the fitting spot)."""
    se = n = 0.0
    for q in quotes:
        iv = surface.iv(q.strike, q.tte, spot=spot)
        err = (iv - q.iv) if _IV_BAND[0] < iv < _IV_BAND[1] else _PENALTY
        se += err * err
        n += 1
    return _finish(se, n)


def iv_rmse_heston(quotes: Sequence[MarketQuote], spot: float, fit: HestonFit,
                   r: float = 0.0, n_int: int = 128) -> Optional[float]:
    """RMSE of frozen Heston parameters priced by characteristic function."""
    se = n = 0.0
    for q in quotes:
        px = heston_price(spot, q.strike, q.tte, fit.v0, fit.kappa, fit.theta,
                          fit.xi, fit.rho, r, q.is_call, n=n_int)
        iv = implied_vol(px, spot, q.strike, q.tte, r, q.is_call)
        err = (iv - q.iv) if _IV_BAND[0] < iv < _IV_BAND[1] else _PENALTY
        se += err * err
        n += 1
    return _finish(se, n)


def iv_rmse_sfv(quotes: Sequence[MarketQuote], spot: float, prior: HestonFit,
                beta: Sequence[float], std: Dict[str, float],
                gate: Tuple[float, float],
                jump: Tuple[float, float, float, float] = (0.0, 0.5, 50.0, 50.0),
                r: float = 0.0, n_paths: int = 400,
                seed: int = 99) -> Optional[float]:
    """RMSE of the frozen SFV law (prior when ``beta`` is all zeros, bridged
    otherwise), MC-priced with one CRN engine per expiry -- so prior and
    bridge arms scored with the same seed are PAIRED on identical shocks."""
    by_tte: Dict[float, List[MarketQuote]] = {}
    for q in quotes:
        by_tte.setdefault(round(q.tte, 6), []).append(q)
    se = n = 0.0
    lam, p_up, eta_up, eta_dn = jump
    for tte, group in sorted(by_tte.items()):
        eng = PathEngine(
            n_paths=n_paths, n_steps=max(40, min(120, int(tte * 500))),
            horizon=tte, v0=prior.v0, kappa=prior.kappa, theta=prior.theta,
            xi=prior.xi, rho=prior.rho, r=r, lambda_S=lam, p_up=p_up,
            eta_up=eta_up, eta_dn=eta_dn, seed=seed,
            sx=std["sx"], v_ref=std["v_ref"], sv=std["sv"],
            gate_m=gate[0], gate_c=gate[1])
        disc = math.exp(-r * tte)
        terminal = eng.terminal_logret(beta)
        for q in group:
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
            iv = implied_vol(px, spot, q.strike, tte, r, q.is_call)
            err = (iv - q.iv) if _IV_BAND[0] < iv < _IV_BAND[1] else _PENALTY
            se += err * err
            n += 1
    return _finish(se, n)

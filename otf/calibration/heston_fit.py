"""Calibrate the Heston (SFV affine-prior) parameters to a market vol surface.

The SFV model's computational backbone is the five-parameter Heston block
``(v0, kappa, theta, xi, rho)``. This module runs the inverse problem: given a
set of market option quotes (an implied-vol smile, or a whole surface across
strikes *and* maturities), find the parameters whose characteristic-function
Heston prices reproduce the observed implied volatilities.

The fit is done in **implied-vol space** (not price space) so every strike and
maturity carries comparable weight regardless of its premium. For each quote we

    1. price the option with :func:`otf.models.heston.heston_price` at the
       trial parameters,
    2. invert that price back to a Black-Scholes implied vol,
    3. accumulate the squared gap to the market implied vol.

Positivity of ``v0, kappa, theta, xi`` and the ``rho in (-1, 1)`` constraint
are enforced by optimising in an unconstrained transformed space (``log`` for
the positive parameters, ``atanh`` for the correlation), so the simplex can
never propose an invalid model.

Ported from the Stocks repository (core/calibration.py).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from otf.models.black_scholes import implied_vol
from otf.models.heston import heston_price
from otf.optim import nelder_mead

__all__ = ["MarketQuote", "HestonFit", "CalibrationResult", "calibrate_heston"]


# --------------------------------------------------------------------------- #
# Bounded <-> unconstrained parameter transforms.                             #
#   v0, kappa, theta, xi  > 0        via  log / exp                           #
#   rho in (-1, 1)                   via  atanh / tanh                        #
# --------------------------------------------------------------------------- #
def _encode(v0: float, kappa: float, theta: float, xi: float, rho: float) -> List[float]:
    r = max(-0.999999, min(0.999999, rho))
    return [math.log(v0), math.log(kappa), math.log(theta), math.log(xi), math.atanh(r)]


def _decode(z: Sequence[float]) -> Tuple[float, float, float, float, float]:
    return (math.exp(z[0]), math.exp(z[1]), math.exp(z[2]), math.exp(z[3]), math.tanh(z[4]))


@dataclass(slots=True)
class MarketQuote:
    """One market observation on the vol surface.

    ``iv`` is the market Black-Scholes implied volatility (annualised) for a
    European option of strike ``strike`` and time-to-expiry ``tte`` (years).
    ``weight`` lets you emphasise liquid at-the-money strikes over the wings.
    """
    strike: float
    tte: float
    iv: float
    is_call: bool = True
    weight: float = 1.0


@dataclass(slots=True)
class HestonFit:
    """Calibrated risk-neutral Heston parameters."""
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.v0, self.kappa, self.theta, self.xi, self.rho)

    def feller_ok(self) -> bool:
        """Feller condition ``2 kappa theta >= xi^2`` (variance stays > 0)."""
        return 2.0 * self.kappa * self.theta >= self.xi * self.xi


@dataclass(slots=True)
class CalibrationResult:
    params: HestonFit
    rmse: float                 # root-mean-square implied-vol error (vol points)
    model_ivs: List[float]      # model implied vol per input quote, same order
    iterations: int
    converged: bool


def _model_iv(z: Sequence[float], S: float, r: float, q: MarketQuote, n: int) -> float:
    v0, kappa, theta, xi, rho = _decode(z)
    px = heston_price(S, q.strike, q.tte, v0, kappa, theta, xi, rho, r, q.is_call, n=n)
    return implied_vol(px, S, q.strike, q.tte, r, q.is_call)


def calibrate_heston(
    quotes: Sequence[MarketQuote],
    S: float,
    r: float = 0.0,
    x0: Optional[HestonFit] = None,
    feller_weight: float = 0.0,
    kappa_max: float = 0.0,
    n: int = 128,
    max_iter: int = 4000,
    restarts: int = 1,
) -> CalibrationResult:
    """Fit Heston ``(v0, kappa, theta, xi, rho)`` to a market vol smile/surface.

    Minimises the weighted RMSE between model and market implied vols. ``x0`` is
    the starting guess (a sensible default is derived from the quotes when
    omitted). ``feller_weight`` adds a soft penalty for violating the Feller
    condition -- keep it at 0 for a pure best-fit, raise it to steer the fit
    toward parameters whose variance process stays strictly positive; a CF-only
    prior that violates Feller hard prices the surface but cannot be SIMULATED
    faithfully, so the bridge would then fit discretization error.
    ``kappa_max`` (0 = off) softly caps the mean-reversion speed: a CF fit on
    a jumpy surface can mimic jumps with extreme ``kappa``/``xi`` pairs that
    price well but describe an implausible variance process. ``n`` is
    the Heston integration resolution; ``restarts`` re-runs the simplex from the
    best point found (a cheap way to polish a Nelder-Mead result).
    """
    if not quotes:
        raise ValueError("need at least one market quote to calibrate")
    if S <= 0:
        raise ValueError("spot S must be positive")

    if x0 is None:
        # ATM-ish market vol as the variance anchor; mild mean reversion; a
        # downward-skew default (rho<0) that matches typical equity smiles.
        atm = min(quotes, key=lambda q: abs(q.strike - S)).iv
        var = max(atm * atm, 1e-4)
        x0 = HestonFit(v0=var, kappa=2.0, theta=var, xi=0.4, rho=-0.3)

    tot_w = sum(max(q.weight, 0.0) for q in quotes) or 1.0

    def objective(z: Sequence[float]) -> float:
        v0, kappa, theta, xi, rho = _decode(z)
        se = 0.0
        for q in quotes:
            iv = _model_iv(z, S, r, q, n)
            d = iv - q.iv
            se += max(q.weight, 0.0) * d * d
        mse = se / tot_w
        if feller_weight > 0.0:
            short = xi * xi - 2.0 * kappa * theta      # > 0 means violation
            if short > 0.0:
                mse += feller_weight * short * short
        if kappa_max > 0.0 and kappa > kappa_max:
            mse += ((kappa - kappa_max) / kappa_max) ** 2
        return mse

    z = _encode(*x0.as_tuple())
    best_z, best_f, iters, converged = nelder_mead(objective, z, max_iter=max_iter)
    for _ in range(max(0, restarts - 1)):
        z2, f2, it2, conv2 = nelder_mead(objective, best_z, step=0.15, max_iter=max_iter)
        iters += it2
        if f2 < best_f:
            best_z, best_f, converged = z2, f2, conv2

    v0, kappa, theta, xi, rho = _decode(best_z)
    fit = HestonFit(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    model_ivs = [_model_iv(best_z, S, r, q, n) for q in quotes]

    # Report the pure (unpenalised) implied-vol RMSE.
    se = sum(max(q.weight, 0.0) * (m - q.iv) ** 2 for m, q in zip(model_ivs, quotes))
    rmse = math.sqrt(se / tot_w)
    return CalibrationResult(params=fit, rmse=rmse, model_ivs=model_ivs,
                             iterations=iters, converged=converged)

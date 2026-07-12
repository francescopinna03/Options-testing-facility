"""Full Heston (1993) option pricing via the characteristic function.

Prices a European option under the Heston stochastic-variance dynamics by
numerically integrating the two probabilities P1, P2 of the characteristic
function. This captures the vol-of-vol *convexity* (the implied-vol smile) that
the average-variance Black-Scholes approximation misses -- most visible on
out-of-the-money strikes.

Uses the Albrecher et al. (2007) "Little Heston Trap" formulation (the
``e^{-dT}`` branch), which is numerically stable across maturities, and Simpson
integration over a truncated frequency range. Pure-stdlib: ``cmath`` + ``math``.

Parameters (risk-neutral): ``v0`` spot variance, ``kappa`` mean-reversion speed,
``theta`` long-run variance, ``xi`` vol-of-vol, ``rho`` leverage correlation,
``r`` rate. As ``xi -> 0`` the price converges to Black-Scholes at the
average-variance vol.

Ported verbatim from the Stocks repository (core/heston.py).
"""

from __future__ import annotations
import cmath
import math

__all__ = ["heston_price"]


def _char(phi: float, j: int, x: float, v0: float, kappa: float, theta: float,
          xi: float, rho: float, r: float, T: float) -> complex:
    u = 0.5 if j == 1 else -0.5
    b = (kappa - rho * xi) if j == 1 else kappa
    xi2 = xi * xi
    rspi = rho * xi * phi * 1j
    d = cmath.sqrt((rspi - b) ** 2 - xi2 * (2.0 * u * phi * 1j - phi * phi))
    # Little Trap: g uses (b - rspi - d)/(b - rspi + d) with e^{-dT}
    num = b - rspi - d
    g = num / (b - rspi + d)
    edt = cmath.exp(-d * T)
    D = (num / xi2) * ((1.0 - edt) / (1.0 - g * edt))
    C = (r * phi * 1j * T
         + (kappa * theta / xi2) * (num * T - 2.0 * cmath.log((1.0 - g * edt) / (1.0 - g))))
    return cmath.exp(C + D * v0 + 1j * phi * x)


def _prob(j: int, S: float, K: float, T: float, v0: float, kappa: float,
          theta: float, xi: float, rho: float, r: float, upper: float,
          n: int) -> float:
    x = math.log(S)
    lnK = math.log(K)

    def integrand(phi: float) -> float:
        cf = _char(phi, j, x, v0, kappa, theta, xi, rho, r, T)
        return (cmath.exp(-1j * phi * lnK) * cf / (1j * phi)).real

    # Simpson over [eps, upper] (integrand has a finite limit at 0; start small).
    eps = 1e-6
    h = (upper - eps) / n
    total = integrand(eps) + integrand(upper)
    for k in range(1, n):
        total += (4.0 if k % 2 else 2.0) * integrand(eps + k * h)
    integral = total * h / 3.0
    return 0.5 + integral / math.pi


def heston_price(S: float, K: float, T: float, v0: float, kappa: float,
                 theta: float, xi: float, rho: float, r: float = 0.0,
                 call: bool = True, upper: float = 120.0, n: int = 512) -> float:
    """European option price under Heston. ``n`` (even) integration points."""
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or v0 <= 0:
        return max((S - K) if call else (K - S), 0.0)
    P1 = _prob(1, S, K, T, v0, kappa, theta, xi, rho, r, upper, n)
    P2 = _prob(2, S, K, T, v0, kappa, theta, xi, rho, r, upper, n)
    call_px = max(S * P1 - K * math.exp(-r * T) * P2, 0.0)
    if call:
        return call_px
    return call_px - S + K * math.exp(-r * T)     # put-call parity

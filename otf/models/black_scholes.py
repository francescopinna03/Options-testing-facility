"""Black-Scholes option pricing, greeks and implied vol (pure-stdlib).

The reference vocabulary of the whole facility: market quotes live in
implied-vol space, every model is scored by inverting its prices back to a
Black-Scholes implied volatility, and the flat-vol fit is the first benchmark
arm of the out-of-sample study.

Ported from the Stocks repository (core/options.py, core/stats.py).
"""

from __future__ import annotations
import math
from dataclasses import dataclass

__all__ = ["norm_cdf", "norm_pdf", "bs_price", "bs_greeks", "implied_vol",
           "Greeks"]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, sigma: float, r: float):
    vol = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol
    return d1, d1 - vol


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.0,
             call: bool = True) -> float:
    """Black-Scholes price. Degenerate inputs return the discounted intrinsic."""
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        intrinsic = (S - K) if call else (K - S)
        return max(intrinsic, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    disc = math.exp(-r * T)
    if call:
        return S * norm_cdf(d1) - K * disc * norm_cdf(d2)
    return K * disc * norm_cdf(-d2) - S * norm_cdf(-d1)


def implied_vol(price: float, S: float, K: float, T: float, r: float = 0.0,
                call: bool = True, lo: float = 1e-4, hi: float = 5.0,
                iters: int = 80) -> float:
    """Black-Scholes implied volatility by bisection (price is monotone in vol)."""
    if S <= 0 or K <= 0 or T <= 0:
        return 0.0
    disc = math.exp(-r * T)
    intrinsic = max((S - K * disc) if call else (K * disc - S), 0.0)
    if price <= intrinsic:
        return lo
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if bs_price(S, K, T, m, r, call) < price:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


@dataclass(slots=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float      # per 1.00 change in vol (divide by 100 for per 1%)
    theta: float     # per year


def bs_greeks(S: float, K: float, T: float, sigma: float, r: float = 0.0,
              call: bool = True) -> Greeks:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        intrinsic = max((S - K) if call else (K - S), 0.0)
        delta = (1.0 if S > K else 0.0) if call else (-1.0 if S < K else 0.0)
        return Greeks(price=intrinsic, delta=delta, gamma=0.0, vega=0.0,
                      theta=0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    disc = math.exp(-r * T)
    pdf = norm_pdf(d1)
    price = bs_price(S, K, T, sigma, r, call)
    delta = norm_cdf(d1) if call else norm_cdf(d1) - 1.0
    gamma = pdf / (S * sigma * math.sqrt(T))
    vega = S * pdf * math.sqrt(T)
    if call:
        theta = -(S * pdf * sigma) / (2.0 * math.sqrt(T)) - r * K * disc * norm_cdf(d2)
    else:
        theta = -(S * pdf * sigma) / (2.0 * math.sqrt(T)) + r * K * disc * norm_cdf(-d2)
    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)

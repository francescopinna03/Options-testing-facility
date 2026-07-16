"""Raw-SVI implied-volatility surface: the market-standard benchmark.

SVI (Gatheral 2004) parameterises one expiry's TOTAL implied variance
w(k) = iv(k)^2 * tte as a function of log-moneyness k = log(K/F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

with b >= 0, |rho| < 1, sigma > 0 and a >= -b*sigma*sqrt(1-rho^2) (so that
w >= 0 everywhere). It is not a dynamic model -- it has no dynamics at all --
which is precisely why it is the right yardstick: any structural model (Heston
prior, SFV bridge) claiming to capture the surface should at least be compared
against the best static smile parameterisation practitioners actually use.

The facility works under the r = 0 / forward = spot convention of its chain
pipeline, so k = log(K / spot).

Fitting is per expiry in total-variance space with the shared Nelder-Mead
(:mod:`otf.optim`); positivity of the smile enters as a penalty on min_k w(k).
:class:`SVISurface` interpolates total variance linearly in maturity at fixed
k between fitted slices, with proportional-in-t extrapolation outside the
fitted range -- enough for the day-ahead (tte shift ~ 1/252) out-of-sample
study this facility runs.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from otf.optim import nelder_mead

__all__ = ["SVISlice", "SVISurface", "fit_svi_slice", "fit_svi_surface"]

_BIG = 1e6


@dataclass(slots=True)
class SVISlice:
    """One expiry's raw-SVI total-variance smile."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    tte: float
    rmse: float          # IV RMSE of the fit on its own quotes (vol points)
    n_quotes: int
    converged: bool

    def total_var(self, k: float) -> float:
        d = k - self.m
        w = self.a + self.b * (self.rho * d + math.sqrt(d * d + self.sigma * self.sigma))
        return max(w, 1e-10)

    def iv(self, k: float, tte: Optional[float] = None) -> float:
        """Implied vol at log-moneyness ``k``; ``tte`` defaults to the slice's
        own maturity (pass the evaluation maturity to reuse the smile at a
        slightly rolled tte)."""
        t = self.tte if tte is None else max(float(tte), 1e-8)
        return math.sqrt(self.total_var(k) / t)


def _min_total_var(a: float, b: float, rho: float, sigma: float) -> float:
    # min_k w(k) = a + b * sigma * sqrt(1 - rho^2)
    return a + b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0))


def fit_svi_slice(ks: Sequence[float], ivs: Sequence[float], tte: float,
                  weights: Optional[Sequence[float]] = None,
                  max_iter: int = 800, restarts: int = 1) -> SVISlice:
    """Fit one raw-SVI slice to (log-moneyness, implied vol) quotes.

    Least squares in total-variance space (comparable weight per quote
    regardless of maturity), unconstrained reparameterisation ``b = e^zb``,
    ``sigma = e^zs``, ``rho = tanh(zr)`` plus a penalty for a negative smile
    minimum. Needs >= 5 quotes (5 parameters).
    """
    if len(ks) != len(ivs) or len(ks) < 5:
        raise ValueError(f"need >= 5 (k, iv) pairs, got {len(ks)}")
    t = float(tte)
    if t <= 0.0:
        raise ValueError("tte must be positive")
    ws = [1.0] * len(ks) if weights is None else [max(float(w), 0.0) for w in weights]
    tot_w = sum(ws) or 1.0
    w_mkt = [iv * iv * t for iv in ivs]

    def unpack(z):
        a, zb, zr, m, zs = z
        return a, math.exp(zb), math.tanh(zr), m, math.exp(zs)

    def objective(z) -> float:
        a, b, rho, m, sigma = unpack(z)
        se = 0.0
        for k, wm, wt in zip(ks, w_mkt, ws):
            d = k - m
            w = a + b * (rho * d + math.sqrt(d * d + sigma * sigma))
            se += wt * (w - wm) ** 2
        pen = 0.0
        mn = _min_total_var(a, b, rho, sigma)
        if mn < 0.0:
            pen = _BIG * mn * mn
        return se / tot_w + pen

    # ATM-anchored start: flat-ish smile around the observed ATM total var.
    w_atm = w_mkt[min(range(len(ks)), key=lambda i: abs(ks[i]))]
    b0 = max(0.1 * w_atm / max(0.1, max(abs(min(ks)), abs(max(ks)))), 1e-4)
    z0 = [max(w_atm - b0 * 0.1, 1e-6), math.log(b0), math.atanh(-0.3), 0.0,
          math.log(0.1)]

    best_z, best_f, iters, conv = nelder_mead(objective, z0, step=0.5,
                                              max_iter=max_iter)
    for _ in range(max(0, restarts)):
        z2, f2, it2, c2 = nelder_mead(objective, best_z, step=0.15,
                                      max_iter=max_iter)
        iters += it2
        if f2 < best_f:
            best_z, best_f, conv = z2, f2, c2

    a, b, rho, m, sigma = unpack(best_z)
    sl = SVISlice(a=a, b=b, rho=rho, m=m, sigma=sigma, tte=t, rmse=0.0,
                  n_quotes=len(ks), converged=conv)
    se = sum(wt * (sl.iv(k) - iv) ** 2 for k, iv, wt in zip(ks, ivs, ws))
    sl.rmse = math.sqrt(se / tot_w)
    return sl


@dataclass(slots=True)
class SVISurface:
    """Fitted slices sorted by maturity + total-variance interpolation."""
    slices: List[SVISlice]
    spot: float

    def iv(self, strike: float, tte: float,
           spot: Optional[float] = None) -> float:
        """Implied vol at (strike, tte): linear interpolation of total
        variance in maturity at fixed k, proportional-in-t extrapolation
        outside the fitted maturity range.

        ``spot`` overrides the fitting-day spot: passing the evaluation day's
        underlying close reads the surface sticky-moneyness, the fair analog
        of respotting the structural models."""
        k = math.log(strike / (self.spot if spot is None else float(spot)))
        t = max(float(tte), 1e-8)
        sl = self.slices
        if t <= sl[0].tte:
            w = sl[0].total_var(k) * t / sl[0].tte
        elif t >= sl[-1].tte:
            w = sl[-1].total_var(k) * t / sl[-1].tte
        else:
            for lo, hi in zip(sl, sl[1:]):
                if lo.tte <= t <= hi.tte:
                    u = (t - lo.tte) / (hi.tte - lo.tte)
                    w = (1.0 - u) * lo.total_var(k) + u * hi.total_var(k)
                    break
        return math.sqrt(max(w, 1e-10) / t)


def fit_svi_surface(quotes: Sequence, spot: float, min_per_slice: int = 5,
                    max_iter: int = 800) -> SVISurface:
    """Fit raw-SVI slices expiry by expiry over ``quotes``.

    ``quotes`` are duck-typed: objects with ``strike``, ``tte``, ``iv`` and
    optionally ``weight`` attributes (``otf.calibration.MarketQuote`` works).
    Expiries with fewer than ``min_per_slice`` quotes are skipped; raises if
    no expiry survives.
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    by_tte: Dict[float, List] = {}
    for q in quotes:
        by_tte.setdefault(round(q.tte, 6), []).append(q)
    slices: List[SVISlice] = []
    for tte, group in sorted(by_tte.items()):
        if len(group) < min_per_slice:
            continue
        ks = [math.log(q.strike / spot) for q in group]
        ivs = [q.iv for q in group]
        ws = [getattr(q, "weight", 1.0) for q in group]
        slices.append(fit_svi_slice(ks, ivs, tte, weights=ws,
                                    max_iter=max_iter))
    if not slices:
        raise ValueError(f"no expiry has >= {min_per_slice} quotes")
    return SVISurface(slices=slices, spot=float(spot))

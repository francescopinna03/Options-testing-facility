"""Estimate an SFV prior from a real price series (realized measures).

Turns a series of closes -- e.g. LSEG Workspace daily history -- into the
parameters the SFV case needs: the diffusive Heston block
``(v0, kappa, theta, xi, rho)`` plus a Kou price-jump block
``(lambda_S, p_up, eta_up, eta_dn)``. It is the real-data counterpart of the
hand-set numbers in ``configs/cases/options_sfv.yaml``.

Decomposition (the SFV split, done honestly):

* **Jumps** are isolated first, by flagging returns beyond ``jump_z`` robust
  sigmas (median / MAD, so the threshold is not inflated by the jumps it is
  meant to catch). Their annualised intensity, up-probability and the two Kou
  decay rates come from the flagged set.
* **Diffusion** is everything else. ``theta`` (long-run variance) is the
  annualised variance of the jump-free returns; ``v0`` is the last EWMA
  instantaneous variance. ``kappa`` and ``xi`` are read off an AR(1) fit of the
  rolling variance series, and ``rho`` is the leverage correlation between
  returns and variance innovations (negative for equities).

The diffusive ``kappa/xi/rho`` from a single return series are *rough* realized
estimates -- noisy by nature. When an option smile is available, calibrating
the Heston block with :func:`core.calibration.calibrate_heston` is the higher
quality route; ``estimate_sfv_prior`` is the returns-only fallback and the
source of the bridge target. Everything here is pure-stdlib and unit-tested;
the only untested surface is the LSEG SDK line in
:func:`data.providers.lseg_history.load_closes`.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

__all__ = ["log_returns", "horizon_returns", "SFVPrior", "estimate_sfv_prior",
           "MarketJumpBlock", "JointSFVPrior", "estimate_joint_sfv"]


def log_returns(prices: Sequence[float]) -> List[float]:
    out = []
    for a, b in zip(prices, prices[1:]):
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def horizon_returns(prices: Sequence[float], k: int) -> List[float]:
    """Non-overlapping ``k``-step log-returns (the bridge target sample)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    out = []
    for i in range(0, len(prices) - k, k):
        a, b = prices[i], prices[i + k]
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _jump_flags(rets: Sequence[float], jump_z: float):
    """Robust (median/MAD) jump detector: returns ``(med, thr, flags)`` where
    ``flags`` is the set of indices ``t`` with ``|rets[t] - med| > thr``."""
    med = _median(rets)
    mad = _median([abs(r - med) for r in rets]) or 1e-12
    thr = jump_z * 1.4826 * mad                    # MAD -> sigma (normal)
    return med, thr, {t for t, r in enumerate(rets) if abs(r - med) > thr}


def _fit_kou(jumps: Sequence[float]):
    """Kou block ``(p_up, eta_up, eta_dn)`` from a sample of jump sizes.
    Decay = 1 / mean(|jump|) per side; guards eta_up > 1 (finite MGF)."""
    ups = [j for j in jumps if j > 0]
    dns = [-j for j in jumps if j < 0]
    p_up = (len(ups) / len(jumps)) if jumps else 0.5
    eta_up = max(1.0 / _mean(ups), 1.0001) if ups else 50.0
    eta_dn = max(1.0 / _mean(dns), 0.5) if dns else 50.0
    return p_up, eta_up, eta_dn


@dataclass(slots=True)
class SFVPrior:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    lambda_S: float
    p_up: float
    eta_up: float
    eta_dn: float
    n_returns: int
    n_jumps: int
    ann_vol_total: float          # annualised vol incl. jumps (sanity readout)

    def as_yaml_fields(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in
                ("v0", "kappa", "theta", "xi", "rho",
                 "lambda_S", "p_up", "eta_up", "eta_dn")}

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def estimate_sfv_prior(
    prices: Sequence[float],
    periods_per_year: float = 252.0,
    jump_z: float = 4.0,
    ewma_lambda: float = 0.94,
) -> SFVPrior:
    """Fit an SFV prior to ``prices`` by realized measures.

    ``periods_per_year`` annualises (252 for daily closes, 52 weekly, ...).
    ``jump_z`` is the robust-sigma threshold that separates jumps from
    diffusion; ``ewma_lambda`` is the RiskMetrics decay for the instantaneous
    variance series.
    """
    rets = log_returns(prices)
    n = len(rets)
    if n < 20:
        raise ValueError(f"need >= 20 returns, got {n}")
    dt = 1.0 / periods_per_year

    total_var_ann = _var(rets) * periods_per_year

    # --- jump / diffusion split via a robust (median/MAD) threshold --------- #
    med, thr, flags = _jump_flags(rets, jump_z)
    jumps = [rets[t] - med for t in sorted(flags)]
    diffusive = [r for t, r in enumerate(rets) if t not in flags]
    n_jumps = len(jumps)

    lambda_S = n_jumps * periods_per_year / n          # annualised intensity
    p_up, eta_up, eta_dn = _fit_kou(jumps)

    theta = max(_var(diffusive) * periods_per_year, 1e-6)   # long-run variance

    # --- instantaneous variance series (EWMA) for kappa / xi / rho ---------- #
    v = None
    v_series: List[float] = []
    for r in diffusive:
        r2 = r * r
        v = r2 if v is None else ewma_lambda * v + (1.0 - ewma_lambda) * r2
        v_series.append(v * periods_per_year)              # annualised
    v0 = max(v_series[-1] if v_series else theta, 1e-6)

    # AR(1) on the variance: dv_t = kappa*dt*(theta - v_t) + eps_t.
    # rho stays at the equity-leverage default unless the returns clearly
    # identify it (they usually do not at daily frequency -- see below).
    kappa, xi, rho = 2.0, 0.4, -0.5                        # defaults / fallback
    if len(v_series) > 30:
        x = [theta - v_series[t] for t in range(len(v_series) - 1)]
        dv = [v_series[t + 1] - v_series[t] for t in range(len(v_series) - 1)]
        sxx = sum(xi_ * xi_ for xi_ in x)
        if sxx > 1e-18:
            slope = sum(xi_ * d for xi_, d in zip(x, dv)) / sxx   # = kappa*dt
            k_est = slope / dt
            if 0.05 < k_est < 50.0:
                kappa = k_est
            # residual variance -> xi^2 * v * dt
            resid = [dv[t] - slope * x[t] for t in range(len(x))]
            zs = [resid[t] / math.sqrt(max(v_series[t], 1e-9) * dt)
                  for t in range(len(x))]
            xi_est = math.sqrt(max(_var(zs), 0.0))
            if 0.02 < xi_est < 5.0:
                xi = xi_est

    # Leverage via the standard next-bar realized proxy corr(r_t, r_{t+1}^2) on
    # the DIFFUSIVE returns (jump skew, already in the Kou block, does not leak
    # in). At daily frequency this is heavily attenuated -- the chi^2 noise of
    # r_{t+1}^2 swamps the variance signal -- so returns essentially cannot
    # identify rho. We keep the equity default and only override when the
    # de-attenuated proxy is clearly significant. An option smile
    # (calibrate_heston) is the right source for rho/kappa/xi.
    if len(diffusive) > 30:
        a = diffusive[:-1]
        b = [x * x for x in diffusive[1:]]
        ma, mb = _mean(a), _mean(b)
        cov = sum((a[t] - ma) * (b[t] - mb) for t in range(len(a)))
        sa = math.sqrt(sum((x - ma) ** 2 for x in a))
        sb = math.sqrt(sum((x - mb) ** 2 for x in b))
        if sa > 1e-15 and sb > 1e-18:
            proxy = 6.0 * cov / (sa * sb)             # de-attenuated
            if abs(proxy) > 0.15:                     # clearly identified only
                rho = max(-0.95, min(0.95, proxy))

    return SFVPrior(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_S=lambda_S, p_up=p_up, eta_up=eta_up, eta_dn=eta_dn,
        n_returns=n, n_jumps=n_jumps, ann_vol_total=math.sqrt(total_var_ann),
    )


# --------------------------------------------------------------------------- #
# Joint (portfolio) estimation: correlation + market vs idiosyncratic jumps.  #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class MarketJumpBlock:
    """Common (systemic) Kou jump shared by every name in the portfolio."""
    lambda_M: float               # annualised intensity of market-wide jumps
    p_up: float
    eta_up: float
    eta_dn: float
    n_jumps: int                  # market-jump days found in the sample


@dataclass(slots=True)
class JointSFVPrior:
    """Per-name SFV priors plus the cross-sectional structure.

    ``priors[name]`` is the *marginal* fit (its jump block contains ALL of the
    name's jumps -- idio + market -- which is what a marginal bridge fit or a
    single-name readout should use). ``idio[name]`` is the idiosyncratic-only
    jump block for the simulator, where market jumps enter once through
    ``market`` and hit every name on the same tick.
    """
    names: List[str]
    priors: Dict[str, SFVPrior]
    idio: Dict[str, Dict[str, float]]     # name -> {lambda_S, p_up, eta_up, eta_dn}
    market: MarketJumpBlock
    corr: List[List[float]]               # diffusive correlation, order = names
    n_aligned: int                        # aligned observations used


def estimate_joint_sfv(
    prices_by_name: Dict[str, Sequence[float]],
    periods_per_year: float = 252.0,
    jump_z: float = 4.0,
    ewma_lambda: float = 0.94,
) -> JointSFVPrior:
    """Joint SFV prior for a portfolio of price series.

    The decomposition keeps the total correlation honest: the **diffusive**
    correlation matrix is estimated on jump-free aligned returns only, and the
    crash co-movement the filter removed comes back through the **market
    jump** channel -- a day on which >= 2 names jump together is systemic, and
    its sizes (pooled across the jumping names) fit one common Kou law. Jumps
    a name has alone stay in its idiosyncratic block. Feeding the *total*
    empirical correlation to the diffusive channel AND adding common jumps
    would double-count exactly the tail dependence this split preserves.

    Series are aligned on their common tail (last ``min(len)`` closes). With
    exports over the same venue/calendar that is a date alignment; mixing
    calendars shifts rows against each other and biases the correlation low.
    """
    names = list(prices_by_name)
    if len(names) < 2:
        raise ValueError("need >= 2 names for a joint estimate")
    m = min(len(p) for p in prices_by_name.values())
    if m < 40:
        raise ValueError(f"need >= 40 aligned closes, got {m}")
    rets = {k: log_returns(list(prices_by_name[k])[-m:]) for k in names}
    n = min(len(r) for r in rets.values())
    rets = {k: r[-n:] for k, r in rets.items()}

    priors = {k: estimate_sfv_prior(list(prices_by_name[k])[-m:],
                                    periods_per_year=periods_per_year,
                                    jump_z=jump_z, ewma_lambda=ewma_lambda)
              for k in names}

    med, flags = {}, {}
    for k in names:
        med[k], _, flags[k] = _jump_flags(rets[k], jump_z)

    # A day is a MARKET jump when at least two names jump together.
    market_days = {t for t in range(n)
                   if sum(1 for k in names if t in flags[k]) >= 2}
    pooled = [rets[k][t] - med[k]
              for k in names for t in sorted(flags[k] & market_days)]
    mp, meu, med_ = _fit_kou(pooled)
    market = MarketJumpBlock(
        lambda_M=len(market_days) * periods_per_year / n,
        p_up=mp, eta_up=meu, eta_dn=med_, n_jumps=len(market_days))

    idio: Dict[str, Dict[str, float]] = {}
    for k in names:
        own = sorted(flags[k] - market_days)
        ip, ieu, ied = _fit_kou([rets[k][t] - med[k] for t in own])
        idio[k] = {"lambda_S": len(own) * periods_per_year / n,
                   "p_up": ip, "eta_up": ieu, "eta_dn": ied}

    # Diffusive correlation on rows where NO name jumped.
    all_jump_days = market_days.union(*flags.values())
    keep = [t for t in range(n) if t not in all_jump_days]
    corr = [[1.0] * len(names) for _ in names]
    if len(keep) > 10:
        clean = {k: [rets[k][t] for t in keep] for k in names}
        mu = {k: _mean(clean[k]) for k in names}
        sd = {k: math.sqrt(sum((x - mu[k]) ** 2 for x in clean[k])) for k in names}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j <= i or sd[a] <= 1e-15 or sd[b] <= 1e-15:
                    continue
                c = sum((clean[a][t] - mu[a]) * (clean[b][t] - mu[b])
                        for t in range(len(keep))) / (sd[a] * sd[b])
                corr[i][j] = corr[j][i] = max(-0.99, min(0.99, c))

    return JointSFVPrior(names=names, priors=priors, idio=idio, market=market,
                         corr=corr, n_aligned=n)

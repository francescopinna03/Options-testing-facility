"""SFV model: affine jump-diffusion prior + restricted Schrödinger bridge.

This is the MOST RECENT formulation of the SFV model (the one implemented in
the Stocks repository, lineage documented in docs/MODEL_HISTORY.md). Under the
prior Q^0:

    d X = (r - 0.5 v - lambda_S kappa_S) dt + sqrt(v) dW_S + dJ_S
    d v = kappa (theta - v) dt + xi sqrt(v) dW_v
    d<W_S, W_v> = rho dt
    J_S ~ Kou(lambda_S, p_up, eta_up, eta_dn)

with the Kou compensator kappa_S = E[e^Y - 1] in closed form so e^X stays a
discounted martingale. The restricted Schrödinger-bridge correction acts on
the VARIANCE channel only:

    nu(x, v) = alpha * gate(v) * v * (rho*xi * dTheta/dx + xi^2 * dTheta/dv)
    Theta(x, v) = b0 + b1 x~ + b2 v~ + b3 x~ v~ + b4 x~^2 + b5 v~^2

on standardized coordinates x~ = x/sx, v~ = (v - v_ref)/sv, with an optional
sigmoid variance gate (transient activation) and an optional tanh cap on nu.
Because the correction cannot manufacture price drift, the forward is
preserved by construction; targets must differ in SHAPE (variance level,
tails, x-v feedback skew).

The :class:`PathEngine` freezes all shocks at construction (common random
numbers), so every quantity is a deterministic function of beta and the same
pure-stdlib Nelder-Mead used everywhere else can calibrate it. No closed form
exists once beta != 0, so pricing under the bridged law is Monte Carlo
(:func:`mc_price`, :func:`mc_smile`) over the frozen shocks -- smooth in beta.

Ported from the Stocks repository (sim/bridge_calibration.py, sim/sfv.py);
the calibration front-ends live in otf.calibration.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from otf.models.black_scholes import implied_vol

__all__ = ["kou_compensator", "PathEngine", "BridgeDiagnostics",
           "w2_distance", "sinkhorn_divergence", "sample_moments",
           "standardization_for", "mc_price", "mc_smile"]


def kou_compensator(p_up: float, eta_up: float, eta_dn: float) -> float:
    """E[e^Y - 1] for the Kou double-exponential jump law.

    Y = +Exp(eta_up) w.p. p_up, -Exp(eta_dn) w.p. 1 - p_up.
    Requires eta_up > 1 for a finite upward MGF.
    """
    if eta_up <= 1.0:
        raise ValueError("Kou eta_up must be strictly > 1 for finite MGF")
    if eta_dn <= 0.0:
        raise ValueError("Kou eta_dn must be > 0")
    up = p_up * eta_up / (eta_up - 1.0)
    dn = (1.0 - p_up) * eta_dn / (eta_dn + 1.0)
    return up + dn - 1.0


# --------------------------------------------------------------------------- #
# Distances between 1-D samples                                               #
# --------------------------------------------------------------------------- #
def _quantiles(sample: Sequence[float], m: int) -> List[float]:
    s = sorted(sample)
    n = len(s)
    out = []
    for k in range(m):
        u = (k + 0.5) / m * n - 0.5          # fractional order statistic
        if u <= 0:
            out.append(s[0])
        elif u >= n - 1:
            out.append(s[-1])
        else:
            i = int(u)
            w = u - i
            out.append(s[i] * (1.0 - w) + s[i + 1] * w)
    return out


def w2_distance(a: Sequence[float], b: Sequence[float], m: int = 256) -> float:
    """Exact 1-D 2-Wasserstein distance via quantile matching.

    Values are clamped to +-1e6: a Nelder-Mead trial beta can blow the
    paths up, and the distance only needs to report "far", not overflow
    in (x - y)^2."""
    qa = _quantiles([min(max(v, -1e6), 1e6) for v in a], m)
    qb = _quantiles([min(max(v, -1e6), 1e6) for v in b], m)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(qa, qb)) / m)


def _sinkhorn_ot(x: List[float], y: List[float], eps: float, iters: int) -> float:
    """Entropic OT value between uniform empirical measures (log-domain)."""
    n, m = len(x), len(y)
    f = [0.0] * n
    g = [0.0] * m
    log_n, log_m = math.log(n), math.log(m)
    for _ in range(iters):
        for i in range(n):
            xi = x[i]
            mx = None
            vals = []
            for j in range(m):
                t = (g[j] - (xi - y[j]) ** 2) / eps
                vals.append(t)
                if mx is None or t > mx:
                    mx = t
            s = sum(math.exp(t - mx) for t in vals)
            f[i] = -eps * (mx + math.log(s) - log_m)
        for j in range(m):
            yj = y[j]
            mx = None
            vals = []
            for i in range(n):
                t = (f[i] - (x[i] - yj) ** 2) / eps
                vals.append(t)
                if mx is None or t > mx:
                    mx = t
            s = sum(math.exp(t - mx) for t in vals)
            g[j] = -eps * (mx + math.log(s) - log_n)
    return sum(f) / n + sum(g) / m


def sinkhorn_divergence(a: Sequence[float], b: Sequence[float],
                        eps: Optional[float] = None, iters: int = 60,
                        atoms: int = 96) -> float:
    """Debiased Sinkhorn divergence S_eps = OT(a,b) - (OT(a,a)+OT(b,b))/2.

    Samples are compressed to ``atoms`` quantile atoms (uniform weights) to
    keep the pure-Python O(atoms^2 * iters) cost tame. As eps -> 0 this
    approaches W2^2. Values are clamped to +-1e6: a Nelder-Mead trial beta
    can blow the paths up, and the divergence only needs to report "far",
    not overflow in (x - y)^2.
    """
    ca = [min(max(v, -1e6), 1e6) for v in a]
    cb = [min(max(v, -1e6), 1e6) for v in b]
    x = _quantiles(ca, atoms)
    y = _quantiles(cb, atoms)
    if eps is None:
        var = max(sample_moments(ca + cb)[1] ** 2, 1e-12)
        eps = 0.1 * var
    ab = _sinkhorn_ot(x, y, eps, iters)
    aa = _sinkhorn_ot(x, x, eps, iters)
    bb = _sinkhorn_ot(y, y, eps, iters)
    return max(ab - 0.5 * (aa + bb), 0.0)


def sample_moments(s: Sequence[float]) -> Tuple[float, float, float, float]:
    """(mean, std, skew, excess kurtosis) of a sample."""
    n = len(s)
    mu = sum(s) / n
    c2 = sum((x - mu) ** 2 for x in s) / n
    sd = math.sqrt(max(c2, 1e-18))
    c3 = sum((x - mu) ** 3 for x in s) / n
    c4 = sum((x - mu) ** 4 for x in s) / n
    return mu, sd, c3 / sd ** 3, c4 / sd ** 4 - 3.0


# --------------------------------------------------------------------------- #
# Path engine with frozen shocks (common random numbers)                      #
# --------------------------------------------------------------------------- #
class PathEngine:
    """Simulates horizon log-returns of the SFV prior under a trial bridge.

    All randomness is drawn once in ``__init__``; ``terminal_logret(beta)``
    replays the SAME shocks under the trial variance drift, so any calibration
    objective built on it is a deterministic function of beta (common random
    numbers). Discretization: full-truncation Euler for v (Lord, Koekkoek,
    Van Dijk 2010), log-Euler for X, Knuth Poisson sampling for the Kou jumps.
    """

    def __init__(
        self,
        n_paths: int = 400,
        n_steps: int = 200,
        horizon: float = 0.1,            # years
        v0: float = 0.0225,
        kappa: float = 3.0,
        theta: float = 0.0225,
        xi: float = 0.4,
        rho: float = -0.5,
        r: float = 0.0,
        lambda_S: float = 0.0,
        p_up: float = 0.5,
        eta_up: float = 50.0,
        eta_dn: float = 50.0,
        seed: int = 0,
        alpha: float = 1.0,
        # standardized ansatz + gate + cap (defaults = raw ansatz, no gate).
        # The engine's x starts at 0, so x0 stays 0; a consumer reusing fitted
        # betas at a different spot must measure the same displacement
        # (write the constants alongside the betas).
        sx: float = 1.0,
        v_ref: float = 0.0,
        sv: float = 1.0,
        gate_m: float = 0.0,
        gate_c: float = 0.0,
        nu_max: float = 0.0,
    ):
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.dt = float(horizon) / float(n_steps)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.r = float(r)
        self.alpha = float(alpha)
        self.sx = float(sx)
        self.v_ref = float(v_ref)
        self.sv = float(sv)
        self.gate_m = float(gate_m)
        self.gate_c = float(gate_c)
        self.nu_max = float(nu_max)
        self.lambda_S = float(lambda_S)
        kou_k = kou_compensator(p_up, eta_up, eta_dn) if lambda_S > 0.0 else 0.0
        self.jump_drift = -lambda_S * kou_k          # martingale compensator

        rng = random.Random(seed)
        sdt = math.sqrt(self.dt)
        orto = math.sqrt(max(1.0 - self.rho * self.rho, 0.0))
        # Frozen shocks: per path/step, the price Gaussian, the variance
        # Gaussian (already correlated), and the summed price-jump increment.
        self.zs: List[List[float]] = []
        self.zv: List[List[float]] = []
        self.js: List[List[float]] = []
        lam_dt = self.lambda_S * self.dt
        for _ in range(self.n_paths):
            zs_row, zv_row, j_row = [], [], []
            for _ in range(self.n_steps):
                z1 = rng.gauss(0.0, 1.0)
                zp = rng.gauss(0.0, 1.0)
                zs_row.append(z1 * sdt)
                zv_row.append((self.rho * z1 + orto * zp) * sdt)
                j = 0.0
                if lam_dt > 0.0:
                    # Knuth Poisson (lam_dt is tiny per step)
                    L = math.exp(-lam_dt)
                    k, p = 0, 1.0
                    while True:
                        k += 1
                        p *= rng.random()
                        if p < L:
                            k -= 1
                            break
                    for _ in range(k):
                        if rng.random() < p_up:
                            j += rng.expovariate(eta_up)
                        else:
                            j -= rng.expovariate(eta_dn)
                j_row.append(j)
            self.zs.append(zs_row)
            self.zv.append(zv_row)
            self.js.append(j_row)

    def terminal_logret(self, beta: Sequence[float]) -> List[float]:
        """Horizon log-returns X_T - X_0 under bridge coefficients ``beta``
        (b0 b1 b2 b3 b4 b5; b0 is inert)."""
        return self._replay(beta)[0]

    def diagnostics(self, beta: Sequence[float]) -> "BridgeDiagnostics":
        """The SFV_M2 'Diagnostic quantities' at ``beta`` (same frozen shocks)."""
        return self._replay(beta, want_diag=True)[1]

    def _replay(self, beta: Sequence[float], want_diag: bool = False):
        b1, b2, b3, b4, b5 = (float(beta[1]), float(beta[2]), float(beta[3]),
                              float(beta[4]), float(beta[5]))
        dt = self.dt
        kappa, theta, xi, alpha0 = self.kappa, self.theta, self.xi, self.alpha
        rho_xi, xi2 = self.rho * xi, xi * xi
        sx, v_ref, sv = self.sx, self.v_ref, self.sv
        gate_m, gate_c, nu_max = self.gate_m, self.gate_c, self.nu_max
        base_drift = self.r + self.jump_drift
        out = []
        ctrl_sum = energy_sum = gate_sum = 0.0
        hits = 0
        for zs_row, zv_row, j_row in zip(self.zs, self.zv, self.js):
            x = 0.0
            v = self.v0
            for t in range(self.n_steps):
                vp = v if v > 0.0 else 0.0                   # full truncation
                sq = math.sqrt(vp)
                # bridge correction on standardized coordinates
                xt = x / sx                                  # engine x0 = 0
                vt = (vp - v_ref) / sv
                gate = 1.0
                if gate_m > 0.0:
                    # clamp the exponent: a sharp gate far from the current
                    # variance must saturate to 0/1, not overflow exp()
                    zg = min(max(-gate_m * (vt - gate_c), -60.0), 60.0)
                    gate = 1.0 / (1.0 + math.exp(zg))
                dxt = (b1 + b3 * vt + 2.0 * b4 * xt) / sx
                dvt = (b2 + b3 * xt + 2.0 * b5 * vt) / sv
                nu = alpha0 * gate * vp * (rho_xi * dxt + xi2 * dvt)
                if nu_max > 0.0:
                    nu = nu_max * math.tanh(nu / nu_max)
                if want_diag:
                    ctrl_sum += nu
                    energy_sum += nu * nu / (xi2 * vp + 1e-8) * dt
                    gate_sum += alpha0 * gate
                x += (base_drift - 0.5 * vp) * dt + sq * zs_row[t] + j_row[t]
                v = vp + (kappa * (theta - vp) + nu) * dt + xi * sq * zv_row[t]
                if v < 0.0:
                    hits += 1
            out.append(x)
        diag = None
        if want_diag:
            n_obs = self.n_paths * self.n_steps
            horizon = dt * self.n_steps
            disc = [math.exp(x - self.r * horizon) for x in out]
            m = sum(disc) / len(disc)
            sd = math.sqrt(sum((d - m) ** 2 for d in disc) / (len(disc) - 1))
            diag = BridgeDiagnostics(
                mean_control=ctrl_sum / n_obs,
                control_energy=energy_sum / self.n_paths,
                martingale_error=m - 1.0,
                martingale_stderr=sd / math.sqrt(len(disc)),
                boundary_hits=hits / n_obs,
                gate_activation=gate_sum / n_obs,
                horizon=horizon,
            )
        return out, diag


@dataclass(slots=True)
class BridgeDiagnostics:
    """SFV_M2 'Diagnostic quantities' for one beta on frozen shocks.

    ``martingale_error`` is E[e^{X_T - rT}] - 1 (0 for an exact risk-neutral
    martingale; judge it against ``martingale_stderr``). ``control_energy`` is
    the variance-normalized proxy E[int nu^2/(xi^2 v + eps) dt] of the KL cost.
    ``boundary_hits`` is the fraction of (path, step) draws the full-truncation
    scheme clipped at v = 0. ``gate_activation`` is E[mean_t alpha_t]: ~always
    ``alpha`` means the model is not transient, ~0 means the bridge layer is
    irrelevant."""
    mean_control: float
    control_energy: float
    martingale_error: float
    martingale_stderr: float
    boundary_hits: float
    gate_activation: float
    horizon: float


def standardization_for(v0: float, kappa: float, theta: float, xi: float,
                        horizon: float) -> dict:
    """Standardization constants for the quadratic ansatz (SFV_M2 sec. 9.1):
    x scaled by the typical horizon log-return, v centred at theta and scaled
    by the stationary CIR standard deviation xi sqrt(theta / 2 kappa)."""
    sx = max(math.sqrt(max(theta, v0, 1e-8) * max(horizon, 1e-6)), 1e-6)
    sv = max(xi * math.sqrt(max(theta, 1e-8) / max(2.0 * kappa, 1e-8)), 1e-6)
    return {"sx": sx, "v_ref": float(theta), "sv": sv}


# --------------------------------------------------------------------------- #
# Monte Carlo pricing under the bridged SFV law (no CF exists once beta != 0) #
# --------------------------------------------------------------------------- #
def mc_price(engine: PathEngine, strike: float, is_call: bool = True,
             beta: Sequence[float] = (0.0,) * 6,
             s0: float = 100.0) -> Tuple[float, float]:
    """European option price under the engine's (bridged) law: discounted MC
    mean over the frozen shocks, so prices are SMOOTH in beta (CRN). Returns
    ``(price, standard_error)``."""
    T = engine.dt * engine.n_steps
    disc = math.exp(-engine.r * T)
    pays = []
    for x in engine.terminal_logret(beta):
        st = s0 * math.exp(x)
        pays.append(max(st - strike, 0.0) if is_call else max(strike - st, 0.0))
    m = sum(pays) / len(pays)
    var = sum((p - m) ** 2 for p in pays) / max(len(pays) - 1, 1)
    return disc * m, disc * math.sqrt(var / len(pays))


def mc_smile(engine: PathEngine, strikes: Sequence[float],
             beta: Sequence[float] = (0.0,) * 6, s0: float = 100.0,
             is_call: Optional[bool] = None) -> List[dict]:
    """MC price + Black-Scholes implied vol per strike (OTM side by default:
    puts below spot, calls above -- the liquid convention)."""
    T = engine.dt * engine.n_steps
    out = []
    for k in strikes:
        call = (k >= s0) if is_call is None else bool(is_call)
        px, se = mc_price(engine, k, call, beta, s0)
        iv = implied_vol(px, s0, k, T, engine.r, call)
        out.append({"strike": float(k), "is_call": call, "price": px,
                    "stderr": se, "iv": iv})
    return out

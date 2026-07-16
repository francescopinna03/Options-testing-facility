"""Vectorized Heston prior with exposed pathwise innovations.

Two schemes behind one PriorModel implementation:

* ``euler_ft`` — full-truncation Euler (Lord, Koekkoek, Van Dijk 2010).
  The M1 *reference* scheme: its Brownian innovations are exact pathwise
  objects, which the BSDE, the Girsanov likelihood, and the energy identity
  require. Weak-order accuracy is bought with small dt.
* ``qe`` — Andersen (2008) quadratic-exponential. Pricing-grade terminal
  marginals at coarse dt, used as a cross-check of the Euler engine. Its
  variance update is *not* driven by a Gaussian increment, so the batch is
  emitted with ``d_w=None`` and must be rejected by the likelihood layer
  (see otf.ssfv.types.PathBatch).

Coordinate convention: ``x`` is the log *forward* price, x = log(S/F_0,t),
so the drift is -v/2 dt, E[e^{x_T}] = 1 is the martingale identity, and the
constraint-family coordinate is simply k = x_T (arch doc §6.1).

Correlation convention: ``d_w[..., 0]`` = G_1 sqrt(dt) drives the price
channel, ``d_w[..., 1]`` = G_2 sqrt(dt) is the orthogonal component;
dW_S = d_w[..., 0], dW_v = rho d_w[..., 0] + sqrt(1-rho^2) d_w[..., 1].
The orthogonal basis (G_1, G_2) is what the martingale projection uses.

The Feller condition is recorded, never enforced (DECISIONS.md D8).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from otf.models.heston import heston_price
from otf.ssfv.types import LocalCharacteristics, PathBatch, ValidationReport

__all__ = ["HestonPrior"]

_QE_PSI_C = 1.5  # Andersen's switching threshold


def _batch_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class HestonPrior:
    """Prior Q^0: correlated Heston diffusion (no jumps), forward coordinates."""

    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    scheme: str = "euler_ft"  # or "qe"

    # -- PriorModel protocol -------------------------------------------------

    def simulate(self, n_paths: int, n_steps: int, horizon: float, seed: int) -> PathBatch:
        if self.scheme == "euler_ft":
            return self._simulate_euler_ft(n_paths, n_steps, horizon, seed)
        if self.scheme == "qe":
            return self._simulate_qe(n_paths, n_steps, horizon, seed)
        raise ValueError(f"unknown scheme {self.scheme!r}")

    def characteristics(self, x: float, v: float, t: float) -> LocalCharacteristics:
        vp = max(v, 0.0)
        return LocalCharacteristics(
            drift=(-0.5 * vp, self.kappa * (self.theta - v)),
            diffusion=(vp, self.rho * self.xi * vp, self.rho * self.xi * vp, self.xi**2 * vp),
        )

    def price_european_cf(self, strike: float, maturity: float, call: bool = True) -> float:
        """Forward (undiscounted, S0=1) European price via the Heston CF."""
        return heston_price(
            S=1.0, K=strike, T=maturity, v0=self.v0, kappa=self.kappa,
            theta=self.theta, xi=self.xi, rho=self.rho, r=0.0, call=call,
        )

    def validate_parameters(self) -> ValidationReport:
        msgs = []
        ok = True
        for name in ("v0", "kappa", "theta", "xi"):
            if getattr(self, name) <= 0.0:
                msgs.append(f"{name} must be positive")
                ok = False
        if not -1.0 < self.rho < 1.0:
            msgs.append("rho must lie in (-1, 1)")
            ok = False
        feller = 2.0 * self.kappa * self.theta - self.xi**2
        if feller < 0.0:
            # Recorded, not enforced: no admissibility constraint (D8).
            msgs.append(f"Feller violated (2*kappa*theta - xi^2 = {feller:.6g}); "
                        "scheme accuracy near v=0 may degrade, existence unaffected")
        return ValidationReport(ok=ok, messages=tuple(msgs))

    # -- exact CIR moments (unit-test anchors) -------------------------------

    def variance_mean(self, t: float) -> float:
        e = math.exp(-self.kappa * t)
        return self.theta + (self.v0 - self.theta) * e

    def variance_variance(self, t: float) -> float:
        k, th, x2 = self.kappa, self.theta, self.xi**2
        e = math.exp(-self.kappa * t)
        return (self.v0 * x2 * e * (1.0 - e) / k
                + th * x2 * (1.0 - e) ** 2 / (2.0 * k))

    # -- schemes --------------------------------------------------------------

    def _simulate_euler_ft(self, n_paths: int, n_steps: int, horizon: float, seed: int) -> PathBatch:
        dt = horizon / n_steps
        sdt = math.sqrt(dt)
        orto = math.sqrt(1.0 - self.rho**2)
        rng = np.random.Generator(np.random.PCG64(seed))

        # Orthogonal N(0,1) basis, exposed as the pathwise innovations.
        g = rng.standard_normal((n_paths, n_steps, 2))
        d_w = g * sdt

        x = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        x[:, 0] = 0.0
        v[:, 0] = self.v0
        for j in range(n_steps):
            vp = np.maximum(v[:, j], 0.0)  # full truncation
            sq = np.sqrt(vp)
            dw_s = d_w[:, j, 0]
            dw_v = self.rho * d_w[:, j, 0] + orto * d_w[:, j, 1]
            x[:, j + 1] = x[:, j] - 0.5 * vp * dt + sq * dw_s
            v[:, j + 1] = v[:, j] + self.kappa * (self.theta - vp) * dt + self.xi * sq * dw_v

        times = np.linspace(0.0, horizon, n_steps + 1)
        return PathBatch(
            times=times, x=x, v=v, d_w=d_w,
            jump_offsets=None, jump_marks=None,
            initial_state=np.array([0.0, self.v0]),
            scheme="euler_ft", seed=seed, batch_hash=_batch_hash(x, v),
        )

    def _simulate_qe(self, n_paths: int, n_steps: int, horizon: float, seed: int) -> PathBatch:
        """Andersen QE with central discretization (gamma_1 = gamma_2 = 1/2).

        Pricing-grade: d_w is None because V_{t+dt} is sampled from a
        moment-matched quadratic/exponential law, not driven by a Gaussian
        increment. K0 uses Andersen's martingale correction so E[e^x] = 1
        holds exactly in the conditional step.
        """
        dt = horizon / n_steps
        k, th, xi, rho = self.kappa, self.theta, self.xi, self.rho
        rng = np.random.Generator(np.random.PCG64(seed))

        e = math.exp(-k * dt)
        c1 = xi**2 * e * (1.0 - e) / k
        c2 = th * xi**2 * (1.0 - e) ** 2 / (2.0 * k)

        g1 = g2 = 0.5
        K1 = g1 * dt * (k * rho / xi - 0.5) - rho / xi
        K2 = g2 * dt * (k * rho / xi - 0.5) + rho / xi
        K3 = g1 * dt * (1.0 - rho**2)
        K4 = g2 * dt * (1.0 - rho**2)
        # Martingale-corrected K0 (Andersen 2008, QE-M): A = K2 + K4/2.
        A = K2 + 0.5 * K4

        x = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        x[:, 0] = 0.0
        v[:, 0] = self.v0
        for j in range(n_steps):
            vt = v[:, j]
            m = th + (vt - th) * e
            s2 = vt * c1 + c2
            psi = s2 / np.maximum(m**2, 1e-300)

            v_next = np.empty(n_paths)
            u = rng.uniform(size=n_paths)
            quad = psi <= _QE_PSI_C
            # Quadratic branch: V = a (b + Z)^2.
            if np.any(quad):
                pq = psi[quad]
                b2 = 2.0 / pq - 1.0 + np.sqrt(np.maximum(2.0 / pq * (2.0 / pq - 1.0), 0.0))
                a = m[quad] / (1.0 + b2)
                zv = _norm_ppf(u[quad])
                v_next[quad] = a * (np.sqrt(b2) + zv) ** 2
            # Exponential branch: P(V=0) = p, else exponential tail.
            expb = ~quad
            if np.any(expb):
                pe = psi[expb]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / np.maximum(m[expb], 1e-300)
                ue = u[expb]
                v_next[expb] = np.where(ue <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - ue, 1e-300)) / beta)

            # Martingale-corrected K0: exact conditional-mgf correction on
            # the quadratic branch, Andersen eq. (36)-(40); exponential
            # branch uses its exact mgf too.
            K0 = np.empty(n_paths)
            if np.any(quad):
                pq = psi[quad]
                b2 = 2.0 / pq - 1.0 + np.sqrt(np.maximum(2.0 / pq * (2.0 / pq - 1.0), 0.0))
                a = m[quad] / (1.0 + b2)
                Aa = A * a
                K0[quad] = -Aa * b2 / (1.0 - 2.0 * Aa) + 0.5 * np.log(np.maximum(1.0 - 2.0 * Aa, 1e-300)) \
                    - (K1 + 0.5 * K3) * vt[quad]
            if np.any(expb):
                pe = psi[expb]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / np.maximum(m[expb], 1e-300)
                K0[expb] = -np.log(np.maximum(p + beta * (1.0 - p) / np.maximum(beta - A, 1e-300), 1e-300)) \
                    - (K1 + 0.5 * K3) * vt[expb]

            zx = rng.standard_normal(n_paths)
            x[:, j + 1] = x[:, j] + K0 + K1 * vt + K2 * v_next \
                + np.sqrt(np.maximum(K3 * vt + K4 * v_next, 0.0)) * zx
            v[:, j + 1] = v_next

        times = np.linspace(0.0, horizon, n_steps + 1)
        return PathBatch(
            times=times, x=x, v=v, d_w=None,
            jump_offsets=None, jump_marks=None,
            initial_state=np.array([0.0, self.v0]),
            scheme="qe", seed=seed, batch_hash=_batch_hash(x, v),
        )


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    """Standard normal inverse CDF (Acklam's rational approximation).

    Avoids a hard SciPy dependency inside the hot loop; max abs error
    ~1.15e-9, adequate for QE variance sampling.
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    u = np.clip(u, 1e-300, 1.0 - 1e-16)
    out = np.empty_like(u)
    lo = u < 0.02425
    hi = u > 1.0 - 0.02425
    mid = ~(lo | hi)
    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(u[lo]))
        out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                  ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log(1.0 - u[hi]))
        out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(mid):
        q = u[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return out

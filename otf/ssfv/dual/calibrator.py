"""Finite dual calibration via the reduced moment map (arch doc §10;
DECISIONS.md D6 — primary algorithm of the projective core).

The calibrator solves the moment equations m(lambda) = a_n rather than
running a certified ascent on the dual D_n(lambda) = lambda^T a_n -
Y_0^sample(Phi_lambda) — hence the name ReducedMomentMapCalibrator. The
two are consistent by a *structural* theorem, not by convention: the
sample-dual gradient is

    dD_n/dlambda = a_n - E^w[Psi] + E^w[d_lambda N^S],

and d_lambda N^S is a stochastic integral against W^S. In the diffusion
sector the Girsanov change of measure moves only W^perp (the projector is
structural), so W^S remains a Q_lambda-Brownian motion and
E^{Q_lambda}[d_lambda N^S] = 0 exactly in continuous time, O(1/sqrt(N))
on the sample. A zero of the moment map is therefore a first-order
condition of the sample dual up to Monte Carlo error; the certified-dual
upgrade (implicit differentiation of the Picard fixed point) is M2 scope.

Two-stage realization:

* **stage 1 — static exponential family.** With no dynamic offset the
  sample dual g(lambda) = lambda^T a - log-mean-exp(lambda^T Psi) is
  strictly concave with exact gradient and Hessian; Newton solves it to
  machine precision. This is the marginal-only tilt: an excellent starting
  point because the SSFV correction to the *marginals* beyond it is the
  martingale projection.

* **stage 2 — Gauss-Newton on the true moment map.** The map
  m(lambda) = E^{Q_lambda}[Psi] (weights exp(Phi_lambda - N^S(lambda)),
  field refreshed by the Picard Hopf-Cole solver) is differentiated by
  finite differences on the frozen paths (the solver is a deterministic,
  smooth function of lambda). The Newton step uses a *pseudo-inverse* with
  a relative singular-value cutoff. This is essential, not cosmetic: in
  directions dynamically replicable by the martingale term, the hedge
  N^S(lambda) cancels almost all of the static tilt's effect on the
  moments — the reduced Jacobian (the Schur complement of §10.3) is
  near-singular there, a naive static update overshoots by 1/(1 - gamma)
  and diverges, and the pseudo-inverse instead takes the minimal-norm step
  in the identifiable subspace.

Safeguards per step: an exact sup-norm trust region; a monotone line
search — the step is accepted only if the moment residual ||a - m||
strictly decreases; and ESS backtracking — for the mild deformations of
interest the true log-weights have std about sqrt(2H), so an ESS collapse
always signals field-estimation noise. Failing candidates are halved; if
no acceptable step exists the fit stops and says so (§20: fail loudly);
the certificates carry the residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from otf.ssfv.constraints.hat_family import LambdaPotential, NestedHatFamily
from otf.ssfv.types import BSDESolution, ConstraintLevel, DualFitResult, PathBatch

__all__ = ["ReducedMomentMapCalibrator"]


@dataclass(frozen=True)
class ReducedMomentMapCalibrator:
    """FiniteDualCalibrator backend: static warm start + reduced Gauss-Newton."""

    family: NestedHatFamily
    solver: Any = None
    gradient_tolerance: float = 1.0e-9  # stage-1 Newton tolerance
    moment_tolerance: float = 1.0e-3  # stage-2 stop: ||a - m(lambda)||
    max_outer: int = 8  # stage-2 Gauss-Newton iterations
    max_newton: int = 50  # stage-1 iterations
    ridge: float = 1.0e-10  # stage-1 Hessian regularization
    fd_phi: float = 0.05  # FD perturbation, in potential sup-norm units
    jac_rcond: float = 3.0e-2  # relative singular cutoff of the moment Jacobian
    delta_phi_max: float = 2.0  # trust region: exact sup-norm change per step
    min_ess_fraction: float = 0.05  # ESS backtracking threshold
    max_backtracks: int = 6

    def build_context(self, level: ConstraintLevel, paths: PathBatch):
        """Build the solver context; passes the constraint knots when the
        backend accepts them (terminal-condition kinks live there)."""
        try:
            return self.solver.build_context(paths, extra_knots=np.asarray(level.knots))
        except TypeError:
            return self.solver.build_context(paths)

    def fit(
        self,
        level: ConstraintLevel,
        targets: np.ndarray,
        paths: PathBatch,
        warm_start: np.ndarray | None = None,
        context: Any = None,
    ) -> DualFitResult:
        if self.solver is None:
            raise ValueError("no BSDE solver configured")
        if level.normalization is None:
            raise ValueError("level must be normalized before calibration")
        d = level.dim
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (d,):
            raise ValueError(f"targets shape {targets.shape} != level dim ({d},)")
        if context is None:
            context = self.build_context(level, paths)

        psi_T = self.family.evaluate_normalized(level, paths.x[:, -1])  # (n_paths, d)
        n_paths = paths.n_paths

        def sup_phi(coeffs: np.ndarray) -> float:
            return LambdaPotential(self.family, level, coeffs).sup_norm_exact()

        def moments_of(lam: np.ndarray) -> tuple[np.ndarray, float, BSDESolution]:
            sol = self._solve(level, lam, paths, context)
            w = _softmax(psi_T @ lam - self._dynamic_offset(sol, paths))
            ess_frac = 1.0 / (float((w**2).sum()) * n_paths)
            return w @ psi_T, ess_frac, sol

        # Stage 1: exact static exponential-family fit (marginal-only tilt).
        if warm_start is None:
            lam = self._static_newton(np.zeros(d), targets, psi_T, np.zeros(n_paths))
        else:
            lam = np.asarray(warm_start, dtype=float).copy()

        # Stage 2: Gauss-Newton on the field-refreshed moment map.
        m, ess, sol = moments_of(lam)
        grad = targets - m
        gnorm = float(np.linalg.norm(grad))
        converged = gnorm < self.moment_tolerance
        n_outer = 0
        status = "stage-1 static fit sufficient" if converged else ""
        while not converged and n_outer < self.max_outer:
            n_outer += 1
            # FD Jacobian dm/dlambda on frozen paths (deterministic solver).
            J = np.empty((d, d))
            for j in range(d):
                e = np.zeros(d)
                e[j] = 1.0
                eps = self.fd_phi / max(sup_phi(e), 1e-12)
                m_j, _, _ = moments_of(lam + eps * e)
                J[:, j] = (m_j - m) / eps
            step = np.linalg.pinv(J, rcond=self.jac_rcond) @ grad
            dphi = sup_phi(step)
            if dphi > self.delta_phi_max:
                step *= self.delta_phi_max / dphi
            # Monotone line search + ESS backtracking: a candidate is
            # accepted only if the moment residual strictly decreases AND
            # the ESS survives. A solver fixed-point failure on the
            # candidate (Picard not self-consistent at that lambda) is a
            # rejected step, not a crash — halve and retry.
            accepted = False
            for _ in range(self.max_backtracks + 1):
                try:
                    m_new, ess_new, sol_new = moments_of(lam + step)
                except ValueError:
                    step *= 0.5
                    continue
                gnorm_new = float(np.linalg.norm(targets - m_new))
                if ess_new >= self.min_ess_fraction and gnorm_new < gnorm:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                status = ("stopped: no residual-decreasing, ESS-preserving "
                          "step (field noise dominates)")
                break
            lam = lam + step
            m, ess, sol = m_new, ess_new, sol_new
            grad = targets - m
            gnorm = float(np.linalg.norm(grad))
            if gnorm < self.moment_tolerance:
                converged = True
        if not status:
            status = (f"gauss-newton {'converged' if converged else 'stopped'} "
                      f"after {n_outer} outer iterations (moment residual {gnorm:.3e})")

        return DualFitResult(
            level=level.n,
            lam=lam,
            dual_value=float(lam @ targets) - sol.y0_sample,
            gradient=grad,
            gradient_norm=gnorm,
            moment_residuals=grad,
            moment_residual_norm=gnorm,
            n_iterations=n_outer,
            converged=converged,
            status=status,
            warm_started=warm_start is not None,
        )

    # -- blocks -------------------------------------------------------------------

    def _dynamic_offset(self, sol: BSDESolution, paths: PathBatch) -> np.ndarray:
        """Pathwise N^S_T from the current martingale field."""
        return (sol.z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1)

    def _static_newton(self, lam0: np.ndarray, targets: np.ndarray,
                       psi_T: np.ndarray, n_s: np.ndarray) -> np.ndarray:
        """Exact Newton on the strictly concave offset exponential family."""
        lam = lam0.copy()
        for _ in range(self.max_newton):
            w = _softmax(psi_T @ lam - n_s)
            mean = w @ psi_T
            grad = targets - mean
            if float(np.linalg.norm(grad)) < self.gradient_tolerance:
                break
            centered = psi_T - mean
            hess = centered.T @ (centered * w[:, None])  # Cov_w(Psi)
            hess[np.diag_indices_from(hess)] += self.ridge
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            # Backtracking on the concave objective.
            g0 = float(lam @ targets) - _log_mean_exp(psi_T @ lam - n_s)
            t = 1.0
            for _ in range(30):
                cand = lam + t * step
                g1 = float(cand @ targets) - _log_mean_exp(psi_T @ cand - n_s)
                if g1 >= g0 + 1e-4 * t * float(grad @ step):
                    break
                t *= 0.5
            lam = lam + t * step
        return lam

    # -- solver plumbing -------------------------------------------------------------

    def solve_at(self, level: ConstraintLevel, lam: np.ndarray, paths: PathBatch,
                 context: Any = None) -> BSDESolution:
        """Public BSDE solve at fixed multipliers (certificates, diagnostics)."""
        if context is None:
            context = self.build_context(level, paths)
        return self._solve(level, lam, paths, context)

    def _solve(self, level, lam, paths, context) -> BSDESolution:
        pot = LambdaPotential(self.family, level, np.asarray(lam, dtype=float))
        return self.solver.solve(paths, pot, context)


def _softmax(a: np.ndarray) -> np.ndarray:
    w = np.exp(a - a.max())
    return w / w.sum()


def _log_mean_exp(a: np.ndarray) -> float:
    m = float(a.max())
    return m + float(np.log(np.mean(np.exp(a - m))))

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
    moment_tolerance: float = 1.0e-3  # stage-2 stop: ||FOC residual||
    # Sobolev regularization strength sigma^2 (M2, D12). The regularized
    # dual is D^sigma(lambda) = lambda^T a - Y_0^sample - sigma^2/2 *
    # lambda^T S lambda with S the exact H^1 Gram of the normalized
    # family; the stage-2 root problem becomes the regularized FOC
    #     a - m(lambda) - sigma^2 S lambda = 0.
    # Zero disables it exactly (M1 behavior). The *raw* moment residual
    # a - m is still reported separately — the duality-certificate defect
    # decomposition quantifies what the regularization gave up:
    # moment_defect ~ -sigma^2 lambda^T S lambda at the optimum.
    sobolev_sigma2: float = 0.0
    # Jacobian backend (M2, D12): "implicit" differentiates the Picard
    # fixed point (exact on the sample modulo frozen clip/cap sets, one
    # tangent solve for all directions — an order of magnitude faster
    # than FD and free of finite-difference noise in the weakly
    # identifiable directions); "fd" keeps the forward-difference probe
    # as the cross-check backend.
    jacobian: str = "implicit"
    max_outer: int = 8  # stage-2 Gauss-Newton iterations
    max_newton: int = 50  # stage-1 iterations
    ridge: float = 1.0e-10  # stage-1 Hessian regularization
    fd_phi: float = 0.05  # FD perturbation, in potential sup-norm units
    # Identifiability floor for singular values of the reduced Jacobian,
    # in *natural units* (the normalized basis is orthonormal on the
    # pilot sample, so the static curvature is the identity and singular
    # values of dm/dlambda measure how much of a static unit response
    # survives dynamic cancellation). Directions below the floor are
    # declared unidentifiable at this sample size — the §10.3
    # near-replicable directions — and convergence is asserted on the
    # identifiable subspace, with the unidentifiable remainder reported,
    # never chased. NOTE: absolute, not relative to the largest singular
    # value — one strongly hedged direction can have |dm/dlambda| >> 1
    # and a relative cutoff would silently discard everything else.
    jac_rcond: float = 3.0e-2
    lm_mu0: float = 1.0e-2  # initial Levenberg-Marquardt damping
    # Dual-ascent acceptance slack: a candidate must also not DECREASE
    # the (regularized) sample dual by more than this. FOC-residual
    # decrease alone is not enough — at large deformations the noisy
    # fields underestimate Y_0^sample, the sample dual is overestimated
    # far from the origin, and a monotone-FOC trail can walk into a
    # spurious large-entropy basin (observed at fine levels). The dual
    # is the quantity being maximized; require it to behave like one.
    dual_slack: float = 1.0e-4
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
        sigma2 = self.sobolev_sigma2
        s_gram = self.family.sobolev_gram(level) if sigma2 > 0.0 else np.zeros((d, d))

        def sup_phi(coeffs: np.ndarray) -> float:
            return LambdaPotential(self.family, level, coeffs).sup_norm_exact()

        def moments_of(lam: np.ndarray) -> tuple[np.ndarray, float, BSDESolution]:
            sol = self._solve(level, lam, paths, context)
            w = _softmax(psi_T @ lam - self._dynamic_offset(sol, paths))
            ess_frac = 1.0 / (float((w**2).sum()) * n_paths)
            return w @ psi_T, ess_frac, sol

        def foc(lam: np.ndarray, m: np.ndarray) -> np.ndarray:
            """Gradient of the (possibly regularized) sample dual."""
            return targets - m - sigma2 * (s_gram @ lam)

        def dual_value(lam: np.ndarray, sol: BSDESolution) -> float:
            """(Regularized) sample dual D^sigma(lambda)."""
            return (float(lam @ targets) - sol.y0_sample
                    - 0.5 * sigma2 * float(lam @ (s_gram @ lam)))

        def implicit_jacobian(lam: np.ndarray, m: np.ndarray,
                              sol: BSDESolution) -> np.ndarray:
            """Exact-on-sample dm/dlambda at the current solution:
            J[k,i] = Cov_w(psi_k, psi_i - dN^S_i)."""
            pot = LambdaPotential(self.family, level, lam)
            dns = self.solver.dn_s_dlam(paths, pot, psi_T, context, sol)
            dl = psi_T - dns
            wv = _softmax(psi_T @ lam - self._dynamic_offset(sol, paths))
            return psi_T.T @ (dl * wv[:, None]) - np.outer(m, wv @ dl)

        # Stage 1: exact static exponential-family fit (marginal-only tilt),
        # restricted to the identifiable subspace of the reduced Jacobian
        # at the starting point. The restriction is not cosmetic: the
        # static problem sees near-replicable directions as perfectly
        # identifiable (full static curvature), so the unrestricted fit
        # loads them with multipliers amplified by 1/(1 - gamma) — pure
        # gauge at this level's resolution (the hedge cancels them; the
        # law barely moves), but *poison* as a warm start, because gauge
        # is resolution-dependent and the finer level no longer cancels
        # it (observed: a clean H ~ 1e-3 level embedded into H ~ 1.9
        # garbage). Restricting the static step keeps every returned
        # multiplier a minimal-gauge representative by construction.
        if warm_start is None:
            lam0 = np.zeros(d)
            sol_ref = self._solve(level, lam0, paths, context)
            n_s_ref = self._dynamic_offset(sol_ref, paths)
            dual0 = 0.0  # zero potential: Y_0^sample = 0 exactly
        else:
            # Warm-started levels still need stage 1 for the *new*
            # directions (embedded coefficients are zero there): freeze the
            # dynamic offset N^S at the warm start and run the restricted
            # static Newton on the offset exponential family from it.
            lam0 = np.asarray(warm_start, dtype=float).copy()
            sol_ref = self._solve(level, lam0, paths, context)
            n_s_ref = self._dynamic_offset(sol_ref, paths)
            dual0 = dual_value(lam0, sol_ref)
        ident_proj = None
        if self.jacobian == "implicit":
            w_ref = _softmax(psi_T @ lam0 - n_s_ref)
            m_ref = w_ref @ psi_T
            j0 = implicit_jacobian(lam0, m_ref, sol_ref)
            _, s0, vt0 = np.linalg.svd(j0)
            v_keep = vt0[s0 > self.jac_rcond].T
            ident_proj = v_keep @ v_keep.T
        lam = self._static_newton(lam0, targets, psi_T, n_s_ref,
                                  s_gram, sigma2, ident_proj)
        # Belt over the braces: exact sup-norm trust region on the whole
        # static move, then backtrack toward the start until the Picard
        # fixed point holds, the ESS survives AND the sample dual has not
        # decreased below the starting value.
        dphi0 = sup_phi(lam - lam0)
        if dphi0 > self.delta_phi_max:
            lam = lam0 + (lam - lam0) * (self.delta_phi_max / dphi0)
        for _ in range(self.max_backtracks + 1):
            try:
                m, ess, sol = moments_of(lam)
            except ValueError:
                lam = lam0 + 0.5 * (lam - lam0)
                continue
            if (ess >= self.min_ess_fraction
                    and dual_value(lam, sol) >= dual0 - self.dual_slack):
                break
            lam = lam0 + 0.5 * (lam - lam0)
        else:
            lam = lam0
            m, ess, sol = moments_of(lam)

        # Stage 2: Levenberg-Marquardt on the (regularized) FOC of the
        # field-refreshed moment map: a - m(lambda) - sigma^2 S lambda = 0.
        # LM replaces the truncated pseudo-inverse: weakly identifiable
        # directions (near-replicable, gamma -> 1) are *damped*, not cut,
        # and the damping adapts to the strong nonlinearity the dynamic
        # feedback induces along them.
        def ident_split(J: np.ndarray, r: np.ndarray) -> tuple[float, float, int]:
            """Residual norms on/off the identifiable subspace of the
            reduced Jacobian (singular values above the absolute floor)."""
            u, s, _ = np.linalg.svd(J)
            keep = s > self.jac_rcond
            r_id = u[:, keep] @ (u[:, keep].T @ r)
            return (float(np.linalg.norm(r_id)),
                    float(np.linalg.norm(r - r_id)), int(keep.sum()))

        grad = foc(lam, m)
        gnorm = float(np.linalg.norm(grad))
        converged = gnorm < self.moment_tolerance
        ident_note = ""
        n_outer = 0
        status = "stage-1 static fit sufficient" if converged else ""
        J = None
        mu = self.lm_mu0
        dual_cur = dual_value(lam, sol)
        while not converged and n_outer < self.max_outer:
            n_outer += 1
            if self.jacobian == "implicit":
                # Implicit differentiation of the Picard fixed point.
                J = implicit_jacobian(lam, m, sol)
            else:
                # FD Jacobian dm/dlambda on frozen paths (deterministic solver).
                J = np.empty((d, d))
                for j in range(d):
                    e = np.zeros(d)
                    e[j] = 1.0
                    eps = self.fd_phi / max(sup_phi(e), 1e-12)
                    m_j, _, _ = moments_of(lam + eps * e)
                    J[:, j] = (m_j - m) / eps
            A = J + sigma2 * s_gram  # FOC Jacobian (up to sign)
            u_a, s_a, vt_a = np.linalg.svd(A)
            keep_a = s_a > self.jac_rcond
            utr = u_a[:, keep_a].T @ grad
            # LM inner loop restricted to the identifiable subspace: the
            # step is damped where the reduced Jacobian is informative and
            # *zero* along unidentifiable directions — moving there is a
            # random walk on field noise and, left free, LM will follow a
            # monotone FOC trail into a spurious large-entropy basin
            # (observed at fine levels). A candidate is accepted only if
            # the FOC residual strictly decreases AND the ESS survives;
            # failures (including Picard fixed-point failures at the
            # candidate) raise the damping and retry.
            accepted = False
            for _ in range(self.max_backtracks + 1):
                step = vt_a[keep_a].T @ (s_a[keep_a] / (s_a[keep_a] ** 2 + mu) * utr)
                dphi = sup_phi(step)
                if dphi > self.delta_phi_max:
                    step *= self.delta_phi_max / dphi
                try:
                    m_new, ess_new, sol_new = moments_of(lam + step)
                except ValueError:
                    mu *= 4.0
                    continue
                gnorm_new = float(np.linalg.norm(foc(lam + step, m_new)))
                dual_new = dual_value(lam + step, sol_new)
                if (ess_new >= self.min_ess_fraction and gnorm_new < gnorm
                        and dual_new >= dual_cur - self.dual_slack):
                    accepted = True
                    mu = max(mu / 3.0, 1e-8)
                    break
                mu *= 4.0
            if not accepted:
                r_id, r_un, rank = ident_split(J, grad)
                status = ("stopped: no dual-ascending, residual-decreasing, "
                          f"ESS-preserving LM step (identifiable residual {r_id:.3e}, "
                          f"unidentifiable {r_un:.3e}, rank {rank}/{d})")
                break
            lam = lam + step
            m, ess, sol = m_new, ess_new, sol_new
            dual_cur = dual_new
            grad = foc(lam, m)
            gnorm = float(np.linalg.norm(grad))
            if gnorm < self.moment_tolerance:
                converged = True
                break
            # Convergence on the identifiable subspace: the remainder
            # lives where the reduced Jacobian says this sample cannot
            # distinguish laws — report it, never chase it (§10.3).
            r_id, r_un, rank = ident_split(J, grad)
            if r_id < self.moment_tolerance:
                converged = True
                ident_note = (f" in identifiable subspace (rank {rank}/{d}, "
                              f"unidentifiable residual {r_un:.3e})")
                break
        if not status:
            status = (f"levenberg-marquardt {'converged' if converged else 'stopped'}"
                      f"{ident_note} after {n_outer} outer iterations "
                      f"(foc residual {gnorm:.3e})")

        # Reduced Jacobian at the returned multipliers, for the §10.3
        # Schur/conditioning certificate. Refresh it at the final point
        # (steps may have moved lambda since the last one was formed).
        if self.jacobian == "implicit":
            J = implicit_jacobian(lam, m, sol)

        moment_res = targets - m
        return DualFitResult(
            level=level.n,
            lam=lam,
            dual_value=float(lam @ targets) - sol.y0_sample,
            gradient=grad,  # regularized-dual gradient (FOC residual)
            gradient_norm=gnorm,
            moment_residuals=moment_res,  # raw a - m, always reported
            moment_residual_norm=float(np.linalg.norm(moment_res)),
            n_iterations=n_outer,
            converged=converged,
            status=status,
            warm_started=warm_start is not None,
            reduced_jacobian=J,
        )

    # -- blocks -------------------------------------------------------------------

    def _dynamic_offset(self, sol: BSDESolution, paths: PathBatch) -> np.ndarray:
        """Pathwise N^S_T from the current martingale field."""
        return (sol.z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1)

    def _static_newton(self, lam0: np.ndarray, targets: np.ndarray,
                       psi_T: np.ndarray, n_s: np.ndarray,
                       s_gram: np.ndarray, sigma2: float,
                       ident_proj: np.ndarray | None = None) -> np.ndarray:
        """Exact Newton on the strictly concave offset exponential family,
        Sobolev-regularized when sigma2 > 0. With ``ident_proj`` (the
        projector onto the identifiable subspace of the reduced Jacobian
        at the starting point) every step — and the convergence check —
        is restricted to that subspace: the static stage must never load
        gauge directions the dynamics will cancel."""
        lam = lam0.copy()
        for _ in range(self.max_newton):
            w = _softmax(psi_T @ lam - n_s)
            mean = w @ psi_T
            grad = targets - mean - sigma2 * (s_gram @ lam)
            if ident_proj is not None:
                grad = ident_proj @ grad
            if float(np.linalg.norm(grad)) < self.gradient_tolerance:
                break
            centered = psi_T - mean
            hess = centered.T @ (centered * w[:, None])  # Cov_w(Psi)
            hess += sigma2 * s_gram
            hess[np.diag_indices_from(hess)] += self.ridge
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            if ident_proj is not None:
                step = ident_proj @ step
            # Backtracking on the concave objective.
            g0 = (float(lam @ targets) - _log_mean_exp(psi_T @ lam - n_s)
                  - 0.5 * sigma2 * float(lam @ (s_gram @ lam)))
            t = 1.0
            for _ in range(30):
                cand = lam + t * step
                g1 = (float(cand @ targets) - _log_mean_exp(psi_T @ cand - n_s)
                      - 0.5 * sigma2 * float(cand @ (s_gram @ cand)))
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

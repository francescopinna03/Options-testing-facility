"""CertificateBundle construction (arch doc §12).

Every level of the projective sequence emits one immutable bundle. A
successful fit without a certificate bundle is not a valid research
result; the decisive first table of the paper is the projective
certificate table, not an OOS leaderboard (§21).

Estimates:

* duality (§12.1): primal entropy H^LR against the dual value
  lambda^T a - Y_0^sample, with the exact sample decomposition of the
  signed gap for approximately calibrated laws:
  H - D = lambda^T (m - a) - E^Q[N^S_T]. The identity residual must sit
  at floating-point noise; the two defects say whether a nonzero gap is
  primal inadmissibility (moments not met, dynamic term not centered) or
  a genuine violation.
* entropy identity (§12.2): H^LR (likelihood route) vs H^EN
  (characteristic-energy route).
* martingale (§12.4): terminal forward error and the projector residual.
* projective Cauchy (§12.3): for consecutive levels n_prev < n,
  KL(Q_n | Q_{n_prev}) <= H_n - H_{n_prev} and
  TV <= sqrt((H_n - H_{n_prev})/2); the direct KL estimate uses the two
  weight vectors on the common paths, and the signed slack is decomposed
  against the coarse potential (moment term minus dynamic term) so
  nesting error, moment-calibration error, dynamic-term error and true
  Pythagorean-identity violations are distinguishable.
* conditioning (§10.3): eigenvalues of the weighted moment covariance —
  the *raw* Jacobian, a conditioning diagnostic only (the reduced Schur
  complement after eliminating dynamic multipliers is a later layer).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otf.ssfv.posterior.reweight import ReweightedPosterior
from otf.ssfv.types import (
    BSDESolution,
    CertificateBundle,
    ConditioningCertificate,
    DualFitResult,
    DualityCertificate,
    EntropyCertificate,
    MartingaleCertificate,
    PathBatch,
    ProjectiveCertificate,
)

__all__ = ["build_certificate_bundle", "direct_kl", "PreviousLevelData"]


def direct_kl(post_fine: ReweightedPosterior, post_coarse: ReweightedPosterior) -> float:
    """KL(Q_fine | Q_coarse) estimated on the common prior paths."""
    wf = np.maximum(post_fine.weights(), 1e-300)
    wc = np.maximum(post_coarse.weights(), 1e-300)
    return float(wf @ (np.log(wf) - np.log(wc)))


@dataclass(frozen=True)
class PreviousLevelData:
    """What the Cauchy decomposition needs from the coarser level, all on
    the same frozen path batch: its posterior, multipliers, normalized
    basis at the terminal, and the pathwise semistatic gain N^{S,n}_T."""

    n: int
    h_lr: float
    posterior: ReweightedPosterior
    lam: np.ndarray
    psi_normalized: np.ndarray  # (n_paths, d_n)
    semistatic_gain: np.ndarray  # (n_paths,)


def build_certificate_bundle(
    paths: PathBatch,
    solution: BSDESolution,
    fit: DualFitResult,
    psi_normalized: np.ndarray,
    targets: np.ndarray,
    previous: PreviousLevelData | None = None,
    identifiable_sv_floor: float = 3.0e-2,
) -> tuple[CertificateBundle, ReweightedPosterior]:
    """Assemble the §12 bundle for one calibrated level.

    ``previous`` carries the preceding projective level on the same path
    batch, enabling the Cauchy certificate; None for the first level.
    ``identifiable_sv_floor`` is the absolute singular-value floor (in
    static-curvature units) below which reduced directions are declared
    unidentifiable — pass the calibrator's ``jac_rcond`` so the
    certificate and the optimizer agree on the split.
    """
    post = ReweightedPosterior(paths, solution)
    h_lr = post.entropy_lr()
    h_en = post.entropy_en()
    w = post.weights()
    lam = np.asarray(fit.lam, dtype=float)

    # E^{Q_n}[N^S_T]: the semistatic gain is a Q_n-martingale increment sum
    # (Girsanov moves only W^perp in the diffusion sector, so W^S stays a
    # Q_n-Brownian motion) — its posterior mean must vanish up to MC error.
    n_s_terminal = (solution.z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1)

    # Exact sample decomposition of the signed gap (see DualityCertificate):
    # H - D = lambda^T (m - a) - E^Q[N^S_T], term by term on the same
    # weight vector, so the identity residual is a pure consistency check.
    m = w @ psi_normalized
    gap = h_lr - fit.dual_value
    moment_defect = float(lam @ (m - np.asarray(targets, dtype=float)))
    semistatic_defect = -float(w @ n_s_terminal)
    duality = DualityCertificate(
        primal_entropy=h_lr,
        dual_value=fit.dual_value,
        gap=gap,
        moment_residual_norm=fit.moment_residual_norm,
        moment_defect=moment_defect,
        semistatic_defect=semistatic_defect,
        duality_identity_residual=gap - (moment_defect + semistatic_defect),
    )
    entropy = EntropyCertificate(h_lr=h_lr, h_en=h_en, discrepancy=abs(h_lr - h_en))

    proj = solution.projector
    martingale = MartingaleCertificate(
        forward_error=post.forward_error(),
        max_conditional_drift_residual=float("nan"),  # multi-horizon layer (M3)
        projector_residual=proj.max_root_residual if proj is not None else float("nan"),
    )

    if previous is None:
        projective = ProjectiveCertificate(
            n_prev=None, h_n=h_lr, delta_h=None, kl_direct=None, tv_bound=None,
            cauchy_slack=None, cauchy_moment_term=None,
            cauchy_semistatic_term=None, cauchy_identity_residual=None,
        )
    else:
        delta_h = h_lr - previous.h_lr
        kl = direct_kl(post, previous.posterior)
        slack = delta_h - kl
        # Exact sample decomposition of the signed Cauchy slack against
        # the coarse potential (see ProjectiveCertificate):
        # slack = lambda_n^T (m_N^{(n)} - m_n)
        #         - (E^{Q_N}[N^{S,n}] - E^{Q_n}[N^{S,n}]).
        w_coarse = previous.posterior.weights()
        m_fine_on_coarse = w @ previous.psi_normalized
        m_coarse = w_coarse @ previous.psi_normalized
        moment_term = float(np.asarray(previous.lam) @ (m_fine_on_coarse - m_coarse))
        semistatic_term = float(w @ previous.semistatic_gain
                                - w_coarse @ previous.semistatic_gain)
        projective = ProjectiveCertificate(
            n_prev=previous.n,
            h_n=h_lr,
            delta_h=delta_h,
            kl_direct=kl,
            tv_bound=float(np.sqrt(max(delta_h, 0.0) / 2.0)),
            # Signed slack of the Cauchy inequality: >= 0 by the theorem
            # for exactly calibrated laws; for approximate fits the
            # decomposition below says which defect is responsible.
            cauchy_slack=slack,
            cauchy_moment_term=moment_term,
            cauchy_semistatic_term=semistatic_term,
            cauchy_identity_residual=slack - (moment_term - semistatic_term),
        )

    centered = psi_normalized - m
    jac = centered.T @ (centered * w[:, None])
    eig = np.linalg.eigvalsh(jac)
    # True reduced (Schur) spectrum when the fit carries the
    # implicit-differentiation Jacobian dm/dlambda: dynamic cancellation
    # included, so near-zero singular values ARE the gamma -> 1
    # replicable directions. The FOC residual is split on the same
    # threshold the optimizer used.
    red_kw: dict = {}
    if fit.reduced_jacobian is not None:
        u, s, _ = np.linalg.svd(np.asarray(fit.reduced_jacobian))
        keep = s > identifiable_sv_floor
        r = np.asarray(fit.gradient, dtype=float)
        r_id = u[:, keep] @ (u[:, keep].T @ r)
        red_kw = {
            "reduced_sv_min": float(s[-1]),
            "reduced_sv_max": float(s[0]),
            "identifiable_dim": int(keep.sum()),
            "identifiable_residual_norm": float(np.linalg.norm(r_id)),
            "unidentifiable_residual_norm": float(np.linalg.norm(r - r_id)),
        }
    conditioning = ConditioningCertificate(
        eigen_min=float(eig[0]),
        eigen_max=float(eig[-1]),
        condition_number=float(eig[-1] / max(eig[0], 1e-300)),
        n_removed_directions=psi_normalized.shape[1] - int((eig > 1e-12 * eig[-1]).sum()),
        **red_kw,
    )

    diagnostics = {
        "ess_fraction": post.ess_fraction(),
        "max_weight_share": post.max_weight_share(),
        "y0": solution.y0,
        "y0_sample": solution.y0_sample,
        "posterior_mean_semistatic_gain": -semistatic_defect,
        **{f"solver_{k}": float(v) for k, v in solution.residuals.items()},
    }
    bundle = CertificateBundle(
        level=fit.level,
        duality=duality,
        entropy=entropy,
        martingale=martingale,
        projective=projective,
        conditioning=conditioning,
        diagnostics=diagnostics,
    )
    return bundle, post

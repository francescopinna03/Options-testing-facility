"""CertificateBundle construction (arch doc §12).

Every level of the projective sequence emits one immutable bundle. A
successful fit without a certificate bundle is not a valid research
result; the decisive first table of the paper is the projective
certificate table, not an OOS leaderboard (§21).

Estimates:

* duality (§12.1): primal entropy H^LR against the dual value
  lambda^T a - Y_0^sample; their gap plus the moment residual.
* entropy identity (§12.2): H^LR (likelihood route) vs H^EN
  (characteristic-energy route).
* martingale (§12.4): terminal forward error and the projector residual.
* projective Cauchy (§12.3): for consecutive levels n_prev < n,
  KL(Q_n | Q_{n_prev}) <= H_n - H_{n_prev} and
  TV <= sqrt((H_n - H_{n_prev})/2); the direct KL estimate uses the two
  weight vectors on the common paths.
* conditioning (§10.3): eigenvalues of the weighted moment covariance —
  the *raw* Jacobian, a conditioning diagnostic only (the reduced Schur
  complement after eliminating dynamic multipliers is a later layer).
"""

from __future__ import annotations

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

__all__ = ["build_certificate_bundle", "direct_kl"]


def direct_kl(post_fine: ReweightedPosterior, post_coarse: ReweightedPosterior) -> float:
    """KL(Q_fine | Q_coarse) estimated on the common prior paths."""
    n = post_fine.paths.n_paths
    wf = np.maximum(post_fine.weights(), 1e-300)
    wc = np.maximum(post_coarse.weights(), 1e-300)
    return float(wf @ (np.log(wf) - np.log(wc)))


def build_certificate_bundle(
    paths: PathBatch,
    solution: BSDESolution,
    fit: DualFitResult,
    psi_normalized: np.ndarray,
    previous: tuple[int, float, ReweightedPosterior] | None = None,
) -> tuple[CertificateBundle, ReweightedPosterior]:
    """Assemble the §12 bundle for one calibrated level.

    ``previous`` is (n_prev, H_prev, posterior_prev) of the preceding
    projective level on the same path batch, enabling the Cauchy
    certificate; None for the first level.
    """
    post = ReweightedPosterior(paths, solution)
    h_lr = post.entropy_lr()
    h_en = post.entropy_en()

    duality = DualityCertificate(
        primal_entropy=h_lr,
        dual_value=fit.dual_value,
        gap=abs(h_lr - fit.dual_value),
        moment_residual_norm=fit.moment_residual_norm,
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
        )
    else:
        n_prev, h_prev, post_prev = previous
        delta_h = h_lr - h_prev
        projective = ProjectiveCertificate(
            n_prev=n_prev,
            h_n=h_lr,
            delta_h=delta_h,
            kl_direct=direct_kl(post, post_prev),
            tv_bound=float(np.sqrt(max(delta_h, 0.0) / 2.0)),
        )

    w = post.weights()
    mean = w @ psi_normalized
    centered = psi_normalized - mean
    jac = centered.T @ (centered * w[:, None])
    eig = np.linalg.eigvalsh(jac)
    conditioning = ConditioningCertificate(
        eigen_min=float(eig[0]),
        eigen_max=float(eig[-1]),
        condition_number=float(eig[-1] / max(eig[0], 1e-300)),
        n_removed_directions=psi_normalized.shape[1] - int((eig > 1e-12 * eig[-1]).sum()),
    )

    diagnostics = {
        "ess_fraction": post.ess_fraction(),
        "max_weight_share": post.max_weight_share(),
        "y0": solution.y0,
        "y0_sample": solution.y0_sample,
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

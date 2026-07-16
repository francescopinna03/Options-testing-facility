"""Projective refinement sequence (arch doc §10.2, §12.3; DECISIONS.md D6).

Runs the nested levels n_0 < n_1 < ... on one frozen path batch:

* each level is normalized on the same prior sample and calibrated by the
  alternating finite-dual calibrator;
* warm starts embed the previous level's multipliers exactly
  (ConstraintFamily.embed) with new coefficients at zero;
* every level emits a CertificateBundle; consecutive levels feed the
  projective Cauchy certificate KL(Q_N|Q_n) <= H_N - H_n.

The observable plateau of H_n across levels is the main convergence
certificate of the whole construction (§12.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from otf.ssfv.certificates.bundle import PreviousLevelData, build_certificate_bundle
from otf.ssfv.dual.calibrator import ReducedMomentMapCalibrator
from otf.ssfv.posterior.reweight import ReweightedPosterior
from otf.ssfv.types import CertificateBundle, ConstraintLevel, DualFitResult, PathBatch

__all__ = ["ProjectiveSequence", "LevelRun"]


@dataclass(frozen=True)
class LevelRun:
    level: ConstraintLevel
    fit: DualFitResult
    bundle: CertificateBundle
    posterior: ReweightedPosterior


@dataclass(frozen=True)
class ProjectiveSequence:
    calibrator: ReducedMomentMapCalibrator

    def run(
        self,
        paths: PathBatch,
        levels: Sequence[int],
        k_market_sample: np.ndarray | None = None,
        targets_fn=None,
        normalization_seed: int = 0,
    ) -> list[LevelRun]:
        """Targets come either from a terminal market/DGP sample
        (``k_market_sample``) or from ``targets_fn(level) -> ndarray``
        (e.g. weighted moments of a known-potential posterior on the same
        paths — arch doc §14.1 DGP 2)."""
        if (k_market_sample is None) == (targets_fn is None):
            raise ValueError("provide exactly one of k_market_sample / targets_fn")
        family = self.calibrator.family
        runs: list[LevelRun] = []
        prev_level: ConstraintLevel | None = None
        prev_lam: np.ndarray | None = None
        prev_cert: PreviousLevelData | None = None

        for n in levels:
            # Threading `previous` inherits the coarser level's transform
            # verbatim (nested normalization plan): C_{n+1} ⊆ C_n is then
            # structural, which is what the Cauchy certificate needs.
            level = family.normalize(family.level(n), paths.x[:, -1],
                                     normalization_seed=normalization_seed,
                                     previous=prev_level)
            if targets_fn is not None:
                targets = np.asarray(targets_fn(level), dtype=float)
            else:
                targets = family.targets_from_sample(level, k_market_sample)

            warm = None
            if prev_level is not None and prev_lam is not None:
                warm = family.embed(prev_lam, prev_level, level)

            context = self.calibrator.build_context(level, paths)
            fit = self.calibrator.fit(level, targets, paths,
                                      warm_start=warm, context=context)
            sol = self.calibrator.solve_at(level, fit.lam, paths, context)
            psi = family.evaluate_normalized(level, paths.x[:, -1])
            bundle, post = build_certificate_bundle(
                paths, sol, fit, psi, targets, previous=prev_cert,
                identifiable_sv_floor=self.calibrator.jac_rcond,
            )

            runs.append(LevelRun(level=level, fit=fit, bundle=bundle, posterior=post))
            prev_level, prev_lam = level, fit.lam
            prev_cert = PreviousLevelData(
                n=n, h_lr=bundle.entropy.h_lr, posterior=post, lam=fit.lam,
                psi_normalized=psi,
                semistatic_gain=(sol.z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1),
            )
        return runs

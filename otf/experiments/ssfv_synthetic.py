"""M1 synthetic experiment: projective SSFV calibration with certificates.

Runs the DGP-2 program of the architecture document (§14.1): a known
bounded cylindrical potential defines the synthetic posterior; the
projective sequence recalibrates it level by level and every level emits a
CertificateBundle. All artifacts (manifest, per-level certificates,
fit summaries) land in a self-contained experiment directory — paper
tables are generated from these files, never from notebook state (§12.6).

Usage:
    python -m otf.experiments.ssfv_synthetic --out runs/ssfv_synth_001 \
        [--paths 8192] [--steps 32] [--horizon 0.5] [--seed 1729] [--levels 0 1]

Requires the ``numerical`` extra (numpy).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from otf.ssfv.bsde.picard import PicardHopfColeSolver
from otf.ssfv.config import ExperimentConfig, PriorConfig, SimulationConfig, derive_seed
from otf.ssfv.constraints.hat_family import LambdaPotential, NestedHatFamily
from otf.ssfv.dual.calibrator import AlternatingDualCalibrator
from otf.ssfv.dual.projective_sequence import ProjectiveSequence
from otf.ssfv.posterior.reweight import ReweightedPosterior
from otf.ssfv.prior.heston import HestonPrior


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paths", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--horizon", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--potential-scale", type=float, default=0.5)
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        prior=PriorConfig(),
        simulation=SimulationConfig(n_paths=args.paths, horizon=args.horizon,
                                    seed=args.seed),
    )
    prior = HestonPrior()
    paths = prior.simulate(args.paths, args.steps, args.horizon,
                           seed=derive_seed(args.seed, "prior"))

    family = NestedHatFamily(k_min=-0.5, k_max=0.5, base_dim=4)
    solver = PicardHopfColeSolver().for_prior(prior)
    cal = AlternatingDualCalibrator(family, solver=solver)

    # DGP 2: known potential at the finest requested level.
    fine = family.normalize(family.level(max(args.levels)), paths.x[:, -1])
    nm = fine.normalization
    kept = list(nm.kept_indices)
    colbound = (1.0 + np.abs(nm.means[kept])) / nm.stds[kept]
    rng = np.random.default_rng(derive_seed(args.seed, "potential"))
    lam_star = rng.normal(0.0, args.potential_scale, fine.dim) / colbound
    ctx_fine = cal.build_context(fine, paths)
    sol_star = solver.solve(paths, LambdaPotential(family, fine, lam_star), ctx_fine)
    post_star = ReweightedPosterior(paths, sol_star)

    def targets_fn(level):
        psi = family.evaluate_normalized(level, paths.x[:, -1])
        return post_star.weights() @ psi

    runs = ProjectiveSequence(cal).run(paths, args.levels, targets_fn=targets_fn)

    manifest = config.manifest(extra={
        "experiment": "ssfv_synthetic_dgp2",
        "n_steps": args.steps,
        "levels": args.levels,
        "path_batch_hash": paths.batch_hash,
        "dgp_lambda_star": lam_star.tolist(),
        "dgp_entropy_lr": post_star.entropy_lr(),
    })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    table = []
    for r in runs:
        (out / f"certificates_level_{r.bundle.level}.json").write_text(r.bundle.to_json())
        table.append({
            "level": r.bundle.level,
            "dim": r.level.dim,
            "converged": r.fit.converged,
            "moment_residual": r.fit.moment_residual_norm,
            "H_lr": r.bundle.entropy.h_lr,
            "H_en": r.bundle.entropy.h_en,
            "entropy_discrepancy": r.bundle.entropy.discrepancy,
            "duality_gap": r.bundle.duality.gap,
            "forward_error": r.bundle.martingale.forward_error,
            "kl_direct": r.bundle.projective.kl_direct,
            "delta_h": r.bundle.projective.delta_h,
            "ess_fraction": r.bundle.diagnostics["ess_fraction"],
        })
    (out / "refinement_table.json").write_text(json.dumps(table, indent=2))

    for row in table:
        print(" ".join(f"{k}={v}" for k, v in row.items()))
    print(f"artifacts written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

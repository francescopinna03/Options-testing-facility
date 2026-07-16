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
from otf.ssfv.dual.calibrator import ReducedMomentMapCalibrator
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
    ap.add_argument("--dt-table", type=int, nargs="+", default=None,
                    metavar="STEPS",
                    help="also emit dt_convergence_table.json across these "
                         "step counts (same horizon, same DGP potential)")
    ap.add_argument("--sobolev-sigma2", type=float, default=0.0,
                    help="Sobolev regularization strength sigma^2 (M2, D12)")
    ap.add_argument("--jacobian", choices=["implicit", "fd"], default="implicit",
                    help="stage-2 Jacobian backend (implicit fixed-point "
                         "differentiation, or finite differences)")
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
    cal = ReducedMomentMapCalibrator(family, solver=solver,
                                     sobolev_sigma2=args.sobolev_sigma2,
                                     jacobian=args.jacobian)

    # DGP 2: known potential at the finest requested level, rescaled so
    # sup|Phi*| equals --potential-scale exactly.
    fine = family.normalize(family.level(max(args.levels)), paths.x[:, -1])
    rng = np.random.default_rng(derive_seed(args.seed, "potential"))
    lam_star = rng.normal(0.0, 1.0, fine.dim)
    sup0 = LambdaPotential(family, fine, lam_star).sup_norm_exact()
    lam_star *= args.potential_scale / max(sup0, 1e-12)
    ctx_fine = cal.build_context(fine, paths)
    sol_star = solver.solve(paths, LambdaPotential(family, fine, lam_star), ctx_fine)
    post_star = ReweightedPosterior(paths, sol_star)

    def targets_fn(level):
        psi = family.evaluate_normalized(level, paths.x[:, -1])
        return post_star.weights() @ psi

    runs = ProjectiveSequence(cal).run(paths, args.levels, targets_fn=targets_fn)

    manifest = config.manifest(
        # The concrete components that actually ran, serialized field by
        # field: the manifest must describe the executed algorithm, never
        # a config default that a code path ignored.
        components={"prior": prior, "family": family, "solver": solver,
                    "calibrator": cal},
        extra={
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
            "cauchy_slack": r.bundle.projective.cauchy_slack,
            "posterior_mean_semistatic_gain": r.bundle.diagnostics["posterior_mean_semistatic_gain"],
            "ess_fraction": r.bundle.diagnostics["ess_fraction"],
        })
    (out / "refinement_table.json").write_text(json.dumps(table, indent=2))

    for row in table:
        print(" ".join(f"{k}={v}" for k, v in row.items()))

    if args.dt_table:
        # dt-refinement of the first-order score discretization gs/ehc.
        # Mean-type gaps carry the systematic (discretization) bias;
        # variance-type metrics accumulate per-step estimation noise and
        # are expected to GROW with the step count at fixed N — both are
        # recorded, labeled, so the table cannot be misread.
        pot = LambdaPotential(family, fine, lam_star)
        dt_rows = []
        for steps in sorted(args.dt_table):
            p = prior.simulate(args.paths, steps, args.horizon,
                               seed=derive_seed(args.seed, "prior"))
            ctx = solver.build_context(p, extra_knots=np.asarray(fine.knots))
            sol = solver.solve(p, pot, ctx)
            post = ReweightedPosterior(p, sol)
            dt = args.horizon / steps
            zp = sol.z[:, :, 1]
            log_l_en = (zp * p.d_w[:, :, 1]).sum(axis=1) - 0.5 * (zp**2).sum(axis=1) * dt
            raw_lr = (pot.terminal_value(p.x[:, -1])
                      - (sol.z[:, :, 0] * p.d_w[:, :, 0]).sum(axis=1) - sol.y0)
            dt_rows.append({
                "steps": steps,
                "dt": dt,
                "y0": sol.y0,
                "systematic_lr_en_gap_mean": abs(float((raw_lr - log_l_en).mean())),
                "entropy_lr": post.entropy_lr(),
                "entropy_en": post.entropy_en(),
                "likelihood_normalization": sol.residuals["likelihood_normalization"],
                "energy_identity_rms_variance_dominated": sol.residuals["energy_identity_rms"],
                "fixed_point_residual": sol.residuals["fixed_point_residual"],
            })
        (out / "dt_convergence_table.json").write_text(json.dumps(dt_rows, indent=2))
        for row in dt_rows:
            print(" ".join(f"{k}={v}" for k, v in row.items()))

    print(f"artifacts written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

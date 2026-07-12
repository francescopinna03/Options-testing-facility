"""Multi-model surface OUT-OF-SAMPLE pricing study: fit day t, price day t+1.

THE experiment this facility exists for: does the SFV bridge price tomorrow's
surface better than the standard alternatives, or does it just memorise
today's noise? For each consecutive pair of collected days and each name,
five arms are fitted on day t and scored -- frozen -- on day t+1's quotes at
day t+1's underlying close:

    flat    single Black-Scholes vol (weighted mean IV): the floor any model
            must clear;
    svi     raw-SVI slices + total-variance interpolation, read
            sticky-moneyness: the market-standard STATIC surface fit;
    heston  the affine prior priced by characteristic function;
    prior   the SAME prior priced by the CRN Monte Carlo engine (beta = 0):
            the honest paired baseline for the bridge, sharing its shocks and
            its discretization error;
    bridge  prior + restricted Schrödinger-bridge correction fitted to day
            t's residuals (SFV_M2 Level-2 objective).

The comparison is PAIRED per day (same quotes; prior/bridge also share the
same engine shocks). Aggregation per name: mean IV RMSE per arm, win counts,
moving-block-bootstrap CI and Diebold-Mariano p-value on the bridge-vs-X RMSE
differentials.

Example:
    python -m otf.experiments.surface_oos --chains-dir chains \\
        --names aapl msft nvda --json surface_oos.json --latex surface_oos.tex

Extends apps/surface_oos.py of the Stocks repository with the flat, SVI and
CF-Heston benchmark arms.
"""

from __future__ import annotations
import argparse
import json
import math
from typing import Dict, List

from otf.calibration.heston_fit import calibrate_heston
from otf.calibration.surface_fit import calibrate_bridge_to_surface
from otf.data.chains import day_dirs, load_day, load_spots
from otf.evaluation.pricing import (fit_flat_vol, iv_rmse_flat,
                                    iv_rmse_heston, iv_rmse_sfv, iv_rmse_svi)
from otf.evaluation.stats import block_bootstrap_ci, dm_test, latex_table
from otf.models.svi import fit_svi_surface

ARMS = ("flat", "svi", "heston", "prior", "bridge")


def run_name(root: str, name: str, days: List[str], args) -> List[Dict]:
    rows = []
    for t_day, t1_day in zip(days, days[1:]):
        sp_t = load_spots(root, t_day)
        sp_t1 = load_spots(root, t1_day)
        if name not in sp_t or name not in sp_t1:
            continue
        q_t = load_day(root, t_day, name, sp_t[name],
                       args.min_tte, args.max_tte, args.moneyness)
        q_t1 = load_day(root, t1_day, name, sp_t1[name],
                        args.min_tte, args.max_tte, args.moneyness)
        if not q_t or not q_t1 or len(q_t) < 10 or len(q_t1) < 10:
            continue

        # ----- fit every arm on day t ----------------------------------- #
        sigma_flat = fit_flat_vol(q_t)
        try:
            svi = fit_svi_surface(q_t, sp_t[name])
        except ValueError:
            svi = None
        res = calibrate_heston(q_t, sp_t[name],
                               feller_weight=args.feller_weight,
                               kappa_max=args.kappa_max)
        h = res.params
        jump = (0.0, 0.5, 50.0, 50.0)
        fit = calibrate_bridge_to_surface(
            q_t, sp_t[name], v0=h.v0, kappa=h.kappa, theta=h.theta,
            xi=h.xi, rho=h.rho, free=(2, 5), gate=(4.0, 1.0),
            n_paths=args.paths, max_iter=args.max_iter)

        # ----- score every arm, frozen, on day t+1 ---------------------- #
        common = dict(spot=sp_t1[name], prior=h, std=fit.std, gate=fit.gate,
                      jump=jump, r=0.0, n_paths=args.paths)
        rmse = {
            "flat": iv_rmse_flat(q_t1, sigma_flat),
            "svi": (iv_rmse_svi(q_t1, svi, spot=sp_t1[name])
                    if svi is not None else None),
            "heston": iv_rmse_heston(q_t1, sp_t1[name], h),
            "prior": iv_rmse_sfv(q_t1, beta=(0.0,) * 6, **common),
            "bridge": iv_rmse_sfv(q_t1, beta=fit.beta, **common),
        }
        if any(rmse[a] is None for a in ARMS):
            continue
        rows.append({"day": t_day, "next": t1_day, "n_quotes": len(q_t1),
                     **{f"rmse_{a}": rmse[a] for a in ARMS},
                     "in_rmse_prior": fit.iv_rmse_before,
                     "in_rmse_bridge": fit.iv_rmse_after,
                     "beta2": fit.beta[2], "beta5": fit.beta[5]})
        print(f"  {name} {t_day} -> {t1_day}: OOS RMSE (vol pts) "
              + "  ".join(f"{a} {rmse[a] * 100:.2f}" for a in ARMS)
              + f"  ({len(q_t1)}q, b2 {fit.beta[2]:+.2f})")
    return rows


def aggregate(name: str, rows: List[Dict]) -> Dict:
    out = {"days": len(rows), "rows": rows}
    for a in ARMS:
        out[f"rmse_{a}"] = sum(r[f"rmse_{a}"] for r in rows) / len(rows)
    # Paired differentials: positive delta = bridge better than the arm.
    for a in ARMS:
        if a == "bridge":
            continue
        dp = [r[f"rmse_{a}"] - r["rmse_bridge"] for r in rows]
        lo, hi = block_bootstrap_ci(dp)
        _, dm_p = dm_test(dp)
        out[f"vs_{a}"] = {"wins": sum(1 for d in dp if d > 0),
                          "delta_ci": [lo, hi], "dm_p": dm_p}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="multi-model surface OOS pricing study")
    ap.add_argument("--chains-dir", default="chains")
    ap.add_argument("--names", nargs="+", default=["aapl", "msft", "nvda"])
    ap.add_argument("--min-tte", type=float, default=0.08)
    ap.add_argument("--max-tte", type=float, default=0.8)
    ap.add_argument("--moneyness", type=float, default=0.20)
    ap.add_argument("--feller-weight", type=float, default=10.0)
    ap.add_argument("--kappa-max", type=float, default=15.0)
    ap.add_argument("--paths", type=int, default=400)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--json")
    ap.add_argument("--latex")
    args = ap.parse_args()

    days = day_dirs(args.chains_dir)
    if len(days) < 2:
        raise SystemExit(f"need >= 2 day dirs under {args.chains_dir}, "
                         f"got {len(days)}")
    print(f"{len(days)} days: {days[0]} .. {days[-1]}\n")

    results, table = {}, []
    for name in args.names:
        rows = run_name(args.chains_dir, name, days, args)
        if not rows:
            continue
        agg = aggregate(name, rows)
        results[name] = agg
        table.append(
            [name.upper(), agg["days"]]
            + [f"{agg[f'rmse_{a}'] * 100:.2f}" for a in ARMS]
            + [f"{agg['vs_prior']['wins']}/{agg['days']}",
               f"{agg['vs_prior']['dm_p']:.3f}",
               f"{agg['vs_svi']['dm_p']:.3f}"])
        print(f"\n{name.upper():<6} {agg['days']} day-pairs   mean OOS RMSE "
              + "  ".join(f"{a} {agg[f'rmse_{a}'] * 100:.2f}" for a in ARMS))
        for a in ARMS:
            if a == "bridge":
                continue
            v = agg[f"vs_{a}"]
            lo, hi = v["delta_ci"]
            print(f"    bridge vs {a:<6} wins {v['wins']}/{agg['days']}   "
                  f"dRMSE CI [{lo * 100:+.2f}, {hi * 100:+.2f}] pts   "
                  f"DM p {v['dm_p']:.3f}")
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"(results written to {args.json})")
    if args.latex and table:
        with open(args.latex, "w") as fh:
            fh.write(latex_table(
                "surface-oos",
                "Out-of-sample surface pricing: fit on day $t$, price day "
                "$t+1$ (IV RMSE, vol points; paired per day).",
                ["Name", "Days", "Flat", "SVI", "Heston", "Prior", "Bridge",
                 "W(pr)", "DM$_{pr}$", "DM$_{svi}$"], table) + "\n")
        print(f"(LaTeX written to {args.latex})")


if __name__ == "__main__":
    main()

"""Calibrate the SFV model to a REAL option surface (SFV_M2 Level 2).

Two-layer pipeline, in the order the theory prescribes:

    option chain (LSEG Workspace export / CSV)
      -> affine PRIOR fit          : Heston (v0,kappa,theta,xi,rho) on the
                                     surface, characteristic function + IV loss
      -> prior residual report     : RMSE per expiry (the residual-gate logic)
      -> BRIDGE fit on the surface : betas of the standardized, gated variance
                                     correction under the full Level-2 loss
                                     (IV + vega-weighted price + martingale +
                                     control energy + L2), MC-priced with CRN
      -> paste-ready parameter block: betas + the standardization/gate
                                     constants that give them meaning

Example:
    python -m otf.experiments.calibrate_surface --chain-csv aapl_chain.csv \\
        --prices-csv aapl.csv --spot 211.5

Ported from the Stocks repository (apps/calibrate_surface.py).
"""

from __future__ import annotations
import argparse
import csv
import math
from typing import List

from otf.calibration.heston_fit import calibrate_heston
from otf.calibration.surface_fit import calibrate_bridge_to_surface
from otf.data.chains import load_chain
from otf.data.realized import estimate_sfv_prior


def _read_closes(path: str) -> List[float]:
    """Close column from a prices CSV (column named close/CLOSE/Close or the
    last numeric column)."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    low = [h.strip().lower() for h in rows[0]]
    idx = low.index("close") if "close" in low else len(rows[0]) - 1
    out = []
    for r in rows[1:]:
        try:
            out.append(float(str(r[idx]).replace(",", "")))
        except (ValueError, IndexError):
            continue
    if len(out) < 21:
        raise SystemExit(f"{path}: need >= 21 closes, got {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SFV Level-2 surface calibration")
    ap.add_argument("--chain-csv", required=True, help="option chain export")
    ap.add_argument("--spot", type=float, required=True, help="underlying spot")
    ap.add_argument("--prices-csv", help="closes CSV -> Kou jump block "
                                         "(omitted: no jumps in the prior)")
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--min-tte", type=float, default=0.08,
                    help="drop sub-month expiries: their smile is jump-driven "
                         "and forces the diffusive CF prior into xi explosion "
                         "/ Feller violation")
    ap.add_argument("--max-tte", type=float, default=1.5)
    ap.add_argument("--moneyness", type=float, default=0.25,
                    help="keep strikes within this fraction of spot")
    ap.add_argument("--free", default="2,5", help="beta indices to fit")
    ap.add_argument("--w-iv", type=float, default=1.0)
    ap.add_argument("--w-px", type=float, default=1.0)
    ap.add_argument("--lambda-mart", type=float, default=100.0)
    ap.add_argument("--lambda-energy", type=float, default=0.05)
    ap.add_argument("--lambda-beta", type=float, default=0.01)
    ap.add_argument("--lambda-sk", type=float, default=0.0,
                    help="Stage-5 marginal Sinkhorn on X_T vs the "
                         "Breeden-Litzenberger implied density (0 = off)")
    ap.add_argument("--gate", default="4.0,1.0")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--paths", type=int, default=800)
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--feller-weight", type=float, default=2.0,
                    help="penalty on 2*kappa*theta < xi^2 in the prior fit: "
                         "a CF-only prior that violates Feller hard prices "
                         "the surface but cannot be SIMULATED faithfully -- "
                         "the bridge would then fit discretization error")
    ap.add_argument("--kappa-max", type=float, default=0.0,
                    help="soft cap on the prior mean-reversion speed (0 = "
                         "off): jumpy surfaces push the CF fit to extreme "
                         "kappa/xi pairs that price well but simulate badly")
    args = ap.parse_args()

    quotes = load_chain(args.chain_csv, min_tte=args.min_tte,
                        max_tte=args.max_tte,
                        moneyness=args.moneyness, spot=args.spot)
    if len(quotes) < 6:
        raise SystemExit(f"only {len(quotes)} usable quotes after filters")
    ttes = sorted({round(q.tte, 4) for q in quotes})
    print(f"surface: {len(quotes)} quotes, {len(ttes)} expiries "
          f"(tte {ttes[0]:.3f}..{ttes[-1]:.3f}y), spot {args.spot:g}\n")

    # --- layer 1: affine prior on the surface --------------------------- #
    res = calibrate_heston(quotes, args.spot, r=args.r,
                           feller_weight=args.feller_weight,
                           kappa_max=args.kappa_max)
    h = res.params
    print(f"prior Heston fit: v0 {h.v0:.5f}  kappa {h.kappa:.3f}  "
          f"theta {h.theta:.5f}  xi {h.xi:.3f}  rho {h.rho:.3f}")
    print(f"prior IV RMSE {res.rmse * 100:.2f} vol pts "
          f"({res.iterations} iters, converged={res.converged})")

    lam, p_up, eta_up, eta_dn = 0.0, 0.5, 50.0, 50.0
    if args.prices_csv:
        rp = estimate_sfv_prior(_read_closes(args.prices_csv))
        lam, p_up, eta_up, eta_dn = rp.lambda_S, rp.p_up, rp.eta_up, rp.eta_dn
        print(f"jump block from {args.prices_csv}: lambda_S {lam:.2f}/yr  "
              f"p_up {p_up:.2f}  eta {eta_up:.1f}/{eta_dn:.1f}")

    # --- layer 2: bridge on what the prior could not price -------------- #
    gate = (0.0, 0.0) if args.no_gate else tuple(
        float(t) for t in args.gate.split(","))
    free = tuple(int(t) for t in args.free.split(",") if t.strip())
    fit = calibrate_bridge_to_surface(
        quotes, args.spot, v0=h.v0, kappa=h.kappa, theta=h.theta, xi=h.xi,
        rho=h.rho, lambda_S=lam, p_up=p_up, eta_up=eta_up, eta_dn=eta_dn,
        r=args.r, free=free, w_iv=args.w_iv, w_px=args.w_px,
        lambda_mart=args.lambda_mart, lambda_energy=args.lambda_energy,
        lambda_beta=args.lambda_beta, lambda_sk=args.lambda_sk, gate=gate,
        n_paths=args.paths, max_iter=args.max_iter)

    b = fit.beta
    print(f"\nbridge fit (free={free}): loss {fit.loss_before:.4f} -> "
          f"{fit.loss_after:.4f}   IV RMSE {fit.iv_rmse_before*100:.2f} -> "
          f"{fit.iv_rmse_after*100:.2f} vol pts")
    print("loss terms: " + "  ".join(f"{k} {v:.4f}" for k, v in fit.terms.items()))
    d = fit.diagnostics
    print(f"diagnostics: martingale {d.martingale_error:+.5f} "
          f"(se {d.martingale_stderr:.5f})  boundary hits {d.boundary_hits:.4f}"
          f"  control energy {d.control_energy:.5f}  "
          f"gate activation {d.gate_activation:.3f}")

    print("\n  # surface-calibrated SFV parameters")
    print(f"  v0: {h.v0:.6f}\n  kappa: {h.kappa:.4f}\n  theta: {h.theta:.6f}")
    print(f"  xi: {h.xi:.4f}\n  rho: {h.rho:.4f}")
    if lam > 0:
        print(f"  lambda_S: {lam:.4f}\n  p_up: {p_up:.4f}")
        print(f"  eta_up: {eta_up:.4f}\n  eta_dn: {eta_dn:.4f}")
    print("  bridge_alpha: 1.0")
    print(f"  bridge_beta: [{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, "
          f"{b[3]:.4f}, {b[4]:.4f}, {b[5]:.4f}]")
    print(f"  bridge_x0: {math.log(args.spot):.6f}")
    print(f"  bridge_sx: {fit.std['sx']:.6f}")
    print(f"  bridge_v_ref: {fit.std['v_ref']:.6f}")
    print(f"  bridge_sv: {fit.std['sv']:.6f}")
    print(f"  bridge_gate_m: {fit.gate[0]:.4f}")
    print(f"  bridge_gate_c: {fit.gate[1]:.4f}")


if __name__ == "__main__":
    main()

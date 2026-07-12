"""Convert WRDS OptionMetrics panel CSVs into the facility's chains/ layout.

The SFV M-series work (2014 AAPL study) produced per-day calibration panels
with OptionMetrics columns (``date``, ``exdate``, ``cp_flag``, ``strike`` in
dollars, ``impl_volatility``, ``spot_close``). This script rewrites them as
``chains/YYYY-MM-DD/{name}_chain.csv`` + ``spots.json`` -- the exact layout
``collect_chains.py`` produces -- so ``otf.experiments.surface_oos`` consumes
historical WRDS panels and live LSEG snapshots identically.

Rows without an implied vol are dropped (OptionMetrics leaves it blank where
its own inversion failed).

Usage:
    python scripts/panels_to_chains.py OUT_DIR PANEL.csv [PANEL.csv ...]
    python scripts/panels_to_chains.py --name aapl chains_2014 panels/*.csv
"""
import argparse
import csv
import json
import os


def convert(panel: str, out_root: str, name: str) -> tuple:
    with open(panel, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("impl_volatility", "").strip()]
    if not rows:
        raise SystemExit(f"{panel}: no rows with impl_volatility")
    day = rows[0]["date"].split()[0]
    spot = float(rows[0]["spot_close"])
    outdir = os.path.join(out_root, day)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}_chain.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["STRIKE_PRC", "EXPIR_DATE", "PUTCALLIND", "IMP_VOLT"])
        for r in rows:
            w.writerow([r["strike"], r["exdate"].split()[0],
                        r["cp_flag"].strip().upper(), r["impl_volatility"]])
    sp_path = os.path.join(outdir, "spots.json")
    spots = {}
    if os.path.exists(sp_path):
        with open(sp_path) as fh:
            spots = json.load(fh)
    spots[name] = spot
    with open(sp_path, "w") as fh:
        json.dump(spots, fh, indent=1)
    return day, spot, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="WRDS panels -> chains/ layout")
    ap.add_argument("--name", default="aapl", help="underlying name key")
    ap.add_argument("out_root", help="output chains directory")
    ap.add_argument("panels", nargs="+", help="panel CSVs (one per day)")
    args = ap.parse_args()
    for p in sorted(args.panels):
        day, spot, n = convert(p, args.out_root, args.name)
        print(f"{day}: {args.name} {n} quotes, spot {spot:.2f}")


if __name__ == "__main__":
    main()

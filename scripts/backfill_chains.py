"""Backfill historical daily surfaces from the LIVE contracts' IV history.

LSEG keeps per-contract daily history (IMP_VOLT, TRDPRC_1) for listed
options. From today's chain we know the live RICs with strike/expiry/type;
pulling each contract's history and pivoting by date rebuilds the surface
as it stood on each past day -- written in the exact same layout as
``collect_chains.py`` so ``apps.surface_oos`` consumes both transparently.

Survivorship note (for the paper): today's chain only contains contracts
still alive, so a surface rebuilt D days back is missing exactly those
expiring within D days -- i.e. quotes with tte < D at that date. A
backfill horizon no longer than the study's min-tte filter (0.08y ~ 20
business days) is therefore free of survivorship bias BY CONSTRUCTION;
beyond that, raise min-tte accordingly when consuming the data.

Usage (lseg-env, Workspace open):
    ~/lseg-env/bin/python backfill_chains.py            # 25 days, AAPL MSFT NVDA
    ~/lseg-env/bin/python backfill_chains.py 40 GOOGL.O
"""
import csv
import datetime as dt
import json
import os
import sys

import pandas as pd
import refinitiv.data as rd

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
RICS = sys.argv[2:] or ["AAPL.O", "MSFT.O", "NVDA.O"]
SRC_DAY = sorted(os.listdir("chains"))[-1]          # newest snapshot as source

rd.open_session()
by_date = {}                                         # date -> name -> rows
spots_by_date = {}                                   # date -> name -> close

for und_ric in RICS:
    name = und_ric.split(".")[0].lower()
    src = os.path.join("chains", SRC_DAY, f"{name}_chain.csv")
    with open(src, newline="") as fh:
        contracts = {r["Instrument"]: r for r in csv.DictReader(fh)}
    rics = list(contracts)
    print(f"{name}: {len(rics)} live contracts, pulling {DAYS}d history...")

    # underlying closes for the spot file
    px = rd.get_history(universe=und_ric, interval="daily", count=DAYS + 5)
    col = next((c for c in px.columns
                if str(c).upper() in ("TRDPRC_1", "CLOSE", "CLOSE_PRC")),
               px.columns[-1])
    closes = px[col].dropna()
    for d, s in closes.items():
        spots_by_date.setdefault(d.date().isoformat(), {})[name] = float(s)

    n_rows = 0
    for i in range(0, len(rics), 50):
        batch = rics[i:i + 50]
        try:
            hist = rd.get_history(universe=batch, fields=["IMP_VOLT", "TRDPRC_1"],
                                  interval="daily", count=DAYS + 5)
        except Exception as e:                       # noqa: BLE001
            print(f"  batch {i}: {type(e).__name__} (skipped)")
            continue
        # single-RIC batches come back unstacked; normalize to (RIC, field)
        if not isinstance(hist.columns, pd.MultiIndex):
            hist.columns = pd.MultiIndex.from_product([[batch[0]], hist.columns])
        for ric in {c[0] for c in hist.columns}:
            meta = contracts.get(ric)
            if meta is None or (ric, "IMP_VOLT") not in hist.columns:
                continue
            ivs = hist[(ric, "IMP_VOLT")].dropna()
            for d, iv in ivs.items():
                day = d.date().isoformat()
                row = {"Instrument": ric,
                       "STRIKE_PRC": meta["STRIKE_PRC"],
                       "EXPIR_DATE": meta["EXPIR_DATE"],
                       "PUTCALLIND": meta["PUTCALLIND"],
                       "IMP_VOLT": float(iv)}
                by_date.setdefault(day, {}).setdefault(name, []).append(row)
                n_rows += 1
    print(f"  {n_rows} historical quotes across "
          f"{len({d for d, m in by_date.items() if name in m})} days")
rd.close_session()

cutoff = (dt.date.today() - dt.timedelta(days=DAYS * 1.6)).isoformat()
written = 0
for day, names in sorted(by_date.items()):
    if day < cutoff or day >= SRC_DAY:               # keep live snapshot as-is
        continue
    outdir = os.path.join("chains", day)
    os.makedirs(outdir, exist_ok=True)
    for name, rows in names.items():
        path = os.path.join(outdir, f"{name}_chain.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["Instrument", "STRIKE_PRC",
                                               "EXPIR_DATE", "PUTCALLIND",
                                               "IMP_VOLT"])
            w.writeheader()
            w.writerows(rows)
    sp = {n: spots_by_date.get(day, {}).get(n) for n in names}
    sp = {k: v for k, v in sp.items() if v is not None}
    with open(os.path.join(outdir, "spots.json"), "w") as fh:
        json.dump(sp, fh, indent=1)
    written += 1
    print(f"{day}: " + "  ".join(f"{n} {len(r)}q" for n, r in sorted(names.items()))
          + f"  spots {sorted(sp)}")
print(f"\n{written} historical days written under chains/")

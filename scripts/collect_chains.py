"""Daily option-chain snapshot collector (LSEG Workspace, lseg-env).

The paper's surface out-of-sample table needs a TIME SERIES of chains:
calibrate on day t, price day t+1. One snapshot per day, per name, into
``chains/YYYY-MM-DD/{name}_chain.csv`` plus ``spots.json`` with the
underlying last closes. Run once per trading day while Workspace is open:

    ~/lseg-env/bin/python collect_chains.py            # AAPL MSFT NVDA
    ~/lseg-env/bin/python collect_chains.py GOOGL.O    # custom list
"""
import datetime as dt
import json
import os
import sys

import refinitiv.data as rd
from refinitiv.data.discovery import Views, search

RICS = sys.argv[1:] or ["AAPL.O", "MSFT.O", "NVDA.O"]
FIELDS = ["STRIKE_PRC", "EXPIR_DATE", "PUTCALLIND", "IMP_VOLT", "CF_CLOSE"]

day = dt.date.today().isoformat()
outdir = os.path.join("chains", day)
os.makedirs(outdir, exist_ok=True)

rd.open_session()
spots = {}
for ric in RICS:
    name = ric.split(".")[0].lower()

    px = rd.get_history(universe=ric, interval="daily", count=1)
    col = next((c for c in px.columns
                if str(c).upper() in ("TRDPRC_1", "CLOSE", "CLOSE_PRC")),
               px.columns[-1])
    spots[name] = float(px[col].dropna().iloc[-1])

    hits = search(
        view=Views.DERIVATIVE_QUOTES,
        filter=f"UnderlyingQuoteRIC eq '{ric}' and RCSAssetCategoryLeaf eq "
               f"'Equity Option'",
        select="RIC", top=2000)
    rics = [r for r in hits["RIC"].tolist()
            if isinstance(r, str) and r.endswith(".U")]

    import pandas as pd
    frames = []
    for i in range(0, len(rics), 100):
        frames.append(rd.get_data(universe=rics[i:i + 100], fields=FIELDS))
    chain = pd.concat(frames, ignore_index=True)
    chain = chain.dropna(subset=["STRIKE_PRC", "IMP_VOLT"])
    path = os.path.join(outdir, f"{name}_chain.csv")
    chain.to_csv(path, index=False)
    print(f"{name}: {len(chain)} quotes -> {path}  (spot {spots[name]:.2f})")
rd.close_session()

with open(os.path.join(outdir, "spots.json"), "w") as fh:
    json.dump(spots, fh, indent=1)
print(f"spots -> {outdir}/spots.json")

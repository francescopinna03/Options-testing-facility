"""Load and clean LSEG Workspace option-chain exports into MarketQuotes.

Chain CSV: one row per option. Recognized columns (case-insensitive):
strike (STRIKE_PRC), expiry date (EXPIR_DATE) or tte in years (TTE), implied
vol (IMP_VOLT, % or decimal), put/call (PUTCALLIND / TYPE). Rows with missing
fields are skipped and counted. This is exactly the layout written by
``scripts/collect_chains.py`` and ``scripts/backfill_chains.py``.

Filters applied when ``spot`` is given: moneyness band around spot, OTM-only
(the liquid side: puts below spot, calls above), tte band, sane-IV band.
Sub-month expiries default to being dropped by the CALLERS (min_tte ~ 0.08y):
their smile is jump-driven and forces a diffusive CF prior into xi explosion /
Feller violation.

Ported from the Stocks repository (apps/calibrate_surface.py).
"""

from __future__ import annotations
import csv
import datetime as dt
import json
import os
from typing import List, Optional

from otf.calibration.heston_fit import MarketQuote

__all__ = ["load_chain", "load_day", "day_dirs"]

_STRIKE = ("strike", "strike_prc", "strike price")
_EXPIRY = ("expir_date", "expiry", "expiry_date", "expiration", "maturity")
_TTE = ("tte", "years", "t")
_IV = ("imp_volt", "iv", "implied_vol", "implied volatility", "impliedvolatility")
_TYPE = ("putcallind", "type", "put_call", "putcall", "cp")


def _num(v) -> Optional[float]:
    s = str(v).strip().replace(",", "").replace("%", "")
    if not s or s.upper() in ("NA", "N/A", "NAN", "NULL", "#N/A", "NONE", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _col(low: List[str], names) -> Optional[int]:
    for n in names:
        if n in low:
            return low.index(n)
    return None


def _tte_from(v, today: dt.date) -> Optional[float]:
    s = str(v).strip()
    x = _num(s)
    if x is not None and x < 20.0:            # already in years
        return x
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            d = dt.datetime.strptime(s.split("T")[0].split()[0], fmt).date()
            return max((d - today).days / 365.25, 0.0)
        except ValueError:
            continue
    return None


def load_chain(path: str, today: Optional[dt.date] = None,
               min_tte: float = 0.01, max_tte: float = 1.5,
               moneyness: float = 0.25, spot: Optional[float] = None,
               otm_only: bool = True, verbose: bool = True) -> List[MarketQuote]:
    """Parse a Workspace chain export into MarketQuotes (skips bad rows)."""
    today = today or dt.date.today()
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} is empty")
    low = [h.strip().lower() for h in rows[0]]
    i_k, i_e, i_t = _col(low, _STRIKE), _col(low, _EXPIRY), _col(low, _TTE)
    i_iv, i_cp = _col(low, _IV), _col(low, _TYPE)
    if i_k is None or i_iv is None or (i_e is None and i_t is None):
        raise ValueError(f"chain csv needs strike, iv and expiry/tte columns; "
                         f"got header {rows[0]}")
    quotes, skipped = [], 0
    for r in rows[1:]:
        def cell(i):
            return r[i] if i is not None and i < len(r) else ""
        k = _num(cell(i_k))
        iv = _num(cell(i_iv))
        tte = (_num(cell(i_t)) if i_t is not None
               else _tte_from(cell(i_e), today))
        if k is None or iv is None or tte is None:
            skipped += 1
            continue
        if iv > 3.0:                          # percent -> decimal
            iv /= 100.0
        cp = str(cell(i_cp)).strip().upper()[:1]
        if spot is not None:
            is_call = cp == "C" if cp in ("C", "P") else k >= spot
            if abs(k / spot - 1.0) > moneyness:
                skipped += 1
                continue
            if otm_only and ((is_call and k < spot) or (not is_call and k > spot)):
                skipped += 1
                continue
        else:
            is_call = cp != "P"
        if not (min_tte <= tte <= max_tte) or not (0.01 < iv < 3.0):
            skipped += 1
            continue
        quotes.append(MarketQuote(strike=k, tte=tte, iv=iv, is_call=is_call))
    if skipped and verbose:
        print(f"({skipped} chain rows skipped: missing/invalid fields or filters)")
    return quotes


def day_dirs(root: str) -> List[str]:
    """Sorted day directories under a ``chains/`` root (those with spots.json)."""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "spots.json")):
            out.append(d)
    return out


def load_day(root: str, day: str, name: str, spot: float,
             min_tte: float = 0.08, max_tte: float = 0.8,
             moneyness: float = 0.20) -> Optional[List[MarketQuote]]:
    """One name's cleaned chain for one collected day (None if absent)."""
    path = os.path.join(root, day, f"{name}_chain.csv")
    if not os.path.exists(path):
        return None
    return load_chain(path, today=dt.date.fromisoformat(day),
                      min_tte=min_tte, max_tte=max_tte,
                      moneyness=moneyness, spot=spot)


def load_spots(root: str, day: str) -> dict:
    """The ``spots.json`` (name -> underlying close) of one collected day."""
    with open(os.path.join(root, day, "spots.json")) as fh:
        return json.load(fh)

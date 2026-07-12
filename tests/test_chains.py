import datetime as dt
import json

from otf.data.chains import day_dirs, load_chain, load_day, load_spots

HEADER = "Instrument,STRIKE_PRC,EXPIR_DATE,PUTCALLIND,IMP_VOLT\n"


def _write_chain(path, rows):
    with open(path, "w") as fh:
        fh.write(HEADER)
        for r in rows:
            fh.write(",".join(str(c) for c in r) + "\n")


def test_load_chain_parses_filters_and_scales(tmp_path):
    p = tmp_path / "chain.csv"
    _write_chain(p, [
        ("AAPL1", 100.0, "2026-10-16", "C", 25.0),    # kept: pct -> 0.25
        ("AAPL2", 90.0, "2026-10-16", "P", 0.30),     # kept: OTM put, decimal
        ("AAPL3", 90.0, "2026-10-16", "C", 0.30),     # dropped: ITM call
        ("AAPL4", 300.0, "2026-10-16", "C", 0.30),    # dropped: moneyness
        ("AAPL5", 100.0, "2026-10-16", "C", ""),      # dropped: missing IV
        ("AAPL6", 100.0, "2026-07-20", "C", 0.30),    # dropped: tte < min
    ])
    today = dt.date(2026, 7, 12)
    quotes = load_chain(str(p), today=today, spot=100.0, min_tte=0.08,
                        max_tte=1.5, moneyness=0.25, verbose=False)
    assert len(quotes) == 2
    call = next(q for q in quotes if q.is_call)
    put = next(q for q in quotes if not q.is_call)
    assert call.strike == 100.0 and abs(call.iv - 0.25) < 1e-12
    assert put.strike == 90.0 and abs(put.iv - 0.30) < 1e-12
    assert abs(call.tte - (dt.date(2026, 10, 16) - today).days / 365.25) < 1e-9


def test_day_layout_roundtrip(tmp_path):
    root = tmp_path / "chains"
    day = "2026-07-10"
    (root / day).mkdir(parents=True)
    (root / day / "spots.json").write_text(json.dumps({"aapl": 210.0}))
    _write_chain(root / day / "aapl_chain.csv",
                 [("AAPL1", 220.0, "2026-11-20", "C", 0.28),
                  ("AAPL2", 200.0, "2026-11-20", "P", 0.31)])
    (root / "not-a-day").mkdir()               # no spots.json: ignored

    assert day_dirs(str(root)) == [day]
    assert load_spots(str(root), day) == {"aapl": 210.0}
    quotes = load_day(str(root), day, "aapl", 210.0)
    assert len(quotes) == 2
    assert load_day(str(root), day, "msft", 500.0) is None

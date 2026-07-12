import math

from otf.models.black_scholes import bs_greeks, bs_price, implied_vol


def test_put_call_parity():
    S, K, T, sig, r = 100.0, 95.0, 0.7, 0.25, 0.03
    c = bs_price(S, K, T, sig, r, call=True)
    p = bs_price(S, K, T, sig, r, call=False)
    assert abs((c - p) - (S - K * math.exp(-r * T))) < 1e-10


def test_implied_vol_round_trip():
    S, K, T, r = 100.0, 110.0, 0.5, 0.01
    for sig in (0.1, 0.25, 0.6):
        px = bs_price(S, K, T, sig, r, call=True)
        assert abs(implied_vol(px, S, K, T, r, call=True) - sig) < 1e-4


def test_degenerate_inputs_return_intrinsic():
    assert bs_price(100.0, 90.0, 0.0, 0.2) == 10.0
    assert bs_price(100.0, 90.0, 0.5, 0.0) == 10.0
    assert bs_price(0.0, 90.0, 0.5, 0.2) == 0.0


def test_greeks_match_finite_differences():
    S, K, T, sig, r = 100.0, 100.0, 0.5, 0.2, 0.02
    g = bs_greeks(S, K, T, sig, r, call=True)
    h = 1e-4
    fd_delta = (bs_price(S + h, K, T, sig, r) - bs_price(S - h, K, T, sig, r)) / (2 * h)
    fd_vega = (bs_price(S, K, T, sig + h, r) - bs_price(S, K, T, sig - h, r)) / (2 * h)
    assert abs(g.delta - fd_delta) < 1e-6
    assert abs(g.vega - fd_vega) < 1e-4
    assert g.gamma > 0.0

import math

from otf.models.black_scholes import bs_price
from otf.models.heston import heston_price


def test_small_xi_converges_to_black_scholes():
    # With xi -> 0 and v0 = theta the variance is frozen: Heston = BS at
    # sigma = sqrt(v0).
    S, K, T, r = 100.0, 105.0, 0.5, 0.02
    v0 = theta = 0.04
    hp = heston_price(S, K, T, v0, 2.0, theta, 1e-4, 0.0, r, call=True)
    bp = bs_price(S, K, T, math.sqrt(v0), r, call=True)
    assert abs(hp - bp) < 5e-3


def test_put_call_parity():
    S, K, T, r = 100.0, 95.0, 0.8, 0.01
    args = (0.05, 1.5, 0.05, 0.5, -0.6, r)
    c = heston_price(S, K, T, *args, call=True)
    p = heston_price(S, K, T, *args, call=False)
    assert abs((c - p) - (S - K * math.exp(-r * T))) < 5e-3


def test_skew_sign_with_negative_rho():
    # Negative leverage correlation: OTM puts richer (higher IV) than OTM
    # calls -- put price above the flat-vol benchmark, call below.
    S, T = 100.0, 0.5
    v0 = theta = 0.04
    common = dict(v0=v0, kappa=2.0, theta=theta, xi=0.6, rho=-0.7, r=0.0)
    put_otm = heston_price(S, 85.0, T, call=False, **common)
    put_bs = bs_price(S, 85.0, T, math.sqrt(v0), 0.0, call=False)
    assert put_otm > put_bs

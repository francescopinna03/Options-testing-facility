import math

from otf.calibration.bridge_fit import calibrate_bridge
from otf.calibration.heston_fit import MarketQuote, calibrate_heston
from otf.models.black_scholes import implied_vol
from otf.models.heston import heston_price
from otf.models.sfv import PathEngine, sample_moments

TRUE = dict(v0=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.6)


def _quotes_from_true(spot: float):
    quotes = []
    for tte in (0.25, 0.6):
        for k in (0.85, 0.95, 1.0, 1.05, 1.15):
            strike = spot * k
            px = heston_price(spot, strike, tte, TRUE["v0"], TRUE["kappa"],
                              TRUE["theta"], TRUE["xi"], TRUE["rho"], 0.0,
                              True, n=128)
            iv = implied_vol(px, spot, strike, tte, 0.0, True)
            quotes.append(MarketQuote(strike=strike, tte=tte, iv=iv))
    return quotes


def test_calibrate_heston_reprices_a_known_surface():
    spot = 100.0
    quotes = _quotes_from_true(spot)
    res = calibrate_heston(quotes, spot, n=64, max_iter=600)
    # The identification target is the SURFACE, not the raw parameters
    # (Heston parameters trade off): sub-vol-point repricing is the claim.
    assert res.rmse < 0.01
    assert res.params.v0 > 0 and res.params.xi > 0
    assert -1.0 < res.params.rho < 0.0        # equity skew recovered


def test_calibrate_bridge_moves_law_toward_wider_target():
    eng = PathEngine(n_paths=150, n_steps=50, horizon=0.2, v0=0.04, kappa=2.0,
                     theta=0.04, xi=0.4, rho=-0.5, seed=11)
    # Target: the same engine under a known variance-level correction.
    target = eng.terminal_logret((0.0, 0.0, 30.0, 0.0, 0.0, 0.0))
    fit = calibrate_bridge(target, eng, free=(2,), max_iter=80, restarts=1)
    assert fit.distance_after < fit.distance_before
    # Law recovery: fitted std much closer to target's than the prior's was.
    sd_t = sample_moments(target)[1]
    assert abs(fit.moments_fit[1] - sd_t) < abs(fit.moments_prior[1] - sd_t)

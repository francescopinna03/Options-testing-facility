from otf.calibration.heston_fit import HestonFit, MarketQuote
from otf.evaluation.pricing import (fit_flat_vol, iv_rmse_flat,
                                    iv_rmse_heston, iv_rmse_sfv, iv_rmse_svi)
from otf.models.black_scholes import implied_vol
from otf.models.heston import heston_price
from otf.models.sfv import standardization_for
from otf.models.svi import fit_svi_surface

FIT = HestonFit(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.5)


def _quotes(spot=100.0, tte=0.3):
    out = []
    for k in (0.9, 0.95, 1.0, 1.05, 1.1):
        strike = spot * k
        px = heston_price(spot, strike, tte, *FIT.as_tuple(), 0.0, True, n=96)
        out.append(MarketQuote(strike=strike, tte=tte,
                               iv=implied_vol(px, spot, strike, tte)))
    return out


def test_flat_vol_fit_and_score():
    quotes = _quotes()
    sigma = fit_flat_vol(quotes)
    assert 0.1 < sigma < 0.4
    assert iv_rmse_flat(quotes, sigma) < iv_rmse_flat(quotes, sigma + 0.1)


def test_heston_scores_its_own_surface_near_zero():
    quotes = _quotes()
    rmse = iv_rmse_heston(quotes, 100.0, FIT)
    assert rmse < 5e-3           # CF round trip: sub-half-vol-point


def test_svi_scores_close_after_fitting():
    quotes = _quotes()
    surf = fit_svi_surface(quotes, 100.0)
    assert iv_rmse_svi(quotes, surf) < 0.01


def test_sfv_prior_and_bridge_arms_are_paired_and_ranked():
    quotes = _quotes()
    std = standardization_for(FIT.v0, FIT.kappa, FIT.theta, FIT.xi, 0.3)
    common = dict(spot=100.0, prior=FIT, std=std, gate=(0.0, 0.0), r=0.0,
                  n_paths=200, seed=42)
    prior_rmse = iv_rmse_sfv(quotes, beta=(0.0,) * 6, **common)
    same = iv_rmse_sfv(quotes, beta=(0.0,) * 6, **common)
    assert prior_rmse == same    # CRN: identical seed, identical score
    # An absurd variance correction must score worse than the prior on a
    # surface the prior generated.
    bad = iv_rmse_sfv(quotes, beta=(0.0, 0.0, 200.0, 0.0, 0.0, 0.0), **common)
    assert bad > prior_rmse

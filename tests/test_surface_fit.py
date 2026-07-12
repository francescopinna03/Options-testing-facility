from otf.calibration.heston_fit import MarketQuote
from otf.calibration.surface_fit import (calibrate_bridge_to_surface,
                                         implied_terminal_sample)
from otf.models.black_scholes import implied_vol
from otf.models.heston import heston_price


def _surface(spot: float, v0: float, xi: float):
    quotes = []
    for tte in (0.2, 0.4):
        for k in (0.9, 0.95, 1.0, 1.05, 1.1):
            strike = spot * k
            px = heston_price(spot, strike, tte, v0, 2.0, 0.04, xi, -0.5,
                              0.0, True, n=96)
            quotes.append(MarketQuote(strike=strike, tte=tte,
                                      iv=implied_vol(px, spot, strike, tte)))
    return quotes


def test_bridge_improves_a_misspecified_prior_in_sample():
    spot = 100.0
    # Market generated at HIGHER variance than the prior we hand the bridge:
    # a variance-level residual the b2/b5 correction is built to absorb.
    quotes = _surface(spot, v0=0.0625, xi=0.5)
    fit = calibrate_bridge_to_surface(
        quotes, spot, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.5,
        free=(2,), gate=(0.0, 0.0), n_paths=120, max_iter=40, seed=3)
    assert fit.loss_after <= fit.loss_before
    assert fit.iv_rmse_after < fit.iv_rmse_before
    # Correction must stay essentially risk-neutral.
    assert abs(fit.diagnostics.martingale_error) < 0.02


def test_implied_terminal_sample_is_centred_and_sized():
    spot = 100.0
    quotes = [q for q in _surface(spot, v0=0.04, xi=0.4) if q.tte == 0.2]
    sample = implied_terminal_sample(quotes, spot, 0.2, n=128)
    assert len(sample) == 128
    m = sum(sample) / len(sample)
    assert abs(m) < 0.1          # log-return atoms centred near zero

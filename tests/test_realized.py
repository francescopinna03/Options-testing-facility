import math
import random

from otf.data.realized import estimate_sfv_prior, horizon_returns, log_returns


def _series_with_jumps(n=600, seed=5):
    rng = random.Random(seed)
    prices = [100.0]
    jump_days = {100, 250, 400, 520}
    for t in range(n):
        r = rng.gauss(0.0, 0.012)
        if t in jump_days:
            r += -0.08 if t % 2 == 0 else 0.07
        prices.append(prices[-1] * math.exp(r))
    return prices, len(jump_days)


def test_log_and_horizon_returns():
    prices = [100.0, 110.0, 121.0]
    lr = log_returns(prices)
    assert len(lr) == 2 and abs(lr[0] - math.log(1.1)) < 1e-12
    hr = horizon_returns([100, 101, 102, 103, 104, 105], 2)
    assert len(hr) == 2


def test_estimate_sfv_prior_splits_jumps_from_diffusion():
    prices, n_planted = _series_with_jumps()
    prior = estimate_sfv_prior(prices)
    assert prior.n_jumps >= n_planted            # planted jumps all flagged
    assert prior.lambda_S > 0.0
    assert prior.eta_up > 1.0 and prior.eta_dn > 0.0
    # Diffusive long-run vol ~ 1.2% daily ~ 19% annualised.
    assert 0.1 < math.sqrt(prior.theta) < 0.3
    assert prior.v0 > 0.0
    assert -1.0 < prior.rho < 1.0

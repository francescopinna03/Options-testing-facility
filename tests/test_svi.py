import math

from otf.models.svi import SVISlice, fit_svi_slice, fit_svi_surface


class _Q:
    def __init__(self, strike, tte, iv):
        self.strike, self.tte, self.iv = strike, tte, iv
        self.weight = 1.0


def _synthetic_slice():
    return SVISlice(a=0.01, b=0.08, rho=-0.4, m=0.02, sigma=0.15, tte=0.5,
                    rmse=0.0, n_quotes=0, converged=True)


def test_fit_recovers_synthetic_svi_smile():
    true = _synthetic_slice()
    ks = [-0.3 + 0.05 * i for i in range(13)]
    ivs = [true.iv(k) for k in ks]
    fit = fit_svi_slice(ks, ivs, true.tte)
    assert fit.rmse < 5e-3        # half a vol point on a noiseless smile
    for k in (-0.25, 0.0, 0.2):
        assert abs(fit.iv(k) - true.iv(k)) < 5e-3


def test_fit_requires_five_quotes():
    try:
        fit_svi_slice([0.0, 0.1, 0.2, -0.1], [0.2] * 4, 0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_surface_interpolates_total_variance_in_maturity():
    spot = 100.0
    quotes = []
    for tte, base in ((0.25, 0.20), (0.75, 0.24)):
        for k in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
            iv = base + 0.10 * k * k          # gentle smile
            quotes.append(_Q(spot * math.exp(k), tte, iv))
    surf = fit_svi_surface(quotes, spot)
    assert len(surf.slices) == 2
    iv_lo = surf.iv(spot, 0.25)
    iv_hi = surf.iv(spot, 0.75)
    iv_mid = surf.iv(spot, 0.5)
    assert min(iv_lo, iv_hi) - 1e-3 <= iv_mid <= max(iv_lo, iv_hi) + 1e-3
    # extrapolation stays finite and positive
    assert 0.0 < surf.iv(spot, 0.1) < 1.0
    assert 0.0 < surf.iv(spot, 1.2) < 1.0


def test_sticky_moneyness_spot_override():
    surf = fit_svi_surface(
        [_Q(100.0 * math.exp(k), 0.5, 0.2 + 0.1 * k * k)
         for k in (-0.2, -0.1, 0.0, 0.1, 0.2)], 100.0)
    # same k, different strike: reading at the new spot must match reading
    # the old spot at the equivalent strike
    assert abs(surf.iv(105.0, 0.5, spot=105.0) - surf.iv(100.0, 0.5)) < 1e-12

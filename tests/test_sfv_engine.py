
import pytest

from otf.models.sfv import (PathEngine, kou_compensator, sample_moments,
                            sinkhorn_divergence, standardization_for,
                            w2_distance, mc_price)

ZERO = (0.0,) * 6


def _engine(**kw):
    base = dict(n_paths=150, n_steps=60, horizon=0.25, v0=0.04, kappa=2.0,
                theta=0.04, xi=0.4, rho=-0.5, seed=7)
    base.update(kw)
    return PathEngine(**base)


def test_kou_compensator_guards():
    with pytest.raises(ValueError):
        kou_compensator(0.5, 1.0, 50.0)      # eta_up must be > 1
    with pytest.raises(ValueError):
        kou_compensator(0.5, 50.0, 0.0)
    assert kou_compensator(0.5, 50.0, 50.0) > 0.0


def test_common_random_numbers_are_deterministic():
    eng = _engine()
    a = eng.terminal_logret(ZERO)
    b = eng.terminal_logret(ZERO)
    assert a == b
    eng2 = _engine()                          # same seed -> same shocks
    assert eng2.terminal_logret(ZERO) == a


def test_prior_is_a_discounted_martingale():
    eng = _engine(n_paths=600)
    d = eng.diagnostics(ZERO)
    # E[e^{X_T}] - 1 = 0 up to MC error (judge against its own stderr).
    assert abs(d.martingale_error) < 4.0 * d.martingale_stderr
    assert d.mean_control == 0.0
    assert d.control_energy == 0.0


def test_prior_martingale_with_kou_jumps():
    eng = _engine(n_paths=800, lambda_S=15.0, p_up=0.4, eta_up=40.0,
                  eta_dn=30.0)
    d = eng.diagnostics(ZERO)
    assert abs(d.martingale_error) < 5.0 * d.martingale_stderr


def test_positive_b2_widens_the_terminal_law():
    # b2 > 0 pushes variance up along every path: the bridged law must be
    # wider than the prior on the SAME shocks.
    eng = _engine()
    sd_prior = sample_moments(eng.terminal_logret(ZERO))[1]
    sd_bridge = sample_moments(eng.terminal_logret((0.0, 0.0, 40.0, 0.0, 0.0, 0.0)))[1]
    assert sd_bridge > sd_prior


def test_gate_suppresses_correction_at_low_variance():
    # A sharp gate centred far above v_ref keeps the correction off: the
    # bridged law collapses back onto the prior.
    std = standardization_for(0.04, 2.0, 0.04, 0.4, 0.25)
    eng_gated = _engine(sx=std["sx"], v_ref=std["v_ref"], sv=std["sv"],
                        gate_m=40.0, gate_c=50.0)
    beta = (0.0, 0.0, 40.0, 0.0, 0.0, 0.0)
    d = w2_distance(eng_gated.terminal_logret(beta),
                    eng_gated.terminal_logret(ZERO))
    assert d < 1e-3


def test_w2_and_sinkhorn_agree_on_far_vs_near():
    eng = _engine()
    prior = eng.terminal_logret(ZERO)
    far = [x + 0.5 for x in prior]
    near = [x + 0.01 for x in prior]
    assert w2_distance(prior, far) > w2_distance(prior, near)
    assert sinkhorn_divergence(prior, far, iters=25, atoms=48) > \
        sinkhorn_divergence(prior, near, iters=25, atoms=48)


def test_mc_price_monotone_in_strike():
    eng = _engine()
    p95, _ = mc_price(eng, 95.0, True, ZERO, s0=100.0)
    p105, _ = mc_price(eng, 105.0, True, ZERO, s0=100.0)
    assert p95 > p105 > 0.0


def test_standardization_constants_positive():
    std = standardization_for(0.04, 2.0, 0.04, 0.4, 0.5)
    assert std["sx"] > 0 and std["sv"] > 0 and std["v_ref"] == 0.04

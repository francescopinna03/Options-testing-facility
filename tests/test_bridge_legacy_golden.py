"""Golden regression freeze of the legacy restricted bridge (M0 gate).

The legacy model in otf.models.sfv is the *legacy restricted
variance-channel deformation* — a different law construction from the
projective SSFV backend, kept as a regression benchmark and ablation arm
(docs/DECISIONS.md D4). Its numerical behavior is frozen here: any change
to these outputs is a breaking change to the ablation baseline and must be
deliberate.

Golden values generated on 2026-07-15 at commit 127ab4c (CPython, pure
stdlib engine — deterministic given the seed). Tolerances are relative
1e-9, not byte equality (arch doc §16.4), to absorb libm differences
across platforms.
"""

import pytest

from otf.models.sfv import (PathEngine, mc_price, sample_moments,
                            standardization_for)

REL = 1e-9


def _approx(x):
    return pytest.approx(x, rel=REL)


def test_standardization_constants_frozen():
    std = standardization_for(v0=0.04, kappa=2.0, theta=0.05, xi=0.5,
                              horizon=0.5)
    assert std["sx"] == _approx(0.15811388300841897)
    assert std["v_ref"] == _approx(0.05)
    assert std["sv"] == _approx(0.05590169943749474)


def test_diffusion_bridge_with_gate_and_cap_frozen():
    # Exercises: standardized ansatz, sigmoid variance gate, tanh nu cap,
    # nonzero r, all six betas.
    std = standardization_for(v0=0.04, kappa=2.0, theta=0.05, xi=0.5,
                              horizon=0.5)
    eng = PathEngine(n_paths=300, n_steps=80, horizon=0.5, v0=0.04,
                     kappa=2.0, theta=0.05, xi=0.5, rho=-0.6, r=0.01,
                     seed=42, alpha=1.0, sx=std["sx"], v_ref=std["v_ref"],
                     sv=std["sv"], gate_m=25.0, gate_c=0.03, nu_max=2.0)
    beta = (0.1, -0.2, 0.3, 0.05, -0.1, 0.15)

    mean, sd, skew, kurt = sample_moments(eng.terminal_logret(beta))
    assert mean == _approx(-0.023352470583301192)
    assert sd == _approx(0.1810208868189655)
    assert skew == _approx(-1.544035998705658)
    assert kurt == _approx(4.675911223383827)

    golden_calls = {
        90.0: (12.100350975579447, 0.649890810885876),
        100.0: (5.761455271164709, 0.46484061780431135),
        110.0: (1.8240989070595868, 0.2796354564607807),
    }
    for strike, (px, se) in golden_calls.items():
        got_px, got_se = mc_price(eng, strike, True, beta)
        assert got_px == _approx(px)
        assert got_se == _approx(se)


def test_kou_jump_bridge_frozen():
    # Exercises: Kou compound-Poisson shocks, martingale jump compensator,
    # raw (unstandardized) ansatz, no gate, put pricing.
    eng = PathEngine(n_paths=300, n_steps=80, horizon=0.25, v0=0.03,
                     kappa=1.5, theta=0.04, xi=0.4, rho=-0.5, lambda_S=20.0,
                     p_up=0.4, eta_up=40.0, eta_dn=25.0, seed=7)
    beta = (0.0, 0.1, -0.15, 0.2, 0.05, -0.05)

    mean, sd, skew, kurt = sample_moments(eng.terminal_logret(beta))
    assert mean == _approx(-0.017713274295841977)
    assert sd == _approx(0.1565940925581988)
    assert skew == _approx(-0.9924192071096152)
    assert kurt == _approx(2.0286396763976526)

    px, se = mc_price(eng, 95.0, False, beta)
    assert px == _approx(3.854269360021472)
    assert se == _approx(0.44507480275831746)

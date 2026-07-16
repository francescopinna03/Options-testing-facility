"""Unit tests for the SSFV Heston prior engine (arch doc §16.1).

Anchors: exact CIR moments, the forward-martingale identity E[e^{x_T}] = 1,
the CF pricer, determinism of the frozen batch, and — critically — that the
exposed innovations d_w are *exactly* the ones that generated the paths
(pathwise replay), since BSDE/likelihood correctness depends on it.
"""

import math

import pytest

np = pytest.importorskip("numpy")

from otf.ssfv.prior.heston import HestonPrior

T = 1.0
N_PATHS = 30_000


@pytest.fixture(scope="module")
def euler_batch():
    return HestonPrior(scheme="euler_ft").simulate(N_PATHS, 256, T, seed=1729)


@pytest.fixture(scope="module")
def qe_batch():
    return HestonPrior(scheme="qe").simulate(N_PATHS, 32, T, seed=1729)


def test_validate_parameters_flags_feller_without_failing():
    rep = HestonPrior().validate_parameters()  # xi=0.5 violates Feller here
    assert rep.ok
    assert any("Feller" in m for m in rep.messages)
    bad = HestonPrior(kappa=-1.0).validate_parameters()
    assert not bad.ok


@pytest.mark.parametrize("fixture", ["euler_batch", "qe_batch"])
def test_martingale_identity(fixture, request):
    b = request.getfixturevalue(fixture)
    ex = np.exp(b.x[:, -1])
    se = ex.std() / math.sqrt(b.n_paths)
    assert abs(ex.mean() - 1.0) < 5.0 * se


@pytest.mark.parametrize("fixture,var_tol", [("euler_batch", 0.05), ("qe_batch", 0.02)])
def test_cir_terminal_moments(fixture, var_tol, request):
    p = HestonPrior()
    b = request.getfixturevalue(fixture)
    vT = b.v[:, -1]
    se_mean = vT.std() / math.sqrt(b.n_paths)
    assert abs(vT.mean() - p.variance_mean(T)) < max(5.0 * se_mean, 0.01 * p.variance_mean(T))
    # Var: relative tolerance (full-truncation Euler carries a small bias
    # in the Feller-violated regime; QE moment-matches by construction).
    assert abs(vT.var() - p.variance_variance(T)) < var_tol * p.variance_variance(T)


@pytest.mark.parametrize("fixture", ["euler_batch", "qe_batch"])
def test_mc_matches_cf_prices(fixture, request):
    p = HestonPrior()
    b = request.getfixturevalue(fixture)
    ex = np.exp(b.x[:, -1])
    for k in (0.9, 1.0, 1.1):
        pay = np.maximum(ex - k, 0.0)
        se = pay.std() / math.sqrt(b.n_paths)
        assert abs(pay.mean() - p.price_european_cf(k, T)) < 5.0 * se


def test_deterministic_and_seed_sensitive():
    p = HestonPrior()
    a = p.simulate(500, 16, 0.25, seed=11)
    b = p.simulate(500, 16, 0.25, seed=11)
    c = p.simulate(500, 16, 0.25, seed=12)
    assert a.batch_hash == b.batch_hash
    assert a.batch_hash != c.batch_hash


def test_euler_innovations_replay_paths_exactly(euler_batch):
    """d_w must be the exact pathwise innovations: replaying the recursion
    from d_w reproduces x and v to the last bit pattern (same float ops)."""
    p = HestonPrior()
    b = euler_batch
    dt = T / b.n_steps
    orto = math.sqrt(1.0 - p.rho**2)
    x = np.full(b.n_paths, 0.0)
    v = np.full(b.n_paths, p.v0)
    for j in range(b.n_steps):
        vp = np.maximum(v, 0.0)
        sq = np.sqrt(vp)
        dw_s = b.d_w[:, j, 0]
        dw_v = p.rho * b.d_w[:, j, 0] + orto * b.d_w[:, j, 1]
        x = x - 0.5 * vp * dt + sq * dw_s
        v = v + p.kappa * (p.theta - vp) * dt + p.xi * sq * dw_v
    assert np.array_equal(x, b.x[:, -1])
    assert np.array_equal(v, b.v[:, -1])


def test_qe_batch_is_flagged_pricing_grade(qe_batch):
    assert qe_batch.d_w is None
    assert not qe_batch.has_innovations

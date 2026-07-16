"""Unit and property tests for the cumulative constraint family
(arch doc §16.1: nesting and normalization, gauge removal; §16.2:
embedded level-n with zero new coefficients reproduces level n; review
fixes R1 — tail tests — and R2 — structural nesting)."""

import numpy as np
import pytest

pytest.importorskip("numpy")

from otf.ssfv.constraints.hat_family import LambdaPotential, NestedHatFamily

RNG = np.random.default_rng(7)
FAMILY = NestedHatFamily(k_min=-1.0, k_max=1.0, base_dim=4)


@pytest.fixture(scope="module")
def prior_sample():
    # Loose stand-in for a terminal log-moneyness marginal.
    return RNG.normal(-0.02, 0.25, size=20_000)


def test_cumulative_dimensions():
    # level 0: (base_dim + 2) hats + 2 ramps; level n+1 adds one midpoint
    # hat per gap, 2 outward hats, 2 ramps.
    assert [FAMILY.dim_at(n) for n in range(4)] == [8, 17, 33, 63]


def test_columns_bounded_and_tails_saturate():
    lvl = FAMILY.level(2)
    k = np.linspace(-4.0, 4.0, 8001)
    b = FAMILY.evaluate(lvl, k)
    assert b.min() >= 0.0 and b.max() <= 1.0
    # Beyond the outermost breakpoint every column is constant: hats are
    # zero, ramps saturate at one — bounded tail tests, not zero tails.
    far_left = FAMILY.evaluate(lvl, np.array([-50.0, -60.0]))
    far_right = FAMILY.evaluate(lvl, np.array([50.0, 60.0]))
    np.testing.assert_array_equal(far_left[0], far_left[1])
    np.testing.assert_array_equal(far_right[0], far_right[1])
    assert far_left[0] @ (lvl.col_kind == 1) > 0  # left ramps alive out there


def test_structural_column_nesting():
    """Level n's column list is a prefix of level n+1's — nesting is an
    identity of functions, not a span argument."""
    coarse, fine = FAMILY.level(1), FAMILY.level(2)
    d = coarse.dim_raw
    np.testing.assert_array_equal(coarse.col_kind, fine.col_kind[:d])
    np.testing.assert_allclose(coarse.col_loc, fine.col_loc[:d], atol=0)
    np.testing.assert_allclose(coarse.col_wl, fine.col_wl[:d], atol=0)
    k = np.linspace(-2.0, 2.0, 2001)
    np.testing.assert_array_equal(FAMILY.evaluate(coarse, k),
                                  FAMILY.evaluate(fine, k)[:, :d])


def test_tail_ramps_distinguish_tail_mass():
    """Two samples identical inside [k_min, k_max] but different in the
    tails must produce different targets (review fix R1: a fixed-interval
    hat family cannot see this)."""
    inside = RNG.normal(0.0, 0.2, size=15_000)
    inside = inside[np.abs(inside) < 0.9]
    # Both tails beyond k_max = 1; they differ inside the resolution the
    # level's ramps and extension hats currently cover (a fixed level only
    # sees tails at its own resolution; the cumulative limit sees all).
    tail_a = np.full(300, 1.2)
    tail_b = np.full(300, 1.7)
    lvl = FAMILY.normalize(FAMILY.level(1), np.concatenate([inside, tail_a, -tail_a]))
    t_a = FAMILY.targets_from_sample(lvl, np.concatenate([inside, tail_a]))
    t_b = FAMILY.targets_from_sample(lvl, np.concatenate([inside, tail_b]))
    assert not np.allclose(t_a, t_b)


def test_normalization_orthonormal_on_pilot_sample(prior_sample):
    lvl = FAMILY.normalize(FAMILY.level(2), prior_sample)
    tilde = FAMILY.evaluate_normalized(lvl, prior_sample)
    np.testing.assert_allclose(tilde.mean(axis=0), 0.0, atol=1e-10)
    gram = tilde.T @ tilde / tilde.shape[0]
    np.testing.assert_allclose(gram, np.eye(lvl.dim), atol=1e-8)


def test_gauge_rejects_only_new_directions():
    # A sample concentrated in [-0.1, 0.1] leaves outer hats and ramps
    # without mass: they are rejected and recorded.
    narrow = RNG.normal(0.0, 0.03, size=5_000)
    lvl0 = FAMILY.normalize(FAMILY.level(0), narrow)
    assert lvl0.dim < lvl0.dim_raw
    assert len(lvl0.normalization.rejected_indices) > 0
    # Inherited directions can never be removed: level 1 normalized with
    # `previous` keeps level 0's accepted columns as an exact prefix.
    lvl1 = FAMILY.normalize(FAMILY.level(1), narrow, previous=lvl0)
    nm0, nm1 = lvl0.normalization, lvl1.normalization
    assert nm1.inherited_dim == lvl0.dim
    assert nm1.kept_indices[:lvl0.dim] == nm0.kept_indices
    np.testing.assert_array_equal(nm1.transform[:lvl0.dim_raw, :lvl0.dim],
                                  nm0.transform)


def test_independent_normalization_nests_on_same_sample(prior_sample):
    """Without threading `previous`, two levels normalized independently
    on the SAME pilot sample still produce exactly nested transforms."""
    lvl1 = FAMILY.normalize(FAMILY.level(1), prior_sample)
    lvl2 = FAMILY.normalize(FAMILY.level(2), prior_sample)
    d1 = len(lvl1.normalization.kept_indices)
    assert lvl2.normalization.kept_indices[:d1] == lvl1.normalization.kept_indices
    np.testing.assert_allclose(
        lvl2.normalization.transform[:lvl1.dim_raw, :d1],
        lvl1.normalization.transform, atol=1e-12,
    )


def test_embedding_is_exact(prior_sample):
    """Zero-padded coefficients reproduce the coarse potential exactly —
    not up to an L2 tolerance (review fix R2)."""
    lvl1 = FAMILY.normalize(FAMILY.level(1), prior_sample)
    lvl2 = FAMILY.normalize(FAMILY.level(2), prior_sample)
    lam1 = RNG.normal(size=lvl1.dim)
    lam2 = FAMILY.embed(lam1, lvl1, lvl2)

    k_eval = np.concatenate([RNG.normal(-0.02, 0.25, size=3_000),
                             np.array([-5.0, -2.0, 2.0, 5.0])])
    phi1 = LambdaPotential(FAMILY, lvl1, lam1).terminal_value(k_eval)
    phi2 = LambdaPotential(FAMILY, lvl2, lam2).terminal_value(k_eval)
    np.testing.assert_allclose(phi2, phi1, atol=1e-12)


def test_embedding_rejects_mismatched_pilot_samples(prior_sample):
    lvl1 = FAMILY.normalize(FAMILY.level(1), prior_sample)
    other = RNG.normal(0.1, 0.3, size=20_000)
    lvl2 = FAMILY.normalize(FAMILY.level(2), other)
    with pytest.raises(ValueError):
        FAMILY.embed(RNG.normal(size=lvl1.dim), lvl1, lvl2)


def test_sup_and_lipschitz_are_exact(prior_sample):
    lvl = FAMILY.normalize(FAMILY.level(2), prior_sample)
    lam = RNG.normal(size=lvl.dim)
    pot = LambdaPotential(FAMILY, lvl, lam)
    k = np.linspace(-8.0, 8.0, 200_001)
    vals = pot.terminal_value(k)
    dense_sup = np.abs(vals).max()
    dense_lip = np.abs(np.diff(vals) / np.diff(k)).max()
    assert dense_sup <= pot.sup_norm_exact() + 1e-9
    assert pot.sup_norm_exact() <= dense_sup + 1e-6
    assert dense_lip <= pot.lipschitz_bound() + 1e-6
    assert pot.sup_norm_exact() <= pot.sup_norm_bound() + 1e-12

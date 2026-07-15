"""Unit and property tests for the nested hat constraint family
(arch doc §16.1: basis nesting and normalization, gauge removal;
§16.2: embedded level-n with zero new coefficients reproduces level n)."""

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


def test_dyadic_dimensions():
    assert [FAMILY.dim_at(n) for n in range(4)] == [4, 9, 19, 39]


def test_hats_bounded_and_compactly_supported():
    lvl = FAMILY.level(2)
    k = np.linspace(-2.0, 2.0, 4001)
    b = FAMILY.evaluate(lvl, k)
    assert b.min() >= 0.0 and b.max() <= 1.0
    assert np.all(b[np.abs(k) > 1.0] == 0.0)  # zero outside the domain


def test_exact_span_nesting():
    """Coarse hat = fine interpolant of itself: psi_j = sum_i psi_j(t'_i) psi'_i,
    exactly, on a dense evaluation grid."""
    coarse, fine = FAMILY.level(1), FAMILY.level(2)
    k = np.linspace(-1.0, 1.0, 2001)
    b_coarse = FAMILY.evaluate(coarse, k)
    b_fine = FAMILY.evaluate(fine, k)
    weights = FAMILY.evaluate(coarse, fine.knots[1:-1])  # (d_fine, d_coarse)
    reproduced = b_fine @ weights
    np.testing.assert_allclose(reproduced, b_coarse, atol=1e-12)


def test_normalization_standardizes_on_the_sample(prior_sample):
    lvl = FAMILY.normalize(FAMILY.level(2), prior_sample)
    tilde = FAMILY.evaluate_normalized(lvl, prior_sample)
    np.testing.assert_allclose(tilde.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(tilde.std(axis=0), 1.0, atol=1e-3)


def test_gauge_removal_drops_empty_hats():
    # A sample concentrated in [-0.1, 0.1] leaves the outer hats of a wide
    # level-3 grid without mass: those directions must be removed.
    narrow = RNG.normal(0.0, 0.03, size=5_000)
    lvl = FAMILY.normalize(FAMILY.level(3), narrow)
    assert lvl.dim < lvl.dim_raw


def test_embedding_preserves_potential_up_to_constant(prior_sample):
    lvl1 = FAMILY.normalize(FAMILY.level(1), prior_sample)
    lvl2 = FAMILY.normalize(FAMILY.level(2), prior_sample)
    # Load only central (mass-rich) directions: there the dyadic nesting is
    # exact. Tail directions dropped by the statistical gauge are preserved
    # only in L2(Q0) — that regime is exercised in the projective tests.
    lam1 = np.zeros(lvl1.dim)
    central = [i for i, raw in enumerate(lvl1.normalization.kept_indices)
               if abs(FAMILY.level(1).knots[raw + 1]) < 0.45]
    lam1[central] = RNG.normal(size=len(central))
    lam2 = FAMILY.embed(lam1, lvl1, lvl2)

    k_eval = RNG.normal(-0.02, 0.25, size=3_000)
    phi1 = LambdaPotential(FAMILY, lvl1, lam1).terminal_value(k_eval)
    phi2 = LambdaPotential(FAMILY, lvl2, lam2).terminal_value(k_eval)
    diff = phi1 - phi2
    # Equal up to an additive constant (gauge direction) on the sample.
    assert diff.std() < 1e-2 * max(np.abs(phi1 - phi1.mean()).max(), 1.0)


def test_sup_norm_bound_dominates_samples(prior_sample):
    lvl = FAMILY.normalize(FAMILY.level(2), prior_sample)
    lam = RNG.normal(size=lvl.dim)
    pot = LambdaPotential(FAMILY, lvl, lam)
    k = np.linspace(-1.5, 1.5, 5001)
    assert np.abs(pot.terminal_value(k)).max() <= pot.sup_norm_bound() + 1e-12

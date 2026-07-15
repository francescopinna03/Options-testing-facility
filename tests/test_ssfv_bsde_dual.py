"""M1 core tests: Picard Hopf-Cole solver, reweighted posterior, alternating
finite-dual calibration (arch doc §16; milestone M1 gate).

Property tests (§16.2): zero potential returns the prior exactly; a
constant added to the potential changes no posterior law; the projected
posterior preserves the forward.

Recovery: multipliers are identified only up to dynamically-replicable
gauge directions (paper §empirical: "the recovered object is the feedback
field, not merely the coefficient vector"), so recovery is asserted on the
law — call prices, entropy, moments — not on the raw lambda vector.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from otf.ssfv.bsde.picard import PicardHopfColeSolver
from otf.ssfv.constraints.hat_family import LambdaPotential, NestedHatFamily
from otf.ssfv.dual.calibrator import AlternatingDualCalibrator
from otf.ssfv.posterior.reweight import ReweightedPosterior
from otf.ssfv.prior.heston import HestonPrior

PRIOR = HestonPrior()
FAMILY = NestedHatFamily(k_min=-0.8, k_max=0.8, base_dim=4)
N_PATHS, N_STEPS, T = 8192, 32, 0.5


@pytest.fixture(scope="module")
def paths():
    return PRIOR.simulate(N_PATHS, N_STEPS, T, seed=1729)


@pytest.fixture(scope="module")
def level(paths):
    return FAMILY.normalize(FAMILY.level(0), paths.x[:, -1])


@pytest.fixture(scope="module")
def solver():
    return PicardHopfColeSolver().for_prior(PRIOR)


@pytest.fixture(scope="module")
def context(paths, level, solver):
    return solver.build_context(paths, extra_knots=level.knots)


@pytest.fixture(scope="module")
def lam_star(level):
    # Multipliers scaled per column so each basis element contributes an
    # O(0.3) potential: the regime real calibration targets live in.
    nm = level.normalization
    colbound = (1.0 + np.abs(nm.means)) / nm.stds
    rng = np.random.default_rng(5)
    return rng.normal(0.0, 0.35, level.dim) / colbound[list(nm.kept_indices)]


@pytest.fixture(scope="module")
def star(paths, level, solver, context, lam_star):
    sol = solver.solve(paths, LambdaPotential(FAMILY, level, lam_star), context)
    return sol, ReweightedPosterior(paths, sol)


def test_zero_potential_returns_the_prior_exactly(paths, level, solver, context):
    sol = solver.solve(paths, LambdaPotential(FAMILY, level, np.zeros(level.dim)), context)
    post = ReweightedPosterior(paths, sol)
    assert sol.y0 == pytest.approx(0.0, abs=1e-9)
    assert np.abs(sol.log_density).max() == pytest.approx(0.0, abs=1e-9)
    assert post.ess_fraction() == pytest.approx(1.0, abs=1e-9)
    assert post.entropy_lr() == pytest.approx(0.0, abs=1e-9)


def test_constant_shift_changes_no_posterior_law(paths, level, solver, context, lam_star, star):
    class Shifted:
        def __init__(self, base, c):
            self.base, self.c = base, c

        def terminal_value(self, k):
            return self.base.terminal_value(k) + self.c

        def sup_norm_bound(self):
            return self.base.sup_norm_bound() + abs(self.c)

        def lipschitz_bound(self):
            return self.base.lipschitz_bound()

    base = LambdaPotential(FAMILY, level, lam_star)
    sol_shift = solver.solve(paths, Shifted(base, 0.7), context)
    post_shift = ReweightedPosterior(paths, sol_shift)
    sol0, post0 = star
    np.testing.assert_allclose(post_shift.weights(), post0.weights(), atol=1e-10)
    # The sample dual absorbs the constant exactly.
    assert sol_shift.y0_sample - sol0.y0_sample == pytest.approx(0.7, abs=1e-6)


def test_known_potential_solver_certificate_quality(star):
    sol, post = star
    r = sol.residuals
    assert r["likelihood_normalization"] < 0.02
    assert r["energy_identity_rms"] < 0.15
    assert r["positivity_clipped_fraction"] < 1e-4
    assert r["score_capped_fraction"] < 1e-4
    assert post.ess_fraction() > 0.9
    assert post.forward_error() < 5e-3  # martingale preserved
    # Entropy double-entry: both routes near the (small) true KL.
    assert abs(post.entropy_lr() - post.entropy_en()) < 5e-3


def test_qe_batch_is_rejected_by_the_likelihood_layer(solver):
    qe = HestonPrior(scheme="qe").simulate(256, 8, 0.25, seed=1)
    with pytest.raises(ValueError, match="innovations"):
        solver.build_context(qe)


def test_out_of_range_multipliers_fail_loudly(paths, level, solver, context):
    lam = np.full(level.dim, 50.0)
    with pytest.raises(ValueError, match="sup-norm"):
        solver.solve(paths, LambdaPotential(FAMILY, level, lam), context)


def test_dual_calibration_recovers_the_law(paths, level, solver, context, lam_star, star):
    sol_star, post_star = star
    psi = FAMILY.evaluate_normalized(level, paths.x[:, -1])
    targets = post_star.weights() @ psi

    cal = AlternatingDualCalibrator(FAMILY, solver=solver)
    fit = cal.fit(level, targets, paths, context=context)
    assert fit.converged
    assert fit.moment_residual_norm < 1e-3

    sol_hat = cal.solve_at(level, fit.lam, paths, context)
    post_hat = ReweightedPosterior(paths, sol_hat)
    strikes = np.array([0.85, 0.95, 1.0, 1.05, 1.15])
    np.testing.assert_allclose(
        post_hat.call_prices(strikes), post_star.call_prices(strikes), atol=5e-4,
    )
    assert post_hat.ess_fraction() > 0.9
    assert post_hat.forward_error() < 5e-3
    assert abs(post_hat.entropy_lr() - post_star.entropy_lr()) < 2e-3


def test_null_targets_recover_the_prior(paths, level, solver, context):
    """Pure-Heston DGP: targets computed from the prior itself must return
    (numerically) zero deformation (arch doc §14.1 DGP 1, §16.2)."""
    psi = FAMILY.evaluate_normalized(level, paths.x[:, -1])
    targets = psi.mean(axis=0)  # prior moments of the normalized basis = 0
    cal = AlternatingDualCalibrator(FAMILY, solver=solver)
    fit = cal.fit(level, targets, paths, context=context)
    assert fit.converged
    sol = cal.solve_at(level, fit.lam, paths, context)
    post = ReweightedPosterior(paths, sol)
    assert post.entropy_lr() < 1e-4
    assert post.ess_fraction() > 0.99

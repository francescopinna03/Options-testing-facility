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

import os

import pytest

pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from otf.ssfv.bsde.picard import PicardHopfColeSolver
from otf.ssfv.constraints.hat_family import LambdaPotential, NestedHatFamily
from otf.ssfv.dual.calibrator import ReducedMomentMapCalibrator
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
    # Multipliers rescaled so sup|Phi*| = 0.35 exactly: the mild-
    # deformation regime real calibration targets live in.
    rng = np.random.default_rng(5)
    lam = rng.normal(0.0, 1.0, level.dim)
    sup = LambdaPotential(FAMILY, level, lam).sup_norm_exact()
    return lam * (0.35 / sup)


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

    cal = ReducedMomentMapCalibrator(FAMILY, solver=solver)
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


def test_implicit_jacobian_matches_finite_differences(paths, level, solver, context, lam_star, star):
    """M2: dm/dlambda by implicit differentiation of the Picard fixed
    point vs central finite differences. Agreement is asserted on
    columns where the clip/cap active sets are quiet (the tangent
    freezes them; FD steps across them)."""
    sol, _ = star
    psi_T = FAMILY.evaluate_normalized(level, paths.x[:, -1])
    pot = LambdaPotential(FAMILY, level, lam_star)
    dns, tdiag = solver.dn_s_dlam(paths, pot, psi_T, context, sol)
    # Tangent certificate (review M2-3): converged linear iteration and a
    # quiet active set are what make the FD comparison below meaningful.
    assert tdiag.converged and tdiag.residual < 1e-5
    assert tdiag.cap_active_fraction < 1e-3
    assert tdiag.clip_active_fraction < 1e-3
    dl = psi_T - dns
    ns0 = (sol.z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1)
    e = psi_T @ lam_star - ns0
    w = np.exp(e - e.max())
    w /= w.sum()
    m0 = w @ psi_T
    j_imp = psi_T.T @ (dl * w[:, None]) - np.outer(m0, w @ dl)

    def moments(lam):
        s = solver.solve(paths, LambdaPotential(FAMILY, level, lam), context)
        ns = (s.z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1)
        le = psi_T @ lam - ns
        wv = np.exp(le - le.max())
        wv /= wv.sum()
        return wv @ psi_T

    eps = 1e-3
    for j in (2, 3):
        ej = np.zeros(level.dim)
        ej[j] = eps
        col_fd = (moments(lam_star + ej) - moments(lam_star - ej)) / (2 * eps)
        rel = np.abs(j_imp[:, j] - col_fd).max() / max(np.abs(col_fd).max(), 1e-12)
        assert rel < 5e-2, f"column {j}: rel error {rel:.2e}"


@pytest.mark.skipif(not os.environ.get("SSFV_SLOW"),
                    reason="extended FD validation (~3 min); set SSFV_SLOW=1")
def test_implicit_jacobian_fd_extended(paths, solver):
    """Review M2-3: FD validation beyond two quiet columns — random
    directions, the roughest (max Sobolev diagonal) and the last-created
    (tail-side) direction, two FD steps, two levels, with an explicit
    tangent certificate (converged, quiet clip/cap sets) as the
    precondition for demanding tight agreement."""
    rng = np.random.default_rng(11)
    for n in (0, 1):
        level = FAMILY.normalize(FAMILY.level(n), paths.x[:, -1])
        psi_T = FAMILY.evaluate_normalized(level, paths.x[:, -1])
        d = level.dim
        lam = rng.normal(0.0, 1.0, d)
        lam *= 0.3 / LambdaPotential(FAMILY, level, lam).sup_norm_exact()
        ctx = solver.build_context(paths, extra_knots=level.knots)
        pot = LambdaPotential(FAMILY, level, lam)
        sol = solver.solve(paths, pot, ctx)
        dns, tdiag = solver.dn_s_dlam(paths, pot, psi_T, ctx, sol)
        assert tdiag.converged
        assert tdiag.certified, tdiag.reason  # quiet clip/cap sets required
        dl = psi_T - dns
        ns0 = (sol.z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1)
        e = psi_T @ lam - ns0
        w = np.exp(e - e.max())
        w /= w.sum()
        m0 = w @ psi_T
        j_imp = psi_T.T @ (dl * w[:, None]) - np.outer(m0, w @ dl)

        def moments(la):
            s = solver.solve(paths, LambdaPotential(FAMILY, level, la), ctx)
            ns = (s.z[:, :, 0] * paths.d_w[:, :, 0]).sum(axis=1)
            le = psi_T @ la - ns
            wv = np.exp(le - le.max())
            wv /= wv.sum()
            return wv @ psi_T

        s_diag = np.diag(FAMILY.sobolev_gram(level))
        dirs = [np.eye(d)[int(np.argmax(s_diag))],  # roughest column
                np.eye(d)[d - 1]]                   # last-created (tail side)
        for _ in range(2):
            v = rng.normal(size=d)
            dirs.append(v / np.linalg.norm(v))
        for v in dirs:
            jv = j_imp @ v
            for eps in (3e-4, 1e-3):
                fd = (moments(lam + eps * v) - moments(lam - eps * v)) / (2 * eps)
                rel = np.abs(jv - fd).max() / max(np.abs(fd).max(), 1e-12)
                assert rel < 5e-2, f"level {n}, eps {eps}: rel {rel:.2e}"


def test_sobolev_gram_and_null_targets(paths, level, solver, context):
    """M2: the Sobolev Gram is symmetric positive definite with unit L2
    block, and a regularized fit on null targets still returns the prior
    (the penalty vanishes at lambda = 0, so exactness survives)."""
    S = FAMILY.sobolev_gram(level)
    np.testing.assert_allclose(S, S.T, atol=1e-10)
    assert np.linalg.eigvalsh(S)[0] >= 1.0 - 1e-8  # I + PSD derivative part
    psi = FAMILY.evaluate_normalized(level, paths.x[:, -1])
    targets = psi.mean(axis=0)
    cal = ReducedMomentMapCalibrator(FAMILY, solver=solver, sobolev_sigma2=1e-3)
    fit = cal.fit(level, targets, paths, context=context)
    assert fit.converged
    assert float(np.abs(fit.lam).max()) < 1e-3


def test_dt_refinement_no_first_order_bias_visible(level, solver, lam_star):
    """The score ratio gs/ehc is a *first-order* discretization of the
    Feynman-Kac score (kill = 1 + O(dt)), classified as such — never an
    exact discrete identity. This test pins the observable consequence:
    mean-type identity gaps (systematic bias) stay at the MC noise floor
    across a dt refinement, and the fixed-point residual does not grow.
    Variance-type metrics (energy_identity_rms) accumulate per-step
    estimation noise and are NOT expected to shrink with dt — the CLI
    --dt-table records both, labeled."""
    pot = LambdaPotential(FAMILY, level, lam_star)
    for steps in (8, 16, 32):
        p = PRIOR.simulate(N_PATHS, steps, T, seed=1729)
        ctx = solver.build_context(p, extra_knots=level.knots)
        sol = solver.solve(p, pot, ctx)
        dt = T / steps
        zp = sol.z[:, :, 1]
        log_l_en = (zp * p.d_w[:, :, 1]).sum(axis=1) - 0.5 * (zp**2).sum(axis=1) * dt
        raw_lr = (pot.terminal_value(p.x[:, -1])
                  - (sol.z[:, :, 0] * p.d_w[:, :, 0]).sum(axis=1) - sol.y0)
        sys_gap = abs(float((raw_lr - log_l_en).mean()))
        assert sys_gap < 1e-3, f"systematic LR/EN gap {sys_gap:.2e} at {steps} steps"
        assert sol.residuals["fixed_point_residual"] < 1e-4
        assert sol.residuals["likelihood_normalization"] < 1e-3


def test_null_targets_recover_the_prior(paths, level, solver, context):
    """Pure-Heston DGP: targets computed from the prior itself must return
    (numerically) zero deformation (arch doc §14.1 DGP 1, §16.2)."""
    psi = FAMILY.evaluate_normalized(level, paths.x[:, -1])
    targets = psi.mean(axis=0)  # prior moments of the normalized basis = 0
    cal = ReducedMomentMapCalibrator(FAMILY, solver=solver)
    fit = cal.fit(level, targets, paths, context=context)
    assert fit.converged
    sol = cal.solve_at(level, fit.lam, paths, context)
    post = ReweightedPosterior(paths, sol)
    assert post.entropy_lr() < 1e-4
    assert post.ess_fraction() > 0.99

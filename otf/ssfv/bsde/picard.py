"""Fixed-multiplier Picard solver for the martingale-projected Hopf-Cole
equation (paper thm:self-consistent-hopf-cole, eq:self-consistent-iteration).

The exact diffusion solver is the self-consistent field iteration

    h^(n)  ->  Lambda^(n) = -D log h^(n),  D = d_x + rho xi d_v
           ->  V^(n) = 1/2 v (Lambda^(n))^2
           ->  h^(n+1) = killed Feynman-Kac backward propagation,

where each h-step is *linear* and positivity-preserving. On a frozen prior
PathBatch the discrete realization is a backward *linear* regression

    h_j = exp(-1/2 v_j Lambda_j^2 dt) * E_j[h_{j+1}],
    h_N = exp(Phi(X_T)),

so — unlike the direct quadratic-driver recursion in
:mod:`otf.ssfv.bsde.regression` — the regression error never feeds through
a squared control: SSFV is recovered as the fixed point of a linear
backward propagation and a scalar monotone martingale projection (paper
§linear-backward-operator).

Score fields come from analytic derivatives of the fitted basis, with the
boundary-clean square-root coordinate identities (y = sqrt(v)):

    sqrt(v) U_v = h_y / (2 h)                  (no division by v)
    Z^S  = sqrt(v) h_x / h + rho xi h_y / (2h)  = -sqrt(v) Lambda
    Z^perp = xi sqrt(1-rho^2) h_y / (2h)

Density (semistatic identity, thm:bsde-semista-verification):

    log L = Phi(X_T) - N^S_T - Y_0,   N^S_T = sum_j Z^S_j dW^S_j,

pathwise-computable with no stochastic-integral regression; the EN route
N^perp - 1/2 <N^perp> is reported against it as the energy-identity
residual.

Positivity and boundedness: killing >= 0 gives h <= e^{||Phi||_inf}
(maximum principle, rigorous); fitted conditional means are capped there.
A small positive floor guards regression undershoot in sparse regions;
the clipped fraction is a reported residual, never silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otf.ssfv.bsde.regression import _hat_tensor_features, _log_mean_exp
from otf.ssfv.projection.diffusion import DiffusionMartingaleProjector
from otf.ssfv.types import BSDESolution, PathBatch

__all__ = ["PicardHopfColeSolver", "PicardContext"]


@dataclass(frozen=True)
class PicardContext:
    """Per-batch cache: features, analytic derivatives, fold Gram inverses.

    Potential-independent; build once per PathBatch and reuse across every
    dual-objective evaluation.
    """

    features: list  # F_j, (n_paths, n_feat)
    dfeat_x: list  # dF_j/dx
    dfeat_y: list  # dF_j/dy, y = sqrt(v)
    fold_masks: list
    gram_pinv: list  # [[(n_feat,n_feat) pinv, train mask] per fold] per step
    batch_hash: str


@dataclass(frozen=True)
class PicardHopfColeSolver:
    """ProjectedBSDESolver backend: linear killed-FK Picard iteration."""

    n_x_knots: int = 8
    y_degree: int = 2
    n_folds: int = 2
    # Relative ridge on the per-step Gram matrix: basis columns with a
    # handful of training points in their support otherwise get wild
    # coefficients that are garbage on the held-out fold (cv error > 1).
    # Shrinking rare-support directions to zero is the statistically
    # honest alternative to extrapolating them.
    gram_ridge: float = 1.0e-6
    # Score-field extraction: "increments" regresses the centered
    # martingale increment (h_{j+1} - E_j h)(dW/dt) — no differentiation of
    # the fitted value, so no 1/knot-spacing noise amplification;
    # "derivatives" uses the analytic basis derivative of the fitted h
    # (step-function in x for hats: noisier, kept as a cross-check).
    score_from: str = "increments"
    n_picard: int = 12
    picard_damping: float = 0.6  # under-relaxation of the Lambda-field update
    picard_tol: float = 1.0e-3  # early stop on rms field change
    # Acceptance threshold on the fixed-point residual of the final
    # consistency sweep, relative to the field rms scale. The solution is
    # only returned when r_FP = ||Z^{S,eval} - Z-bar^S|| is below it; a
    # violation means the Picard iteration did not reach a self-consistent
    # field and the solve fails loudly (§20). Sized a few multiples of
    # picard_tol: at convergence r_FP = O(picard_delta).
    fp_tolerance: float = 5.0e-3
    h_floor_log: float = 3.0  # floor = e^{-||Phi|| - h_floor_log}
    # Score-field caps. |U_x| <= Lip(Phi) is a theorem
    # (prop:tangential-lipschitz: global, translation invariance). The
    # orthogonal score has no pointwise theorem (that is the Bessel-
    # Bernstein frontier); its cap is an engineering guard against
    # division by vanishing fitted h in sparse regions, sized in units of
    # xi * Lip(Phi), and the capped fraction is a reported residual.
    score_cap_multiplier: float = 5.0
    rho: float | None = None  # taken from prior; must be supplied
    xi: float | None = None

    def for_prior(self, prior) -> "PicardHopfColeSolver":
        """Bind the prior's (rho, xi) — the D-operator needs them."""
        return PicardHopfColeSolver(
            n_x_knots=self.n_x_knots, y_degree=self.y_degree,
            n_folds=self.n_folds, gram_ridge=self.gram_ridge,
            score_from=self.score_from, n_picard=self.n_picard,
            picard_damping=self.picard_damping, picard_tol=self.picard_tol,
            fp_tolerance=self.fp_tolerance, h_floor_log=self.h_floor_log,
            score_cap_multiplier=self.score_cap_multiplier,
            rho=prior.rho, xi=prior.xi,
        )

    # -- context ---------------------------------------------------------------

    def build_context(self, paths: PathBatch, extra_knots: np.ndarray | None = None) -> PicardContext:
        """``extra_knots``: constraint-family knots — the terminal condition
        kinks there, so the regression grid must contain them even in
        low-density state regions."""
        if not paths.has_innovations:
            raise ValueError(
                f"scheme {paths.scheme!r} exposes no pathwise innovations; "
                "the likelihood requires them (PathBatch.d_w is None)"
            )
        n_paths = paths.n_paths
        idx = np.arange(n_paths)
        fold_masks = [idx % self.n_folds == f for f in range(self.n_folds)]
        features, dxs, dys, gram_pinv = [], [], [], []
        for j in range(paths.n_steps):
            F, Fx, Fy = _hat_tensor_features(
                paths.x[:, j], np.sqrt(np.maximum(paths.v[:, j], 0.0)),
                self.n_x_knots, self.y_degree, with_derivatives=True,
                extra_knots=extra_knots,
            )
            per_fold = []
            for mask in fold_masks:
                train = ~mask
                A = F[train]
                gram = A.T @ A
                gram[np.diag_indices_from(gram)] += self.gram_ridge * max(
                    float(np.trace(gram)) / gram.shape[0], 1e-30
                )
                per_fold.append((np.linalg.pinv(gram, rcond=1e-12), train))
            features.append(F)
            dxs.append(Fx)
            dys.append(Fy)
            gram_pinv.append(per_fold)
        return PicardContext(features, dxs, dys, fold_masks, gram_pinv, paths.batch_hash)

    # -- solve -------------------------------------------------------------------

    def solve(self, paths: PathBatch, potential, context: PicardContext | None = None,
              init_fields: tuple | None = None) -> BSDESolution:
        """``init_fields``: optional (zs, zp) warm start for the Picard
        fields — successive dual-objective evaluations differ by small
        multiplier moves, so the previous fixed point is one or two
        iterations away."""
        if self.rho is None or self.xi is None:
            raise ValueError("solver not bound to a prior; call for_prior(prior) first")
        if context is None:
            context = self.build_context(paths)
        if context.batch_hash != paths.batch_hash:
            raise ValueError("picard context was built for a different path batch")

        rho, xi = self.rho, self.xi
        n_paths, n_steps = paths.n_paths, paths.n_steps
        dt = float(paths.times[1] - paths.times[0])

        phi = potential.terminal_value(paths.x[:, -1])
        # Exact sup when the potential provides it (piecewise-linear
        # families do); the triangle-inequality bound overestimates by
        # orders of magnitude for overlapping normalized hats.
        sup_fn = getattr(potential, "sup_norm_exact", potential.sup_norm_bound)
        b = float(sup_fn())
        if b > 40.0:  # fail loudly (§20): e^Phi out of trustworthy range
            raise ValueError(
                f"potential sup-norm {b:.1f} > 40: the multipliers are "
                "outside the numerically trustworthy region"
            )
        h_cap = np.exp(b)
        h_floor = np.exp(-b - self.h_floor_log)
        h_T = np.exp(phi)
        sqv = np.sqrt(np.maximum(paths.v[:, :-1], 0.0))  # (n_paths, n_steps)

        # Score budgets: |sqrt(v) U_x| <= sqrt(v) Lip(Phi) (theorem);
        # orthogonal channel capped at score_cap_multiplier * xi * Lip(Phi).
        lip = float(getattr(potential, "lipschitz_bound", sup_fn)())
        vbar = float(np.quantile(paths.v, 0.999)) ** 0.5
        zs_cap = self.score_cap_multiplier * max(vbar, 1e-3) * (1.0 + abs(rho) * xi) * max(lip, 1e-12)
        zp_cap = self.score_cap_multiplier * xi * max(lip, 1e-12)

        if init_fields is not None:
            zs = np.clip(init_fields[0], -zs_cap, zs_cap)
            zp = np.clip(init_fields[1], -zp_cap, zp_cap)
        else:
            zs = np.zeros((n_paths, n_steps))  # Z^S field, defines the killing
            zp = np.zeros((n_paths, n_steps))
        picard_delta = float("inf")

        for it in range(self.n_picard):
            kill = np.exp(-0.5 * zs**2 * dt)  # frozen-multiplier FK weight
            _, zs_new, zp_new, _ = self._backward_pass(
                paths, context, kill, h_T, h_floor, h_cap, zs_cap, zp_cap, sqv, dt,
            )
            picard_delta = float(np.sqrt(np.mean((zs_new - zs) ** 2)))
            scale = float(np.sqrt(np.mean(zs_new**2))) + 1e-12
            # Under-relaxed field update: the full-slab fixed-point map is
            # only guaranteed contractive on short slabs (prop:hopf-slab);
            # damping restores convergence on one long slab.
            omega = self.picard_damping if it > 0 else 1.0
            zs = (1.0 - omega) * zs + omega * zs_new
            zp = (1.0 - omega) * zp + omega * zp_new
            if picard_delta < self.picard_tol * scale:
                break

        # Final consistency sweep. Freeze the converged field Z-bar = (zs,
        # zp), run one last linear propagation with no field update, and
        # build every returned object — h, Y_0, density — from that single
        # pass. The returned solution is therefore an approximate solution
        # of the *frozen linear problem* at Z-bar plus a fixed-point
        # residual r_FP = ||Z^{S,eval} - Z-bar^S||_{L2}; the fields
        # extracted from this sweep are a diagnostic only and are never
        # substituted back (that would re-associate h with the previous
        # field). In the limit r_FP -> 0 the two objects coincide.
        kill = np.exp(-0.5 * zs**2 * dt)
        h_path, zs_eval, zp_eval, stats = self._backward_pass(
            paths, context, kill, h_T, h_floor, h_cap, zs_cap, zp_cap, sqv, dt,
        )
        n_clipped, n_score_capped, cv_num, cv_den = stats
        y0 = float(np.log(h_path[:, 0].mean()))
        field_scale = float(np.sqrt(np.mean(zs**2)))
        r_fp = float(np.sqrt(np.mean((zs_eval - zs) ** 2)))
        if r_fp > self.fp_tolerance * field_scale + 1e-8:
            raise ValueError(
                f"picard fixed-point residual {r_fp:.3e} exceeds "
                f"{self.fp_tolerance:.1e} x field scale {field_scale:.3e}: "
                "the returned (h, Z) pair is not self-consistent"
            )

        # Semistatic (LR) density and EN cross-check — from the frozen field.
        n_s = (zs * paths.d_w[:, :, 0]).sum(axis=1)
        n_perp = (zp * paths.d_w[:, :, 1]).sum(axis=1)
        energy = 0.5 * (zp**2).sum(axis=1) * dt
        log_l_lr = phi - n_s - y0
        log_l_en = n_perp - energy
        lse = _log_mean_exp(log_l_lr)

        z = np.stack([zs, zp], axis=2)
        z_orth, proj_diag = self.projector().project(zs, zp, np.maximum(paths.v[:, :-1], 0.0))

        residuals = {
            "likelihood_normalization": abs(float(lse)),
            "energy_identity_rms": float(np.sqrt(np.mean((log_l_lr - log_l_en) ** 2))),
            "terminal_residual": 0.0,
            "regression_cv_error": float(cv_num / cv_den) if cv_den > 0 else 0.0,
            "picard_delta_zs_rms": picard_delta,
            "fixed_point_residual": r_fp,
            "positivity_clipped_fraction": n_clipped / (n_paths * n_steps),
            "score_capped_fraction": n_score_capped / (2 * n_paths * n_steps),
        }
        return BSDESolution(
            y0=y0, y=np.log(np.maximum(h_path, h_floor)), z=z, z_orth=z_orth,
            u_jump=None, log_density=log_l_lr - lse,
            projector=proj_diag, residuals=residuals,
            y0_sample=y0 + float(lse),
        )

    def _backward_pass(self, paths: PathBatch, context: PicardContext,
                       kill: np.ndarray, h_T: np.ndarray, h_floor: float,
                       h_cap: float, zs_cap: float, zp_cap: float,
                       sqv: np.ndarray, dt: float):
        """One linear killed-FK backward propagation at a frozen killing
        field: h_j = clip(E_j[h_{j+1}]) * kill_j, with score fields
        extracted along the way.

        The score ratio gs/ehc divides by the *pre-kill* conditional mean
        while h stores kill * ehc; since kill = 1 + O(dt) this is a
        first-order-in-dt discretization of the Feynman-Kac score, not an
        exact discrete identity — its Delta-t convergence is measured by
        the dt-refinement table (experiments CLI --dt-table and
        test_ssfv_bsde_dual dt-convergence test).
        """
        rho, xi = self.rho, self.xi
        orto = np.sqrt(1.0 - rho**2)
        n_paths, n_steps = paths.n_paths, paths.n_steps
        zs_new = np.empty((n_paths, n_steps))
        zp_new = np.empty((n_paths, n_steps))
        h_path = np.empty((n_paths, n_steps + 1))
        h_path[:, -1] = h_T
        n_clipped = 0
        n_score_capped = 0
        cv_num = cv_den = 0.0
        for j in range(n_steps - 1, -1, -1):
            target = h_path[:, j + 1]
            if j == 0:
                eh = np.full(n_paths, target.mean())
                resid = target - eh
                hs = (resid * paths.d_w[:, 0, 0]).mean() / dt
                hy_incr = (resid * paths.d_w[:, 0, 1]).mean() / dt
                # Z fields at the deterministic initial state via the
                # increment estimators (no fitted derivative exists).
                zs_new[:, 0] = hs / eh
                zp_new[:, 0] = hy_incr / eh
                h0v = eh * kill[:, 0]
                h_path[:, 0] = np.clip(h0v, h_floor, h_cap)
            else:
                F = context.features[j]
                eh = np.empty(n_paths)
                gs = np.empty(n_paths)  # score numerator, W^S channel
                gp = np.empty(n_paths)  # score numerator, W^perp channel
                for f, mask in enumerate(context.fold_masks):
                    gram_inv, train = context.gram_pinv[j][f]
                    At = F[train].T
                    # Ridge shrinks toward the *fold-mean predictor*,
                    # not toward zero: fit the centered target and add
                    # the mean back. A constant target (zero potential)
                    # is then reproduced exactly — the §16.2 property
                    # "zero potential returns the prior" survives the
                    # regularization.
                    ybar = float(target[train].mean())
                    coef = gram_inv @ (At @ (target[train] - ybar))
                    eh[mask] = ybar + F[mask] @ coef
                    if self.score_from == "increments":
                        resid = target[train] - ybar - F[train] @ coef
                        gs[mask] = F[mask] @ (gram_inv @ (At @ (resid * paths.d_w[train, j, 0]))) / dt
                        gp[mask] = F[mask] @ (gram_inv @ (At @ (resid * paths.d_w[train, j, 1]))) / dt
                    else:  # analytic derivatives of the fitted h
                        ehx = context.dfeat_x[j][mask] @ coef
                        ehy = context.dfeat_y[j][mask] @ coef
                        gs[mask] = sqv[mask, j] * ehx + rho * xi * ehy / 2.0
                        gp[mask] = xi * orto * ehy / 2.0
                cv_num += float(((target - eh) ** 2).mean())
                cv_den += float(target.var()) + 1e-30
                over = (eh < h_floor) | (eh > h_cap)
                n_clipped += int(over.sum())
                ehc = np.clip(eh, h_floor, h_cap)
                # Z = (score numerator) / h, capped at the
                # theoretical/engineering budgets above. For the
                # increment route gs, gp are the W-coefficients of dh
                # (both channels directly); for the derivative route
                # they are assembled from (h_x, h_y).
                zs_raw = gs / ehc
                zp_raw = gp / ehc
                n_score_capped += int((np.abs(zs_raw) > zs_cap).sum())
                n_score_capped += int((np.abs(zp_raw) > zp_cap).sum())
                zs_new[:, j] = np.clip(zs_raw, -zs_cap, zs_cap)
                zp_new[:, j] = np.clip(zp_raw, -zp_cap, zp_cap)
                h_path[:, j] = ehc * kill[:, j]
        return h_path, zs_new, zp_new, (n_clipped, n_score_capped, cv_num, cv_den)

    def projector(self) -> DiffusionMartingaleProjector:
        return DiffusionMartingaleProjector()

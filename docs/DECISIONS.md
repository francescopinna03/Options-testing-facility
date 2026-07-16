# SSFV implementation decisions

Confirmed 2026-07-15 against the paper
*Schrödinger Stochastic Feedback Volatility — complete formulation and
technical closure* (Pinna) and
[`SSFV_IMPLEMENTATION_ARCHITECTURE.md`](SSFV_IMPLEMENTATION_ARCHITECTURE.md).
Where the paper and the architecture document diverge, this file is the
binding record.

## D1 — Canonical interface taxonomy

One vocabulary across code, docs, and paper revisions:

| canonical API                | paper name (§migration)                  |
| ---------------------------- | ---------------------------------------- |
| `PriorModel`                 | `PriorGenerator`                         |
| `ConstraintFamily`           | — (builds the test family Ψₙ)            |
| `CylindricalPotential`       | `SchrodingerPotential` (Φₙ = λₙᵀΨₙ)      |
| `MartingaleProjector`        | `MartingaleProjector`                    |
| `ProjectedCharacteristics`   | `DoobCharacteristics`                    |
| `ProjectedBSDESolver`        | `ProjectedBSDE`                          |
| `FiniteDualCalibrator`       | `SSFVCalibrator` (with `ProjectiveSequence`) |
| `ProjectiveSequence`         | `SSFVCalibrator` (level loop)            |
| `ReweightedPosteriorMeasure` | — (likelihood route, implemented)         |
| `CharacteristicPosteriorModel` | — (transformed characteristics, future direct simulation; split per D11.6) |
| `CertificateBundle`          | —                                        |
| `SSFVEvaluator`              | `SSFVEvaluator`                          |
| `ImaginaryTimePropagator`    | `ImaginaryTimePropagator` — **out of core**, see D5 |

`ConstraintFamily` and `CylindricalPotential` are distinct: the former
constructs Ψₙ, the latter represents Φₙ = λₙᵀΨₙ. The paper's
`MarginalScaler` (IPF) becomes an optional backend of
`FiniteDualCalibrator` (see D6).

## D2 — Dependencies

`dependencies = []` stays. Extras: `numerical` (numpy, scipy), `torch`,
`data` (pandas, pyarrow, duckdb), `lseg`, `dev` (pytest, hypothesis, ruff,
mypy). Import discipline: `import otf` and `import otf.ssfv` load only
stdlib types/interfaces; concrete numerical modules import NumPy/SciPy at
module level and are imported explicitly by their users. CI lanes:
`legacy-minimal` (no NumPy), `ssfv-numerical`, `ssfv-torch`,
`large/scheduled` (GPU, QMC, large MC).

## D3 — Greenfield scope

The architecture tree (~40 modules) is a destination, not the first
commit. First code drop: typed result objects, Protocol interfaces,
config/manifest, one bounded cylindrical basis, diffusion-only simulator,
legacy regression freeze. Grow module-by-module behind the interfaces.

## D4 — Legacy bridge is a *different law construction*, not an approximation

Legacy: `δb_V = α·g(v)·v·(ρξ ∂ₓΘ + ξ² ∂ᵥΘ)`, quadratic Θ, martingale
*penalty*. Projected SSFV (diffusion): `δb_X* = 0`,
`δb_V* = ξ²(1−ρ²)·v·U_v`, martingale root Λ structural
(paper cor:diffusion-ssfv). They coincide only under `U_x = −ρξU_v`,
which the legacy ansatz does not impose. Banned phrasings: "low-order
approximation of exact SSFV", "coarser precision of the same drift".
Correct labels: **legacy restricted variance-channel deformation** vs
**exact martingale-projected entropic deformation**. Two orthogonal
ablation axes, never conflated:

1. *law-construction ablation*: prior / legacy bridge / projective SSFV;
2. *potential-class ablation*: P0…P5, entirely inside the projective backend.

## D5 — PDE backend out of M0–M6

`ImaginaryTimePropagator` is not in the core milestones. It remains a
theoretical representation, a future independent backend (milestone M7),
an audit of the probabilistic solution, and a platform for classical
regularity. In the paper: *optional deterministic backend*, not core
requirement.

## D6 — Finite dual is the primary algorithm; IPF is a backend

Primary solver: the projective sequence of finite duals
`Dₙ(λ) = λᵀaₙ − Y₀(λᵀΨₙ)` with L-BFGS / trust-region Newton on the
reduced dual. IPF is block-coordinate ascent on the same convex dual —
useful for maturity-grouped constraints and density-grid representations;
kept as an optional comparable backend. Paper algorithm naming to be
updated accordingly:

- Algorithm I — Projective finite-dual calibration
- Algorithm II — Particle and reweighting realization
- Algorithm III — Optional block-IPF / density-scaling solver
- Algorithm IV — Optional imaginary-time solver

## D7 — Data layer: interface now, lake later

No Parquet/DuckDB rewrite before the numerical core. Introduce a single
in-memory `MarketSurface` interface with `LegacyCSVSurfaceAdapter` now;
`ParquetSurfaceAdapter` later. Order: M0–M3 current CSV; M4 normalized
schema; M5 Parquet/DuckDB for OOS panels. The architecture document's
"Market-data architecture" §5 is a *target*, not a prerequisite.

## D8 — Feller: three separate senses, corrected wording

Theoretical admissibility ≠ scheme stability ≠ prior-fit regularization
(architecture doc §7.2 stands). Correction to MODEL_HISTORY.md: Feller
violation does **not** prevent existence or simulation of the CIR; it can
make a *specific scheme* (notably legacy full-truncation Euler)
inaccurate near zero. With QE or better schemes, non-Feller is not a
structural failure. The Feller penalty in the legacy prior fit is kept and
relabeled *scheme-robustness regularizer*, never an admissibility
constraint. The SSFV probabilistic realization requires no Feller
condition (paper thm:global-bsde-realization).

## D9 — Target repository

`francescopinna03/Options-testing-facility`, branch `main`; architecture
document at commit `127ab4c23bd1731cbb8f8f2ce73a0e4b9bf4dd7b`. Local
checkout: `~/Options-testing-facility`.

## D10 — M1 numerical realization (added 2026-07-15, implementation)

Decisions taken while building the M1 core; each is documented at the
point of use in the code.

1. **Reference BSDE solver = fixed-multiplier Picard Hopf-Cole**
   (`otf/ssfv/bsde/picard.py`), the paper's self-consistent-field
   iteration (thm:self-consistent-hopf-cole): each pass is a *linear*
   killed-FK regression, so regression error never feeds through a squared
   control. The direct quadratic-driver recursion
   (`otf/ssfv/bsde/regression.py`) is kept as a diagnostic backend only.
2. **Score extraction from centered increments**, not from derivatives of
   the fitted value: hats' step-function derivative amplifies coefficient
   noise by 1/knot-spacing. Derivative route retained as cross-check.
3. **Dual optimizer = static exponential-family Newton warm start +
   Gauss-Newton on the field-refreshed moment map with pseudo-inverse**
   (`otf/ssfv/dual/calibrator.py`), replacing the architecture document's
   joint L-BFGS: the finite-sample gradient a − E^w[Psi] is not the exact
   gradient of the sample dual (missing E^w[d_lambda N^S]), and in
   dynamically-replicable directions the reduced Jacobian is
   near-singular — a static update overshoots by 1/(1−gamma) and diverges.
   The pseudo-inverse takes the minimal-norm step in the identifiable
   subspace. Safeguards: exact sup-norm trust region, ESS backtracking.
4. **Multipliers are identified only up to replicable gauge**: synthetic
   recovery is asserted on the *law* (call prices, entropy, moments,
   feedback field), never on the raw lambda vector (paper §empirical).
5. **Statistical gauge floor** `var_floor = 1e-3` in the hat family:
   directions without prior mass have no dual curvature and drive
   multipliers toward the relative-interior boundary (§6.3). Principled
   replacement: the paper's Sobolev regularization (M2).
6. **Bounds used by the solver are exact, not triangle-inequality**:
   sup|Phi| and Lip(Phi) evaluated at the knots
   (`LambdaPotential.sup_norm_exact/lipschitz_bound`). |U_x| <= Lip(Phi)
   is a theorem (prop:tangential-lipschitz); the orthogonal-score cap is an
   engineering guard with its capped fraction reported.
7. **Gram ridge shrinks toward the fold-mean predictor**, so the §16.2
   property "zero potential returns the prior" holds exactly under
   regularization.
8. **External-DGP calibration (variance-shifted marginals, §14.1 DGP 3)
   is M2 scope**: at M1 path counts the martingale field's estimation
   noise dominates its true size (true log-weight std ~ sqrt(2H)); the
   refinement study + Sobolev regularization are the cure, and the
   certificates already detect the failure mode (ESS collapse, entropy
   discrepancy).

## D11 — M1 review closure (added 2026-07-16, PR #2 review)

Outcomes of the foundational review of the first code drop; each item is
implemented, tested, and documented at the point of use.

1. **Cumulative tail-testing family.** A fixed-interval hat family is
   *not* convergence determining on R (normalized hats are constant in
   the tails). Levels are now cumulative: level n+1 keeps every level-n
   column unchanged and appends midpoint hats, outward-extension hats
   (observed support grows without bound) and capped tail ramps.
   Convergence determination is claimed for the cumulative limit only.
2. **Nested normalization plan.** Per-level independent standardization
   plus gauge removal destroyed exact nesting. Normalization is now a
   block-triangular plan: inherited columns keep their transform verbatim
   and can never be removed; new columns are residualized (modified
   Gram-Schmidt on the pilot sample) against everything already accepted;
   rejections (raw or residual variance below `var_floor`) are recorded.
   `C_{n+1} ⊆ C_n` is structural; upward embedding is zero-padding
   verified at 1e-12. The projective Cauchy certificate is a theorem
   again, not an empirical diagnostic.
3. **Frozen-field solution semantics.** The Picard solver ends with a
   final consistency sweep: the converged field Z-bar is frozen, one last
   linear propagation rebuilds (h, Y_0, density) in a single pass, and
   the sweep-extracted fields are a diagnostic only. The returned object
   is an approximate solution of the frozen linear problem plus a
   reported fixed-point residual r_FP = ||Z^{S,eval} − Z-bar^S||; the
   solve fails loudly above `fp_tolerance`. Substituting Z^eval back
   would re-associate h with the previous field and is never done.
   The score ratio gs/ehc is classified as a first-order-in-dt
   discretization; the dt table (CLI `--dt-table`, dual-solver test)
   records mean-type (bias) and variance-type (noise-accumulating)
   metrics separately.
4. **Calibrator is a reduced-moment-map solver, and says so.**
   `ReducedMomentMapCalibrator` (was `AlternatingDualCalibrator`). The
   moment-map zero coincides with the sample-dual FOC by a structural
   theorem: Girsanov moves only W^perp in the diffusion sector, so W^S
   stays a Q-Brownian motion and E^Q[d_lambda N^S] = 0 (documented in
   the module docstring). Steps are accepted only if the moment residual
   strictly decreases AND the ESS survives; the stage-1 static move is
   trust-regioned (exact sup-norm) and ESS-backtracked. Certified dual
   ascent via implicit differentiation of the Picard fixed point is M2.
5. **Signed certificates.** `DualityCertificate.gap = H_LR − D_n`
   (signed; weak duality demands ≥ 0 up to MC error),
   `ProjectiveCertificate.cauchy_slack = ΔH − KL` (signed), and
   `posterior_mean_semistatic_gain = E^{Q_n}[N^S_T]` (must vanish) are
   reported; negative values beyond MC tolerance fail tests, never get
   absolute-valued away.
6. **Honest interfaces and manifests.** `PosteriorMeasure` split into
   `ReweightedPosteriorMeasure` (implemented) and
   `CharacteristicPosteriorModel` (future direct simulation); config
   defaults name the algorithms that actually run; the experiment
   manifest serializes the concrete component dataclasses field by field;
   the diffusion projector reports min(v) instead of a placeholder.
7. **Independent external DGP with known answers.** Girsanov tilt of
   W^perp with constant control c: exact entropy c²T/2, one-sided
   I-projection bound on the calibrated entropy, independent seeds for
   calibration/targets/holdout. At M1 path counts the fit stalls loudly;
   the measured N-scaling of the stall (5.7e-2 / 3.9e-2 / 7.9e-3 at
   8k/16k/32k paths, test env-gated behind SSFV_SLOW=1) confirms the
   field-noise attribution of D10.8 as a *measurement*, closing the
   review's demand that it not remain an unfalsified hypothesis.

### D11 addenda (2026-07-16, second review pass)

8. **Triangular scheme, not the literal limit.** The variance gauge is
   sample selection: at fixed pilot-batch size N the active family
   stabilizes in a finite space and is not convergence determining even
   though the raw cumulative family is. The code at fixed N realizes one
   row Psi_{n(N),N} of a triangular scheme; the theorem's full limit is
   the double limit N → ∞, n(N) → ∞, var_floor_N ↓ 0. The clean
   alternative — regularization that never permanently removes
   directions — is the Sobolev route (M2). Not an M1 blocker: nesting of
   the levels actually built is exact, and rejections are recorded so
   the realized row is auditable.
9. **Certificates decompose, defects are named.** For approximately
   calibrated laws the signed gap obeys the exact sample identity
   H − D_n = λᵀ(m − a) − E^Q[N^S_T]; the bundle stores
   `moment_defect`, `semistatic_defect` and `duality_identity_residual`
   (must sit at fp/MC noise — tested at 1e-10). A negative gap explained
   by the defects is primal inadmissibility of the approximate fit, not
   a weak-duality violation. The Cauchy slack likewise decomposes
   against the coarse potential (`cauchy_moment_term`,
   `cauchy_semistatic_term`, `cauchy_identity_residual`), separating
   nesting error, moment error, dynamic-term error and genuine
   Pythagorean-identity violations.
10. **External DGP is symmetric holdout.** The calibrated multipliers
    are frozen and the BSDE is re-solved on the holdout batch (fresh
    Picard context); DGP, prior and fitted calls are expectations on the
    same holdout sample. The reported law is out of sample with respect
    to the solver's regression error, not only the moment noise.

## D12 — M2 numerical realization (added 2026-07-16)

Order followed: Sobolev regularization, then implicit differentiation
*before* the definitive Schur certificate (the implicit-diff Jacobian is
the clean way to compute the true reduced Jacobian), refinement tables
throughout. Findings are recorded as measurements.

1. **Sobolev geometry.** `NestedHatFamily.sobolev_gram(level)` is the
   exact H^1 Gram of the normalized columns, S = I + W^T D_raw W: L^2
   block under the prior-pilot measure (identity, by orthonormality),
   derivative block under Lebesgue (exact — derivatives are piecewise
   constant on the knot union; ramps saturate, so S is finite despite
   the tail tests not being L^2(dk)). The calibrator solves the
   regularized FOC a − m(λ) − σ²Sλ = 0; σ² = 0 reproduces M1 exactly.
   The raw moment residual stays reported; the duality-defect
   decomposition (D11.9) quantifies what regularization gave up.
2. **Measured: Sobolev does not cure the external-DGP stall at M1 path
   counts.** A σ² grid on the Girsanov DGP at 8k paths shrinks H toward
   0 and worsens holdout error monotonically: the stall is not
   multiplier runaway (which Sobolev bounds) but a *target-attainability
   floor* — targets from an independent batch differ from anything the
   calibration batch can represent by O(sqrt(d/N)) ≈ 3e-2 at 8k, and
   the fit stalls there. The refinement axis for external accuracy is
   N (measured in D11.7), not σ².
3. **Implicit differentiation of the Picard fixed point**
   (`PicardHopfColeSolver.dn_s_dlam`). The tangent of the frozen-field
   solution solves the linear equation dZ = L dZ + b (b = terminal
   perturbation through the killed-FK pass, L = the killing-weight
   coupling d kill = −dt·Z̄·dZ·kill); it is solved by the same damped
   iteration as the field, with regression operators fixed (they are
   linear maps cached in the context) and clip/cap active sets frozen at
   the solution. Exact on the sample modulo those measure-zero kinks:
   validated against central finite differences (5e-2 max-relative,
   exact where the caps are quiet), 13× faster than the FD Jacobian,
   one tangent solve for all directions. Default Jacobian backend; FD
   retained as cross-check.
4. **Relative pseudo-inverse cutoff was a bug in disguise.** The reduced
   Jacobian can have one strongly hedged direction with |dm/dλ| ≫ 1
   (measured: singular value 21 against a mid-block of 0.1–0.5); a
   cutoff *relative to the largest singular value* then silently
   discards every other direction and Gauss-Newton degenerates to rank
   one — the actual mechanism behind part of the external stall.
   `jac_rcond` is now an *absolute* floor in static-curvature units
   (the basis is orthonormal on the pilot sample, so singular values of
   dm/dλ measure directly what survives dynamic cancellation).
5. **Levenberg-Marquardt replaces truncated-pinv Gauss-Newton.** Weakly
   identifiable directions are damped, not cut; the damping adapts to
   the strong nonlinearity (overshoot 1/(1−γ)) along them. Acceptance
   still requires strict FOC-residual decrease AND ESS survival.
6. **Convergence is asserted on the identifiable subspace.** Directions
   with reduced singular value below the floor are declared
   unidentifiable at this sample size; the FOC residual is split and
   both parts are reported (status string + Schur certificate). Chasing
   the unidentifiable part is exactly the field-noise chase of D10.8.
7. **Schur certificate** (`ConditioningCertificate` extension): reduced
   singular range, identifiable dimension, and the identifiable/
   unidentifiable FOC-residual split, computed from the implicit-diff
   Jacobian carried by `DualFitResult.reduced_jacobian`, with the same
   floor the optimizer used (threaded by `ProjectiveSequence`).

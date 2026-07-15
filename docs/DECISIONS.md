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
| `PosteriorMeasure`           | — (likelihood + transformed characteristics) |
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

"""Immutable result objects for the SSFV projective core.

Every class here corresponds to a mathematical object in the SSFV paper
(Pinna, *Schrödinger Stochastic Feedback Volatility*), per the governing
rule of docs/SSFV_IMPLEMENTATION_ARCHITECTURE.md §0: every numerical object
must correspond to a mathematical object defined in the paper.

This module is stdlib-only at import time (DECISIONS.md D2). Array-typed
fields are annotated lazily; concrete arrays are produced by the numerical
modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import stdlib-only
    import numpy as np

    Array = np.ndarray
else:
    Array = Any


# ---------------------------------------------------------------------------
# Prior simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathBatch:
    """A frozen batch of prior paths with exposed pathwise innovations.

    The BSDE solver, the energy identity, and the Girsanov likelihood all
    require the *innovations* that generated each path, not only the states
    (architecture doc §3.1). ``d_w`` stores the orthogonal standard-normal
    basis increments scaled by sqrt(dt): shape (n_paths, n_steps, 2), with
    ``d_w[..., 0]`` driving the price channel and ``d_w[..., 1]`` the
    orthogonal component; the engine documents its correlation convention.
    ``d_w`` may be None for schemes (e.g. Andersen QE) whose variance update
    is not driven by a Gaussian increment — such batches are pricing-grade
    and must be rejected by the BSDE/likelihood layer.
    """

    times: Array  # (n_steps + 1,)
    x: Array  # (n_paths, n_steps + 1) log forward-price
    v: Array  # (n_paths, n_steps + 1) variance
    d_w: Array | None  # (n_paths, n_steps, 2) orthogonal N(0,1)*sqrt(dt)
    jump_offsets: Array | None
    jump_marks: Array | None
    initial_state: Array  # (2,) = (x0, v0)
    scheme: str
    seed: int
    batch_hash: str

    @property
    def n_paths(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.x.shape[1]) - 1

    @property
    def has_innovations(self) -> bool:
        return self.d_w is not None


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of PriorModel.validate_parameters()."""

    ok: bool
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalCharacteristics:
    """Prior local characteristics (b0, a, nu0) at a state/time.

    ``drift`` is (b0_X, b0_V); ``diffusion`` is the 2x2 matrix a(v) flattened
    row-major; jump data are None in the diffusion-only sector.
    """

    drift: tuple[float, float]
    diffusion: tuple[float, float, float, float]
    jump_intensity: float | None = None


@dataclass(frozen=True)
class ProjectedCharacteristics:
    """Transformed (Doob/posterior) characteristics at a state/time.

    Paper: eq:ssfv-drift-feedback and eq:ssfv-jump-kernel. In the
    diffusion-only sector Lambda = -(U_x + rho*xi*U_v) in closed form,
    delta_b_x = 0 and delta_b_v = xi^2 (1-rho^2) v U_v
    (cor:diffusion-ssfv).
    """

    delta_b_x: float
    delta_b_v: float
    lam: float
    jump_tilt_log: float | None = None


# ---------------------------------------------------------------------------
# Constraint families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationMap:
    """Nested normalization plan of a cumulative basis level (arch doc
    §6.2, review fix R2).

    The normalized columns are psi_tilde = (psi_raw - means) @ transform,
    where ``transform`` is block-triangular in column-creation order:
    every accepted column is the previous levels' column *unchanged*, and
    columns new at this level are standardized then residualized against
    the span of all previously accepted columns on the pilot prior
    sample. Consequences, both structural:

    * span(Psi_tilde_n) ⊆ span(Psi_tilde_{n+1}) exactly — the first
      ``inherited_dim`` columns of level n+1 ARE level n's columns, so
      C_{n+1} ⊆ C_n holds by construction and the projective Cauchy
      certificate applies as a theorem, not an approximation;
    * embedding coefficients upward is zero-padding (machine-exact).

    Inherited columns can never be removed. A new column whose raw or
    residual variance falls below the floor is *rejected* and recorded in
    ``rejected_indices`` — the statistical gauge (§6.3) acts only on new
    directions.
    """

    means: Array  # (d_raw,) raw column means on the pilot sample
    stds: Array  # (d_raw,) raw column stds (diagnostic; transform is authoritative)
    transform: Array  # (d_raw, d_kept), block-triangular in creation order
    kept_indices: tuple[int, ...]  # raw column generating each kept direction
    rejected_indices: tuple[int, ...]  # raw columns rejected by the variance gauge
    inherited_dim: int  # leading kept columns identical to the previous level
    eps: float
    normalization_seed: int


@dataclass(frozen=True)
class ConstraintLevel:
    """One level n of the *cumulative* bounded test family Psi_n.

    Columns are bounded piecewise-linear functions of forward
    log-moneyness k = log(S_T / F_{0,T}): hats (compact support) and
    capped tail ramps (constant in the far tails). Levels are cumulative:
    level n+1 contains every level-n column unchanged and appends new
    ones (interior midpoint hats, outward-extension hats, new tail
    ramps), so nesting is an identity of column lists, not a span
    argument. ``knots`` is the sorted union of all column breakpoints —
    the potential is piecewise linear exactly there.
    """

    n: int
    family: str
    knots: Array  # sorted union of all column breakpoints
    col_kind: Array  # (d_raw,) int: 0 = hat, 1 = left ramp, 2 = right ramp
    col_loc: Array  # (d_raw,) hat center / ramp edge
    col_wl: Array  # (d_raw,) hat left width / ramp width
    col_wr: Array  # (d_raw,) hat right width (unused for ramps)
    k_min: float
    k_max: float
    normalization: NormalizationMap | None

    @property
    def dim_raw(self) -> int:
        return int(self.col_kind.shape[0])

    @property
    def dim(self) -> int:
        if self.normalization is None:
            return self.dim_raw
        return len(self.normalization.kept_indices)


# ---------------------------------------------------------------------------
# Martingale projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectorDiagnostics:
    """Per-batch diagnostics of the scalar martingale root (arch doc §8.2).

    A non-attained state is a structural result, not a numerical nuisance;
    the fraction must be reported, never clipped away.
    """

    max_root_residual: float
    min_derivative: float
    max_abs_multiplier: float
    newton_steps: int
    bisection_fallbacks: int
    non_attained_fraction: float
    overflow_protected_fraction: float


# ---------------------------------------------------------------------------
# BSDE solution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BSDESolution:
    """Output of ProjectedBSDESolver.solve (arch doc §3.3, §9.5).

    ``y0`` is the scalar Y_0 entering the dual D_n(lambda) = lambda^T a_n
    - Y_0. ``y`` and ``z`` are pathwise values of Y_t and the Brownian
    control Z_t on the simulation grid; ``z_orth`` is the
    martingale-projected orthogonal component actually driving the entropy.
    ``log_density`` is the pathwise log dQ/dQ0 reconstructed from the
    solution (likelihood layer), and ``residuals`` carries the §9.5 solver
    residuals by name.
    """

    y0: float
    y: Array  # (n_paths, n_steps + 1)
    z: Array  # (n_paths, n_steps, 2) full Brownian control
    z_orth: Array  # (n_paths, n_steps) projected orthogonal control
    u_jump: Array | None
    log_density: Array  # (n_paths,)
    projector: ProjectorDiagnostics | None
    residuals: Mapping[str, float]
    # Sample dual estimate log-mean-exp(Phi - N^S): exactly consistent with
    # the self-normalized weights, hence the dual objective the optimizer
    # must use; |y0_sample - y0| is the likelihood-normalization residual.
    y0_sample: float = 0.0


# ---------------------------------------------------------------------------
# Dual optimization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualFitResult:
    """Output of FiniteDualCalibrator.fit at one level (arch doc §3.4, §10).

    ``gradient`` is the gradient of the (possibly Sobolev-regularized)
    sample dual — the FOC residual the optimizer drives to zero;
    ``moment_residuals`` is always the *raw* a_n - m(lambda).
    ``reduced_jacobian`` is dm/dlambda at the returned multipliers when
    the backend provides it (implicit differentiation of the Picard
    fixed point): the true reduced Jacobian, dynamic cancellation
    included, feeding the §10.3 Schur/conditioning certificate.
    """

    level: int
    lam: Array  # (d_n,) multipliers in the normalized basis
    dual_value: float
    gradient: Array  # (d_n,) regularized-dual FOC residual
    gradient_norm: float
    moment_residuals: Array  # (d_n,) raw a_n - m(lambda)
    moment_residual_norm: float
    n_iterations: int
    converged: bool
    status: str
    warm_started: bool
    reduced_jacobian: Array | None = None  # (d_n, d_n) dm/dlambda


# ---------------------------------------------------------------------------
# Certificates (arch doc §12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualityCertificate:
    """§12.1 finite duality: primal entropy vs dual value.

    ``gap`` is the *signed* primal-dual gap H^LR - D_n. For the returned
    (approximately calibrated) law Q-bar_lambda the certifiable statement
    is the exact sample identity

        H - D_n = lambda^T (m(lambda) - a) - E^{Q-bar}[N^S_T],

    so the gap decomposes into ``moment_defect`` = lambda^T (m - a) and
    ``semistatic_defect`` = -E^{Q-bar}[N^S_T];
    ``duality_identity_residual`` = (H - D) - (moment_defect +
    semistatic_defect) must sit at floating-point/MC noise. A negative
    gap explained by the defects is *primal inadmissibility of the
    approximate fit*, not a weak-duality violation; an identity residual
    beyond noise is a real inconsistency and must fail.
    """

    primal_entropy: float
    dual_value: float
    gap: float  # signed: primal_entropy - dual_value
    moment_residual_norm: float
    moment_defect: float = 0.0  # lambda^T (m(lambda) - a)
    semistatic_defect: float = 0.0  # -E^{Q-bar}[N^S_T]
    duality_identity_residual: float = 0.0  # gap - (moment + semistatic defects)


@dataclass(frozen=True)
class EntropyCertificate:
    """§12.2 entropy double-entry: likelihood (LR) vs characteristic energy (EN)."""

    h_lr: float
    h_en: float
    discrepancy: float


@dataclass(frozen=True)
class MartingaleCertificate:
    """§12.4 martingale certificate at each maturity/terminal horizon."""

    forward_error: float
    max_conditional_drift_residual: float
    projector_residual: float


@dataclass(frozen=True)
class ProjectiveCertificate:
    """§12.3 projective Cauchy certificate for the pair (n_prev, n).

    KL(Q_N | Q_n) <= H_N - H_n and ||Q_N - Q_n||_TV <= sqrt((H_N - H_n)/2).
    ``kl_direct`` is the direct estimate of the left-hand side when
    available; the observable plateau of H_n is the main convergence
    certificate. ``cauchy_slack`` is the *signed* slack
    H_N - H_n - KL(Q_N | Q_n).

    For approximately calibrated posteriors the slack is not
    automatically nonnegative; its exact sample decomposition against
    the coarse potential is

        slack = lambda_n^T (m_N^{(n)} - m_n)
                - (E^{Q_N}[N^{S,n}_T] - E^{Q_n}[N^{S,n}_T]),

    stored as ``cauchy_moment_term`` and ``cauchy_semistatic_term``
    (the parenthesized dynamic difference, entering with a minus sign).
    ``cauchy_identity_residual`` = slack - (moment_term -
    semistatic_term) must sit at floating-point/MC noise: it separates
    nesting error, moment-calibration error and dynamic-term error from
    a genuine numerical violation of the Pythagorean identity.
    """

    n_prev: int | None
    h_n: float
    delta_h: float | None
    kl_direct: float | None
    tv_bound: float | None
    cauchy_slack: float | None = None
    cauchy_moment_term: float | None = None  # lambda_n^T (m_N^{(n)} - m_n)
    cauchy_semistatic_term: float | None = None  # E^{Q_N}[N^{S,n}] - E^{Q_n}[N^{S,n}]
    cauchy_identity_residual: float | None = None


@dataclass(frozen=True)
class ConditioningCertificate:
    """§10.3 conditioning: static covariance spectrum plus, when the
    calibrator provides the implicit-differentiation Jacobian, the *true
    reduced* (Schur) spectrum.

    The normalized basis is orthonormal on the pilot sample, so the
    static curvature is the identity and the singular values of
    dm/dlambda are directly the fraction of a unit static response that
    survives dynamic cancellation by the semistatic hedge (gamma -> 1
    directions have singular values -> 0). ``identifiable_dim`` counts
    singular values above the absolute floor; the FOC residual is split
    into its identifiable and unidentifiable parts — the latter is what
    this sample size cannot distinguish and must be reported, never
    chased.
    """

    eigen_min: float
    eigen_max: float
    condition_number: float
    n_removed_directions: int
    reduced_sv_min: float | None = None
    reduced_sv_max: float | None = None
    identifiable_dim: int | None = None
    identifiable_residual_norm: float | None = None
    unidentifiable_residual_norm: float | None = None


@dataclass(frozen=True)
class CertificateBundle:
    """Immutable, serializable certificate record (arch doc §3.6, §12).

    A successful fit without a certificate bundle is not a valid research
    result. ``diagnostics`` carries auxiliary scalars (ESS, max weight
    share, exponential-moment grid values, ...).
    """

    level: int
    duality: DualityCertificate
    entropy: EntropyCertificate
    martingale: MartingaleCertificate
    projective: ProjectiveCertificate
    conditioning: ConditioningCertificate | None
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        def convert(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                return {k: convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, Mapping):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        return convert(self)  # type: ignore[return-value]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, sort_keys=True)

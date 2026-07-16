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
    """Prior-marginal standardization of a basis level (arch doc §6.2).

    psi_tilde_j = (psi_j - mean_j) / sqrt(var_j + eps), followed by gauge
    removal encoded in ``kept_indices`` (dropped constant direction and
    exact linear dependencies). Saved so coefficients stay interpretable and
    refinement levels comparable.
    """

    means: Array  # (d_raw,)
    stds: Array  # (d_raw,)  = sqrt(var + eps)
    eps: float
    kept_indices: tuple[int, ...]  # gauge-reduced raw indices retained
    normalization_seed: int


@dataclass(frozen=True)
class ConstraintLevel:
    """One level n of the nested bounded test family Psi_n.

    ``knots`` are the dyadic nodes in forward log-moneyness
    k = log(S_T / F_{0,T}); nesting span(Psi_n) ⊆ span(Psi_{n+1}) is exact
    for piecewise-linear hats on nested knot grids and is unit-tested.
    """

    n: int
    family: str
    knots: Array  # (d_raw + 2,) including boundary
    k_min: float
    k_max: float
    normalization: NormalizationMap | None

    @property
    def dim_raw(self) -> int:
        return int(self.knots.shape[0]) - 2

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
    """Output of FiniteDualCalibrator.fit at one level (arch doc §3.4, §10)."""

    level: int
    lam: Array  # (d_n,) multipliers in the normalized basis
    dual_value: float
    gradient: Array  # (d_n,) = a_n - E^{Q_n}[Psi_n]
    gradient_norm: float
    moment_residuals: Array  # (d_n,)
    moment_residual_norm: float
    n_iterations: int
    converged: bool
    status: str
    warm_started: bool


# ---------------------------------------------------------------------------
# Certificates (arch doc §12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualityCertificate:
    """§12.1 finite duality: primal entropy vs dual value.

    ``gap`` is the *signed* primal-dual gap H^LR - D_n. Weak duality
    requires gap >= 0 up to Monte Carlo error: a negative value beyond
    the MC tolerance is a violation to be explained, never folded into
    an absolute value.
    """

    primal_entropy: float
    dual_value: float
    gap: float  # signed: primal_entropy - dual_value
    moment_residual_norm: float


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
    H_N - H_n - KL(Q_N | Q_n): the theorem requires it >= 0, and a
    negative value beyond tolerance must fail the certificate.
    """

    n_prev: int | None
    h_n: float
    delta_h: float | None
    kl_direct: float | None
    tv_bound: float | None
    cauchy_slack: float | None = None


@dataclass(frozen=True)
class ConditioningCertificate:
    """§10.3 reduced-Jacobian conditioning (diagnostic, not existence test)."""

    eigen_min: float
    eigen_max: float
    condition_number: float
    n_removed_directions: int


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

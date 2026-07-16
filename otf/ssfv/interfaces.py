"""Protocol interfaces for the SSFV projective core.

Canonical taxonomy per docs/DECISIONS.md D1; paper-name correspondences in
docstrings. Package boundaries are enforced here: the outer dual optimizer
must not reach into solver internals — all communication happens through
the typed result objects in otf.ssfv.types (arch doc §2).

Stdlib-only at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from otf.ssfv.types import (
    BSDESolution,
    ConstraintLevel,
    DualFitResult,
    LocalCharacteristics,
    PathBatch,
    ProjectedCharacteristics,
    ProjectorDiagnostics,
    ValidationReport,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    Array = np.ndarray
else:
    Array = Any


@runtime_checkable
class PriorModel(Protocol):
    """Prior law Q^0 (paper: PriorGenerator). Not posterior calibration.

    Must expose the innovations used to generate every path: a
    terminal-state-only simulator is insufficient for the BSDE, energy
    identity, and Girsanov likelihood (arch doc §3.1).
    """

    def simulate(self, n_paths: int, n_steps: int, horizon: float, seed: int) -> PathBatch: ...

    def characteristics(self, x: float, v: float, t: float) -> LocalCharacteristics: ...

    def price_european_cf(self, strike: float, maturity: float, call: bool) -> float | None: ...

    def validate_parameters(self) -> ValidationReport: ...


@runtime_checkable
class ConstraintFamily(Protocol):
    """Nested bounded test families Psi_n in forward log-moneyness.

    Builds Psi_n; distinct from CylindricalPotential, which represents
    Phi_n = lambda_n^T Psi_n (DECISIONS.md D1). The nesting invariant
    span(Psi_n) ⊆ span(Psi_{n+1}) must hold exactly and be tested.
    """

    def level(self, n: int) -> ConstraintLevel: ...

    def evaluate(self, level: ConstraintLevel, k_terminal: Array) -> Array:
        """Raw basis matrix (n_points, dim_raw) at log-moneyness points."""
        ...

    def evaluate_normalized(self, level: ConstraintLevel, k_terminal: Array) -> Array:
        """Standardized, gauge-reduced basis matrix (n_points, dim)."""
        ...

    def normalize(self, level: ConstraintLevel, k_prior_sample: Array) -> ConstraintLevel:
        """Return the level with its NormalizationMap fixed on a prior sample."""
        ...

    def embed(self, coefficients: Array, level_from: ConstraintLevel, level_to: ConstraintLevel) -> Array:
        """Exact coefficient embedding level n -> n+1 (warm starts)."""
        ...


@runtime_checkable
class CylindricalPotential(Protocol):
    """Phi_n = sum_m lambda_m^T Psi_m(X_{T_m}) (paper: SchrodingerPotential).

    Bounded by construction when the family is bounded and lambda finite.
    """

    def terminal_value(self, k_terminal: Array) -> Array:
        """Phi_n at terminal log-moneyness points, shape (n_points,)."""
        ...

    def sup_norm_bound(self) -> float: ...


@runtime_checkable
class MartingaleProjector(Protocol):
    """Scalar multiplier Lambda solving the local martingale root
    (paper eq:lambda-root; closed form -(U_x + rho*xi*U_v) in the
    diffusion-only sector, cor:diffusion-ssfv)."""

    def project(self, z_price: Array, z_orth: Array, v: Array) -> tuple[Array, ProjectorDiagnostics]:
        """Return the projected orthogonal control and diagnostics."""
        ...


@runtime_checkable
class ProjectedBSDESolver(Protocol):
    """Solves the martingale-projected (quadratic) BSDE for Y_0(Phi)
    (paper: ProjectedBSDE, eq:ssfv-quadratic-bsde / eq:ssfv-jump-bsde)."""

    def solve(self, paths: PathBatch, potential: CylindricalPotential) -> BSDESolution: ...


@runtime_checkable
class FiniteDualCalibrator(Protocol):
    """Maximizes D_n(lambda) = lambda^T a_n - Y_0(lambda^T Psi_n)
    (paper: SSFVCalibrator; primary algorithm per DECISIONS.md D6).

    IPF, when added, is an alternative backend of this interface —
    block-coordinate ascent on the same convex dual."""

    def fit(
        self,
        level: ConstraintLevel,
        targets: Array,
        paths: PathBatch,
        warm_start: Array | None = None,
    ) -> DualFitResult: ...


@runtime_checkable
class ReweightedPosteriorMeasure(Protocol):
    """Posterior Q_n identified by its likelihood on prior paths: the
    reweighting representation available *before* direct posterior
    simulation exists (arch doc §11.1). ReweightedPosterior implements
    exactly this — no more."""

    def expectation(self, values: Array) -> float:
        """E^{Q_n}[G] = E^{Q^0}[L G] via stabilized log-weights."""
        ...

    def log_weights(self) -> Array: ...

    def effective_sample_size(self) -> float: ...


@runtime_checkable
class CharacteristicPosteriorModel(Protocol):
    """Posterior identified by its transformed local characteristics —
    the object direct posterior simulation will need (drift feedback,
    projected multiplier field). No implementation exists yet; kept as a
    separate protocol so the reweighting layer does not have to pretend
    to provide pointwise characteristics it does not have."""

    def characteristics(self, x: float, v: float, t: float) -> ProjectedCharacteristics: ...


@runtime_checkable
class SSFVEvaluator(Protocol):
    """Feeds a frozen posterior to the existing OOS/statistical layers."""

    def price_calls(self, strikes: Sequence[float], maturity: float) -> Array: ...

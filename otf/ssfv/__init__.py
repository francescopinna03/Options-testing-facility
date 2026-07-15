"""SSFV projective core: exact martingale-projected entropic deformation.

This package implements the full Schrödinger Stochastic Feedback Volatility
(SSFV) model — the information projection of an affine (jump-)diffusion prior
onto the market-calibrated martingale set — as specified in
docs/SSFV_IMPLEMENTATION_ARCHITECTURE.md with the binding decisions of
docs/DECISIONS.md.

It is a *separate backend*: the legacy restricted variance-channel bridge in
otf.models.sfv stays untouched as a regression benchmark and ablation arm
(DECISIONS.md D4). The two are different law constructions, not two
precisions of the same drift.

Import discipline (DECISIONS.md D2): importing ``otf.ssfv`` loads only
stdlib types, Protocol interfaces, and configuration. Numerical modules
(``otf.ssfv.prior``, ``otf.ssfv.constraints``, ...) import NumPy at module
level and must be imported explicitly by their users.
"""

from otf.ssfv.types import (
    PathBatch,
    ValidationReport,
    LocalCharacteristics,
    ProjectedCharacteristics,
    ConstraintLevel,
    NormalizationMap,
    ProjectorDiagnostics,
    BSDESolution,
    DualFitResult,
    DualityCertificate,
    EntropyCertificate,
    MartingaleCertificate,
    ProjectiveCertificate,
    ConditioningCertificate,
    CertificateBundle,
)
from otf.ssfv.interfaces import (
    PriorModel,
    ConstraintFamily,
    CylindricalPotential,
    MartingaleProjector,
    ProjectedBSDESolver,
    FiniteDualCalibrator,
    PosteriorMeasure,
    SSFVEvaluator,
)
from otf.ssfv.config import (
    PriorConfig,
    SimulationConfig,
    ConstraintConfig,
    BSDEConfig,
    DualConfig,
    CertificateConfig,
    ExperimentConfig,
    derive_seed,
)

__all__ = [
    # types
    "PathBatch",
    "ValidationReport",
    "LocalCharacteristics",
    "ProjectedCharacteristics",
    "ConstraintLevel",
    "NormalizationMap",
    "ProjectorDiagnostics",
    "BSDESolution",
    "DualFitResult",
    "DualityCertificate",
    "EntropyCertificate",
    "MartingaleCertificate",
    "ProjectiveCertificate",
    "ConditioningCertificate",
    "CertificateBundle",
    # interfaces
    "PriorModel",
    "ConstraintFamily",
    "CylindricalPotential",
    "MartingaleProjector",
    "ProjectedBSDESolver",
    "FiniteDualCalibrator",
    "PosteriorMeasure",
    "SSFVEvaluator",
    # config
    "PriorConfig",
    "SimulationConfig",
    "ConstraintConfig",
    "BSDEConfig",
    "DualConfig",
    "CertificateConfig",
    "ExperimentConfig",
    "derive_seed",
]

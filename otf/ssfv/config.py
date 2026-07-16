"""Experiment configuration and reproducibility manifest (arch doc §4, §18).

Stdlib-only. No mutable global random state anywhere in the SSFV core:
explicit hierarchical seeds derived with :func:`derive_seed`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from typing import Any, Mapping


def derive_seed(root_seed: int, stream: str) -> int:
    """Deterministic child seed for a named stream (arch doc §4).

    Example streams: "prior", "normalization", "optimizer", "bootstrap",
    "posterior_validation".
    """
    digest = hashlib.sha256(f"{root_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@dataclass(frozen=True)
class PriorConfig:
    model: str = "heston_euler_ft"  # reference innovations-exact scheme
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    rate: float = 0.0
    jumps: str = "none"


@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int = 65536
    steps_per_year: int = 252
    horizon: float = 1.0
    random_engine: str = "pcg64"
    seed: int = 1729
    dtype: str = "float64"


@dataclass(frozen=True)
class ConstraintConfig:
    family: str = "nested_hat"
    levels: tuple[int, ...] = (0, 1, 2, 3)
    base_dim: int = 4  # interior nodes at level 0; d_{n+1} = 2 d_n + 1 (dyadic knot refinement)
    k_min: float = -1.0
    k_max: float = 1.0
    normalization_eps: float = 1.0e-12


@dataclass(frozen=True)
class BSDEConfig:
    # Reference backend actually used by the projective core (DECISIONS D10):
    # the regression backend is the diagnostic cross-check only.
    backend: str = "picard_hopf_cole"
    state_basis: str = "hat_tensor"
    cross_fitting_folds: int = 2


@dataclass(frozen=True)
class DualConfig:
    # Actual algorithm: static Newton warm start + Gauss-Newton with
    # pseudo-inverse on the reduced moment map (ReducedMomentMapCalibrator).
    optimizer: str = "reduced_moment_map_gauss_newton"
    gradient_tolerance: float = 1.0e-9
    moment_tolerance: float = 1.0e-3
    max_iterations: int = 8


@dataclass(frozen=True)
class CertificateConfig:
    entropy_tolerance: float = 1.0e-4
    martingale_tolerance: float = 1.0e-4
    min_ess_fraction: float = 0.1


@dataclass(frozen=True)
class ExperimentConfig:
    prior: PriorConfig = field(default_factory=PriorConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    bsde: BSDEConfig = field(default_factory=BSDEConfig)
    dual: DualConfig = field(default_factory=DualConfig)
    certificates: CertificateConfig = field(default_factory=CertificateConfig)

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def manifest(self, extra: dict | None = None,
                 components: Mapping[str, Any] | None = None) -> dict:
        """Self-contained experiment manifest (arch doc §4).

        ``components`` maps role names to the *concrete* component
        instances the experiment actually ran (solver, calibrator,
        family, prior); their dataclass fields are serialized verbatim so
        the manifest describes the executed algorithm, not a default.
        """
        m = {
            "git_sha": _git_sha(),
            "config_hash": self.config_hash(),
            "config": asdict(self),
            "seed_streams": {
                s: derive_seed(self.simulation.seed, s)
                for s in ("prior", "normalization", "optimizer", "bootstrap", "posterior_validation")
            },
        }
        if components:
            m["components"] = {k: component_manifest(v) for k, v in components.items()}
        if extra:
            m.update(extra)
        return m


def component_manifest(obj: Any) -> Any:
    """JSON-safe description of a concrete component: class name plus its
    dataclass fields. Scalars pass through, nested dataclasses recurse,
    anything else (arrays, callables) is reported by type name — the
    manifest must never silently misdescribe what ran."""
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"class": type(obj).__name__}
        for f in fields(obj):
            out[f.name] = component_manifest(getattr(obj, f.name))
        return out
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [component_manifest(v) for v in obj]
    return f"<{type(obj).__name__}>"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"

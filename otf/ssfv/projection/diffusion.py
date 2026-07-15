"""Diffusion-sector martingale projection (paper cor:diffusion-ssfv).

In the pure-diffusion sector the martingale projection is *structural*, not
a root search: the entropic tilt uses only the Brownian direction
orthogonal to the return innovation (the BSDE density is the stochastic
exponential of the W-perp component alone, thm:global-bsde-realization),
so the log-price drift is unchanged by construction and the scalar
multiplier has the closed form

    Lambda = -(U_x + rho * xi * U_v) = -Z^S / sqrt(v)   on {v > 0}
    (eq:lambda-diffusion; Z^S = sqrt(v) (U_x + rho xi U_v)).

The root is always attained: non_attained_fraction = 0 identically. This
module exists to (a) expose the implied multiplier field for diagnostics
and the explicit strategy Delta* = Lambda / S-bar (thm:explicit-delta),
and (b) pin the interface that the jump-sector safeguarded scalar solver
(arch doc §8.2) will implement later — where non-attainment is structural
and must be reported, never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otf.ssfv.types import ProjectorDiagnostics

__all__ = ["DiffusionMartingaleProjector"]


@dataclass(frozen=True)
class DiffusionMartingaleProjector:
    """Structural projector for the no-jump sector.

    ``v_floor`` only guards the *diagnostic* division Z^S/sqrt(v); the
    posterior law itself needs no value at v = 0 (lem:cir-nonsticky:
    the zero set has zero occupation, and the density uses Z-perp only).
    """

    v_floor: float = 1.0e-12

    def project(self, z_price: np.ndarray, z_orth: np.ndarray, v: np.ndarray
                ) -> tuple[np.ndarray, ProjectorDiagnostics]:
        """Return the admissible (orthogonal) control and diagnostics.

        In the diffusion sector the projected control *is* z_orth: the
        price-channel component z_price enters the posterior density only
        through the dynamic hedging term -N^S, never through the entropy.
        """
        lam = self.implied_multiplier(z_price, v)
        diag = ProjectorDiagnostics(
            max_root_residual=0.0,
            min_derivative=1.0,  # F'(lambda) = v > 0: strictly monotone
            max_abs_multiplier=float(np.max(np.abs(lam))) if lam.size else 0.0,
            newton_steps=0,
            bisection_fallbacks=0,
            non_attained_fraction=0.0,
            overflow_protected_fraction=0.0,
        )
        return z_orth, diag

    def implied_multiplier(self, z_price: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Lambda = -Z^S / sqrt(v) on {v > 0}; diagnostic field only."""
        sq = np.sqrt(np.maximum(v, self.v_floor))
        return -z_price / sq

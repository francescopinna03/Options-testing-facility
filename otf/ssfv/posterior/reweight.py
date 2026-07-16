"""Reweighted posterior representation (arch doc §11.1).

E^{Q_n}[G] = E^{Q^0}[L G] on the frozen prior paths, with stabilized
log-weights. Direct posterior simulation is a later, independent audit
(§11.2) — it must not exist before reweighting passes the synthetic tests.

Low marginal error with collapsed ESS is not an acceptable fit (§11.1,
failure policy §20): consumers must check effective_sample_size().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from otf.ssfv.types import BSDESolution, PathBatch

__all__ = ["ReweightedPosterior"]


@dataclass(frozen=True)
class ReweightedPosterior:
    """ReweightedPosteriorMeasure backed by BSDE log-densities on prior paths."""

    paths: PathBatch
    solution: BSDESolution
    _w: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        logw = self.solution.log_density
        m = logw.max()
        w = np.exp(logw - m)
        w /= w.sum()
        object.__setattr__(self, "_w", w)

    # -- expectations ----------------------------------------------------------

    def weights(self) -> np.ndarray:
        return self._w

    def log_weights(self) -> np.ndarray:
        return self.solution.log_density

    def expectation(self, values: np.ndarray) -> float:
        """E^{Q}[G] for pathwise terminal values G (self-normalized)."""
        return float(self._w @ values)

    def call_prices(self, strikes: np.ndarray) -> np.ndarray:
        """Undiscounted forward call prices E^Q[(e^{x_T} - K)^+]."""
        ex = np.exp(self.paths.x[:, -1])
        return np.array([self.expectation(np.maximum(ex - k, 0.0)) for k in np.atleast_1d(strikes)])

    # -- diagnostics -------------------------------------------------------------

    def effective_sample_size(self) -> float:
        return float(1.0 / (self._w**2).sum())

    def ess_fraction(self) -> float:
        return self.effective_sample_size() / self.paths.n_paths

    def max_weight_share(self) -> float:
        return float(self._w.max())

    # -- entropy double-entry (arch doc §12.2) -----------------------------------

    def entropy_lr(self) -> float:
        """H^LR = E^Q[log dQ/dQ0] from the likelihood route."""
        n = self.paths.n_paths
        return float(self._w @ (np.log(np.maximum(self._w, 1e-300)) + np.log(n)))

    def entropy_en(self) -> float:
        """H^EN = E^Q[ integral 1/2 |u*|^2 dt ] from the energy route."""
        dt = float(self.paths.times[1] - self.paths.times[0])
        energy = 0.5 * (self.solution.z_orth**2).sum(axis=1) * dt
        return float(self._w @ energy)

    def entropy_lr_ci(self, n_boot: int = 200, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float]:
        """Monte Carlo confidence interval for H^LR by bootstrap over
        paths (weights are pathwise, so no re-solve is needed): the
        plateau evidence of the refinement table must carry error bars,
        not bare point estimates (review M2-5)."""
        rng = np.random.default_rng(seed)
        n = self.paths.n_paths
        logw = self.solution.log_density
        hs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            lw = logw[idx]
            w = np.exp(lw - lw.max())
            w /= w.sum()
            hs[b] = float(w @ (np.log(np.maximum(w, 1e-300)) + np.log(n)))
        lo, hi = np.quantile(hs, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(lo), float(hi)

    # -- martingale certificate ----------------------------------------------------

    def forward_error(self) -> float:
        """|E^Q[e^{x_T}] - 1|: the terminal-horizon martingale residual."""
        return abs(self.expectation(np.exp(self.paths.x[:, -1])) - 1.0)

# Options Testing Facility

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20facility-orange.svg)](#status)

Research infrastructure for one question: **does the SFV model work on real
option data — and does it price better than the standard alternatives?**

The repository now contains **two distinct law constructions** of the SFV
(Stochastic Feedback Volatility) research line — they are compared, never
conflated (binding record: [`docs/DECISIONS.md`](docs/DECISIONS.md), D4):

1. the **legacy restricted variance-channel bridge** (`otf/models/sfv.py`,
   `otf/calibration/`): the SFV_M2 formulation, calibrated to real LSEG
   Workspace option chains and benchmarked out of sample against
   Black-Scholes, raw-SVI and CF-Heston;
2. the **SSFV projective core** (`otf/ssfv/`): the exact
   martingale-projected entropic deformation of the paper *Schrödinger
   Stochastic Feedback Volatility — complete formulation and technical
   closure*, built as an information projection with a full certificate
   system. Architecture: [`docs/SSFV_IMPLEMENTATION_ARCHITECTURE.md`](docs/SSFV_IMPLEMENTATION_ARCHITECTURE.md).

The model lineage is documented in
[`docs/MODEL_HISTORY.md`](docs/MODEL_HISTORY.md).

---

## Construction 1 — legacy restricted bridge (benchmark arm)

Under the affine jump-diffusion **prior** Q⁰:

$$
dX_t=\Big(r-\tfrac12 v_t-\lambda_S\,\kappa_S\Big)dt+\sqrt{v_t}\,dW^S_t+dJ^S_t,
\qquad
dv_t=\kappa(\theta-v_t)\,dt+\xi\sqrt{v_t}\,dW^v_t,
$$

with $d\langle W^S,W^v\rangle_t=\rho\,dt$ and Kou double-exponential price
jumps $J^S$ whose compensator $\kappa_S=\mathbb E[e^Y-1]$ keeps $e^{X}$ a
discounted martingale. The **restricted Schrödinger-bridge correction**
deforms the prior on the variance channel only:

$$
dv_t \;\mathrel{+}= \alpha\,g(v_t)\,v_t\Big(\rho\xi\,\partial_x\Theta+\xi^2\,\partial_v\Theta\Big)dt,
\qquad
\Theta(x,v)=b_0+b_1\tilde x+b_2\tilde v+b_3\tilde x\tilde v+b_4\tilde x^2+b_5\tilde v^2,
$$

on standardized coordinates $\tilde x,\tilde v$, with an optional sigmoid
variance gate $g$. Calibration is two-layer: prior fit (CF Heston, IV-space
loss, scheme-robustness regularisation) then bridge fit (Level-2 objective,
CRN Monte Carlo).

This is a *different law construction* from SSFV, not an approximation of
it: the two coincide only under $U_x=-\rho\xi U_v$, which the legacy ansatz
does not impose. It stays as the paired benchmark arm and one axis of the
ablation program.

## Construction 2 — SSFV projective core

SSFV is the **information projection** of the affine prior onto the
market-calibrated martingale set: the posterior is the minimal-entropy law
matching bounded terminal test moments while leaving the forward a
martingale *structurally* (Girsanov acts only on the Brownian direction
orthogonal to the return innovation). The `otf/ssfv/` package implements
the diffusion sector end to end:

* `prior/` — Heston prior with **exposed pathwise innovations** (full-
  truncation Euler; Andersen QE is pricing-grade and is rejected by the
  likelihood layer);
* `constraints/` — cumulative bounded test family (hats + capped tail
  ramps) with a block-triangular **nested normalization plan**: exact level
  nesting, zero-padding embeddings, recorded variance-gauge rejections;
* `bsde/` — the reference **Picard Hopf–Cole solver** (linear killed
  Feynman–Kac propagation + scalar martingale projection, frozen-field
  final sweep with reported fixed-point residual) and a direct
  quadratic-driver regression backend as diagnostic cross-check;
* `dual/` — `ReducedMomentMapCalibrator` (static exponential-family Newton
  warm start + Gauss–Newton with pseudo-inverse on the reduced moment map)
  and the `ProjectiveSequence` level loop;
* `posterior/`, `certificates/` — reweighted posterior and the mandatory
  **CertificateBundle** per level: signed duality gap with its exact defect
  decomposition, entropy double-entry (likelihood vs energy route),
  martingale certificate, projective Cauchy certificate with signed slack
  decomposition, conditioning diagnostics.

A successful fit without a certificate bundle is not a valid research
result. The synthetic certificate experiment produces the artifacts
(manifest with concrete component serialization, per-level certificate
JSONs, refinement table, optional dt-convergence table):

```bash
python -m otf.experiments.ssfv_synthetic --out runs/ssfv_synth_001 \
    --paths 8192 --steps 32 --levels 0 1 [--dt-table 8 16 32 64]
```

## The verdict machinery (real data)

`otf.experiments.surface_oos` runs the study the question needs: **fit every
arm on day t, price day t+1 frozen**, same quotes, same shocks:

| arm | what it is | why it is there |
| --- | --- | --- |
| `flat` | single BS vol (weighted mean IV) | the floor any model must clear |
| `svi` | raw-SVI slices + total-variance interpolation | the market-standard *static* surface fit |
| `heston` | the affine prior, CF-priced | the structural benchmark |
| `prior` | same prior, CRN-MC-priced (β=0) | the *paired* baseline for the bridge (same shocks, same discretization error) |
| `bridge` | prior + fitted bridge correction | the legacy construction under test |

Aggregation per name: mean IV RMSE per arm, win counts, moving-block-bootstrap
CIs and Diebold–Mariano p-values on the paired RMSE differentials. The SSFV
projective arm joins this table once the real-surface adapter lands (M5).

---

## Repository layout

```text
otf/
├── optim.py                  shared pure-stdlib Nelder-Mead
├── models/                   BS / Heston CF / raw-SVI / legacy SFV engine
├── calibration/              legacy two-layer calibration
├── data/                     LSEG chain loader, realized measures
├── evaluation/               IV-RMSE scorers, DM test, bootstrap, LaTeX
├── experiments/
│   ├── calibrate_surface.py  one-day two-layer calibration CLI
│   ├── surface_oos.py        multi-model day-ahead OOS study CLI
│   └── ssfv_synthetic.py     SSFV projective certificate experiment CLI
└── ssfv/                     SSFV projective core (see above)
    ├── types.py, interfaces.py, config.py   stdlib-only result objects,
    │                         protocols, reproducibility manifest
    ├── prior/  constraints/  bsde/  projection/  dual/
    ├── posterior/  certificates/
scripts/                      LSEG chain collectors
tests/                        pytest suite (both lanes, see below)
docs/
├── DECISIONS.md              binding decision record D1–D11
├── SSFV_IMPLEMENTATION_ARCHITECTURE.md
└── MODEL_HISTORY.md          SFV model lineage and provenance
```

## Install & test

The **legacy facility is pure stdlib** (`dependencies = []`); the SSFV
numerical core lives behind extras (`docs/DECISIONS.md` D2):

```bash
# legacy lane: stdlib only — SSFV numerical tests skip themselves
python -m venv venv && venv/bin/pip install -e '.[dev]'
venv/bin/python -m pytest                # 40 passed, SSFV modules skipped

# SSFV numerical lane
venv/bin/pip install -e '.[numerical,dev]'
venv/bin/python -m pytest                # full suite (~2 min CPU)
SSFV_SLOW=1 venv/bin/python -m pytest tests/test_ssfv_external_dgp.py  # N-scaling probe
```

Extras: `numerical` (numpy, scipy), `torch`, `data` (pandas, pyarrow,
duckdb), `dev` (pytest, hypothesis, ruff, mypy), `lseg`. CI runs both
lanes plus a non-blocking lint/type pass
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)); `import otf` and
`import otf.ssfv` must never pull in NumPy (asserted in CI).

## Workflow on real data

**1. Collect chains** (machine with LSEG Workspace open, e.g. `lseg-env`):

```bash
python scripts/collect_chains.py                 # AAPL MSFT NVDA, one snapshot/day
python scripts/backfill_chains.py 25             # rebuild ~25 past days from IV history
```

Snapshots land in `chains/YYYY-MM-DD/{name}_chain.csv` + `spots.json`.
Survivorship note: a surface rebuilt D days back misses contracts that expired
within D days; a backfill horizon no longer than the study's `--min-tte`
filter (0.08y ≈ 20 business days) is bias-free by construction.

**2. Calibrate one day** (prior + bridge, paste-ready parameter block):

```bash
python -m otf.experiments.calibrate_surface \
    --chain-csv chains/2026-07-10/aapl_chain.csv --spot 211.5
```

**3. Run the verdict** (needs ≥ 2 collected days):

```bash
python -m otf.experiments.surface_oos --chains-dir chains \
    --names aapl msft nvda --json surface_oos.json --latex surface_oos.tex
```

---

## Status

Working facility. The legacy construction is a faithful port of the Stocks
implementation of SFV_M2 (see `docs/MODEL_HISTORY.md`); the SSFV projective
core has completed milestones M0–M1 on synthetic data (self-consistent DGP
and independent Girsanov-tilt DGP with measured N-scaling of the
field-noise stall). Roadmap and open M2 items (Sobolev regularization,
refinement tables, Schur conditioning certificate) live in
`docs/DECISIONS.md` D10–D11.

## Disclaimer

Academic research code. Not investment advice, not a production pricing
library.

## License

Apache License 2.0.

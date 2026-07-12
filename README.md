# Options Testing Facility

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20facility-orange.svg)](#status)

Research infrastructure for one question: **does the SFV model work on real
option data — and does it price better than the standard alternatives?**

SFV (Stochastic Feedback Volatility, in its current *Schrödinger-bridge*
formulation) is calibrated to real LSEG Workspace option chains and compared,
**out of sample and paired per day**, against Black-Scholes, raw-SVI and
characteristic-function Heston. The model implemented here is the most recent
formulation of the SFV line of work; the full lineage is documented in
[`docs/MODEL_HISTORY.md`](docs/MODEL_HISTORY.md).

---

## The model under test

Under the affine jump-diffusion **prior** Q⁰:

$$
dX_t=\Big(r-\tfrac12 v_t-\lambda_S\,\kappa_S\Big)dt+\sqrt{v_t}\,dW^S_t+dJ^S_t,
\qquad
dv_t=\kappa(\theta-v_t)\,dt+\xi\sqrt{v_t}\,dW^v_t,
$$

with $d\langle W^S,W^v\rangle_t=\rho\,dt$ and Kou double-exponential price
jumps $J^S$ whose compensator $\kappa_S=\mathbb E[e^Y-1]$ keeps $e^{X}$ a
discounted martingale. The **restricted Schrödinger-bridge correction** then
deforms the prior minimally, acting on the variance channel only:

$$
dv_t \;\mathrel{+}= \alpha\,g(v_t)\,v_t\Big(\rho\xi\,\partial_x\Theta+\xi^2\,\partial_v\Theta\Big)dt,
\qquad
\Theta(x,v)=b_0+b_1\tilde x+b_2\tilde v+b_3\tilde x\tilde v+b_4\tilde x^2+b_5\tilde v^2,
$$

on standardized coordinates $\tilde x,\tilde v$, with an optional sigmoid
variance gate $g$ (transient activation). Because the correction cannot
manufacture price drift, the forward is preserved by construction; the betas
absorb what the prior could *not* price — variance level, tails, x–v feedback
skew.

Calibration is two-layer, in the order the theory prescribes:

1. **prior fit** — Heston $(v_0,\kappa,\theta,\xi,\rho)$ on the surface via
   characteristic function, IV-space loss, Feller/κ-cap regularisation;
2. **bridge fit** — betas under the SFV_M2 Level-2 objective (IV +
   vega-weighted price + martingale + control-energy + L2 [+ Sinkhorn on the
   Breeden–Litzenberger implied terminal density]), Monte-Carlo priced with
   common random numbers so the loss is deterministic in beta.

## The verdict machinery

`otf.experiments.surface_oos` runs the study the question needs: **fit every
arm on day t, price day t+1 frozen**, same quotes, same shocks:

| arm | what it is | why it is there |
| --- | --- | --- |
| `flat` | single BS vol (weighted mean IV) | the floor any model must clear |
| `svi` | raw-SVI slices + total-variance interpolation | the market-standard *static* surface fit |
| `heston` | the affine prior, CF-priced | the structural benchmark |
| `prior` | same prior, CRN-MC-priced (β=0) | the *paired* baseline for the bridge (same shocks, same discretization error) |
| `bridge` | prior + fitted bridge correction | the model under test |

Aggregation per name: mean IV RMSE per arm, win counts, moving-block-bootstrap
CIs and Diebold–Mariano p-values on the paired RMSE differentials — the
numbers a referee asks for.

---

## Repository layout

```text
otf/
├── optim.py                  shared pure-stdlib Nelder-Mead
├── models/
│   ├── black_scholes.py      BS price / greeks / implied vol
│   ├── heston.py             Heston CF pricer (Little-Trap formulation)
│   ├── svi.py                raw-SVI slice fit + surface interpolation
│   └── sfv.py                SFV PathEngine (CRN), W2/Sinkhorn, MC pricing
├── calibration/
│   ├── heston_fit.py         MarketQuote + CF Heston calibration
│   ├── bridge_fit.py         bridge betas -> target terminal law
│   └── surface_fit.py        bridge betas -> real surface (Level-2 loss)
├── data/
│   ├── chains.py             LSEG chain CSV loader + filters + day layout
│   └── realized.py           realized-measure SFV prior (jump/diffusion split)
├── evaluation/
│   ├── pricing.py            per-model IV-RMSE scorers (common yardstick)
│   └── stats.py              DM test, block bootstrap, LaTeX tables
└── experiments/
    ├── calibrate_surface.py  one-day two-layer calibration CLI
    └── surface_oos.py        multi-model day-ahead OOS study CLI
scripts/
├── collect_chains.py         daily chain snapshot (LSEG Workspace SDK)
└── backfill_chains.py        rebuild past surfaces from contract IV history
tests/                        pytest suite (pure stdlib, seconds)
legacy/base_pricing.py        the original monolithic thesis script
docs/MODEL_HISTORY.md         SFV model lineage and provenance
```

The numerical core has **zero dependencies** (Python ≥ 3.10 stdlib only).
Only the two collector scripts need the LSEG SDK.

---

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

## Install & test

```bash
python -m venv venv && venv/bin/pip install -e '.[dev]'
venv/bin/python -m pytest        # 37 tests, ~4 s
```

---

## Status

Working facility, actively used for the SFV research program. The
implementation is a faithful port of the most recent SFV code (see
`docs/MODEL_HISTORY.md` for what "most recent" means and what was left
behind); the SVI benchmark and the multi-arm OOS harness are new here.
Planned next steps live at the end of `docs/MODEL_HISTORY.md`.

## Disclaimer

Academic research code. Not investment advice, not a production pricing
library.

## License

Apache License 2.0.

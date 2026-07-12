# SFV model lineage — which version is implemented here, and why

Audit of the SFV material as of **2026-07-12**, across the `SFV.zip` archive
(Desktop/SFV), the Downloads folder scripts, and the
[`francescopinna03/Stocks`](https://github.com/francescopinna03/Stocks)
repository. Conclusion first:

> **The version implemented in this facility is the Stocks-repository
> formulation** (affine Heston+Kou prior with a restricted, standardized,
> gated Schrödinger-bridge correction on the variance channel, quadratic
> ansatz for log h) — the most recent code lineage, last touched 2026-07-10,
> and the direct implementation of the SFV_M2 paper. Nothing newer was found
> in the zip or in Downloads.

## Timeline

| date (2026) | artifact | contribution |
| --- | --- | --- |
| 07–08 Apr | `sv_sfv_testing_facility.py`, `sb_sfv_run.py`, WRDS extraction scripts | first real-data facility: WRDS OptionMetrics AAPL 2014 panels, single-expiry Schrödinger-bridge runs |
| 08 Apr | `testing_facility_m2.py`, `_m3.py` | multi-expiry calibration, composite loss |
| 17 Apr | `sfv_schrodinger_notes.tex` | formal notes: path-space Schrödinger problem, Doob transform, restricted variance-channel bridge |
| 18–23 Apr | `m3_1` + `_fixed` + `_v3` + `_v4`, `aapl_batch_m3_1.py` | batch runs over March-2014 AAPL days; stability fixes |
| 24 Apr | `testing_facility_M4.py`, `M5.py`, `M6.py` | **M5 lessons** (kept, see below); M6 = last standalone facility |
| 17 May (compiled 8 Jul) | `SFV_M2.tex` — *A Transient Schrödinger-Bridge Correction for the SFV Model* | the reference theory: affine prior, market-implied terminal constraints, restricted bridge, transient gates, ansatz hierarchy, control-energy/entropic interpretation, calibration objective, diagnostics |
| Apr → 10 Jul | **Stocks repo** (`sim/sfv.py`, `sim/bridge_calibration.py`, `sim/surface_calibration.py`, `core/*`, `apps/*`) | the paper's program implemented and hardened: CRN PathEngine, W2/Sinkhorn, Level-2 surface objective, LSEG chain collectors, surface-OOS study, ablations, DM tests, bootstrap CIs |

The M-series scripts in the zip are an *earlier, parallel* implementation
lineage (numpy/scipy, differential evolution + Sobol QMC), superseded by the
Stocks implementation but still worth mining (below).

## What this facility ports (and from where)

| here | from Stocks | notes |
| --- | --- | --- |
| `otf/optim.py` | `core/calibration.py` | Nelder-Mead |
| `otf/models/black_scholes.py` | `core/options.py`, `core/stats.py` | |
| `otf/models/heston.py` | `core/heston.py` | verbatim |
| `otf/models/sfv.py` | `sim/bridge_calibration.py` (engine, distances, MC pricing), `sim/sfv.py` (kou_compensator, doc model) | + overflow guard on the gate sigmoid (new fix) |
| `otf/calibration/heston_fit.py` | `core/calibration.py` | |
| `otf/calibration/bridge_fit.py` | `sim/bridge_calibration.py` | |
| `otf/calibration/surface_fit.py` | `sim/surface_calibration.py` | |
| `otf/data/chains.py` | `apps/calibrate_surface.py` (load_chain), `apps/surface_oos.py` (day layout) | |
| `otf/data/realized.py` | `data/realized.py` | verbatim |
| `otf/evaluation/stats.py` | `sim/benchmarks.py` (dm_test, bootstrap), `apps/thesis_report.py` (latex_table) | |
| `otf/experiments/calibrate_surface.py` | `apps/calibrate_surface.py` | |
| `otf/experiments/surface_oos.py` | `apps/surface_oos.py` | **extended**: flat-BS, SVI and CF-Heston arms added |
| `scripts/collect_chains.py`, `scripts/backfill_chains.py` | repo root | verbatim (LSEG SDK) |
| `otf/models/svi.py` | — | **new**: raw-SVI benchmark |

Deliberately *not* ported: the RIT trading simulator, order-book/market-making
stack, portfolio machinery, dashboards — trading-sim infrastructure orthogonal
to the pricing question this facility answers.

## Lessons already paid for (do not relearn)

From the M5 header (zip, 24 Apr) — kept because they bit once:

1. **Loss scaling**: point-wise relative price scaling turns the loss into a
   relative MSE that up-weights OTM quotes by orders of magnitude and is not
   monotone with the reported dollar/vol RMSE. Use one dataset-level scale
   (M5) or, as here, IV-space + vega-weighted price terms.
2. **QMC truncation**: slicing a Sobol batch breaks base-2 equidistribution;
   only use full powers of two. (Relevant if/when a QMC engine is added here.)
3. **Diagnostic consistency**: always report both the metric the solver
   minimised and the metric you quote, so divergence is visible in the JSON.

From the Stocks calibration experience (flag help strings preserve it):

4. **min-tte ≈ 0.08y**: sub-month smiles are jump-driven; they force a
   diffusive CF prior into ξ explosion / Feller violation.
5. **Feller weight on the prior fit**: a CF-only prior that violates Feller
   prices the surface but cannot be *simulated* faithfully — the bridge would
   then fit discretization error, not structure.
6. **kappa cap**: jumpy surfaces push the CF fit to extreme κ/ξ pairs that
   price well and simulate badly.
7. **Backfill survivorship**: a surface rebuilt D days back is missing
   exactly the contracts with tte < D at that date; keep the backfill horizon
   ≤ the min-tte filter and it is bias-free by construction.

## Differences between the two implementation lineages (future work candidates)

The zip M6 facility has features the Stocks lineage dropped or never had:

* **terminal-ramp time gate** `alpha(t)` (bridge switches on near expiry)
  vs. the Stocks *variance* gate `g(v)`; SFV_M2 §"Transient activation"
  covers both — worth an ablation on real chains;
* **structural martingale correction on log S** paired with the variance
  drift (M4/M6) vs. relying on the martingale penalty (Stocks);
* **numpy-vectorized paths + Sobol QMC + differential evolution**: ~50×
  faster than the pure-stdlib engine. A vectorized `PathEngine` with the same
  semantics is the single highest-value speed upgrade for large-scale OOS
  runs.

Planned next steps:

1. collect a real multi-week chain panel (scripts/, LSEG Workspace);
2. run `surface_oos` on it — first real five-arm verdict table;
3. ablations: gate on/off, free-beta sets (2), (2,5), (1..5), λ_sk > 0;
4. optional numpy engine behind the same interface;
5. richer benchmarks if referees ask: Bates CF, SSVI with calendar-arb
   constraints.

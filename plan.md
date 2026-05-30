# Dissertation — Master Plan

_Forward-looking. Last updated 2026-05-29. Points to deep specifics in `docs/`; do not duplicate large sections here._

## Time budget

**4 days remaining.** Items 1 + 2 collapse into **one day** (next session). Items 3–4 fill the remainder.

## Hardware constraint (applies everywhere)

- 16 GB RAM
- 8 GB VRAM (RTX 4060)
- All planned items have been sized to fit. Per-region GAT-GRU training estimated at ~10–25 min on the 4060.

## Item map (coarse)

| # | Title | Day | Track docs | Status |
|---|-------|-----|------------|--------|
| 1 | GNN on IMD + IMD-vs-ERA5 reality check | next | `docs/track_1a_imd_minimap/README.md`, `docs/track_1b_imd_era5_corr/README.md` | scoped |
| 2 | Regional ablation (Northern Plains + Himalayan) | next | `docs/track_2_regional_ablation/README.md` | scoped |
| 3 | Reuse of `minor_project_rl` findings | TBD | `docs/track_3_rl_reuse/README.md` | placeholder — Risheek to scope next prompt |
| 4 | Writeup / dissertation chapter polish | last day(s) | — | not scoped yet |

## Item 1 — GNN on IMD data + correlation check

**Why:** Guide asked for results on IMD (observational) data, not just ERA5 reanalysis. IMD is sparse (69.4% missing overall, many stations 100% missing from 1970+).

**Strategy:** Dual track.

- **Track A — minimap GAT-GRU on a clean IMD subset.** Pick 12–20 stations with <20–35% missing from 1970 on. Train GAT-GRU with rain-only + cyclic time features (`use_latent=False`). Deliverable: notebook + result CSVs + comparison row.  
  Full specifics: **`docs/track_1a_imd_minimap/README.md`**.

- **Track B — IMD vs ERA5 per-station correlation.** Pearson + bias + mean absolute difference, on three slices: full series, monsoon-only, rainy-day-only. Visualize on the WB shapefile.  
  Full specifics: **`docs/track_1b_imd_era5_corr/README.md`**.

**Risks:** If <12 stations survive any reasonable missingness cut, Track A is infeasible — lean on Track B alone.

## Item 2 — Regional ablation: Northern Plains + Himalayan

**Why:** Show whether GAT-GRU generalizes across distinct rainfall climatologies.

**Strategy:** Fetch ERA5 fresh for two new regions (bounding boxes Northern Plains lat 24–31, lon 75–88; Himalayan lat 28–34, lon 75–95). Subsample to ~80–100 grid points per region. Mirror cleaning, pivoting, EDA from the existing WB pipeline. Train GAT-GRU per region with **identical** hyperparams to exp_13 for fair comparison.

Full specifics: **`docs/track_2_regional_ablation/README.md`**.

**Fix the scaler/split alignment bug** (see `status.md` known issue #2) in the regional runs — set `scale_pivots(train_end=...)` to the 70%-split date, not 2015-12-31.

## Item 3 — RL reuse

**Status:** placeholder.

Full specifics: **`docs/track_3_rl_reuse/README.md`** (to be filled after research).

## Item 4 — Writeup

Not yet scoped. Once Items 1–3 land, the writeup will pull from:

- `experiments/results/exp_*/overall_metrics.csv` for the headline numbers.
- The EDA plots in `data/eda_outputs/`.
- The comparison table below.

## Master comparison table (fills in as runs complete)

Real-space test-set metrics. All numbers in mm/day.

| # | Experiment | Data source | N nodes | RMSE | MAE | Bias | Monsoon RMSE | NRMSE |
|---|------------|-------------|---------|------|-----|------|--------------|-------|
| 1 | Baseline LSTM | IMD WB | per-station | 15.82 | 6.87 | +0.16 | 23.03 | 2.83 |
| 9 | ConvLSTM (Keras weighted) | IMD WB (interp grid) | grid | 13.37 | 3.83 | **-3.22** | 20.65 | — |
| 10 | Transformer (regr-as-cls) | IMD WB | per-station | 18.53 | 6.00 | **-4.94** | 28.15 | 3.17 |
| 12 | GCN-GRU | ERA5 WB | 291 | 8.99 | 4.08 | +0.17 | 13.19 | 1.67 |
| 13 | **GAT-GRU** | ERA5 WB | 291 | **8.79** | 4.00 | +0.25 | n/a saved | 1.63 |
| 14 | **GAT-GRU minimap (IMD rain-only)** | IMD WB subset | 16 | **17.60** | 7.67 | -0.00 | 27.60 | 2.72 |
| 14b | GAT-GRU minimap (ERA5 rain-only, matched) | ERA5 WB subset | 16 | 10.42 | 4.95 | +0.23 | 15.65 | 1.61 |
| 14c | GAT-GRU minimap (ERA5 12-feat, matched) | ERA5 WB subset | 16 | 10.03 | 4.60 | +0.04 | 15.20 | 1.55 |
| 15a | GAT-GRU NP | ERA5 NP | ~100 | TBD | TBD | TBD | TBD | TBD |
| 15b | GAT-GRU Himalayan | ERA5 Himalayan | ~100 | TBD | TBD | TBD | TBD | TBD |

(exp_4–8 omitted: stale RMSE values reported in their notebooks; real-space re-eval pending and not on the current path. See `status.md` known issue #5.)

**Item 1 results (2026-05-29).** Rows 14 / 14b / 14c are the matched exp_14 runs (same 16 stations, 2000–2020, identical config) — compare them to **each other**, not to row 13 (different N / features / window). IMD-test baselines: persistence 22.53, climatology 18.50 mm. **Track 1B** (IMD-vs-ERA5 correlation, 291 stations): median r 0.37 (full) / 0.24 (rainy), bias ≈ 0, wet-day frequency IMD 30% vs ERA5 51%. Full records: `docs/track_1a_imd_minimap/implementation_notes.md`, `docs/track_1b_imd_era5_corr/implementation_notes.md`.

## Cross-cutting implementation notes

1. **Always use `models/gat_gru.py` and `models/gcn_gru.py` as-is.** Hyperparam changes go in the notebook CONFIG, not the model file.
2. **Always set `scale_pivots(train_end=...)` to the actual 70%-split date** for any new run. Compute it from the date index. The exp_12/exp_13 runs used `'2015-12-31'`, a known small leakage.
3. **Never copy cells 64 and 69 from `exp_12_gnn_baseline.ipynb`** — they cosmetically adjust predictions and are dead code in any clean run.
4. **Always report best-val-loss-checkpoint metrics, not epoch-30 metrics**, especially on small graphs (Track 1A). The training driver already saves `_best.pt`.
5. **Result CSVs to save for every new run** (matching exp_12's full layout, not exp_13's incomplete one):
   - `overall_metrics.csv`
   - `seasonal_metrics_real.csv`
   - `station_metrics_real.csv`
   - `observed_vs_predicted_scatter.png` (raw preds only)
6. **Random seeds:** none of the existing experiments set a torch/numpy seed. New runs should set seed 42 in the notebook header for reproducibility — note this in the change log when first done.

## Pointers

- High-level project state: `status.md`
- Most-recent session notes + next-session lookouts: `handover.md`
- Documentation policy (where new files go, naming, linking, token rationale): **`docs/track_usages.md`**
- Track specifics: `docs/track_<id>_<name>/README.md` (one per item, plus track-local files alongside)
- Literature: `docs/references.md`

## Documentation policy (short version)

**Write deep, link shallow.** Track-specific content goes inside `docs/track_<id>_<name>/`. Cross-cutting content goes in a root doc. Track entry point is always `README.md`. Future additions specific to a track go into the track's subdirectory — not into root files. Full policy in `docs/track_usages.md`.

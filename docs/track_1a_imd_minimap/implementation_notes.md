# Track 1A — Implementation notes (what was actually done)

_Status: **COMPLETE** 2026-05-29 (Claude + Risheek). This file records the real implementation and results. The `README.md` in this folder is the original **plan** and was partly **superseded** — see "Key reframing" below. Read this file first._

## Outcome in one line

Trained the GAT-GRU on a clean IMD gauge subset **and** on matched ERA5 (same stations, same window, same config) to isolate the **data-source effect**. Real IMD gauge rainfall is genuinely harder to predict than ERA5; the gap is a property of the data (explained by Track 1B), not a model failure.

## Key reframing — the plan changed (read this)

- **README plan:** restrict to 1970+, cut by missingness, keep 12–20 stations.
- **Empirical finding (verified in-session):** 1970+ IMD missingness is **74%** and **block-structured** — every low-missing station still has at least one **full-year** gap (longest continuous NaN runs 365–641 days for the cleanest 14 stations; up to 2272 days for the <20%-missing set), and **2021 is 100% empty** for these subsets. A naive 1970–2021 temporal split would drop a fully-imputed (fabricated) year into the test set → indefensible.
- **Revised approach (window-first):** restrict to **2000–2020 first**, then take the **top-16 lowest-missing** stations. Result: max missing 0.46%, 10 districts, ~566×303 km spread, **~98% of days fully co-observed, no whole-year gaps**. Test window (2017-11 → 2020-12) is real data.

## Data (which files)

- **USE** `data/preprocessed_rain_data.parquet` — canonical IMD (3.96M rows, 293 stations, 1901–2021, **zero NaN rainfall**, full lat/lon).
- **DO NOT** use `data/processed_rain.parquet` — same observed values **plus** 782k explicit-NaN rows and missing coords (an earlier intermediate). The README's "4.74M rows" figure actually describes `processed_rain`, not `preprocessed` — a doc error.
- **ERA5:** `data/era5_pivot_data/rain_pivot.parquet` (1970–2023, 291 stations, gap-free). 291 of 293 IMD station names are present (only SAGAR ISLAND and SANDHEADS missing).

## Experiment design (exp_14)

Three matched GAT-GRU runs, **identical architecture & config** (SEQ_LEN 7, HORIZON 1, hidden 64, heads 4, 30 epochs, lr 1e-3, MSELoss, seed 42), differing only in data/features:

1. **imd_rain** — IMD rain-only (7 features = rain + 6 cyclic time).
2. **era5_rain** — ERA5 rain-only, **same 16 stations & window** (7 features). ← apples-to-apples control.
3. **era5_full** — ERA5 12 features (6 vars + 6 time).

Plus **persistence** and **climatology** baselines on the IMD test targets.

- Graph: `build_edge_index_radius` @ **150 km** (avg degree 4.62, 0 isolated nodes).
- Scaler fix: `scale_pivots(train_end=...)` set to the **actual 70%-split date (2014-09-12)**, not the hard-coded 2015-12-31 (fixes the exp_12/13 leakage, status.md issue #2).
- Eval uses the **best-val checkpoint** (`train_model` saves `<name>_best.pt`). exp_12's cosmetic cells 64+69 were NOT reproduced; scatter uses raw predictions.

## Results (real-space test set, mm/day)

| run | RMSE | MAE | Bias | Monsoon RMSE | NRMSE | skill vs own climatology |
|-----|------|-----|------|--------------|-------|--------------------------|
| persistence (floor) | 22.53 | 8.32 | ~0 | — | 3.48 | — |
| climatology (floor) | 18.50 | 7.90 | -0.18 | — | 2.86 | — |
| **imd_rain** | **17.60** | 7.67 | -0.00 | 27.60 | 2.72 | **+0.10** |
| **era5_rain** | **10.42** | 4.95 | +0.23 | 15.65 | 1.61 | **+0.19** |
| **era5_full** | **10.03** | 4.60 | +0.04 | 15.20 | 1.55 | **+0.25** |

## Interpretation (defensible)

- **Data-source gap (17.60 vs 10.42, identical everything else) is the headline result, and it is intrinsic.** ERA5 lag-1 autocorrelation 0.56 vs IMD 0.31; ERA5 std 13.7 vs 19.5; ERA5 dry-day 52% vs IMD 71%; means nearly identical (~6.4 mm). ERA5 is smoother → more predictable.
- **IMD "bad" = intrinsic unpredictability + the MSE objective, NOT overfitting.** IMD has the *smallest* train–val gap (it converges and stops, doesn't memorize); the scatter shows predictions capped ~70 mm against observations to 350 mm (under-dispersion / mean-reversion); failure concentrates in monsoon. It still beats persistence and climatology (real, if weak, skill).
- **Physics features add little at 1-day horizon:** ERA5 rain→full is only ~4% RMSE (10.42→10.03) but +0.06 skill score; `era5_full` overfits hardest (train R² ≈0.71 vs val ≈0.33) — the extra channels mostly add memorization capacity.

## Comparison discipline (important for the writeup)

Compare **WITHIN exp_14** (IMD vs ERA5 on the same 16 nodes). Do **not** compare exp_14's IMD number to exp_13's 8.79 — that is 291 nodes / 12 features / full window, confounded on three axes. The `partial_network_map.png` documents this honestly (16 of 291 nodes → less spatial context).

## Files produced (all new; no existing code modified)

- `experiments/experimentation_notebooks/exp_14_gat_gru_imd_minimap.ipynb` — 3 runs + baselines + comparison table.
- `experiments/experimentation_notebooks/exp_14_analysis_plots.ipynb` — partial-network map + ERA5-vs-IMD characterization (`%matplotlib inline` + `plt.show()` so figures render in-cell).
- `experiments/results/exp_14_gat_gru_imd_minimap/` — `selected_stations.csv`; `{imd_rain,era5_rain,era5_full}_{overall_metrics,seasonal_metrics_real,station_metrics_real}.csv`; `*_observed_vs_predicted_scatter.png`; `baseline_metrics.csv`; `comparison_table.csv`; `partial_network_map.png`; `era5_vs_imd_characterization.png`.
- `experiments/saved_models/exp_14_gat_gru_imd_minimap/{run}/<name>_best.pt`; `experiments/logs/exp_14_gat_gru_imd_minimap/{run}/*.log`.
- **Reused unchanged:** `utils/data_utils/data_helper_utils.py`, `utils/data_utils/dataset_files/gnn_dataset.py`, `models/gat_gru.py`, `utils/train_utils.py`, `utils/metric_utils/metrics.py`.

## Open / optional next steps

- `log1p` target variant (reduce MSE's extreme-domination; repo precedent exp_6).
- Occurrence-vs-amount split (rain/no-rain accuracy + wet-day-conditional RMSE; precedent exp_10).
- Report skill scores in-notebook and add ERA5 baselines.
- Light dropout/weight-decay for `era5_full` only (won't help IMD — it isn't overfitting).

# Track 1A — GAT-GRU on IMD minimap

_Item 1, Track A. Scoped 2026-05-29. Owner: next session._

> **STATUS: COMPLETE (2026-05-29).** This README is the original **plan**. What was actually built — and an important deviation from it — is in [`implementation_notes.md`](implementation_notes.md). **Key deviation:** the "1970+ then missingness cut" plan below was superseded by a **window-first** approach (restrict to **2000–2020**, then take the **top-16 lowest-missing** stations), because 1970+ IMD missingness is block-structured (whole-year gaps; 2021 fully empty). Headline: IMD rain-only test RMSE **17.60** vs matched ERA5 rain-only **10.42** (16 stations, 2000–2020).

## Goal

Train GAT-GRU on a small, low-missingness subset of IMD stations to give the guide a defensible "GNN on observational data" result. Avoid the noise of the full 293-station IMD pivot (69.4% missing overall, many stations 100% missing from 1970 onward).

## Repo files to read before coding

| File | What it tells you |
|------|-------------------|
| `data/preprocessed_rain_data.parquet` | Long-form IMD: `station_id, date, year, month, day, day_of_year, rainfall, lat, lon, district, station_type`. 4.74M rows, 293 stations, 1901–2021. |
| `experiments/experimentation_notebooks/exp_11_gnn_ver_1.ipynb` cells 41–67 | Prior missingness analysis on the full IMD pivot. Source of the 69.4% / 57-stations-under-50%-missing numbers. Cell 47 = stations with <50% missing across full timeline. |
| `utils/data_utils/data_helper_utils.py` | Reuse: `scale_pivots`, `temporal_split` (line 187 — array variant), `build_edge_index_radius`, `get_lat_lon_aligned`. |
| `utils/data_utils/dataset_files/gnn_dataset.py` | Reuse: `build_time_features`, `build_feature_tensor(use_latent=False)`, `add_time_features`, `SpatioTemporalDataset`. |
| `models/gat_gru.py` | Use unchanged. `in_channels` will be 7 (rain + 6 time features) instead of 12. |
| `experiments/experimentation_notebooks/exp_13_gat_gru.ipynb` | Template for the new notebook — CONFIG, training driver, eval cells, result-CSV layout. **Do not copy cells 64+69 from exp_12** (they cosmetically modify preds; see `status.md` known issues). |
| `experiments/results/exp_13_gat_gru/` | Where exp_13 saved its CSVs (incomplete — see Open Issues in `handover.md`). |

## Station selection logic

1. Load `preprocessed_rain_data.parquet`, pivot to (date × station) on `rainfall`.
2. Restrict to **1970-01-01 onward** (matches ERA5 timeline).
3. For each station, compute `missing_frac = isna().mean()`.
4. **Strict cut**: keep stations with `missing_frac < 0.20`.
5. **Relaxed cut**: keep stations with `missing_frac < 0.35`.
6. Pick the cut that yields **12–20 stations**. If both yield <12, drop to 0.50; if both yield >20, tighten to 0.15. **Document the cut used.**
7. Geographic-spread check (lightweight): plot the surviving stations on a WB map. If they cluster in 1–2 districts, accept it and note as a limitation in the writeup. Do **not** force spatial uniformity by hand-picking.

## Graph construction

- Reuse `build_edge_index_radius(lat, lon, threshold_km=100)`.
- With 12–20 stations spread across WB (~250 km tall, ~400 km wide), 100 km may produce too few edges. After building, check `edge_index.shape[1] / N`. If average degree < 2, bump threshold to **150 km**; if still < 2, bump to **200 km**.
- Save threshold used to the notebook config so it's auditable.

## Node features

- `use_latent=False` in `build_feature_tensor`. IMD has only `rainfall`.
- Append 6 cyclic time features (`build_time_features`): doy_sin, doy_cos, month_sin, month_cos, year_sin, year_cos.
- Final per-timestep node feature dim = **7**.

## Preprocessing

- Pivot → ffill → bfill → 0-fill (mirroring `scale_pivots`). Be explicit that 0-fill at the start of a series is acceptable here because zero is a valid rainfall value.
- Z-score scale the rain channel using **train-period only** mu/sigma. **Do not** repeat exp_12's scaler/split misalignment — set `train_end` to the actual 70%-split date, not 2015-12-31. Compute the date from `idx[int(0.7*len(idx))-1]`.

## Training config

Mirror exp_13 except `in_channels=7`:

```python
SEQ_LEN = 7
HORIZON = 1
BATCH_SIZE = 32
HIDDEN_DIM = 64
HEADS = 4
EPOCHS = 30  # but track best-val checkpoint
LR = 1e-3
CRITERION = torch.nn.MSELoss()
```

**Important:** with 12–20 nodes, the model has very few graph degrees of freedom. Expect overfitting by epoch 10–15. **Report metrics from the best-val-loss checkpoint**, not epoch 30. The training driver in `models/gat_gru.py` already saves `<exp_name>_best.pt`.

## Evaluation

- Save the **full result set** to `experiments/results/exp_14_gat_gru_imd_minimap/`:
  - `overall_metrics.csv`
  - `seasonal_metrics_real.csv`
  - `station_metrics_real.csv`
  - Scatter plot `observed_vs_predicted_scatter.png` — **use raw predictions, not adjusted**.
- Compare against the exp_13 ERA5 numbers in the comparison table inside `plan.md`.

## Risks and fallbacks

- **<12 stations survive the 0.50 cut** → Track A is infeasible. Skip Track A. Note in handover: "IMD too sparse for graph-level training in WB." Lean on Track B (correlation) instead.
- **Sparse graph (avg degree < 1.5 even at 200 km)** → fall back to a **fully connected** graph as a sanity baseline. Document.
- **Validation loss never improves past random init** → likely the 7-feature input is too thin. Try `seq_len=14` or `30` (more temporal context). Do not add ERA5 features — that breaks the "IMD-only" framing.

## Deliverable

- One notebook: `experiments/experimentation_notebooks/exp_14_gat_gru_imd_minimap.ipynb`.
- One result folder: `experiments/results/exp_14_gat_gru_imd_minimap/`.
- One comparison row added to `plan.md`'s comparison table.

## Open questions for Risheek

- Preferred station count target: 12, 16, or 20?
- If a 200 km threshold still produces a sparse graph, fall back to fully connected, or stop and report "not feasible"?
- Is including pre-1970 IMD data acceptable (improves missingness but loses ERA5 comparability) or strictly 1970+?

---

## Session Execution Plan

_Designed so the next session can execute end-to-end without re-deriving any planning. Treat each step as a checkpoint: do not advance until it completes cleanly. Estimated total wall-time: **2–3 hours** on RTX 4060._

### Pre-flight (5 min)

- Confirm files exist: `data/preprocessed_rain_data.parquet`, `data/wb_station_coords.csv`, `models/gat_gru.py`, `utils/data_utils/data_helper_utils.py`.
- Check open clarifications in `handover.md`. If unanswered, defaults: **16 stations**, threshold ramp **100 → 150 → 200 km**, then fully connected as last fallback.
- Create notebook skeleton `experiments/experimentation_notebooks/exp_14_gat_gru_imd_minimap.ipynb`. Copy CONFIG cell from `exp_13_gat_gru.ipynb`.
- Set torch + numpy seeds to **42** at the top of the notebook (project convention going forward — see `plan.md` cross-cutting note 6).

### Step 1 — Station selection (15 min)

1. Load `preprocessed_rain_data.parquet`, pivot to (date × station) on `rainfall`.
2. Restrict to `date >= '1970-01-01'`.
3. Compute `missing_frac = pivot.isna().mean()`.
4. Apply cascading cuts in order: **0.20 → 0.35 → 0.50**. Stop at the first cut that yields **12–20 stations**.
5. Save chosen list to `experiments/results/exp_14_gat_gru_imd_minimap/selected_stations.csv` with `[station_id, missing_frac, lat, lon, district]`.
6. **Validation gate:** print count, bounding box, per-district count. Abort if <12 stations even at 0.50 — write a note in `handover.md` and pivot to Track 1B-only path.

### Step 2 — Graph construction (10 min)

1. Build `lat`/`lon` for selected stations via `get_lat_lon_aligned`.
2. Call `build_edge_index_radius(lat, lon, threshold_km=100)`. Print `edge_index.shape[1]`; compute avg degree.
3. **Validation gate:** if avg degree < 1.5, retry at 150 then 200 km. If still < 1.5, fall back to fully connected. Log the threshold used in the notebook config.
4. Save adjacency stats (threshold, edge count, avg degree, isolated nodes) to notebook output.

### Step 3 — Dataset prep (15 min)

1. Apply `pivot.ffill().bfill().fillna(0)`.
2. Compute the actual **70%-split date**: `train_end_date = dates[int(0.7*len(dates)) - 1]`. Use this for scaling — **do not** call `scale_pivots(train_end='2015-12-31')`. This fixes the misalignment bug from exp_12/exp_13.
3. Z-score the rain channel with train-period μ/σ.
4. `build_feature_tensor(scaled, use_latent=False)` → `(T, N, 1)`. Then `add_time_features` → `(T, N, 7)`.
5. Split via `temporal_split` (array variant, line 187 of `data_helper_utils.py`).
6. Build `SpatioTemporalDataset` for train/val/test, then DataLoaders (batch 32).
7. **Validation gate:** print all shapes; confirm `len(train_loader) > 0`.

### Step 4 — Train (45–90 min)

1. `GAT_GRU_Model(in_channels=7, hidden_dim=64, heads=4)`.
2. `train_model(..., epochs=30, lr=1e-3, criterion=nn.MSELoss(), experiment_name='exp_14_gat_gru_imd_minimap')`.
3. **Validation gate:** after epoch 5, abort if loss is flat or NaN.
4. Identify **best-val-loss epoch** from log; load `_best.pt`. Do not use epoch-30.

### Step 5 — Evaluate + save (30 min)

1. Inference on test loader → `preds_scaled, targets_scaled`. Back-scale to mm.
2. Save full result set (mirror exp_12 layout, not exp_13's partial one):
   - `overall_metrics.csv`, `seasonal_metrics_real.csv`, `station_metrics_real.csv`, `observed_vs_predicted_scatter.png` (raw preds only).
3. Print headline numbers to fill `plan.md`'s master comparison table.

### Step 6 — Handover (10 min)

1. Append row to `plan.md`'s comparison table.
2. Update `status.md` Research Depth Map for `exp_14`.
3. Add change-log entry to `status.md` + session note to `handover.md`. Mention any deviation (station count, threshold, fully-connected fallback).
4. Create `docs/track_1a_imd_minimap/implementation_notes.md` only if something non-obvious surfaced; otherwise skip.
5. Mark task #9 complete.

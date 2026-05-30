# Track 2 — Regional ablation: Northern Plains + Himalayan

_Item 2. Scoped 2026-05-29. Owner: next session._

## Goal

Show whether GAT-GRU generalizes across distinct Indian rainfall climatologies. WB is the baseline. Two new domains:

- **Northern Plains** — flatter, lower mean rainfall, more uniform spatial structure.
- **Himalayan** — orographic, heavy and spatially variable, ERA5 known to smooth peaks.

Both use ERA5 reanalysis only (we don't have observational data for these regions in this repo). Ablation is **within ERA5 proxy**, not against ground truth — state this in the writeup.

## Repo files to read before coding

| File | What it tells you |
|------|-------------------|
| `experiments/experimentation_notebooks/gee_data_fetch.py` | Existing GEE fetch script. Adapt the bounding box and the output directory. **Read it first** to understand the export format and any quota considerations. |
| `experiments/experimentation_notebooks/cds_data_fetch.ipynb` | Alternate fetch path via Copernicus CDS. Likely slower but bypasses GEE quotas if hit. |
| `EDA/EDA_00_data_validation.ipynb` | Cleaning template — converts yearly CSVs into `era5_station_level_clean.parquet`. Mirror this for each region. |
| `experiments/experimentation_notebooks/exp_11_gnn_ver_1.ipynb` cells 70+ | Pivot creation from cleaned ERA5 long-form. Mirror this. |
| `EDA/EDA_01_rainfall_spatial_core.ipynb` | Spatial/seasonal EDA template — mirror per region. |
| `experiments/experimentation_notebooks/exp_13_gat_gru.ipynb` | Model run template. Reuse config unchanged; only the pivot path and station-coord file change. |
| `utils/data_utils/data_helper_utils.py` | Reuse all helpers as-is. |
| `models/gat_gru.py` | Reuse unchanged. |

## Region definitions (refine in session)

### Northern Plains

- Bounding box: lat **[24°, 31°]**, lon **[75°, 88°]**.
- Sample ERA5 at **0.5° spacing** → 15 × 27 = 405 candidate grid points.
- Subsample to **~80–100 points** by stratified random sampling within the box.
- **Exclude any point within the WB bounding box** (lat 20.85–27.22, lon 85.97–89.84) to keep the ablation honest.
- Station IDs synthesized as `NP_<lat>_<lon>` or sequential `NP_0001..NP_0100`.

### Himalayan

- Bounding box: lat **[28°, 34°]**, lon **[75°, 95°]**.
- Sample ERA5 at **0.5° spacing** → ~13 × 41 = ~530 candidate points.
- Subsample to **~80–100 points**.
- **Optional elevation mask**: drop points with elevation < 1000 m (Risheek to confirm — adds complexity). Use SRTM via GEE `USGS/SRTMGL1_003` if going this route.
- Station IDs synthesized as `HM_<lat>_<lon>` or sequential `HM_0001..HM_0100`.

## Data fetch — implementation notes

1. **Variables**: same 6 as WB ERA5: `total_precipitation_sum, temperature_2m, dewpoint_temperature_2m, surface_pressure, u_component_of_wind_10m, v_component_of_wind_10m`.
2. **Period**: 1970-01-01 → 2023-12-31 (same as WB for direct comparability).
3. **Output format**: one CSV per year, per region. Path: `data/gee_era5_data_northern_plains/ERA5_Station_Data_<year>.csv` and `data/gee_era5_data_himalayan/ERA5_Station_Data_<year>.csv`.
4. **GEE quota**: a region of ~100 points × 365 days × 6 vars per year is ~219K cells, well under the 5M-cell single-export limit. But **total ERA5 daily for both regions is ~24M cells across 54 years** — set up batched exports (one per year) and let it run in background.
5. **Expected latency**: 30 min – 2 hr per region depending on GEE responsiveness. **Start this first thing in the morning.**
6. **Disk footprint**: ~14 MB/year × 54 years × 2 regions ≈ 1.5 GB.

## Cleaning + pivoting

For each region, produce:

- `data/processed/era5_<region>_station_level_clean.parquet` — long-form, same schema as WB's `era5_station_level_clean.parquet`.
- `data/era5_pivot_data_<region>/{rain,temp,dew,pressure,u10,v10}_pivot.parquet` — wide-form.
- `data/<region>_station_coords.csv` — `[station_id, latitude, longitude]`.

## EDA per region

New notebook `EDA/EDA_03_<region>_rainfall_spatial_core.ipynb` per region, mirroring `EDA_01`. Minimum content:

- Mean monthly rainfall heatmap (months × stations).
- Monsoon vs non-monsoon comparison.
- Station correlation heatmap.
- Annual rainfall map.
- One paragraph in markdown summarizing differences from WB.

## Model run

- Notebook: `experiments/experimentation_notebooks/exp_15_gat_gru_<region>.ipynb`.
- Reuse exp_13 hyperparams unchanged (30 epochs, Adam, lr 1e-3, hidden 64, heads 4, batch 32, MSE).
- Reuse `build_edge_index_radius(threshold_km=100)`. For 100 stations on a 0.5° grid (~55 km spacing), 100 km gives ~3-station neighbors on average. Should work; verify after building.
- **Fix the scaler/split alignment bug** from exp_12/exp_13: set `train_end` to the date at the 70% split, not 2015-12-31. The Himalayan and NP datasets span 1970–2023, so the 70% cut lands around 2007-10. Use that.
- Results to `experiments/results/exp_15_gat_gru_<region>/` — include the full set (overall, seasonal, station, district-free since regions have no district mapping).

## Comparison table (final deliverable)

One table comparing the three regions, to live in `plan.md`:

| Region | N stations | RMSE | MAE | Bias | Monsoon RMSE | Non-monsoon RMSE | NRMSE |
|--------|------------|------|-----|------|--------------|------------------|-------|
| West Bengal (exp_13) | 291 | 8.79 | 4.00 | +0.25 | — | — | 1.63 |
| Northern Plains | ~100 | — | — | — | — | — | — |
| Himalayan | ~100 | — | — | — | — | — | — |

Add a short interpretation paragraph: which regions are easier/harder, and what climate-physics signal we think explains it.

## Hardware budget

- ~100 stations × 19,723 days × 6 vars × 4 bytes ≈ 47 MB per variable → easy in 16 GB RAM.
- Forward pass batch 32 × seq 7 × 100 × 12 features × 4 bytes ≈ 1 MB input per batch. GAT with heads=4 inflates internals but stays well under 8 GB VRAM.
- Per-region training: extrapolating from exp_13's ~50 sec/epoch on 291 nodes, ~15–25 sec/epoch on ~100 nodes → ~10 min per region. **Both regions in <30 min compute.**

## Risks and fallbacks

- **GEE export fails / quota hit** → switch to CDS via `cds_data_fetch.ipynb`. Slower but reliable.
- **One region's data is late** → ship Item 2 with just the other region, mark the missing one as "data pending" in handover.
- **Region overlaps WB** → filter explicitly in code, not by hand.
- **Himalayan elevation mask logic adds a day of work** → drop it. The bounding box restriction is enough to make the point.

## Open questions for Risheek

- Are the bounding boxes acceptable, or do you want specific sub-regions (Uttarakhand, Sikkim, etc.)?
- Apply elevation mask in Himalayan, or skip for simplicity?
- Is 100 stations per region the right size, or do you want larger (~200) for closer parity with WB's 291?

---

## Session Execution Plan

_Multi-phase because the GEE fetch is async and slow (~30 min – 2 hr per region). Phase 1 starts FIRST and runs in the background while Track 1 work proceeds. Phase 2 + 3 run once data arrives._

### Phase 1 — Data fetch (kick off at start of day, background)

**Estimated wall-time:** 30 min – 2 hr per region. Both kicked off in parallel.

#### Pre-flight (10 min)

- Confirm `experiments/experimentation_notebooks/gee_data_fetch.py` runs end-to-end on current WB config (sanity test on a small year range like 2020–2021).
- Confirm GEE auth is alive. If not, re-auth.
- Defaults if Risheek hasn't answered open questions: **bounding boxes as given**, **no elevation mask**, **~100 stations per region**.

#### Step 1.1 — Northern Plains fetch

1. Copy `gee_data_fetch.py` to `gee_data_fetch_northern_plains.py`.
2. Change bounding box to `lat [24, 31], lon [75, 88]`.
3. Generate 0.5° grid; subsample to ~100 points via stratified random sampling (seed 42). **Filter out any point in WB box** (lat in [20.85, 27.22] AND lon in [85.97, 89.84]).
4. Synthesize station IDs as `NP_<idx>` (zero-padded to 4 digits).
5. Save station coords to `data/northern_plains_station_coords.csv`.
6. Launch the GEE export loop (54 years × 6 variables). Output directory: `data/gee_era5_data_northern_plains/`.

#### Step 1.2 — Himalayan fetch (parallel with 1.1)

1. Copy to `gee_data_fetch_himalayan.py`.
2. Change bounding box to `lat [28, 34], lon [75, 95]`.
3. Generate grid, subsample ~100 points, filter WB overlap, synthesize `HM_<idx>` IDs.
4. Save `data/himalayan_station_coords.csv`.
5. Launch export to `data/gee_era5_data_himalayan/`.

#### Phase 1 validation gate

- Both directories contain 54 yearly CSVs each (1970 → 2023).
- Spot-check: load 1 file per region, confirm shape and that no station lies inside the WB box.
- If either fetch fails or stalls > 3 hr: switch that region to CDS via `cds_data_fetch.ipynb`. If still failing, ship Item 2 with just the successful region.

### Phase 2 — Cleaning + pivoting + EDA (after Phase 1 done, ~1.5 hr)

For each region (do Northern Plains first, Himalayan second):

#### Step 2.1 — Cleaning (15 min per region)

1. Mirror `EDA/EDA_00_data_validation.ipynb` as `EDA/EDA_00_<region>_data_validation.ipynb`.
2. Concatenate yearly CSVs, apply unit conversions (precipitation m → mm, temp K → C), drop bad rows.
3. Save: `data/processed/era5_<region>_station_level_clean.parquet`.
4. **Validation gate:** print shape, station count, date range, missing %.

#### Step 2.2 — Pivoting (10 min per region)

1. Reuse the pivot logic from `exp_11_gnn_ver_1.ipynb` cells 70+.
2. Output 6 files: `data/era5_pivot_data_<region>/{rain,temp,dew,pressure,u10,v10}_pivot.parquet`.
3. **Validation gate:** all 6 files exist; each is `(T, N)` with `T = 19723` and `N ≈ 100`.

#### Step 2.3 — Regional EDA (30 min per region)

1. Copy `EDA/EDA_01_rainfall_spatial_core.ipynb` to `EDA/EDA_03_<region>_rainfall_spatial_core.ipynb`.
2. Adjust the shapefile section (no district shapefile for NP/Himalayan — replace with a simple lat/lon scatter colored by rainfall metric).
3. Generate: monthly climatology heatmap, monsoon vs non-monsoon by station, station correlation heatmap, annual rainfall map.
4. End with a 2-sentence markdown summary of how the region differs from WB.

#### Phase 2 handover

- Add `EDA_00_<region>_data_validation`, `EDA_03_<region>_rainfall_spatial_core` to `status.md` Research Depth Map.

### Phase 3 — Model runs per region (~30 min)

For each region:

#### Step 3.1 — Notebook setup (10 min)

1. Copy `exp_13_gat_gru.ipynb` to `exp_15_gat_gru_<region>.ipynb`.
2. Change `DATA_PATH` to the region's pivot directory.
3. Change `station_df` source to `data/<region>_station_coords.csv`.
4. **Fix the scaler/split bug**: compute `train_end_date = pivots['rain'].index[int(0.7*T)-1]` and pass to scaling. **Do not** keep `train_end='2015-12-31'`.
5. Set seeds to 42 at the top.
6. Confirm `in_channels = 12` (6 ERA5 vars + 6 time features) and `use_latent = True`.

#### Step 3.2 — Train + eval (15 min per region on RTX 4060)

1. Build graph with `build_edge_index_radius(threshold_km=100)`. Print stats. Note: on a 0.5° grid (~55 km spacing), 100 km gives ~3-station neighbors.
2. Train with exp_13's exact hyperparams (30 epochs, Adam, lr=1e-3, hidden 64, heads 4, batch 32, MSE).
3. Save the **full result set** under `experiments/results/exp_15_gat_gru_<region>/`: overall, seasonal, station (NO district — synthetic stations have no districts), scatter plot.
4. **Important:** mark exp_13's bug as fixed in this run by logging `train_end_date` in the notebook output.

#### Phase 3 handover

1. Append both rows to `plan.md`'s master comparison table.
2. Add a 4-sentence interpretation paragraph below the table: which region was easier/harder, what climate-physics signal we think dominates each, and what the ERA5 smoothing caveat means for the Himalayan result.
3. Update `status.md` Research Depth Map for both exp_15 runs.
4. Append session note to `handover.md`.
5. Mark tasks #11, #12, #13 complete.

### Combined daily timeline (for orchestration)

| Time | Track 2 (NP) | Track 2 (Himalayan) | Track 1A | Track 1B |
|------|--------------|---------------------|----------|----------|
| 09:00 | Kick off fetch | Kick off fetch | — | — |
| 09:30 | (running) | (running) | Start | — |
| 10:30 | (running) | (running) | Train | Start |
| 11:30 | (running) | Likely done | Eval | Plots |
| 12:00 | Likely done | Clean+pivot+EDA | Handover | Handover |
| 13:00 | Clean+pivot+EDA | Train | — | — |
| 14:30 | Train | Eval | — | — |
| 15:30 | Eval + handover | Eval + handover | — | — |

Aggressive but feasible with 2 GPU-bound runs and 2 CPU-bound analyses.

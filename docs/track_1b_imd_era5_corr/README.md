# Track 1B — IMD vs ERA5 correlation EDA

_Item 1, Track B. Scoped 2026-05-29. Owner: next session._

> **STATUS: COMPLETE (2026-05-29).** This README is the original **plan**. What was actually built is in [`implementation_notes.md`](implementation_notes.md), with a full methodology + viva-defence write-up in `EDA_02_explained.docx` (this folder). Headline: across 291 shared stations (199 with ≥1 yr overlap), median IMD–ERA5 correlation **0.37** (→ **0.24** on rainy days), bias ≈ 0, but ERA5 reports rain on **51%** of days vs IMD's **30%** (drizzle bias).

## Goal

Quantify how different IMD (observation) and ERA5 (reanalysis) rainfall are, station-by-station. This is the fallback if Track 1A fails, and the supplementary if Track 1A succeeds. **Avoid heavy statistics** — Risheek wants results that are easy to defend without deep statistical theory.

## Repo files to read before coding

| File | What it tells you |
|------|-------------------|
| `data/preprocessed_rain_data.parquet` | IMD long-form (1901–2021, 293 stations). |
| `data/era5_pivot_data/rain_pivot.parquet` | ERA5 rainfall pivot (1970–2023, 291 stations). |
| `data/wb_station_coords.csv` | Station coords for the ERA5 pivot's 291 columns. |
| `EDA/EDA_01_rainfall_spatial_core.ipynb` | Style and plotting template for spatial figures — reuse the WB shapefile loading, choropleth-style maps, district normalization helpers. |
| `data/West_Bengal/District_shape_West_Bengal.shp` | District shapefile for choropleths. |

## Statistical operations (kept deliberately shallow)

For each station shared by both datasets:

1. **Pearson correlation** on overlapping daily series 1970–2021.
2. **Mean bias** = `mean(ERA5 - IMD)` (in mm/day).
3. **Mean absolute difference** = `mean(abs(ERA5 - IMD))`.

These three numbers are enough. Risheek explicitly said no Spearman, no higher-order tests, no quantile regression — usage must be justifiable without deep statistical knowledge.

### Why correlation alone is insufficient (for the writeup)

Daily rainfall is **zero-inflated** — most days are zero in both series, which inflates correlation toward 1 just because both agree on "no rain". Address this by computing the same three numbers on **two slices** in addition to the full series:

- **Monsoon slice**: months 6, 7, 8, 9 only.
- **Rainy slice**: only days where `IMD > 0`.

Report all three slices in the same table. The user can defend the slicing in plain language: "We checked the full series, then the monsoon-only series, then the rainy-day-only series, to see if the agreement holds when zeros are removed."

## Visualizations

- **Histogram of per-station Pearson correlations** (one per slice → three subplots).
- **Choropleth map of WB**: color stations by Pearson correlation (full slice). Use the WB shapefile.
- **Choropleth map of WB**: color stations by mean bias.
- **Scatter**: x = station latitude, y = mean bias. Look for systematic patterns (e.g., does ERA5 over-predict in the Himalayas?).
- **One-station example**: pick the median-correlation station and plot IMD vs ERA5 daily series for a single monsoon year (e.g., 2010). Caption: "what a typical agreement looks like."

## Output paths

- Notebook: `EDA/EDA_02_imd_vs_era5_correlation.ipynb`.
- Plots: `data/eda_outputs/imd_vs_era5_*.png` (one PNG per figure).
- Summary CSV: `data/eda_outputs/imd_vs_era5_summary.csv` — one row per station, columns `[station_id, lat, lon, corr_full, corr_monsoon, corr_rainy, bias_full, bias_monsoon, mad_full, mad_monsoon, n_overlap]`.

## Open question for Risheek (not blocking)

- IMD station_ids and ERA5 station_ids should match by name — but ERA5 has 291 stations and IMD has 293. Two missing in ERA5. Confirm with Risheek why the mismatch (likely the 2 stations dropped during `EDA_00` because of coordinate issues — see `data/missing_station_coordinates.csv`).

## Deliverable for guide

One slide with two panels: (left) histogram of correlations (3 slices), (right) WB map colored by correlation. Plus the table comparing GAT-GRU on IMD minimap (Track 1A) vs ERA5 (exp_13) numbers.

---

## Session Execution Plan

_Self-contained. Designed to run **in parallel with Track 2's GEE fetch** since this track is pure CPU analysis. Estimated total wall-time: **1.5–2 hours**._

### Pre-flight (5 min)

- Confirm files exist: `data/preprocessed_rain_data.parquet`, `data/era5_pivot_data/rain_pivot.parquet`, `data/wb_station_coords.csv`, `data/West_Bengal/District_shape_West_Bengal.shp`.
- Create notebook `EDA/EDA_02_imd_vs_era5_correlation.ipynb`. Import: pandas, numpy, scipy.stats (for pearsonr), matplotlib, seaborn, geopandas.

### Step 1 — Align both datasets on shared stations + dates (15 min)

1. Load IMD long-form and pivot to (date × station) on `rainfall`.
2. Load ERA5 pivot (`era5_pivot_data/rain_pivot.parquet`) directly.
3. Find `shared = sorted(set(imd_pivot.columns) & set(era5_pivot.columns))`. **Expected: 289–291**. Note the missing ones (likely the 2 dropped in EDA_00).
4. Restrict both to `1970-01-01 .. 2021-12-31` (overlap window).
5. **Validation gate:** print `shared` count, date range, and `imd_aligned.shape == era5_aligned.shape`.

### Step 2 — Compute per-station stats on three slices (20 min)

For each shared station, for each of three slices `{full, monsoon_only, rainy_only}`:

1. **full** = all overlapping non-NaN day-pairs.
2. **monsoon_only** = full ∩ `month in [6,7,8,9]`.
3. **rainy_only** = full ∩ `IMD > 0`.

Compute and store:

- `pearson_r` (use `scipy.stats.pearsonr`; skip stations with <30 valid days in the slice).
- `mean_bias = mean(ERA5 - IMD)`.
- `mean_abs_diff = mean(abs(ERA5 - IMD))`.
- `n_valid` = count of pairs used.

Output: long DataFrame `(station_id, slice, pearson_r, mean_bias, mean_abs_diff, n_valid)`. Pivot to wide: `(station_id, lat, lon, corr_full, corr_monsoon, corr_rainy, bias_full, bias_monsoon, mad_full, mad_monsoon, n_overlap)`. Save to `data/eda_outputs/imd_vs_era5_summary.csv`.

### Step 3 — Visualizations (30 min)

Produce, in order:

1. `imd_vs_era5_corr_histograms.png` — 3 subplots (one per slice). x-axis = Pearson r. Annotate median + IQR on each.
2. `imd_vs_era5_corr_map.png` — WB choropleth; load shapefile via geopandas; color stations by `corr_full`. Use the canonical_map normalization from `EDA_01` to align district names.
3. `imd_vs_era5_bias_map.png` — same as above but colored by `bias_full`.
4. `imd_vs_era5_bias_vs_latitude.png` — scatter, x=lat, y=`bias_full`. Add LOWESS or linear fit (use seaborn's `regplot`, not heavier statistical tests).
5. `imd_vs_era5_example_station_2010.png` — pick the median-`corr_full` station; plot IMD vs ERA5 daily series for monsoon 2010 only. Two-line plot.

**Validation gate:** all 5 PNGs exist in `data/eda_outputs/` after this step.

### Step 4 — Summary text + handover (15 min)

1. In the notebook's final markdown cell, write a 3-sentence summary suitable for the guide slide: median correlation across slices, where the biggest disagreements are, and what this implies for "IMD GNN" feasibility.
2. Append a row to `plan.md` (no new comparison table needed — just a note under Item 1 with median corr per slice).
3. Update `status.md` Research Depth Map: add `EDA_02_imd_vs_era5_correlation` at L3.
4. Add change-log entry to `status.md` + session note to `handover.md`.
5. Mark task #10 complete.

### Failure modes

- **`pearsonr` returns NaN for a station** → almost always due to constant input (all zeros). Drop those stations from the histogram but report their count as a footnote.
- **Geopandas choropleth alignment fails** → reuse the district `normalize_text` + `canonical_map` from `EDA_01` exactly. Do not invent new normalization.

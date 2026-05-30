# Track 1B — Implementation notes (what was actually done)

_Status: **COMPLETE** 2026-05-29 (Claude + Risheek). Records the real implementation and results. The `README.md` is the original plan, followed with minor deviations noted below._

## Outcome in one line

Quantified, network-wide, how far ERA5 reanalysis departs from real IMD gauge rainfall, using only simple statistics. Built the `EDA_02` notebook plus a 7-page explainer Word doc. Confirms ERA5 is a **smoothed proxy** — which explains the exp_14 IMD-vs-ERA5 gap — and the result is corroborated by published ERA5-over-India literature.

## Method (deliberately simple, no advanced statistics)

For each station shared by both datasets (291; **199** with ≥365 overlapping days), over **1970–2021**, using **observed days only** (missing gauge values never filled): **Pearson r**, **mean bias** (ERA5 − IMD), **mean absolute difference (MAD)**, and **wet-day frequency** (% days ≥1 mm). Each computed on three slices: **full**, **monsoon** (JJAS), **rainy** (IMD ≥ 1 mm). No Spearman / quantile / categorical skill scores — a conscious scope choice for defensibility.

## Results (medians across stations)

- Pearson r: **0.37 full → 0.29 monsoon → 0.24 rainy** (range 0.02–0.57). Correlation drops as the easy dry-day zeros are removed.
- Mean bias: **≈ −0.04 mm/day (≈ 0)** — averages agree.
- MAD: **7.58 mm/day** — large next to the ~6.4 mm mean, so individual days disagree a lot.
- Wet-day frequency: **IMD 30% vs ERA5 51%** — ERA5's drizzle bias, quantified.
- Spatially: bias near-zero almost everywhere except scattered strong departures in the **hilly north**, where correlation is also lowest.

## The defensible story (lock this)

ERA5 ≈ IMD on **climatology** (bias ≈ 0) so it is a sound basis for the main GNN work where gauges are too sparse to build a full graph; but it is only a **moderate** match at the **daily** scale, **over-reports light rain** (drizzle) and **smooths extremes**, **worst over complex terrain**. This matches the published record (Lavers et al. 2022 QJRMS; Atmospheric Research 2021 India assessments; Scientific Reports 2024). It explains exp_14: ERA5 is smoother and ~2× more autocorrelated day-to-day, so the same model predicts it better — the RMSE gap is a **data property**, not a model failure. Always compare **within-study** (not against exp_13's 8.79).

## Files produced (all new; no existing code modified)

- `EDA/EDA_02_imd_vs_era5_correlation.ipynb` — `%matplotlib inline`, renders figures in-cell.
- `docs/track_1b_imd_era5_corr/EDA_02_explained.docx` — 7-page methodology + reading guide + viva Q&A + references, with the four figures embedded. The teaching/defence document.
- `data/eda_outputs/imd_vs_era5_summary.csv` — one row per station. Columns: `station_id, lat, lon, corr_full, corr_monsoon, corr_rainy, bias_full, bias_monsoon, mad_full, mad_monsoon, wetfreq_imd, wetfreq_era5, n_overlap`.
- `data/eda_outputs/imd_vs_era5_{corr_histograms,maps,bias_drizzle,example_station}.png`.

## Deviations from the README plan

- Rainy slice = **IMD ≥ 1 mm** (WMO wet-day threshold) rather than `> 0`.
- Added a **wet-day-frequency** figure (the clearest drizzle evidence).
- Maps are **colored station points over district boundaries** (not a per-district choropleth); correlation + bias combined into one `imd_vs_era5_maps.png`; bias-vs-latitude + drizzle combined into one figure.
- Example station is **auto-selected** as the median-correlation station (RAJNAGAR) over its best-covered monsoon (1982), not a fixed 2010.
- Map stations require **≥365** overlapping days (README suggested ≥30) for stable per-station statistics.

## Open / optional next steps

- Per-district choropleth (true area aggregation); seasonal bias maps.

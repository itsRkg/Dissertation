# Dissertation — Status (Master Index)

_Coarse master index. Loaded into every session — kept lean. For detail follow the pointers._

| Doc | What it contains |
|-----|------------------|
| `status.md` (this file) | Coarse overview, file map, known issues, depth map, change log, pointers. |
| `plan.md` | Forward-looking master plan, item map, master comparison table, cross-cutting implementation notes. |
| `handover.md` | Most-recent session notes, next-session lookouts, open clarifications. **Read after this file.** |
| `docs/track_usages.md` | **Documentation policy: directory hierarchy, where new files go, naming + linking conventions, token-budget rationale.** Read once per new session, follow always. |
| `docs/references.md` | Cross-cutting external literature, datasets, internal design doc index. |
| `docs/track_1a_imd_minimap/README.md` | Item 1 Track A: GAT-GRU on IMD station subset. |
| `docs/track_1b_imd_era5_corr/README.md` | Item 1 Track B: IMD vs ERA5 correlation EDA. |
| `docs/track_2_regional_ablation/README.md` | Item 2: Northern Plains + Himalayan ablation. |
| `docs/track_3_rl_reuse/README.md` | Item 3: minor_project_rl reuse (placeholder). |

## Documentation policy (one-line version — full version in docs/track_usages.md)

**Write deep, link shallow.** Track-specific content lives in `docs/track_<id>_<name>/`. Cross-cutting content lives at the root. When adding a new file: ask "is this specific to one track?" — yes → goes in that track's subdirectory; no → goes in a root doc or `docs/references.md`. Track entry point is always `README.md`. Full rules + linking conventions: `docs/track_usages.md`.

---

## Goal

Predict daily rainfall over West Bengal (and ablation regions), with emphasis on **extreme-event accuracy** during monsoon. Storyline: temporal-only LSTMs → focal/seasonal losses → multi-scale + attention → ConvLSTM grid → regression-as-classification transformer → **spatio-temporal graph models (GAT-GRU current best)**.

## Current Status (coarse)

- **Best model:** GAT-GRU on ERA5 WB. Real-space test RMSE **8.79** (exp_13).
- **GNN pipeline:** validated end-to-end. Radius-100km symmetric graph, plain MSE loss, 12 input features (6 ERA5 vars + 6 cyclic time), 7-day window, 1-day horizon.
- **Data pipelines:** IMD (1901–2021, 293 stations, rain only) and ERA5/GEE (1970–2023, 291 stations, 6 vars) both functional.
- **Upcoming:** Items 1 + 2 (one day, next session). Item 3 (RL reuse) scoping in next prompt.

Detailed metrics table → `plan.md`.

## Known Issues (one-liners — full detail in handover.md / track docs)

1. **exp_12 cells 64+69** cosmetically modify preds. Saved CSVs verified safe; visual-only.
2. **Scaler/split misalignment**: `scale_pivots(train_end='2015-12-31')` sees 85% of timeline but train is only 70%. Mild leakage. Fix for new runs.
3. **Two `temporal_split` functions** in `data_helper_utils.py` (lines 5 and 187). Second wins on import; first is dead.
4. **`gnn_dataset.build_edge_index(k=5)` is unused.** Actual experiments use `data_helper_utils.build_edge_index_radius(threshold_km=100)`.
5. **exp_4–exp_10 notebooks have a stale "RMSE 15.82" comparison cell** — not their real metric.
6. **`exp_5_seasonal` log shows NaN** — failed run; valid one is `exp_5_lstm_with_seasonal_focal_loss`.
7. **exp_13 result CSVs are incomplete** — only `overall_metrics` + `station_metrics`; missing seasonal / district.
8. **No random seeds** set in any notebook.

## File / Folder Map (short)

| Path | Purpose |
|------|---------|
| `models/` | One file per architecture (`baseline_lstm`, `lstm_with_*`, `decoder_only`, `gcn_gru`, `gat_gru`). |
| `utils/data_utils/` | `data_helper_utils.py` (split/scale/graph/tokenizer); `dataset_files/` (per-architecture Dataset classes). |
| `utils/metric_utils/metrics.py` | rmse, mae, bias, nrmse. |
| `utils/loss.py` | `FocalMSELoss`, `SeasonalFocalMSELoss`. |
| `utils/logging_utils.py` | `setup_logger`. |
| `utils/plotting_utils/` | Plot helpers. |
| `experiments/experimentation_notebooks/` | `exp_1.ipynb` … `exp_13.ipynb`, plus `gee_data_fetch.py`, `cds_data_fetch.ipynb`. |
| `experiments/logs/<exp>/` | Per-epoch training logs (single `.log` file each). |
| `experiments/saved_models/<exp>/` | Per-epoch `.pt` checkpoints + `_best.pt`. |
| `experiments/results/<exp>/` | Final metrics CSVs and plots (exp_12 complete, exp_13 partial). |
| `EDA/` | `EDA_1` (IMD, old), `EDA_00` + `EDA_01` (ERA5, current). |
| `data/` | Raw + processed. Key: `Rainfallwlatlongshort.xlsx` (raw IMD), `preprocessed_rain_data.parquet` (cleaned IMD), `gee_era5_data/` (raw ERA5 yearly CSVs), `era5_pivot_data/*.parquet` (wide-form ERA5), `processed/era5_station_level_clean.parquet` (cleaned long-form ERA5), `wb_station_coords.csv`, `West_Bengal/*.shp`. |
| `constants/general.py` | `SEASONS` dict. |
| `docs/` | This documentation tree. |

## Confirmed Metrics (compressed — full table in plan.md)

Real-space test RMSE: Baseline LSTM 15.82 → ConvLSTM Keras 13.37 → Transformer 18.53 → **GCN-GRU 8.99 → GAT-GRU 8.79**.

## Research Depth Map

L0 = not looked at, L1 = surface scan, L2 = code read end-to-end + confirmed against notebooks, L3 = code + numerical verification.

| Area | Depth | Notes |
|------|-------|-------|
| Data: IMD pipeline | L2 | Schema + dates confirmed. Cleaning code not re-traced. |
| Data: ERA5/GEE pipeline | L2 | Pivots, scaler, splits read. Leakage flagged. |
| EDA notebooks | L1 | Cell headers scanned. |
| `models/baseline_lstm.py` | L1 | Not read directly. |
| `models/lstm_with_custom_criterion.py` | L1 | Not read directly. |
| `models/lstm_with_seasonal_criterion.py` | L1 | Args extracted from notebook. |
| `models/lstm_with_simple_self_cross_attention.py` | L0 | Multi-scale model. |
| `models/decoder_only.py` | L0 | Transformer. |
| `models/gcn_gru.py` | L3 | Full read, shapes verified, results CSV cross-checked. |
| `models/gat_gru.py` | L3 | Full read, heads=4 concat confirmed. |
| `utils/loss.py` | L1 | Args known; math not derived. |
| `utils/data_utils/data_helper_utils.py` | L3 | All helpers verified. |
| `utils/data_utils/dataset_files/gnn_dataset.py` | L3 | Verified `build_edge_index` is unused. |
| `utils/data_utils/dataset_files/multiscale_dataset.py` | L0 | Used by exp_7–8. |
| `utils/data_utils/dataset_files/classification_dataset.py` | L0 | Used by exp_10. |
| `utils/metric_utils/metrics.py` | L1 | Implementations not verified. |
| `utils/plotting_utils/` | L0 | — |
| exp_1–3 | L2 | Final metrics extracted. |
| exp_4–6 | L2 | Stale comparison cell flagged. |
| exp_7–8 | L1 | Scaled-space only. |
| exp_9 ConvLSTM (Keras) | L2 | RMSE 13.37 confirmed. |
| exp_9 ConvLSTM (PyTorch) | L2 | Confirmed leftover copy of exp_8. |
| exp_10 transformer | L2 | RMSE 18.53 extracted. |
| exp_11 GNN data prep | L3 | Verified data-prep only; pivot missingness numbers extracted. |
| exp_12 GCN-GRU | L3 | Full cell-order trace; cosmetic-adjustment safety verified. |
| exp_13 GAT-GRU | L2 | Numbers confirmed; result CSVs incomplete. |
| exp_14 GAT-GRU IMD minimap | L3 | Built + run; IMD vs matched ERA5 (same 16 stations/window); full results saved. |
| EDA_02 IMD-vs-ERA5 corr | L3 | Built + run; 291 stations; summary CSV + 5 figs + explainer docx. |

---

## Change Log

Append-only. One entry per substantive session.

### 2026-05-30 — track_c reframed + Session 0/1 (Claude)

- **Dissertation reframed** (decided with Risheek): retitle away from "extreme rainfall events" → primary thesis = comparative spatio-temporal models + IMD-vs-ERA5 data-source study (built from completed work). Extremes → a private diagnostic / measured limitation (models under-disperse, preds cap ~70 mm). track_c (RL-reuse/SSL) is now an **upside chapter**. Full plan: `docs/track_3_rl_reuse/session_plan.md` (read §0). **Pending (needs Risheek's OK):** propagate the retitle into this file's Goal line + `plan.md`; reflect the single-seed-now/multi-seed-final policy in the track_3 README (v5→v6, Codex-loop doc).
- **track_c plan**: trimmed-but-greedy. Backbone Tier 0 (12-feat) → Tier 0b (15-feat +lat/lon/elev) → Tier 2A (masked-recon pretrain→finetune); greedy pool (2B/1/2C/3) by value×cost×parallelism + reroute gates. **Seed policy (HARD): single seed now, multi-seed final/parallel/winners-only.**
- **Session 0 DONE**: NEW `utils/metric_utils/extreme_skill.py` (POD/FAR/CSI/HSS/freq-bias at IMD 35.6/64.5/124.5 mm; self-test passes) + NEW `experiments/experimentation_notebooks/fetch_srtm_elevation.py` → Risheek ran it → NEW `data/wb_station_elevation.csv` (293/293, web-verified). Running log: `docs/track_3_rl_reuse/implementation_notes.md`.
- **Session 1 RUNNING (~4.5–5 hr; outputs pending)**: `experiments/experimentation_notebooks/exp_16_gat_gru_baseline_seeds.ipynb` — Tier 0 + Tier 0b + **depth2 (2-layer GAT)**, single seed 42, scaler/split fix + best-val-ckpt fix, saves full metrics + raw preds + private `extreme_skill_real.csv` + `embedding_diag.csv` (oversmoothing check). nbformat-valid; code cells compile. Also NEW: `models/gat_gru_multilayer.py` (depth-configurable; num_layers=1==original) and `utils/metric_utils/embedding_diag.py` (Dirichlet/MAD/eff-rank + GAT-output hook; self-test passes; reusable for Tier 2A). Measured the 100km graph: 1-hop=21% of nodes, diameter ~7 → depth has real range; literature caps useful depth at 2–3 (oversmoothing).
- **Session 2 BUILT (not yet run)**: NEW `models/gat_gru_pretrain.py` (masked-recon pretext; encoder identical to GAT_GRU_Model for 1:1 transfer + MLP decoder) and `experiments/experimentation_notebooks/exp_18_gat_gru_mae_pretrain.ipynb` (BERT-style 15% masking of the 6 physical channels; cyclic/static visible; loss on masked only; lr 3e-4; batch 64; single seed; saves `pretext_best.pt` + loss curve + encoder embedding health check). py_compile + nbformat-valid. Mask-ratio regime web-verified (15–25% for value-corruption masking, not 75% token-drop).
- **Kaggle workflow set up**: NEW `utils/env_paths.py` (local vs Kaggle path resolution), `.gitignore` rewritten (keeps `data/`; adds results/saved_models/logs/caches/secrets), `docs/kaggle_workflow.md` (push → Kaggle-dataset → run → retrieve, + private-repo PAT). `exp_18` made env-portable (bootstrap clone + get_paths) → runs local AND Kaggle unchanged. Results retrieved via Kaggle Output/`kaggle kernels output`, not git. **Pending (Risheek runs):** `git rm --cached` results/saved_models (still tracked, 284MB) + commit + push; upload data subset to Kaggle.
- No model/util logic or data modified; only `.gitignore` (config) + this session's own new files touched.

### 2026-05-29 — Item 1 executed: exp_14 (Track 1A) + EDA_02 (Track 1B) (Claude)

- **Track 1A (exp_14):** trained GAT-GRU on a clean IMD subset and on matched ERA5 (same 16 stations, 2000–2020, identical config). **Reframed the plan to window-first** (2000–2020 then top-16 lowest-missing) after verifying 1970+ IMD missingness is block-structured (whole-year gaps; 2021 empty). Real-space test RMSE: IMD rain 17.60, ERA5 rain 10.42, ERA5 full 10.03; baselines persistence 22.53 / climatology 18.50. The data-source gap is intrinsic (ERA5 smoother; lag-1 autocorr 0.56 vs 0.31). Used `preprocessed_rain_data.parquet` (the README's "4.74M rows" label is wrong — that figure is `processed_rain`). Scaler/split leakage (issue #2) fixed in this run.
- **Track 1B (EDA_02):** per-station IMD-vs-ERA5 Pearson / bias / MAD + wet-day frequency on 3 slices, 291 stations (199 with ≥1 yr). Median r 0.37 / 0.29 / 0.24, bias ≈ 0, wet-day IMD 30% vs ERA5 51%. Corroborated by ERA5-over-India literature. Built `EDA_02_explained.docx` (methodology + viva defence).
- Full records: `docs/track_1a_imd_minimap/implementation_notes.md`, `docs/track_1b_imd_era5_corr/implementation_notes.md`. No existing code/notebooks modified; only new notebooks/results/docs added. `handover.md` left for Risheek's end-of-session prompt.

### 2026-05-29 — Two Opus research passes applied; track 3 README bumped to v5 (Claude)

- Opus pass 1 evaluated whether v4's V_DIST-as-peer-primary direction was justified. Returned a major correction: HyMePre's "2048 stations" are 5.625° grid cells, NOT real meteorological stations. **Verified by Claude via direct paper fetch.**
- Opus pass 2 verified three load-bearing claims: GPT-ST's masked > contrastive ordering (qualitative, exact margins not extractable); STEP/STD-MAE/GPT-ST all unambiguously pretrain-then-finetune topology; no published joint masked-recon + forecasting precedent in weather domain. **STD-MAE and STEP topologies independently verified by Claude via web search.**
- The key mechanistic insight that flipped my prior v5 thinking: **STGCL's "joint > pretrain" finding is contrastive-specific, NOT a universal STGNN-SSL rule.** Mechanism: contrastive pretext = instance discrimination (misaligned with regression downstream); masked-recon pretext = continuous value prediction (aligned with regression downstream). Different SSL families have different optimal usage topologies.
- v5 changes applied:
  - **Tier 2A (PRIMARY)**: masked-recon **pretrain-then-finetune**. Was a v4 peer primary; now lead direction. Anchored on STEP/STD-MAE/GPT-ST lineage.
  - **Tier 2B (NOVEL COMPARISON)**: masked-recon **joint loss** (forecast + recon in single training stage). No published weather-domain precedent — this is the dissertation's empirical contribution.
  - **Tier 2C (V_DIST documentation)**: V_DIST demoted from full pretraining run to 35-minute preprobe only. Single-seed negative result documented in the writeup.
  - **Tier 3 (optional BC analog)**: 3-stage SSL → multi-task supervised pretrain → rain finetune. Addresses Risheek's earlier "BC is missing" concern.
  - **Seed policy locked**: 3 seeds at Tier 0 anchor (non-negotiable), 1 seed elsewhere → 3 seeds only if a tier beats its comparison anchor by ≥0.2 RMSE.
  - **HyMePre removed from references.md** as a station-level precedent.
  - **STEP, GPT-ST, STD-MAE, W-MAE, SpaT-SparK added** to references with verified topologies.
- README version: `track3-readme-2026-05-29-v4` → `track3-readme-2026-05-29-v5`.
- No code or data modified.

### 2026-05-29 — Codex pass 3 (informal) applied; track 3 README bumped to v4 (Claude)

- Codex pass 3 reviewed Claude's chat clarification of Path A sampling structure. Flagged Strategy A as compute-wasteful (only ~62 of 291 nodes get meaningful gradient per pair with 1-layer GAT) and identified two stale mean-pool remnants in v3 README. Recommended Strategy C-subset (K date-pairs × M target stations).
- Claude verified all technical claims against ground truth: **293 stations in coords file (confirmed), 100km graph avg degree 62.2 (Codex said 62, confirmed exactly), batch-256 expected unique stations 170.4 (Codex said 170, confirmed exactly), VRAM activations ~1.4 GB for two windows at batch-16 (fits 8 GB with headroom)**.
- v4 changes applied:
  - **Strategy C-subset becomes V_DIST default sampler**: K=16 date-pairs per batch × M=32 target stations per date-pair = 512 loss terms over 32 graph forwards (vs Strategy A's 256 forwards for 256 loss terms — ~16× more gradient per compute unit).
  - **Strategy A retained as preprobe Stage B probe** for direct comparison against C-subset. Whichever has higher held-out ρ after 5-epoch frozen-encoder probe becomes the full 1.5hr pretext run.
  - **Strategy B (all-station supervision) dropped entirely** — reasoned-out as over-constraining the encoder (forcing every station on same calendar gap to satisfy same lag-distance is physically unrealistic given dry-phase vs wet-phase trajectories).
  - **Two stale mean-pool remnants cleaned** from open-questions list and corrections table (v3 had selected-node fix in the Tier 2A spec but left the older mean-pool language elsewhere).
  - **Preprobe Stage B re-scoped** as Strategy A vs Strategy C-subset bake-off (5 epochs each); winner runs full pretext.
- README version: `track3-readme-2026-05-29-v3` → `track3-readme-2026-05-29-v4`.
- No code or data modified.

### 2026-05-29 — Codex pass 2 (informal) applied; track 3 README bumped to v3 (Claude)

- Codex pass 2 reviewed Claude's chat reasoning about proposed v3 changes (not a formal version-stamped pass). Claude verified the load-bearing empirical claim — V_DIST lag-distance Spearman — at scale: **30-station probe confirms mean ρ ≈ 0.41 at lag 7 (anomaly), collapsing to ρ < 0.15 by lag 90.** Pattern is robust (std ~0.03 across stations).
- Two literature claims (Weather2K time-invariant constants; HiSTGNN hierarchical STGNN) verified via web search.
- v3 changes applied:
  - **Feature-treatment lock**: anomaly normalization on aux variables consistent across all Item 3 tiers. Rain stays raw. Cyclic time excluded from V_DIST pretext.
  - **Tier 0b** added as separate static-feature baseline (15 features with lat/lon/elev). All SSL methods compare against matched baseline (12 or 15 features).
  - **Tier 1 Path C aux head targets** changed to next-day anomalies (input/output distribution match).
  - **Tier 2A V_DIST**: max lag capped at 14 days (empirically justified). Pair embedding = selected station's node final GRU hidden state (NOT mean-pool — fixes v2 contradiction). Two-stage preprobe (all-lag documentation + 14-day feasibility gate).
  - **Tier 2B Path B (masked reconstruction)** promoted from "Tier 3 fallback" to peer primary alongside Tier 2A.
  - **Dissertation narrative reframed** as "we compared faithful-RL-transfer vs domain-adapted alternative" — defensible regardless of which path wins.
- Spearman probe table baked into README as empirical evidence.
- Risheek confirmed: (a) OK losing direct comparability to exp_13's 8.79 (improvement over current is the goal); (b) OK fetching SRTM elevation via GEE for Tier 0b.
- No code or data modified.

### 2026-05-29 — Codex pass 1 applied; track 3 README bumped to v2 (Claude)

- Codex returned 12 corrections-table flags + 4 design critiques + 3 missed concerns on Track 3 README v1.
- Each flag re-verified against source files in `minor_project_rl/rnd_baseline/track_c/`. **All 12 flags confirmed by source.** No flag rejected.
- Most consequential correction: canonical s13_run7 recipe is **distance-only SmoothL1 on log1p(lag), 20 epochs**, NOT ordinal CE + 40 epochs as v1 claimed. Confirmed via `runs/encoder/s13_run7/metadata.json` (loss=`smooth_l1_log1p_lag`, ordinal=false, epochs=20, script=`experiments/distance_loss/train_distance.py`).
- Tier 2 restructured: **V_DIST is primary**, V2 (ordinal CE) and V1 (BCE) demoted to predecessor ablations.
- Augmentation: ±1 day window-jitter REMOVED (corrupts the lag label — Codex's critical insight). Replaced with rain-noise + optional cutout. Sources for source-aug clarified.
- Three NEW guardrails added: (1) imputation-gap masking (ffill/bfill/0 creates fake lag pairs across data gaps), (2) seasonality leak via temp/dew/pressure with three mitigation options (default: day-of-year anomaly normalization), (3) 30-min preprobe before full pretraining (analog of `probe_projfit` from the source).
- Several wording corrections: "rainy-day balance" is NOT source-equivalent to NOOP balance (it's a domain bridge); "must match augmentation" is encoder→BC in source, our pretext↔finetune is the analogous bridge; "default unfrozen" mirrors deployed recipe not CLI default.
- README version stamp: `track3-readme-2026-05-29-v1` → `track3-readme-2026-05-29-v2`.
- No code or data modified.

### 2026-05-29 — Review-loop workflow locked + version stamp on track 3 (Claude)

- Added `**REVIEW VERSION: track3-readme-2026-05-29-v1**` stamp to the top of `docs/track_3_rl_reuse/README.md`. The Codex prompt at the bottom of that file now requires Codex to echo the stamp as the first line of its response.
- Codified the **review-loop workflow** in `docs/track_usages.md` § "Review-loop with Codex (or any external reviewer)". Key rules: Codex never writes; Claude is the only writer; version mismatch = stale review (do not apply); each Codex flag is re-verified against source files before editing; no verbatim adoption of reviewer prose.
- This pattern is reusable for any future track that wants external review (1A, 1B, 2 if needed). Same shape: place a `REVIEW VERSION` stamp + a `[reviewer-only] no-write` prompt at the bottom of the track README.
- No code or data modified.

### 2026-05-29 — Track 3 verified against minor_project_rl source (Claude)

- Risheek granted read-only access to `C:\Users\rishe\minor_project_rl`.
- Read `rnd_baseline/track_c/encoder_tdc.py`, `train_encoder.py`, `bc_teacher.py`, `RESULTS_5M_ABLATION_TRACK_C.md`, `SESSION18_NOTES.md` §1.
- Reconciled Tier 2 spec in `docs/track_3_rl_reuse/README.md` against the verified canonical recipe (s13_run7). Significant corrections: loss is BCE / ordinal CE (not InfoNCE), thresholds are close=4/far=50 (not 7/30), rainy-day balance 50/50 (not 70/30), LR 3e-4 for pretext, batch 256, 40 epochs, projection head 128-d, finetune default is fully unfrozen (not frozen-first-5). Cross-station pairs are always-far semantic class. Single-source-sequence rule applies (one station's time series per pair).
- Full corrections table appended to track 3 README. Codex verification prompt drafted and embedded so Risheek can request an independent review.
- No code or data modified.

### 2026-05-29 — Item 3 ideation locked: tiered SSL ladder (Claude)

- Idea 3 (RL → rainfall transfer) scoped from the knowledge-transfer prompt Risheek pasted. `minor_project_rl` folder access was declined; ideation worked from the prompt content only, flagged in track 3 README.
- **Locked framing:** methodology-transfer paper — "RL encoder-pretraining recipe applied to rainfall forecasting."
- **Locked strategy:** tiered experiment ladder (Tier 0 baseline multi-seed → Tier 1 multi-task aux heads → Tier 2 TDC-analog SSL → Tier 3 conditional MAE backup → Tier 4 ablation matrix). Each tier has explicit decision gates. **No single-bet on a path.**
- 6 web searches conducted to ground literature claims: STEP, TNC, TS2Vec, Adaptive Spatio-Temporal SSL Weather (arxiv 2511.00049), Springer 2025 MTL rainfall, CAMT. All added to `docs/references.md`.
- 5 new tasks created (Tier 0–4) replacing the old single Item 3 placeholder task.
- `docs/track_3_rl_reuse/README.md` rewritten end-to-end with full ladder + decision gates + guardrails.
- No code or data modified.

### 2026-05-29 — Session execution plans locked for Items 1 + 2 (Claude)

- Added Session Execution Plans to all three Item 1+2 track READMEs.
- Combined daily orchestration timetable added to `handover.md`.
- No code or data modified.

### 2026-05-29 — Per-track subdirectories + working policy (Claude)

- Promoted each track to its own subdirectory: `docs/track_<id>_<name>/README.md`. Future track-specific files (implementation notes, scratch analyses, refs) land in the same subdir, keeping root files lean.
- Added `docs/track_usages.md` to codify the documentation policy: where new files go, naming + linking conventions, token-budget rationale.
- Updated cross-references in `status.md`, `plan.md`, `handover.md`, `docs/references.md`.
- No code or data modified.

### 2026-05-29 — Doc restructure into status / plan / handover / docs/* (Claude)

- Split the monolithic `status.md` into a doc tree: this slim `status.md` (master index), `plan.md` (forward-looking master plan), `handover.md` (session continuity), `docs/track_*.md` (per-track specifics with file references), `docs/references.md` (literature + internal docs).
- Open clarifications consolidated into `handover.md`.
- No code or data modified.

> _Superseded by the 2026-05-29 per-track-subdirectories entry above. The flat `docs/track_*.md` files were moved into `docs/track_*/README.md`._

### 2026-05-29 — Future-work roadmap added (Claude)

- Scoped Item 1 (IMD GNN + IMD-vs-ERA5 correlation) and Item 2 (regional ablation NP + Himalayan), both fitted to RTX 4060 / 16 GB RAM budget.
- Item 3 (`minor_project_rl` reuse) deferred to the next prompt per Risheek.

### 2026-05-29 — Initial code-grounded audit (Claude)

- Walked EDA → 13 experiments → GNN models → data helpers end-to-end.
- Corrected GPT's "deep research report" on multiple points (radius not kNN, plain MSE not seasonal, exp_11 is data prep, etc.).
- Verified exp_12 cosmetic-adjustment cells do not corrupt saved metrics.
- Documented all known issues that propagate forward.

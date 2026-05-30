# Track 3 (track_c) — Implementation Notes

_Running log of what was actually built/run for track_c, per [`session_plan.md`](session_plan.md) §8. Append a dated entry per session. Read [`session_plan.md`](session_plan.md) first for the plan; this file records reality + deviations + reroute decisions._

---

## Session 0 — Diagnostics harness + elevation fetch (2026-05-30, Claude)

### Goal
Build the private extreme-skill scorer and the Tier-0b elevation input, so Sessions 1+ can run without setup. (Diagnostics, not a headline metric — see `session_plan.md` §0/§4.)

### What was built (all NEW files; no existing code modified)
1. **`utils/metric_utils/extreme_skill.py`** — standalone numpy/pandas module for categorical (dichotomous) heavy-rain verification. Primitives `contingency(obs, pred, tau)` → (hits, false_alarms, misses, correct_neg); scores `pod`, `far`, `success_ratio`, `csi`, `frequency_bias`, `hss`; tables `extreme_skill_table(obs, pred, thresholds=(35.6, 64.5, 124.5))` and `extreme_skill_by_group(...)`. Default thresholds = IMD rather-heavy/heavy/very-heavy (mm/day). `n_events` (observed events ≥ τ) is a first-class column.
2. **`experiments/experimentation_notebooks/fetch_srtm_elevation.py`** — one-off GEE lookup of `USGS/SRTMGL1_003` elevation at the 293 `wb_station_coords.csv` points → `data/wb_station_elevation.csv` `[station_id, elevation_m]`. Mirrors `gee_data_fetch.py` (project `ee-risheekghosh1`, same auth/station-load pattern); uses a direct `.getInfo()` (293 points from one image — no Drive export needed). **Risheek runs this** (sandbox has no EE auth).

### Verification (done, not assumed)
- `python utils/metric_utils/extreme_skill.py` → **self-test ALL PASS.** Cases: a hand-built table (a=5,b=2,c=3,d=90 at τ=10) matching POD 0.625 / FAR 2/7 / SR 5/7 / CSI 0.5 / freq-bias 0.875 / HSS 888/1388; a perfect-forecast case (POD/CSI/HSS=1, FAR=0); a no-observed-events case (all scores NaN, not crash); NaN-pair dropping; table columns/shape.
- `python -m py_compile` clean on both files.
- Score definitions verified against CAWCR / EUMETRAIN / NWS forecast-verification docs (2026-05-30 web search). FAR encoded as the false-alarm **ratio** `b/(a+b)` (NOT the rate `b/(b+d)`) — convention noted in the docstring.

### Decisions / deviations
- **FAR = ratio convention** (documented in module). `success_ratio = 1 − FAR` included for performance-diagram use.
- **Zero-denominator → NaN** (not 0 or error), so empty thresholds (esp. very-heavy ≥124.5) surface honestly via `n_events`/NaN rather than fake zeros.
- **Elevation fetch uses `.unmask(0)`** so water-masked pixels (coastal stations e.g. SAGAR ISLAND / SANDHEADS) return 0 m = sea level; the script prints which stations came back exactly 0 for a sanity check. Point sample at 30 m (native); optional `BUFFER_M` mean is available but off by default.
- exp_13/exp_14 extreme retrofit (the optional Session-0 step) was **deferred**: both saved only metric CSVs / scatter PNGs (no raw `(obs,pred)` arrays), so scoring them needs re-inference from checkpoints (`exp_13_gat_gru_best.pt` exists) — a model run, i.e. Risheek's to do. Cleaner to get the extreme diagnostics for free from the Tier 0 run (Session 1) instead.

### Research notes (grounding, per session_plan §5)
- Extreme-skill scores: CAWCR Forecast Verification; EUMETRAIN Frequency Bias; NWS/OWP glossary.
- IMD daily thresholds (rather-heavy 35.6 / heavy 64.5 / very-heavy 124.5 mm): IMD Heavy Rainfall Warning Services.
- Tier 0b elevation precedent (for Session 1): MetNet-2 (Nature Comms 2022) feeds lat/lon/elevation; Weather2K uses them as station constants. Honest nuance: some studies find lat/lon redundant — Tier 0b is the ablation that settles it here.

### What the next session (Session 1) needs
1. Run `fetch_srtm_elevation.py` → confirm `data/wb_station_elevation.csv` (293 rows, sane min/mean/max; eyeball the 0-m list).
2. Build `exp_16_gat_gru_baseline_seeds.ipynb` (Tier 0, 12-feat, **single seed 42**) + Tier 0b variant (15-feat, +lat/lon/elev). Apply the two fixes (scaler/split `train_end` from the 70%-index; eval from `_best.pt`). Score with `extreme_skill.extreme_skill_table(...)` on real-space preds → `extreme_skill_real.csv`.
3. Reminder: **single seed everywhere now; multi-seed is the final step (Session 5)**.

### Files
- NEW: `utils/metric_utils/extreme_skill.py`, `experiments/experimentation_notebooks/fetch_srtm_elevation.py`, this file.
- NOT modified: any existing notebook/model/util/data.

### Post-run update (2026-05-30) — elevation fetch DONE + verified
Risheek ran `fetch_srtm_elevation.py` → `data/wb_station_elevation.csv` written, **293/293 stations, 0 missing**. Range min 0.0 / mean 180.6 / max 3208.0 m. Web-cross-checked: PHALUT 3208 / SANDAKPHU 3119 m (WB's two highest summits are Sandakphu 3636 / Phalut 3595 — station points sit just below the actual peaks), DARJEELING 2013 m (town ~2045 m), SILIGURI 129, ALIPUR/Kolkata 14. The exactly-0 m stations (DIAMOND HARBOUR, HOOGHLY, RAIDIGHI, SANDHEADS, ULUBERIA) are all the low-lying delta/Hooghly/coastal points — `.unmask(0)` behaving as designed. **Elevation input verified good; Tier 0b can use it.**

---

## Session 1 — Tier 0 + Tier 0b baseline notebook built (2026-05-30, Claude)

### Goal
Build the corrected baselines (the anchors for every later tier), scored on RMSE + train-val gap, with the private extreme diagnostic attached. Single seed (42) per the §1 hard rule.

### What was built (NEW file; nothing existing modified)
- **`experiments/experimentation_notebooks/exp_16_gat_gru_baseline_seeds.ipynb`** (13 cells). One notebook, two configs in a loop:
  - **tier0** — 12-feat (6 ERA5 vars + 6 cyclic time), `in_channels=12`. Anchor.
  - **tier0b** — 15-feat (+ lat/lon/elev, z-scored across stations, appended after feature 0=rain), `in_channels=15`.
  - Both: exp_13 hyperparams (SEQ_LEN 7, hidden 64, heads 4, batch 32, 30 ep, Adam 1e-3, MSELoss), 100 km radius graph, single seed 42.
- Built programmatically via `nbformat` (a sandbox builder script) so the JSON is valid; **Risheek runs it** on the `torch_cuda` env.

### The two fixes (vs exp_13)
1. **Scaler/split alignment (issue #2):** `train_end_date = dates[int(0.7*T)-1]` (≈2007-10) passed to `scale_pivots(train_end=...)` — exactly matches `temporal_split`'s 70% train cut. Printed in-notebook. (exp_13 used the default `'2015-12-31'`.)
2. **Best-val checkpoint (issue, plan note #4):** evaluate from `{exp_name}_best.pt`, not epoch-30. Uses the existing `train_utils.evaluate(...)` for test/train/val loss → explicit **train-val gap** at best ckpt (clean: GAT-GRU has no dropout/BN).

### Outputs the run will produce (per config, under `experiments/results/exp_16_gat_gru_baseline_seeds/<config>/seed_42/`)
`overall_metrics.csv`, `seasonal_metrics_real.csv`, `station_metrics_real.csv`, `extreme_skill_real.csv` (PRIVATE diagnostic), `test_predictions_real.npz` (raw obs/pred → re-scoreable without re-inference), plus a top-level `comparison_seed42.csv`.

### Verification (done, not assumed)
- Grounded on the actual APIs (read this session): `data_helper_utils` (`load_pivots`, `scale_pivots(train_end=...)`, `temporal_split`, `build_edge_index_radius`, `get_lat_lon_aligned`), `gnn_dataset` (`build_feature_tensor` returns `(X, order)`; feature 0 = rain = target; `SpatioTemporalDataset` seq_len=7), `train_utils` (`train_model` saves `_best.pt`; `evaluate` returns `(loss, preds, targets)`), `gat_gru.GAT_GRU_Model(in_channels, hidden_dim, heads)`.
- `nbformat.validate` passed; **all 11 code cells compile** (`py_compile`). Static features asserted to leave rain at feature 0.
- NOT run by me (needs GPU + torch_geometric + Risheek's env; he runs experiments).

### Decisions / deviations
- **One notebook, two configs** (not two files) → identical pipeline, matched comparison, less duplication.
- **Static features z-scored across stations** (lat ~21–27, lon ~86–90, elev 0–3208 would otherwise dominate the z-scored dynamic channels). Appended at the end so rain stays feature 0 (target extraction unaffected).
- **`SEEDS=[42]`** (a list) — single seed now; Session 5 extends headline winners to `[42,123,7]`. No multi-seed here (§1 hard rule).
- Train-val gap computed via `evaluate` on train/val loaders at the best ckpt (adds a couple of forward passes; fine).

### What the next session (Session 2) needs
1. Risheek runs `exp_16...ipynb`; confirm tier0 RMSE is in a sane range and `train_end_date` prints ≈2007-10; note tier0 vs tier0b RMSE + gap; eyeball `extreme_skill_real.csv` (expect low CSI@64.5).
2. Then build `models/gat_gru_pretrain.py` + the masked-recon pretext notebook (15-feat; mask only the 6 physical channels). I can build that during/after the tier0 run (parallel build-ahead).

### Files
- NEW: `experiments/experimentation_notebooks/exp_16_gat_gru_baseline_seeds.ipynb`.
- NOT modified: any existing notebook/model/util/data.

### Methodological decision (2026-05-30) — chronological split kept (for the writeup)
Confirmed with Risheek: keep the **strict chronological** train/val/test split (train oldest 70% ≈1970→2007, val ≈2007→2016, test newest 15% ≈2016→2023), as used by exp_12/13/14. **Verified in code**: all GNN experiments call the array `temporal_split` (single past→future cut); the only shuffle is `shuffle=True` on the *train* DataLoader (window order within an epoch — never crosses splits). Considered and **rejected** an all-timelines / random-interleaved split: with ~0.56 lag-1 autocorrelation + 7-day sliding windows it would leak near-duplicate windows (6/7 shared days) across train/test and inflate RMSE — a classic time-series leakage trap. The chronological cut is the correct forecasting setup, keeps comparability with exp_12/13/14, and the test window already spans ~8 years/many monsoons. Regime-robustness, if ever needed, is to be addressed by per-year/season test breakdowns or blocked/embargoed held-out-years CV — **not** random splitting. (Risheek chose "keep as-is"; per-year breakdown + blocked-year CV remain optional.)

### Session 1 addition (2026-05-30) — GAT depth ablation + oversmoothing check
Risheek asked to add a **2-layer GAT** variant to exp_16 (3-layer later if time), plus a post-training check that node embeddings stay distinct (oversmoothing) — flagged as reusable for Tier 2A.

**Grounding (researched, not assumed).** Measured the actual 100 km / 291-node graph: 1 hop reaches only **21%** of nodes (avg degree 62), 2 hops 47%, diameter ~7 — so 1→2 layers genuinely widens spatial context. Literature: GNN optimum is **2–4 layers (2–3 best)**, degrades beyond (oversmoothing); 2 layers is the canonical STGNN depth. Oversmoothing metrics = **Dirichlet energy** + **MAD** (cosine distance), standard/citable, with the honest caveat (Rusch survey; arXiv 2502.04591) that they can lag actual RMSE degradation → diagnostic, not verdict.

**Built (NEW files; nothing existing modified):**
- `models/gat_gru_multilayer.py` — `GAT_GRU_MultiLayer(in_channels, hidden_dim, heads, num_layers, residual)`. Stacks `num_layers` GATConv (ReLU between; optional residual on hidden layers), then the same GRU + linear head. **`num_layers=1` is architecturally identical to `GAT_GRU_Model`** → tier0 IS the 1-layer point; depth2 adds exactly one GAT layer. Has a `node_embeddings()` convenience.
- `utils/metric_utils/embedding_diag.py` — reusable oversmoothing diagnostic: `dirichlet_energy` (on unit-normalised embeddings), `mad` (mean pairwise cosine distance), `effective_rank`, `embedding_report`, and `last_gat_embeddings(model, x, edge_index, device)` — a forward-hook extractor that pulls the **last GATConv** output at the final timestep for *any* GAT model here (original, multilayer, or the future Tier 2A encoder) **without modifying it**. Self-test PASSES (collapsed → energy/MAD 0, rank 1; diverse → energy ≈2, MAD ≈1, rank ≈full).
- exp_16 regenerated: now 3 configs — `tier0` (1-layer 12f), `tier0b` (1-layer 15f), **`depth2` (2-layer 12f)** — each saving `embedding_diag.csv` (Dirichlet/MAD/eff-rank averaged over a test batch) alongside the metrics. `depth3` (residual on) is a one-line uncomment if time permits. nbformat-valid; all 11 code cells compile.

**Reasoning on "specific checks for 2 layers".** For depth=2 the only real risks are (a) oversmoothing → the `embedding_diag` check covers it directly (compare depth2 vs tier0: a large drop in Dirichlet/MAD/eff-rank = the 2nd layer collapsed embeddings), and (b) extra-parameter overfitting → the train-val gap (already reported). No additional depth-2-specific check needed. Special handling (residual, normalisation) only becomes relevant at depth≥3 — handle there.

**For Tier 2A ("plan 2"):** reuse `last_gat_embeddings` + `embedding_report` on the pretrained encoder — a collapsed encoder can't reconstruct distinct per-station values, so this is a health check on the pretext.

NOT run by me (needs GPU/torch_geometric; Risheek runs). py_compile + nbformat.validate clean; the depth ablation is single-seed, anchored to tier0.

---

## Session 1 RUNNING + Session 2 built: masked-recon pretraining (2026-05-30, Claude)

### Status
- **Session 1 (exp_16) is RUNNING** on Risheek's machine (~4.5–5 hr; 3 configs × train + 3 evaluate passes + diagnostics). **Outputs pending** — Risheek will paste tier0/tier0b/depth2 RMSE, gaps, extreme + embedding diagnostics next session.
- **Session 2 (Tier 2A Phase A, masked-recon pretraining) is BUILT, NOT yet run.** No runtime conflict (exp_16 writes `results/exp_16…`; this writes `results/exp_18…`; both only read the shared pivots/elevation).

### What was built (NEW files; nothing existing modified)
- `models/gat_gru_pretrain.py` — `GAT_GRU_Pretrain(in_channels, hidden_dim, heads, n_recon)`. **Encoder (`self.gat` + `self.gru`) is byte-for-byte the same definition as `GAT_GRU_Model`** (GATConv(15,64,heads=4,concat) + GRU(256,64)), so Session 3 transfers weights with a 1:1 `load_state_dict` of the `gat`/`gru` submodules. Head = the full GRU sequence `(B*N, L, 64)` → 2-layer MLP decoder → `(B, L, N, 6)` reconstructing the 6 physical channels at every (timestep, node). Decoder is discarded at finetune.
- `experiments/experimentation_notebooks/exp_18_gat_gru_mae_pretrain.ipynb` — the pretext run: builds the 15-feat tensor (same recipe + scaler/split fix as exp_16), trains the masked-recon task on the **train+val periods only**, saves `pretext_best.pt`, the loss curve (csv+png), and a node-embedding health check on the pretrained encoder.

### Design decisions (researched, not assumed)
- **BERT/denoising-style masking, NOT ViT-MAE token-drop.** We zero ~15% of cells but still feed the whole window through the GAT-GRU and reconstruct the masked cells. Web-checked: the 75% ratio quoted for MAE/TSFormer applies only when masked tokens are physically *dropped* and the encoder sees a subset; for value-corruption masking where all positions are processed, the right regime is 15–25% (BERT=15%). So `mask_ratio=0.15` is correct here, not 75%. Configurable.
- **Mask only the 6 physical channels** (rain + temp/dew/pressure/u10/v10). Cyclic time (6) + static lat/lon/elev (3) stay **visible and are never reconstruction targets** → anti-shortcut (calendar/geography are context, not answers) and keeps `in_channels=15` fixed across pretext↔finetune (clean weight transfer, no input-projection surgery — session_plan §2.4).
- **Loss on masked cells only** (standard MAE/BERT convention). Data is z-scored, so the zero we write for a masked cell = the neutral/mean value (a sensible mask token).
- **lr 3e-4** (plan/source pretext convention; gentler than the 1e-3 finetune lr). **Full passes over the ~train windows each epoch** (≈ len/batch steps), NOT the "800 steps/epoch" from the source domain — that figure came from random-pair sampling on traffic; we have a fixed finite window set, so an epoch is one pass. 30 epochs.
- **batch 64** default (safe for 8GB; exp_13 used 32 on this graph). Documented OOM fallback to 32 and the option to try 128. NaN-guard stops the run and tells the user to lower lr/batch/mask_ratio.
- **Val-based checkpointing:** val recon loss uses *fixed* masks each epoch (seeded generator) so it is comparable across epochs; `pretext_best.pt` saved on best val recon MSE. This is also the **stability gate** (does val recon decrease + flatten, no NaN?).
- **Encoder health check (the reusable bit Risheek asked for):** after pretraining, run `embedding_diag` on the encoder's node embeddings → `pretext_embedding_diag.csv`. A collapsed encoder (Dirichlet/MAD→0, rank→1) can't reconstruct distinct stations, so this flags a degenerate pretext before we waste Session 3 on it.

### Verification (done, not assumed)
- `py_compile` clean on `gat_gru_pretrain.py`; nbformat.validate + all 9 code cells compile. Tensor-shape reasoning checked end-to-end (rain stays feature 0; rec `(B,L,N,6)` aligns with `phys`/`mask`; GRU-output→decoder reshape verified). NOT run (needs GPU/torch_geometric; Risheek runs).

### What the next session needs
1. Risheek pastes **Session 1 (exp_16) outputs** + says whether **Session 2 (exp_18) ran bug-free** (and its loss curve / embedding diag if so).
2. Then build **Session 3** (finetune): copy `pretext_best.pt` `gat`+`gru` into `GAT_GRU_Model(in_channels=15)`, fresh rain head, finetune on next-day rain, compare to Tier 0b on RMSE + train-val gap (+ private extreme/embedding diagnostics).

### Files
- NEW: `models/gat_gru_pretrain.py`, `experiments/experimentation_notebooks/exp_18_gat_gru_mae_pretrain.ipynb`.
- NOT modified: any existing notebook/model/util/data.

### Kaggle-portability addition (2026-05-30)
exp_18 was regenerated to run on **both local and Kaggle** unchanged: a bootstrap cell clones the repo + sets `sys.path` on Kaggle (no-op locally), and paths now come from NEW `utils/env_paths.py::get_paths()` (local → `repo/data` + `repo/experiments`; Kaggle → `/kaggle/input/<slug>` + `/kaggle/working`). So Session 2 can run on a free Kaggle GPU in parallel with Session 1 locally. Full procedure + the git-cleanup/push steps + results-retrieval (via Kaggle, not git) are in `docs/kaggle_workflow.md`. nbformat-valid; all code cells compile.

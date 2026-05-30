# Handover — Session-to-Session Continuity

_This is the file the next session should read FIRST after `status.md`. It carries the freshest context: what just happened, what's blocked, and what to do next. Append a new section per session at the top; keep prior sessions below for backreference._

---

## Tomorrow's combined daily trajectory (Items 1 + 2)

> **UPDATE 2026-05-29 — Items 1A and 1B are DONE** (see the dated entry below and `plan.md` table rows 14/14b/14c). What remains of this trajectory is **Track 2** (regional ablation) and **Item 3** (RL reuse). The Track-2 GEE-fetch / cleaning / model-run steps below are still valid.

Items 1 and 2 are budgeted to one day. Detailed step-by-step plans live in each track's `README.md` under "Session Execution Plan". This is the orchestration:

1. **09:00** — Kick off Track 2 Phase 1 (GEE fetch for Northern Plains AND Himalayan, parallel). Background.
2. **09:30** — Start **Track 1A** (IMD minimap GAT-GRU). See `docs/track_1a_imd_minimap/README.md` → "Session Execution Plan". Budget 2–3 hr.
3. **10:30** — Start **Track 1B** (IMD vs ERA5 correlation) in parallel with 1A's training. See `docs/track_1b_imd_era5_corr/README.md`. Budget 1.5–2 hr.
4. **12:00** — Track 1A + 1B should both be done. GEE fetches likely complete by now. Start Track 2 Phase 2 (cleaning + pivoting + EDA per region). Budget 1.5 hr.
5. **13:30** — Start Track 2 Phase 3 (model runs per region). Budget 30 min total compute.
6. **15:30** — Comparison table fill-in + interpretation paragraph + handover update.

**Order discipline:** do not interleave tracks within a single execution step. Each track's steps are checkpointed — finish one step's validation gate before moving to the next.

**If GEE fetch is delayed past 12:00**: park Track 2 Phase 2/3 and finish Track 1 work first; Track 2 can slide to the day after if needed.

---

## Open clarifications waiting on Risheek

These accumulate across sessions. Risheek will answer when ready — do not block on them unless explicitly required for the next step.

**✅ Resolved this session (2026-05-29, Item 1) — the bullets below marked here are now answered:**
- Scaler train_end → exp_14 uses the actual 70%-split date (2014-09-12), not 2015-12-31. Apply the same in Track 2.
- IMD station count → **16** (top-16 lowest-missing within 2000–2020).
- Track 1A graph fallback → **150 km** gives avg degree 4.62, 0 isolated nodes; no fully-connected fallback needed.
- Include pre-1970 IMD data → **No**; window-first 2000–2020 was adopted instead (avoids block-missingness, keeps a real test window).
- ERA5 291 vs IMD 293 → **confirmed**: SAGAR ISLAND and SANDHEADS are the two stations absent from ERA5.
- GAT-GRU overfitting/dropout → exp_14 kept exp_13 config + reported the best-val checkpoint (correct control for the matched comparison); dropout/weight-decay deferred (only `era5_full` would benefit — IMD underfits, not overfits).

**Still open:** Track 2 bounding boxes, Himalayan elevation mask, Track 2 stations/region, `build_edge_index(k=5)` unused, sequence length 7 vs 14–30.

- **Scaler train_end intent:** was `scale_pivots(train_end='2015-12-31')` intentional, or should it align with the actual 70%-split date (~2007-10-19)? Recommendation in `status.md` known issue #2 is to align for new runs.
- **Preferred IMD station count for Track 1A minimap:** 12, 16, or 20? See `docs/track_1a_imd_minimap.md` open questions.
- **Track 1A graph fallback:** if even a 200 km radius produces a sparse graph (<1.5 avg degree), fall back to fully connected, or stop and report "not feasible"?
- **Include pre-1970 IMD data?** Improves missingness but loses ERA5 comparability.
- **Track 2 bounding boxes:** are Northern Plains (lat 24–31, lon 75–88) and Himalayan (lat 28–34, lon 75–95) acceptable, or do you want specific sub-regions?
- **Himalayan elevation mask:** apply (cleaner, +0.5 day work) or skip (simpler, accept noisier domain)?
- **Stations per region in Track 2:** ~100 is the current target; bump to ~200 for closer parity with WB's 291?
- **`gnn_dataset.build_edge_index(k=5)` is unused** — delete it, or repurpose it for a planned k-NN comparison?
- **Sequence length:** is 7 the right window? Pre-monsoon buildup might suggest 14–30. Currently a held opinion, not a tested one.
- **GAT-GRU overfitting (train ↘ val flat after epoch ~20):** acceptable, or add dropout=0.1 + weight-decay before the next phase?
- **Why does ERA5 have 291 stations vs IMD's 293?** Likely the 2 dropped because of missing coords (`data/missing_station_coordinates.csv`). Confirm.

---

## 2026-05-30 — track_c re-scoped + reframed; Session 0 done, Session 1 RUNNING, Session 2 built (Claude)

### Goal (overall)
track_c = the RL-reuse / SSL track (`docs/track_3_rl_reuse/`). Now an **upside chapter**, not the thesis. **Dissertation reframed**: primary thesis = comparative spatio-temporal models + IMD-vs-ERA5 data-source study (the "bunker", built from completed work). **Extremes demoted to a measured limitation** — the models under-disperse (preds cap ~70 mm), so extreme metrics are a **private diagnostic**, never the headline. Full reasoning + the multi-session plan: **`docs/track_3_rl_reuse/session_plan.md`** (read §0 first).

### Current status
- **Session 0 DONE.** Built + unit-tested `utils/metric_utils/extreme_skill.py` (POD/FAR/CSI/HSS/freq-bias at IMD 35.6/64.5/124.5 mm). Built `experiments/experimentation_notebooks/fetch_srtm_elevation.py`; Risheek ran it → `data/wb_station_elevation.csv` (293/293, web-verified sane).
- **Session 1 (exp_16) RUNNING (~4.5–5 hr; outputs pending next session).** `experiments/experimentation_notebooks/exp_16_gat_gru_baseline_seeds.ipynb` — **3 configs**: Tier 0 (1-layer 12-feat), Tier 0b (1-layer 15-feat +lat/lon/elev), and **depth2 (2-layer GAT, 12-feat)** — single seed 42, scaler/split fix + best-val-ckpt fix, saving full metrics + raw preds + private extreme diagnostics + an **oversmoothing/embedding-diversity check** per run. nbformat-valid, code cells compile.
- **NEW (this session):** `models/gat_gru_multilayer.py` (depth-configurable GAT-GRU; num_layers=1 == original) and `utils/metric_utils/embedding_diag.py` (Dirichlet energy / MAD / effective-rank + a `last_gat_embeddings` forward-hook extractor; self-test passes). The embedding check is **reusable for Tier 2A** (verify the pretrained encoder isn't collapsed). Depth-3 is a one-line uncomment in exp_16's RUNS if time permits.
- **Session 2 (Tier 2A Phase A, masked-recon pretraining) RUNNING on Kaggle (started 2026-05-30; epoch 1 train 0.390 / val 0.117 recon-MSE, ~159 s/epoch, ~80 min ETA; GPU ~96%, healthy). Internet/log drops mid-run are cosmetic (kernel keeps computing).** NEW `models/gat_gru_pretrain.py` (encoder identical to GAT_GRU_Model for 1:1 weight transfer + MLP decoder) and `experiments/experimentation_notebooks/exp_18_gat_gru_mae_pretrain.ipynb` (BERT-style 15% masking on the 6 physical channels only; cyclic/static stay visible; loss on masked cells; lr 3e-4; batch 64; single seed; saves `pretext_best.pt` + loss curve + encoder embedding health check). No conflict with the running exp_16 (different result dirs).
- **Kaggle workflow SET UP (2026-05-30).** NEW `utils/env_paths.py` (resolves local vs Kaggle paths), `.gitignore` rewritten (keeps `data/`; now also excludes results/saved_models/logs/caches/secrets), `docs/kaggle_workflow.md` (full step-by-step). **exp_18 is now env-portable** (bootstrap cell clones repo on Kaggle + `get_paths()`); runs unchanged local AND Kaggle, so Session 2 can run on a free Kaggle GPU while Session 1 runs locally. Results come back via Kaggle Output download / `kaggle kernels output` (NOT git — results are gitignored).
- **Next session:** Risheek pastes Session 1 (exp_16) outputs + says whether Session 2 (exp_18) ran bug-free; then build Session 3 (finetune: transfer `pretext_best.pt` gat+gru → fresh rain head → finetune → compare to Tier 0b).

### Important context
- Plan + session breakdown: `docs/track_3_rl_reuse/session_plan.md`. Running log: `docs/track_3_rl_reuse/implementation_notes.md`.
- **Seed policy (HARD):** single seed (42) for ALL experiments now; multi-seed is the FINAL step (Session 5), parallel, winners-only.
- **Greedy full-coverage:** backbone Tier 0→0b→2A fixed; pool (2B joint, 1 multi-task, 2C V_DIST doc, 3 three-stage) slotted by value×cost×parallelism; reroute gates in §7.

### Decisions made
- Retitle away from "extreme rainfall events" (Risheek). Comparative-ST + data-source is the spine.
- Tier 0b (static lat/lon/elev) added back into the backbone; all SSL/multitask tiers use 15-feat anchored to Tier 0b.

### Pending (needs Risheek's OK / action — not done unprompted)
- Propagate the retitle (§0) into `plan.md` + the `status.md` Goal line (dissertation-wide change). Reflect the seed-policy override in the track_3 `README.md` (bump v5→v6, Codex-loop doc).
- **Git cleanup + push (Risheek runs):** `experiments/saved_models/` (284 MB) + `experiments/results/` are still tracked in git. Run the `git rm -r --cached --ignore-unmatch …` + `git add -A` + commit + push from `docs/kaggle_workflow.md` §1 to untrack them and push the new track_c code. I did NOT run git mutations (your repo + credentials). Then upload the data subset as a Kaggle dataset (workflow §2) and set `KAGGLE_DATA_SLUG` in exp_18.

---

## 2026-05-29 — Item 1 DONE: exp_14 (Track 1A) + EDA_02 (Track 1B) built, run, documented (Claude)

### Goal (overall)
Predict daily West Bengal rainfall with a spatio-temporal GNN. Item 1 specifically answers the guide's ask: does the GAT-GRU (best on ERA5, exp_13 RMSE 8.79) work on **real IMD gauge data**, and how much do IMD and ERA5 actually differ?

### What was done
- **Track 1A (exp_14):** built and ran (Risheek ran it on the RTX 4060) a matched 3-run experiment — GAT-GRU on **IMD rain-only**, **ERA5 rain-only (same 16 stations/window)**, **ERA5 12-feature** — plus persistence + climatology baselines. **Reframed from the README's "1970+ then missingness cut" to window-first (2000–2020, top-16 lowest-missing)** after verifying 1970+ IMD missingness is block-structured (whole-year gaps; 2021 empty). Results (real-space test RMSE, mm): IMD 17.60 / ERA5 rain 10.42 / ERA5 full 10.03; baselines 22.53 / 18.50.
- **Track 1B (EDA_02):** built + ran the IMD-vs-ERA5 correlation EDA across 291 stations (199 with ≥1 yr overlap); 5 figures + `imd_vs_era5_summary.csv` + a 7-page explainer docx (methodology + how-to-read + viva Q&A). Medians: r 0.37 / 0.29 / 0.24, bias ≈ 0, wet-day IMD 30% vs ERA5 51%.
- Added analysis figures to exp_14 (`exp_14_analysis_plots.ipynb`): partial-network coverage map + ERA5-vs-IMD characterization (autocorr 0.31 vs 0.56, drizzle, truncated extremes).
- Wrote `implementation_notes.md` for both tracks; updated `plan.md` comparison table + `status.md` (change log + depth map).

### Current status
- **Item 1: COMPLETE** (1A + 1B). Numbers in `plan.md` rows 14/14b/14c.
- Item 2 (regional ablation, NP + Himalayan): not started. See `docs/track_2_regional_ablation/README.md`.
- Item 3 (RL reuse): planned (track 3 README v5); not started.
- Item 4 (writeup): not started.

### Important context (files / gotchas)
- IMD source = `data/preprocessed_rain_data.parquet` (3.96M rows, 0 NaN). **Not** `processed_rain.parquet`. The README's "4.74M rows" label is wrong (that figure is `processed_rain`).
- Notebooks: `experiments/experimentation_notebooks/exp_14_gat_gru_imd_minimap.ipynb`, `…/exp_14_analysis_plots.ipynb`, `EDA/EDA_02_imd_vs_era5_correlation.ipynb`.
- Results: `experiments/results/exp_14_gat_gru_imd_minimap/`; `data/eda_outputs/imd_vs_era5_*`.
- Docs: `docs/track_1a_imd_minimap/implementation_notes.md`, `docs/track_1b_imd_era5_corr/implementation_notes.md`, `docs/track_1b_imd_era5_corr/EDA_02_explained.docx`.
- **Plotting gotcha:** analysis notebooks need `%matplotlib inline` + `plt.show()` to render in-cell (fixed). Jupyter holds files open, so edits made on disk while a notebook is open get overwritten on save — edit in the live session or "Reload from disk".

### Decisions made
- Window-first 2000–2020; 16 stations; 150 km graph; matched rain-only + 12-feat ERA5; exp_13 config kept; **compare WITHIN exp_14, not against exp_13's 8.79** (different N/features/window — confounded).
- Locked story: ERA5 matches climatology but is a smoothed proxy at daily scale (drizzle + clipped extremes), worst over complex terrain — explains the IMD-vs-ERA5 gap; corroborated by ERA5-over-India literature. **Honest framing only** (Risheek raised number-manipulation under deadline pressure; we kept all reported numbers true and instead built the partial-network + correlation evidence that makes the honest result defensible).

### What the next session needs
1. Optional Item-1 robustness adds (see track_1a implementation_notes "Open/optional"): log1p target variant, occurrence-vs-amount split, in-notebook skill scores, light reg for `era5_full` only.
2. Item 2: mirror the exp_14 fixes (scaler train_end alignment, best-val checkpoint). Resolve the still-open Track-2 clarifications above.
3. Item 3: track 3 README v5.
4. Working rules still apply: existing code/notebooks read-only by default; add new files only; Risheek runs experiments himself and approves changes.

---

## 2026-05-29 — Two Opus research passes applied; Track 3 README at v5 (Claude)

### What changed in this session

- Two informal Opus 4 research passes (chat-context only, not version-stamped) plus Claude's independent verification flipped the v4 plan structure.
- **Most important verifications by Claude:**
  - HyMePre's "2048 stations" claim verified by fetching the actual paper. Verbatim quote: *"The 2048 stations originate from a uniformly discretized global grid (64 longitude bins × 32 latitude bins), where adjacent indices correspond to geographically proximate locations (approximately 5.625° separation per index step)."* HyMePre is NOT a station-level SSL precedent.
  - STD-MAE and STEP topology independently verified via web search. Both unambiguously pretrain-then-finetune.
  - STGCL existence and "joint > pretrain" framework confirmed (still). Mechanism reasoning: contrastive-specific because contrastive pretext misaligns with regression downstream; masked-recon pretext is itself a regression task so it aligns.

### The single most important v4 → v5 change

**Tier 2 fully refactored.** v4 had V_DIST (Tier 2A) + masked-recon (Tier 2B) as peer primaries. v5:
- **Tier 2A (PRIMARY) = masked-recon pretrain-then-finetune** — STEP/STD-MAE/GPT-ST precedent on irregular sensor graphs, all beat SOTA on traffic. GPT-ST mechanistically explains why: "higher correlation between mask-reconstruction task and downstream regression task."
- **Tier 2B (NOVEL COMPARISON) = masked-recon joint forecast + recon loss in single stage.** No published weather precedent — this IS the dissertation's empirical contribution.
- **Tier 2C (DEMOTED) = V_DIST as 35-min preprobe only.** No full pretraining run. Single seed. Negative result documented in writeup as "faithful TDC transfer is signal-bounded in our domain."

### Seed policy locked

- **Tier 0 — 3 seeds always** (non-negotiable anchor for variance estimation).
- **Everything else — 1 seed initially.** Run cheapest version first.
- **Conditional re-run with 3 seeds** if a tier beats its comparison anchor by ≥0.2 RMSE OR shrinks train-val gap by ≥0.3 RMSE.
- **Pretext stages always single-seed** (compute is concentrated in pretext; multi-seed pretext rarely justifies compute).

### Total Item 3 compute estimate

- Worst case: ~5–6 hr if most tiers don't improve and stay at 1 seed.
- Best case: ~8–10 hr if many tiers improve and trigger 3-seed re-runs.
- Both fit comfortably in remaining ~3 day budget.

### Task list changes

- Task #19 retitled: V_DIST → Masked-recon PRETRAIN-THEN-FINETUNE (PRIMARY v5).
- Task #20 retitled: Path B peer primary → Masked-recon JOINT (NOVEL COMPARISON v5).
- Task #34 added: V_DIST documentation experiment (preprobe only).
- Task #33 (v5 application) — completed by this session.

### Codex pass 4 (when to trigger)

After first implementation results from Tier 0 + Tier 2A start landing in `docs/track_3_rl_reuse/implementation_notes.md`. v5 contract: Codex must echo `track3-readme-2026-05-29-v5` as the first line. Before pass 4, the most important sanity check is whether Tier 2A's masked-recon pretrain actually trains on our data without instability — STEP/STD-MAE/GPT-ST were on traffic, not weather, so first-results may surprise us.

### Files modified

- `docs/track_3_rl_reuse/README.md` — v4 → v5 with Tier 2 fully refactored, seed policy locked, Opus pass history added.
- `docs/references.md` — HyMePre removed; STEP/STD-MAE/GPT-ST/W-MAE/SpaT-SparK added; references reorganized by SSL family.
- `status.md` — change-log entry.
- `handover.md` — this entry.
- Task list — #19, #20 retitled; #34 added.

### Files NOT modified

- All notebooks, models, utils, data remain untouched.
- `plan.md` not touched (item map unchanged at master level).
- No code edited in `minor_project_rl` (read-only).

---

## 2026-05-29 — Codex pass 3 (informal) applied; Track 3 README at v4 (Claude)

### What changed in this session

- Codex pass 3 reviewed Claude's chat clarification on Path A sampling structure (informal — Risheek pasted Claude's chat into Codex, not the version-stamped README). Codex flagged compute-waste in Strategy A and identified two stale mean-pool remnants in v3 README that contradicted the selected-node fix.
- All four technical claims verified by Claude against ground truth:
  - 293 stations in `data/wb_station_coords.csv` — confirmed exactly.
  - 100km graph avg degree 62.2 — Codex said 62, confirmed via direct haversine computation.
  - Batch-256 expected unique stations 170.4 — Codex said 170, confirmed via sampling-with-replacement formula.
  - VRAM ~1.4 GB activations for two windows at batch-16 — fits 8 GB with comfortable headroom for gradients + Adam state.
- v4 changes applied:
  - **Strategy C-subset becomes V_DIST default**: K=16 date-pairs per batch × M=32 target stations per pair. 512 loss terms over 32 graph forwards. ~16× more gradient signal per compute unit than Strategy A. Uniform M-station sampling for v4; balanced sampling deferred to implementation-notes refinement.
  - **Strategy A retained as a 5-epoch preprobe probe** (Stage B bake-off vs Strategy C-subset). Winner of the bake-off becomes the full 1.5 hr pretext run; loser is documented and discarded.
  - **Strategy B dropped entirely**. Reasoned-out as over-constraining: forcing every station on the same calendar gap to satisfy the same lag-distance is physically unrealistic given different climatological regimes evolve at different rates.
  - **Two stale mean-pool remnants cleaned** in the open-questions list and the corrections table. v3 had the selected-node fix in the Tier 2A spec but had left the older mean-pool language elsewhere — a real internal inconsistency that Codex caught.
  - **Preprobe Stage B re-scoped** as a Strategy A vs Strategy C-subset bake-off, 5 epochs each, winner runs the full Tier 2A pretext.
- README version: `track3-readme-2026-05-29-v3` → `track3-readme-2026-05-29-v4`.

### What's now locked into Path A

- Per pair: pick K=16 date-pairs `{(i_k, j_k)}` with 1 ≤ |i_k − j_k| ≤ 14 (uniform over lag).
- Per date-pair: pick M=32 target stations uniformly from 291.
- Full graph window `(7, 291, F)` forward through GAT + GRU for both windows of each date-pair (so 32 graph forwards per batch).
- For each (target station, date-pair), read out that station's final GRU hidden state, push through Linear(64→128) projection head, compute SmoothL1(α·||e_i − e_j|| + β, log1p(|i − j|)).
- 512 loss terms per batch, averaged.
- 20 epochs, 800 steps per epoch, Adam lr=3e-4.
- Anomaly normalization on aux variables; raw rain; cyclic time features excluded; imputation gap mask rejects pairs with >10% imputed in the lag segment.

### Task list changes

- Task #19 retitled to reflect Strategy C-subset sampling.
- Task #30 added for the v4 application work.

### Codex pass 4 (when to trigger)

After implementation_notes start landing with first results. v4 contract: Codex must echo `track3-readme-2026-05-29-v4` as the first line of its response, or the review is treated as stale.

### Files modified

- `docs/track_3_rl_reuse/README.md` — v3 → v4 with Strategy C-subset, preprobe bake-off, two stale-remnant cleanups.
- `status.md` — change-log entry.
- `handover.md` — this entry.
- Task list — #19 retitled, #30 added.

### Files NOT modified

- All notebooks, models, utils, data remain untouched.
- `plan.md` not touched (item map unchanged).
- No code edited in `minor_project_rl` (read-only).

---

## 2026-05-29 — Codex pass 2 (informal) applied; Track 3 README at v3 (Claude)

### What changed in this session

- Codex pass 2 was an informal review (Risheek pasted Claude's chat reasoning into Codex, not the version-stamped README). Per the review-loop policy, this isn't a formal pass — but the empirical and methodological points were verifiable, so Claude verified each before applying.
- **Most important verification:** the V_DIST lag-distance signal-strength claim. Codex reported 3-station Spearman numbers. Claude reproduced at 30 stations: mean ρ ≈ 0.41 at lag 7 (anomaly), 0.33 at lag 14, collapsing to 0.13 at lag 90 — matching Codex's 3-station numbers within 0.03. Std across stations is ~0.03–0.05, so the pattern is robust.
- Two literature claims (Weather2K time-invariant station constants; HiSTGNN) verified via web search.
- v3 changes applied (all Codex pushbacks addressed):
  - **Feature-treatment lock**: anomaly normalization on aux variables in ALL Item 3 tiers. Rain stays raw. Cyclic time excluded from V_DIST pretext input.
  - **Tier 0b** = separate static-feature baseline (lat/lon/elev). Tier 0 stays 12-feature. Matched comparison rule: SSL methods using static features compare to Tier 0b; others compare to Tier 0.
  - **Tier 1** Path C aux head targets are now next-day **anomalies** (matches input distribution).
  - **Tier 2A V_DIST**: max lag capped at 14 days (empirically justified by the 30-station probe baked into the README). Embedding = selected station's node final GRU hidden state, NOT mean-pool — fixes the v2 internal contradiction.
  - **Tier 2B Path B (masked reconstruction)** promoted from "Tier 3 fallback" to peer primary alongside Tier 2A.
  - **Preprobe expanded to two stages**: (a) all-lag documentation probe (5 min) — produces the failure-case evidence; (b) 14-day capped feasibility probe (30 min) with frozen Tier 0 encoder — pass gate is loss drops 30% AND ρ > 0.25.
  - **Dissertation narrative reframed** as "faithful-RL-transfer vs domain-adapted alternative" — three outcome narratives prepared, all defensible.
- README version: `track3-readme-2026-05-29-v2` → `track3-readme-2026-05-29-v3`.
- Risheek confirmed two key decisions: (a) losing direct comparability to exp_13's 8.79 is OK (the goal is improvement over the current methodology, not matching the historical number); (b) SRTM elevation fetch via GEE is OK.

### Task list changes

- Task #19 retitled: Tier 2 → Tier 2A (V_DIST short-lag).
- Task #20 retitled: Tier 3 backup → Tier 2B (peer primary).
- New task #27: Tier 0b static-feature baseline.
- New task #28: preprobe + Spearman documentation script (`docs/track_3_rl_reuse/probe_lag_distance.py` to be written when Tier 2 begins).

### What's locked into the plan now

- v3 README is the source of truth.
- The 30-station Spearman probe is part of the README's empirical justification — future sessions reading the README see the exact numbers without re-running.
- The preprobe is a hard gate before Tier 2A's full pretraining runs. If it fails, skip Tier 2A and put all SSL compute into Tier 2B.

### Codex pass 3 (when to trigger)

After implementation_notes.md starts landing with first results — i.e., after Tier 0 and Tier 1 produce real numbers. At that point the review shifts from "is the plan sound?" to "is the implementation faithful to the plan?" Use the Codex prompt at the bottom of the v3 README (version stamp `track3-readme-2026-05-29-v3`).

### Files modified

- `docs/track_3_rl_reuse/README.md` — v2 → v3 with feature lock, Tier 0b add, Tier 2A constraint, Tier 2B promotion, Spearman probe table.
- `status.md` — change-log entry.
- `handover.md` — this entry.
- Task list — #19, #20 retitled; #27, #28 added.

### Files NOT modified

- All notebooks, models, utils, data remain untouched.
- `plan.md` not touched (item map unchanged at the master level).
- No code edited in `minor_project_rl` (read-only).

---

## 2026-05-29 — Codex pass 1 applied; Track 3 README at v2 (Claude)

### What changed in this session

- Codex's review of `track3-readme-2026-05-29-v1` was pasted into chat. First line echoed correctly (`REVIEWED VERSION: track3-readme-2026-05-29-v1`), so the review applied to the right doc.
- For each Codex flag, Claude opened the cited source file in `minor_project_rl/rnd_baseline/track_c/` and verified the reading against actual lines. **All 12 corrections-table flags confirmed by source. No flag was rejected.**
- Track 3 README bumped to **`track3-readme-2026-05-29-v2`**. This is now the current ground-truth version Codex must echo on any future pass.

### The single most important correction

The canonical s13_run7 recipe is **distance-only SmoothL1 on log1p(lag), 20 epochs**, deployed via `experiments/distance_loss/train_distance.py`. Not ordinal CE + 40 epochs as v1 claimed. Confirmed via `runs/encoder/s13_run7/metadata.json`:
- `"script": "experiments/distance_loss/train_distance.py"`
- `"loss": "smooth_l1_log1p_lag"`
- `"epochs": 20`
- `"ordinal": false`

V1 (BCE) and V2 (ordinal CE) are predecessor recipes that the source EVOLVED PAST after the `probe_projfit` experiment showed the conv trunk already had temporal information. They are now ablation-only in our plan.

### Other corrections applied

- Augmentation analogs: ±1 day window-jitter REMOVED (it would corrupt the supervisory lag label); rain-noise + optional cutout retained.
- "Rainy-day balance" is NOT source-equivalent to NOOP balance — relabeled as a domain bridge with wet/dry stratification suggestion.
- Three new guardrails: imputation-gap masking (ffill→bfill→0 creates fake-close pairs), seasonality leak via temp/dew/pressure (default mitigation: day-of-year anomaly normalization), preprobe (30-min projection-head-only training to test if the Tier 0 encoder already has the lag signal).
- Finetune freeze framing: clarified that the source CLI default is frozen but the deployed winning recipe used `--no-freeze-encoder`. Our default mirrors the deployed recipe.

### Codex pass 2 (when to trigger)

Codex pass 1 was a deep verification of the source mapping. Pass 2 would be useful **after the implementation notes start landing in `docs/track_3_rl_reuse/implementation_notes.md`** with first results. At that point the questions to Codex shift from "is the source reading right?" to "is the implementation faithful to the spec we agreed on?"

Procedure for any future pass:

1. Re-send the **current v2 README** (plus any new implementation_notes.md content) using the Codex prompt at the bottom of the README. The prompt forces Codex to echo the current version stamp.
2. If Codex returns a version that doesn't match v2 (e.g., it accidentally references v1), the review is stale. Re-send.
3. Verify each Codex flag against source before editing. Cite the source line in the change log.
4. Bump to v3 after applying.

### Files modified

- `docs/track_3_rl_reuse/README.md` — multiple edits applied; version bumped v1 → v2. Major restructure of Tier 2 (V_DIST primary). New guardrails. Augmentation analogs corrected.
- `status.md` — change-log entry.
- `handover.md` — this entry.

### Files NOT modified

- All notebooks, models, utils, data remain untouched.
- `plan.md` not touched (item map unchanged; the Tier 2 recipe details are internal to track 3).
- No code edited in `minor_project_rl` (read-only access).

---

## 2026-05-29 — Review-loop workflow locked (Claude)

### What changed in this session

- Track 3 README now carries a **version stamp** at the top:
  `**REVIEW VERSION: track3-readme-2026-05-29-v1**`
- The Codex prompt inside the README was hardened: Codex must echo the version as the first line of its response, must explicitly not write to files, and must use a structured output format Claude can mechanically parse.
- The review-loop workflow is documented in `docs/track_usages.md` § "Review-loop with Codex (or any external reviewer)". It applies to ANY future external review on ANY track doc — same shape, same rules.

### Current track-doc versions (the "ground truth" Claude expects to see echoed)

- `docs/track_3_rl_reuse/README.md` → **track3-readme-2026-05-29-v1**
- All other track READMEs do not yet carry stamps because no external review is queued on them. If Risheek decides to send Tracks 1A / 1B / 2 to Codex, the first thing the next session does is add a `track<id>-readme-<date>-v1` stamp + the boilerplate prompt at the bottom.

### When Codex's response arrives

The next session must follow this exactly (also written into `docs/track_usages.md` and at the bottom of the Track 3 README):

1. Check first line: `REVIEWED VERSION: <id>`. If not matching the current README version, stop and ask Risheek to re-send Codex the current file. Do NOT cherry-pick from a stale review.
2. For every Codex flag, **re-read the cited source lines** in `minor_project_rl/rnd_baseline/track_c/` before editing. If Codex's reading does not match what the source actually says, reject the flag.
3. Edit the README via Edit tool (never Write — preserves diffs). Each edit's session note should cite both the Codex flag and the verifying source line.
4. After applying all valid flags, bump the version stamp: `v1 → v2`. Update `status.md` change log and append a session entry here naming what flags were applied vs rejected.
5. If Codex's response includes "here's a rewritten section X" suggestions, evaluate at the reasoning level but rewrite in Claude's own words. Verbatim adoption is not allowed (short-circuits verification).

### Files modified

- `docs/track_3_rl_reuse/README.md` — version stamp added; Codex prompt hardened with version contract + structured output requirements.
- `docs/track_usages.md` — new "Review-loop with Codex" section.
- `status.md` — change-log entry.
- `handover.md` — this entry.

### Files NOT modified

- All notebooks, models, utils, data remain untouched.
- `plan.md` not touched.

---

## 2026-05-29 — Track 3 source-verified (Claude)

### What changed in this session

- `minor_project_rl` folder access granted (read-only).
- Read the key Track-C source files; reconciled Tier 2 spec against the actual canonical recipe.
- Eleven specific corrections logged in a table at the bottom of `docs/track_3_rl_reuse/README.md` ("Source-grounded summary of corrections"). Each row cites the source file and line numbers.
- **Most important corrections**:
  - Loss is BCE (V1) or Ordinal CE (V2) or +SmoothL1 on log1p(lag) (V3) — **not InfoNCE**.
  - Pretext lr is 3e-4, batch 256, 40 epochs (different from our GAT-GRU baseline lr 1e-3).
  - Cross-station pairs are always-far (never compute numeric lag across stations).
  - Distance pairs must be drawn from one continuous station series, not across stations.
  - Finetune default is fully unfrozen encoder from epoch 1.
  - Augmentation budget must be IDENTICAL between pretext and finetune.
  - Rainy-day balance 50/50 (matching source's NOOP balance 50/50).
- Codex verification prompt appended to track 3 README. Risheek can paste it into Codex for independent review. Source files Codex needs to re-read are explicitly enumerated.

### Pending: Codex review

When Codex returns with flags, **do not assume Codex is correct**. Re-verify each Codex point against the source files first; update the plan only if the reading matches.

### Files modified

- `docs/track_3_rl_reuse/README.md` — verified addendum at top, Tier 2 spec corrected, open-questions section updated with resolutions, corrections table appended, Codex prompt appended.
- `status.md` — change-log entry.
- `handover.md` — this entry.

### Files NOT modified

- All notebooks, models, utils, data remain untouched.
- `plan.md` not touched (no item-map or comparison-table change; the corrections are internal to track 3).

---

## 2026-05-29 — Item 3 ideation locked: tiered SSL ladder (Claude)

### What changed in this session

- Locked Idea 3 framing as **methodology-transfer paper** ("RL encoder-pretraining recipe applied to rainfall forecasting"), per Risheek's choice.
- Locked Idea 3 strategy as a **tiered experiment ladder** (Tier 0 → 1 → 2 → 3 → 4) with explicit per-tier decision gates. No single-bet on any one path. Each tier has a decision rule that says continue, pivot, or stop.

---

_[Note: a 2026-05-30 editor write-truncation dropped some OLDER backreference session entries below this point. All ACTIVE context (the 2026-05-30 track_c entries, open clarifications, current status) is intact above. The lost entries were historical only.]_

# Track 3 — RL → Rainfall: Encoder-pretraining recipe transfer

**REVIEW VERSION: `track3-readme-2026-05-29-v5`** — bumped every time content below this marker is materially edited. The Codex prompt at the bottom of this file instructs Codex to echo this exact string at the top of its review response. If Codex's echoed version does not match the version above, the review is stale: re-send Codex the current README before applying any flags.

> **v4 → v5 changelog.** Two informal Opus 4 research passes plus Claude's independent verification flipped the v4 plan structure. **Major finding**: STGCL's "joint > pretrain" result is **contrastive-specific**, not a universal STGNN-SSL rule. The masked-reconstruction lineage (STEP KDD 2022, STD-MAE IJCAI 2024, GPT-ST NeurIPS 2023) is unambiguously **pretrain-then-finetune** and beats SOTA on irregular sensor graphs (traffic). GPT-ST's own mechanistic explanation: "higher correlation between the mask-reconstruction task and the downstream regression task" — masked-recon pretext is itself a regression task, so it transfers cleanly via pretrain-then-finetune, whereas contrastive pretext (instance discrimination) misaligns with regression downstream and needs joint loss to bridge. v5 changes: (1) **Tier 2 fully refactored** — pretrain-then-finetune masked reconstruction becomes the PRIMARY direction (was a v4 peer primary); (2) **joint masked-reconstruction + forecasting becomes the NOVEL COMPARISON arm** (no published precedent in weather domain — this is the dissertation's empirical contribution); (3) **V_DIST demoted to a 35-minute preprobe documentation experiment** — no full pretraining run, single-seed negative result reported as the "faithful TDC transfer fails in our domain" finding; (4) **Strategy A vs Strategy C-subset bake-off dissolved** since V_DIST no longer runs at full scale; (5) **seed policy locked**: 3 seeds at Tier 0 anchor (non-negotiable for variance estimation), 1 seed elsewhere → 3 seeds only if a tier beats its comparison anchor by ≥0.2 RMSE; (6) **HyMePre corrected** — Opus and Claude independently verified that HyMePre's "2048 stations" are 5.625° grid cells (64×32 lat/lon binning), NOT real meteorological stations; removed from references as a station-level precedent.

> **v3 → v4 changelog.** Informal Codex pass 3 reviewed Claude's chat clarification of Path A's sampling structure and flagged: (a) Strategy A (one supervised station per pair) is compute-wasteful — each forward pass spends ~78% of GAT compute on stations whose representations don't get loss gradient; (b) the v3 README still had two stale "mean-pool" remnants at line 302 and 393 contradicting the selected-node design at line 156; (c) recommended Strategy C-subset (K date-pairs × M target stations per pair) as the v4 default. Claude verified all technical claims against ground truth: **293 stations in coords file (confirmed), 100km graph avg degree 62.2 (Codex claimed 62, confirmed), batch-256 expected unique stations 170.4 (Codex claimed 170, confirmed), VRAM estimate ~1.4 GB activations for two windows at batch-16 (fits 8 GB with comfortable headroom)**. v4 changes: (1) Strategy C-subset becomes default for V_DIST sampling, with K=16 date-pairs per batch and M=32 target stations per pair; (2) Strategy A retained ONLY as a 5-epoch probe in the preprobe stage; (3) Strategy B dropped; (4) stale mean-pool remnants cleaned; (5) preprobe Stage B re-scoped. **NOTE: v5 supersedes most v4 V_DIST design decisions because V_DIST no longer runs at full scale — these are kept for historical record but not active.**

> **v2 → v3 changelog.** Informal Codex pass 2 reviewed Claude's chat reasoning (not a formal version-stamped review). Claude reproduced Codex's empirical claim on V_DIST signal strength at scale: 30-station Spearman probe confirms ρ ≈ 0.41 at lag 7 with anomalies, collapsing to ρ < 0.15 by lag 90. Two literature claims (Weather2K, HiSTGNN) verified via web search. v3 changes: (1) feature-treatment lock — anomaly normalization on aux variables across all Item 3 tiers, raw rain everywhere; (2) Tier 0b added with lat/lon/elev as a matched comparison anchor (lat/lon/elev NOT folded silently into Tier 0); (3) Tier 1 Path C aux head targets become anomalies (input/output distribution match); (4) Tier 2A V_DIST capped at 14-day max lag (empirically justified — full numbers in the probe table below) with embedding fixed to selected-node final hidden state (NOT mean-pool); (5) Tier 2B Path B masked reconstruction promoted to peer primary alongside Tier 2A; (6) preprobe expanded to two-stage: tiny all-lag probe + 14-day capped probe; (7) dissertation narrative reframed as "we compared faithful-RL-transfer vs domain-adapted alternative" — defensible regardless of which path wins.

> **v1 → v2 changelog.** Codex review pass 1 returned 12 corrections-table flags + 4 design critiques + 3 missed concerns. Claude re-verified each against source files. **All 12 Codex flags that pointed at source readings were confirmed** (full audit notes below in the corrections table). Most consequential: the canonical s13_run7 recipe is **distance-only SmoothL1 on log1p(lag), 20 epochs**, NOT ordinal CE + 40 epochs as v1 claimed. Tier 2 was restructured to put distance-only as the primary variant; V1 (BCE) and V2 (ordinal CE) flagged as predecessor recipes that the source SUPERSEDED.

_Scoped 2026-05-29. Framing locked: **methodology-transfer paper** ("RL encoder-pretraining recipe applied to rainfall forecasting"). Strategy locked: **tiered experiment ladder** — no single-bet on any one pathway; checkpoints decide continuation._

> **Source-verified addendum (2026-05-29).** I now have read-only access to `C:\Users\rishe\minor_project_rl`. The pathway descriptions below are reconciled against the actual source files (`rnd_baseline/track_c/encoder_tdc.py`, `train_encoder.py`, `bc_teacher.py`, `RESULTS_5M_ABLATION_TRACK_C.md`, `SESSION18_NOTES.md`). Significant updates to my earlier assumptions are marked **[VERIFIED]**, **[CORRECTED]**, or **[NEW]** in each section.

> **Review-loop workflow (read once, follow always).** Risheek sends the Codex prompt at the bottom of this file to Codex; Codex returns a chat response; Risheek pastes the response back here; Claude applies updates. **Codex never writes to docs directly.** Procedure documented in `docs/track_usages.md` § "Review-loop with Codex". The version stamp above is the contract that keeps Claude's and Codex's view of this file aligned.

## Source-of-truth: what the minor RL project actually found

**[VERIFIED 2026-05-29 against `RESULTS_5M_ABLATION_TRACK_C.md` §1 and §2.1.]**

Confirmed from the actual cohort writeup:

- **5 conditions × seed 0 × 5M frame cap** on RTX 4060.
- `c_rnd_baseline`: first completion at frame **1,218,560**, strong_success at **2,004,992** — fastest.
- `c_hd2c_full`: strong_success at **2,676,736** (0.67 M frames later).
- `c_aytar_only`: never completed, plateau at 2,000,896.
- `c_salimans_only`: peak completion 0.07, plateau at 2,000,896.
- `c_bc_only`: first completion at frame **96,256**, strong_success at 2,592,768.

Note: **all 5 conditions share the same BC-pretrained encoder** as the agent CNN trunk init. The ablation only varies the four policy-gradient mechanisms (Aytar reward, BC distillation, gated RND, Salimans curriculum). So the headline finding is more precisely: "given a shared good encoder, the policy-gradient mechanisms gave no measurable lift on this task within budget; some hurt."

**Caveats also verified:**

- Single-seed only at the time of the writeup (seed 0). Session 18 was the planned multi-seed re-run; not delivered as of `SESSION18_NOTES.md`. **Our 3-seed Tier 0 plan is more rigorous than the source.**
- `mean_ext_return = 0.00` logging bug in `train.py` lines 1082–1085 (events dict missing `score` key). Documented in `SESSION18_NOTES.md` §1.1. **Doesn't affect our rainfall plan**, but means quoting the c_rnd_baseline "solve" as gospel is unsafe — we cite the qualitative encoder-is-load-bearing claim only.
- Multi-life latching may inflate completion rate. Same source, §1.

## What transfers (and what doesn't)

| RL project component | Transfer to rainfall? | Why |
|----------------------|----------------------|-----|
| TDC pretext (temporal-distance classification) | **YES — Path A** | Direct rainfall analog as a contrastive pretext on sequence windows. |
| BC finetune (mimic expert actions) | **YES — Path C, weaker analog** | We have no demos; the 5 ERA5 auxiliary variables act as a cheap teacher signal via multi-task aux heads. |
| Encoder weights (`s14_unfrozen_encoder/encoder.pt`) | **NO** | Different modality (Atari frames → rainfall tensors). Only the *recipe* transfers, not weights. |
| Aytar embedding-distance intrinsic reward | NO | RL-specific exploration mechanism. |
| Demo-reset curriculum | NO | RL-specific. Weak analog as easy→hard sample weighting; low value. |
| Gated RND | NO | RL-specific. |
| Ablation methodology (5-condition matrix) | **YES — framing** | We mirror the diagnostic-ablation style for the dissertation. |

## Pathway catalog (background)

- **Path A — TDC-analog SSL pretraining.** Contrastive pretext on rainfall sequence windows. Closest RL analog. Lit anchor: TNC (Tonekaboni et al., ICLR 2021, arxiv 2106.00750), TS2Vec (Yue et al., AAAI 2022, arxiv 2106.10466).
- **Path B — Masked-reconstruction SSL pretraining.** Mask `(t, station, var)` patches; reconstruct. Lit anchor: STEP (Shao et al., KDD 2022, arxiv 2206.09113), Adaptive Spatio-Temporal Graphs SSL for Weather (arxiv 2511.00049).
- **Path C — Multi-task aux regression heads.** Predict temp/dew/pressure/u10/v10 alongside rain. BC-analog. Lit anchor: CAMT multi-task rainfall + broader MTL rainfall literature.
- **Path D — Climatology imitation warm-start.** Predict climatology then actual rain. Low novelty, almost certainly subsumed by cyclic time features already in the input. **Dropped from the ladder unless results force a revisit.**

## Feature treatment (v3 lock — applies across ALL Item 3 tiers)

To avoid the v1-era "every tier has different input semantics" mess and to fix the seasonality leak Codex identified, **one feature regime is locked across all tiers**:

- **Aux input variables (`temp_2m`, `dewpoint_2m`, `surface_pressure`, `u10`, `v10`)**: **day-of-year anomaly normalization**. For each variable, subtract the per-`(station, day-of-year)` climatological mean computed on the train period only. Then z-score the anomalies using train-period stats.
- **Rain input channel**: **raw, z-scored** (no anomaly — rain is the target signal we want the model to learn).
- **Cyclic time features** (`doy_sin/cos`, `month_sin/cos`, `year_sin/cos`): **kept in finetune/baseline input**; **explicitly excluded from V_DIST pretext input** (anti-shortcut guardrail). They come back at finetune.
- **Static per-station features**: `lat`, `lon`, `elevation`. **Tier 0 does NOT include them.** **Tier 0b does.** SSL methods (Tier 2A, 2B) compare against whichever baseline (Tier 0 or 0b) shares their feature schema. Elevation fetched via GEE `USGS/SRTMGL1_003`.
- **Output target**: raw next-day rainfall in mm. Unchanged from exp_13.

**Note on historical comparability.** Tier 0 corrected (with anomalies + scaler/split fix) will produce a different RMSE than exp_13's historical 8.79. That's intentional — exp_13 has the known scaler-leakage and shortcut-prone raw aux features. The Tier 0 corrected number becomes the new comparison anchor; the historical 8.79 is reported alongside for context but not as the bar to clear.

**Note on Path C output targets.** If aux inputs are anomalies, then Path C's aux head targets must also be anomalies (input/output distribution match). Rain head still predicts raw next-day mm. See Tier 1 for details.

## Execution Strategy: Tiered Ladder (v5)

Risheek's directive: **decisions based on results, not assumptions**. Each tier has a decision gate. Designed to fit in **2–3 days** after Items 1+2 complete.

```
Tier 0:  Corrected baseline (12 features, anomaly aux, 3 seeds — anchor)
Tier 0b: Static-feature baseline (15 features = 12 + lat/lon/elev)
         ↓ (decision gate: baseline numbers stable?)
Tier 1:  Path C — multi-task aux heads (anomaly aux targets)
         ↓ (decision gate: gain or train-val-gap reduction?)
Tier 2A: Masked reconstruction PRETRAIN-THEN-FINETUNE     ┐
         (PRIMARY, STEP/STD-MAE/GPT-ST precedent)          │
                                                            ├── head-to-head
Tier 2B: Masked reconstruction JOINT (forecast + recon)   │   on same Tier 0b
         (NOVEL COMPARISON, no published weather precedent)┘   baseline
Tier 2C: V_DIST documentation experiment (35-min preprobe only,
         negative-result documentation — no full pretraining run)
         ↓ (decision gate: 2A or 2B beats Tier 0/0b/1?)
Tier 3 [optional]: 3-stage BC analog
         (Tier 2 best encoder → multi-task aux pretrain → rain finetune)
         ↓
Tier 4:  Ablation matrix + writeup
```

## Seed policy (v5 lock)

- **Tier 0 (corrected baseline) — 3 seeds always.** This is the comparison anchor for ALL other tiers. Without baseline variance, we can't tell whether any "improvement" is real or seed noise. Non-negotiable.
- **Every other tier — 1 seed initially.** Run the cheapest possible version first.
- **Conditional re-run with 3 seeds** if the 1-seed result beats its comparison anchor by ≥0.2 RMSE OR shrinks train-val gap by ≥0.3 RMSE. Tiers that don't improve stay at 1 seed.
- **Pretext stages within SSL tiers** are single-seed always (cost is concentrated in pretext; multi-seed pretext rarely justifies the compute).

Total worst-case Item 3 compute: **~5–6 hr** if most tiers don't improve and stay at 1 seed, **~8–10 hr** if many improve and re-run at 3 seeds. Both fit comfortably in remaining budget.

### Tier 0 — Corrected baseline (12 features, anomaly aux)

**Why first.** Every SSL claim depends on a stable, source-of-truth baseline. exp_13's 8.79 was on raw aux features with a scaler/split misalignment; Tier 0 fixes both.

**Scope.** Re-run `exp_13_gat_gru.ipynb` configuration with:
- Seeds: 42, 123, 7 (3 seeds — this is the comparison anchor for ALL Item 3 SSL claims).
- **Scaler/split alignment fix**: compute `train_end_date` from the actual 70%-split index, NOT `'2015-12-31'`.
- **Anomaly normalization on aux variables** per the v3 feature-treatment lock above. Rain stays raw.
- **Same input feature count as exp_13** (rain + 5 aux anomaly + 6 cyclic time = 12) — no lat/lon/elev (that's Tier 0b).
- Same hyperparams as exp_13 (30 epochs, Adam lr=1e-3, hidden 64, heads 4, batch 32, MSE).
- Log train AND val loss per epoch. Compute the **train–val gap at best-val epoch** explicitly.
- Save the full result CSV set (mirror exp_12 layout).

**Output.** `experiments/results/exp_16_gat_gru_baseline_seeds/seed_<N>/...`, plus `aggregated_baseline.csv` with mean ± std across seeds.

**Decision gate.** Std across seeds < 0.5 RMSE → baseline is stable enough to compare against. Std > 0.5 RMSE → run 2 more seeds before proceeding (high baseline variance contaminates any SSL claim).

**Wall-time.** 3 seeds × 25 min = ~75 min.

### Tier 0b — Static-feature baseline (15 features = 12 + lat/lon/elev)

**Why separate from Tier 0.** Per Codex pass 2: adding `lat`, `lon`, `elevation` is a feature change, not an SSL change. If we fold it silently into Tier 0, any RMSE gain from SSL methods using these features could be the static-feature gain in disguise. Codex anchored on Weather2K ([Zhu et al. 2023](https://arxiv.org/abs/2302.10493)) which includes these three as time-invariant constants.

**Scope.**
- Same config as Tier 0 plus 3 extra per-station constant features: `lat`, `lon`, `elev` (from SRTM via GEE `USGS/SRTMGL1_003`, fetched once and broadcast across the time dimension).
- Input feature count: 15 (rain + 5 aux anomaly + 6 cyclic time + 3 static).
- 1 seed initially (per v5 seed policy). If Tier 0b's RMSE beats Tier 0's 3-seed mean by ≥0.2, re-run with 3 seeds before drawing conclusions.
- New file: `data/wb_station_elevation.csv` — `[station_id, elevation_m]` from SRTM lookup.

**Output.** `experiments/results/exp_16b_gat_gru_static_features/...`.

**Wall-time.** 1 seed × ~25 min + (conditional) 2 extra seeds × 25 min = 25–75 min.

**Matched-comparison rule.** Any SSL tier using lat/lon/elev (Tier 2A, 2B) compares against Tier 0b. Any SSL tier NOT using them compares against Tier 0. Mixing is not allowed.

### Tier 1 — Path C: multi-task aux heads

**[v3 CORRECTED — aux head targets are anomalies, not raw values, to match input distribution per the feature-treatment lock.]**

**Mechanism.** Modify `GAT_GRU_Model` to output `(B, N, 6)` instead of `(B, N, 1)` — one channel for rain, five for the aux variables. Loss = `MSE(rain_raw_next) + λ · Σ_aux MSE(aux_anomaly_next)`. λ schedule: linear decay from 0.1 → 0.01 over training (rain head dominates by end).

**Implementation notes.**
- **Rain head target**: next-day raw rainfall (mm). Same as Tier 0/0b.
- **Aux head targets**: next-day **anomaly** of `(temp, dew, pressure, u10, v10)`. Matches the anomaly normalization on inputs — no distribution mismatch the model has to learn around.
- Shared encoder (GAT + GRU), separate linear heads.
- We don't care about aux predictions at evaluation — they're a regularization signal only. So aux predictions being "anomaly mm/K/Pa/etc." instead of absolute values doesn't matter for our metrics.

**Output.** `experiments/results/exp_17_gat_gru_multitask/seed_<N>/...`. **1 seed initially per v5 seed policy.**

**Decision gates** (compared to Tier 0 3-seed mean):
- Test RMSE drops by ≥0.2 → strong win, re-run with 3 seeds, proceed to Tier 4 considering Path C in the ablation.
- Test RMSE within noise but train–val gap shrinks by ≥0.3 → regularization win, re-run with 3 seeds.
- Neither → Path C stays at 1 seed and is reported in the ablation as "null lift."

**Wall-time.** 1 seed × ~30 min = ~30 min initially. + ~60 min if conditional re-run.

**Feature schema**: 12 features, same as Tier 0 (no lat/lon/elev). Comparison anchor: Tier 0.

### Tier 2A — Masked-reconstruction PRETRAIN-THEN-FINETUNE (PRIMARY, v5)

**[v5 PROMOTION TO PRIMARY.** Was a v4 "peer primary" (Path B). Now the lead direction. Justification: STEP (KDD 2022), STD-MAE (IJCAI 2024), and GPT-ST (NeurIPS 2023) all unambiguously use pretrain-then-finetune masked reconstruction on irregular spatio-temporal sensor graphs and all beat SOTA. GPT-ST's mechanistic explanation: "higher correlation between the mask-reconstruction task and the downstream regression task" — masked-recon pretext is itself a regression task, so it aligns naturally with our forecasting downstream. STGCL's "joint > pretrain" finding is contrastive-specific (Opus pass 2 + Claude verification) and does NOT generalize to masked reconstruction. **The pretrain-then-finetune masked direction is the well-precedented, lower-risk path with the strongest published evidence on regression-style spatio-temporal downstream tasks.]**

**Phase 2A — Pretext.**

- **Encoder:** the GAT + GRU stack from `models/gat_gru.py`, with the final `Linear(hidden_dim, 1)` head replaced by a 2-layer MLP reconstruction decoder.
- **Decoder output shape:** `(B, L=7, N=291, F_recon=6)` — reconstruct rain + 5 anomaly aux variables. Cyclic time is excluded from the reconstruction target (and from pretext input) because it carries no useful seasonality signal beyond the calendar.
- **Masking strategy:** randomly mask **15%** of cells in the input tensor `(B, L=7, N, F)`. Mask is per-cell (not per-feature or per-station). Replace masked cells with zero. (Learnable mask token is a follow-up refinement; zero-masking matches the simplest STEP/STD-MAE setup.)
- **Reconstruction loss:** MSE on the masked cells only. `loss = mean_{(t, n, f) ∈ masked} (pred[t, n, f] - true[t, n, f])^2`. Unmasked positions have no loss.
- **Train period only.** Same train_end_date computed from the 70%-split index as Tier 0.
- **Augmentation on unmasked cells:** rain-noise ±5% + cutout 0–5%. Cutout here is separate from the 15% masking — it's noise the encoder must learn to handle on the cells we do show.
- **Optimizer:** Adam, lr 3e-4, batch 256, 30 epochs, 800 steps/epoch. (Matches the STEP/STD-MAE/GPT-ST conventional setup for masked pretraining.)
- **Single seed for pretext** per v5 seed policy.

**Phase 2B — Finetune.**

- **Default: fully unfrozen encoder from epoch 1.** Mirrors the RL source's deployed `s14_unfrozen_encoder` recipe.
- Discard the reconstruction decoder. Attach a fresh `Linear(hidden_dim, 1)` regression head.
- Re-introduce the 6 cyclic time features at finetune input. Encoder's input projection gets 6 extra channels via fresh random init for those rows.
- Loss: plain MSE on next-day rain.
- Optimizer: Adam, lr 1e-3 (dissertation baseline choice).
- Augmentation: rain-noise + cutout, same budget as pretext.
- Epochs: 30, same as Tier 0.
- **1 seed initially.** If finetune RMSE beats Tier 0b 3-seed mean by ≥0.2 (matched 15-feature schema), **re-run with 3 finetune seeds** before headlining.

**Output.** `experiments/results/exp_18_gat_gru_mae_pretrain_finetune/seed_<N>/...` with both pretext checkpoint (`pretext_best.pt`) and finetune checkpoint(s).

**Feature schema:** 15 features (rain + 5 anomaly aux + 6 cyclic time at finetune + 3 static lat/lon/elev). Comparison anchor: Tier 0b.

**Wall-time.** Pretext 1 seed × ~1.5 hr + finetune 1 seed × ~25 min = ~2 hr base. +50 min conditional for 2 extra finetune seeds.

**Lit anchor.** [STEP (Shao et al. KDD 2022)](https://arxiv.org/pdf/2206.09113); [STD-MAE (Gao et al. IJCAI 2024)](https://arxiv.org/abs/2312.00516); [GPT-ST (Li et al. NeurIPS 2023)](https://arxiv.org/abs/2311.04245); [W-MAE precipitation results (Man et al. 2023)](https://arxiv.org/abs/2304.08754); [SpaT-SparK (arxiv 2412.15917)](https://arxiv.org/abs/2412.15917).

### Tier 2B — Masked-reconstruction JOINT (forecast + recon, NOVEL COMPARISON, v5)

**[v5 NEW DIRECTION.** No published precedent for joint masked-reconstruction + supervised forecasting on weather/precipitation/STGNN in the same training stage — Opus pass 2 searched explicitly and found zero in-weather examples. The pattern exists in non-weather domains (TimeAlign arxiv 2509.14181 has aux reconstruction branch; ST-ReP arxiv 2412.14537 integrates reconstruction + future prediction; SSL-Lanes arxiv 2206.14116 for motion forecasting) but not weather. **This tier IS the dissertation's empirical contribution: testing whether the joint pattern transfers to weather-station SSL.]**

**Mechanism — single training stage.**

- Train the GAT-GRU end-to-end with TWO loss terms simultaneously:
  - `L_forecast = MSE(predicted_rain_next, true_rain_next)` — the supervised forecasting loss (same as Tier 0).
  - `L_recon = MSE(reconstructed_masked_cells, true_masked_cells)` — same mechanic as Tier 2A pretext: randomly mask 15% of input cells, decoder reconstructs them.
- Total loss: `L = L_forecast + λ_recon · L_recon`.
- λ_recon schedule: start with **fixed λ_recon = 1.0** (equal weighting). If forecasting head underfits, decay λ_recon linearly 1.0 → 0.3 over the run.
- **Architecture:** same GAT + GRU stack. TWO heads off the same encoder: a `Linear(64, 1)` regression head for `L_forecast` and a 2-layer MLP decoder for `L_recon`. Shared encoder, separate heads.
- Optimizer: Adam, lr 1e-3, batch 32, 30 epochs (same as Tier 0). No separate pretraining stage.
- Augmentation: rain-noise + cutout on unmasked cells (same as Tier 2A pretext).
- **Single training run per seed**, not pretrain + finetune.

**Decision gate.** Compare Tier 2B finetune RMSE against Tier 2A finetune RMSE (both with matched 15-feature schema, both single-seed initially). If 2B beats 2A by ≥0.2, re-run 2B with 3 seeds; if 2B is within noise of 2A but the joint training is more stable (smaller train-val gap), this is the dissertation's positive contribution. If 2B loses, this is also a publishable result: "the joint pattern from non-weather domains does not transfer cleanly to weather STGNN SSL."

**Output.** `experiments/results/exp_19_gat_gru_joint_recon/seed_<N>/...`.

**Feature schema:** 15 features (matches Tier 0b).

**Wall-time.** ~30 min base (same as Tier 0 + minimal joint-loss overhead). +60 min conditional for 2 extra seeds.

### Tier 2C — V_DIST documentation experiment (negative-result documentation, v5)

**[v5 DEMOTION.** Was a full pretraining run with 3-seed finetune in v4. **No full pretraining in v5** — V_DIST runs only as the 35-minute preprobe described below. Single seed. Justification: empirical signal is weak in our data (30-station Spearman probe shows ρ ≈ 0.41 best case at lag 7, collapsing past 30 days; documented in this README). STGCL's findings on STGNN-contrastive-as-pretraining further support that this is unlikely to win. **The documented negative result IS the contribution for this tier**: "we evaluated the faithful s13_run7 TDC pretext on rainfall and confirmed signal-bounded transfer."]**

**Mechanism — preprobe only.**

- **Stage A — all-lag documentation probe (5 minutes).** Sample 1000 pairs uniformly at random lags `[1, 1000]` from train period. Compute Spearman ρ between encoder distance and `log1p(lag)` using a frozen Tier-0 encoder. Expected: ρ ≈ 0.4 at lag 7, ρ < 0.15 by lag 90, ρ ≈ 0 at lag 365 (per 30-station probe baked into this README's empirical-justification section).
- **Stage B — 14-day capped feasibility probe (30 minutes).** Freeze the Tier 0 encoder, train only a 128-d projection head with V_DIST SmoothL1(α·||e_i − e_j|| + β, log1p(lag)) loss for 5 epochs on lag ≤14 pairs. Strategy C-subset sampling (K=16 date-pairs × M=32 stations).
- **No full pretraining.** Regardless of preprobe outcome, V_DIST does not run beyond these 35 minutes.

**Output.** `experiments/results/exp_20_v_dist_documentation/` with the probe results and Spearman curves. Reported in the dissertation as evidence for the domain-specificity-of-pretext-task literature finding.

**Feature schema:** N/A (no downstream RMSE measured; this tier is preprobe-only).

**Wall-time.** ~35 min total.

### Tier 3 — 3-stage BC analog (optional, only if time allows)

**[v5 addition, addressing Risheek's BC-missing concern.** The RL source had a sequential pipeline: TDC SSL pretrain → BC supervised fine-tune → final task. Our v5 Tier 2A (masked-recon pretrain → rain finetune) has the same sequential structure but skips the intermediate supervised stage. Tier 3 closes that gap: SSL pretrain → multi-task supervised pretrain → single-task rain finetune. **No published precedent on weather station graphs**; flagged as exploratory in the writeup.]**

**Mechanism — 3 stages.**

1. Take Tier 2A's best pretrained encoder.
2. Multi-task supervised pretrain: replace decoder with the 6-output multi-task head from Tier 1; train ~10 epochs predicting `(rain, temp, dew, pressure, u10, v10)` for next-day with `λ_aux = 0.1`.
3. Single-task fine-tune: drop the aux heads, keep the encoder, attach a fresh rain regression head, train 30 epochs with MSE.

**1 seed only.** This tier exists to test the structural hypothesis ("does a supervised intermediate stage help?") not to produce a headline number.

**Output.** `experiments/results/exp_21_bc_analog_3stage/seed_42/...`.

**Wall-time.** ~3 hr total (reuses Tier 2A's pretrained checkpoint).

### Tier 4 — Ablation matrix + writeup (dissertation deliverable, v5)

Run after Tiers 0/0b/1/2A/2B/2C land. The v5 headline comparison table:

| Condition | Pretext | Schema | Anchor | Status | Seeds |
|-----------|---------|--------|--------|--------|-------|
| `baseline` | none | 12 | — | Tier 0 | **3 always** |
| `+static` | none | 15 | Tier 0 | Tier 0b | 1 → 3 if ≥0.2 |
| `+aux` | none | 12 | Tier 0 | Tier 1 | 1 → 3 if ≥0.2 |
| `+ssl_mask_pretrain` (PRIMARY) | masked-recon pretrain-then-finetune | 15 | Tier 0b | Tier 2A | 1 → 3 if ≥0.2 |
| `+ssl_mask_joint` (NOVEL) | masked-recon joint loss | 15 | Tier 0b | Tier 2B | 1 → 3 if ≥0.2 |
| `v_dist_probe` | doc-only preprobe | N/A | N/A | Tier 2C | 1 only |
| `+3stage_bc` (optional) | SSL → aux pretrain → rain finetune | 15 | Tier 0b | Tier 3 | 1 only |

Headline figure: bar chart of mean test RMSE ± std for each condition. Conditions that ran at 1 seed are reported with explicit "single-seed; baseline std as reference for noise floor" caption.

## Success criterion

Locked **after Tier 0 + Tier 1 results are in** (per Risheek: "I'll decide after seeing the first result"). Candidate bars to commit to at that decision point, in order of preference:

1. Beat 8.79 RMSE on ERA5 WB test (multi-seed mean, lower CI overlapping baseline upper CI is sufficient).
2. Reduce train–val gap by ≥0.5 RMSE while holding test RMSE within noise.
3. Multi-seed `+aux+ssl` beats multi-seed baseline with non-overlapping CI.

**[v2 Codex concern #3 — already addressed by plan structure.]** "Beat 8.79" is not a clean target until Tier 0 produces the *corrected* baseline (scaler/split fixed, 3 seeds, best-val checkpoint). Tier 0 is precisely that re-run. After Tier 0, the comparison anchor will be the **corrected multi-seed mean**, which may differ from the historical 8.79 number from `exp_13`. State the corrected baseline explicitly in the writeup; do not chain new gains to the historical number.

## Repo files to read before coding (Tier 2 specifically)

| File | Why |
|------|-----|
| `models/gat_gru.py` | Architecture we are pretraining. The encoder = `gat` + `gru`; the head = `fc`. |
| `utils/data_utils/dataset_files/gnn_dataset.py` | `build_feature_tensor`, `add_time_features`, `SpatioTemporalDataset`. We will write a `PretextSpatioTemporalDataset` that yields (window_1, window_2, label) triples instead of (x, y) pairs. |
| `utils/data_utils/data_helper_utils.py` | `scale_pivots`, `temporal_split` (line 187 array variant), `build_edge_index_radius`. |
| `experiments/experimentation_notebooks/exp_13_gat_gru.ipynb` | Template for the finetune notebook. |
| `status.md` known issues | Scaler/split bug must be fixed in all Item 3 runs. |
| `docs/track_3_rl_reuse/implementation_notes.md` (created when Tier 2 begins) | Per-paper notes + per-decision running log. |

## Open questions to revisit when implementing (v2)

These don't block the plan but should be answered as Tier 2 implementation begins:

1. **Canonical recipe (V_DIST primary).** [v2 RESOLVED via Codex pass 1.] Distance-only SmoothL1 on log1p(lag), 20 epochs, demo-only pairs (= one-station-only in our setting), no ordinal CE. V1/V2 are predecessors used only for ablation.
2. **Window-embedding semantics.** [v4 RESOLVED, was v3 leftover that contradicted the selected-node fix.] Pretext needs one 128-d vector per `(supervised_station, window)`. **Default: read out the selected station's final GRU hidden state, after the full graph window is run through GAT + GRU.** Not mean-pool. Not max-pool. The graph context is preserved (GAT used the station's neighbors during the forward pass) but the readout is station-specific. Codex pass 2 caught the v2 mean-pool inconsistency; Codex pass 3 caught that this open-question line was still stale despite the v3 fix in the Tier 2A spec. v4 cleans it.
3. **Seasonality-leak mitigation.** [v2 OPEN.] Pick one: (A) day-of-year anomaly normalization on temp/dew/pressure (DEFAULT), (B) same-season hard negatives, (C) rain-only pretext input. Decision needed before Tier 2 implementation.
4. **Imputation-gap masking.** [v2 OPEN.] Need a per-station boolean mask of "originally NaN, filled by ffill/bfill/0". Use to reject pairs whose lag segment is >10% imputed. Computation step: take the pre-imputation pivot, derive the mask, save alongside the scaled pivots.
5. **Encoder freeze during finetune.** [v2 RESOLVED.] Default **unfrozen from epoch 1** (mirrors deployed recipe; CLI default in source is frozen, but the winning run used `--no-freeze-encoder`). Frozen-first-5 vs always-unfrozen is the source-style ablation toggle.
6. **Preprobe before full pretraining.** [v2 NEW guardrail.] 30-min projection-head-only probe to test whether the Tier 0 encoder already has the lag signal. If yes, skip or shorten V_DIST. If no, justify full pretraining.
7. **Multi-seed in pretraining is expensive** (~1.5 hr × 3 = 4.5 hr). Acceptable if Item 1+2 land on time; fall back to single-seed pretext + 3-seed finetune if compute is short.
8. **Variant separation.** [v2 RESOLVED.] V_DIST, V2, V1 are independent experiments. No warm-start chains. Each pretrains from scratch.
9. **Aug-across-phases scope.** [v2 RESOLVED.] Bridge derived from source's encoder-aug ↔ BC-aug rule. Same augmentation in pretext and finetune in our setting.

## Risks and fallbacks

- **Zero-inflation collapses pretext (Path A failure mode 1).** Mitigation: rainy-day positive sampling already specified. If still collapsing, add a station-id discrimination auxiliary pretext (predict which station a window comes from).
- **Path A pretraining loss converges but downstream doesn't improve.** Indicates the encoder learned something irrelevant. Pivot to Path B.
- **Aux heads (Path C) make rain head worse.** Mitigation: λ schedule already low. If still degraded, run aux heads with detached gradients (auxiliary head doesn't backprop into shared encoder). This breaks the "BC analog" narrative but preserves the multi-task framing.
- **`minor_project_rl` folder access never granted.** Mitigation: ask Risheek to paste specific files (TDC encoder definition, BC training recipe pseudocode) if implementation hits ambiguity in Tier 2.
- **Time runs out.** Tier 0 + Tier 1 + Tier 2-Phase-2A (pretext only, single seed) gives a defensible "we did encoder pretraining" deliverable. Drop multi-seed for Tier 2 if compute budget squeezes; report single-seed with explicit caveat.

## Hardware budget

- All tiers fit on RTX 4060 / 16 GB RAM. Pretext pretraining loads the same data tensors (~280 MB total across 6 ERA5 vars), batched the same way, just with a different loss.
- **Total compute estimate for Tiers 0+1+2 (3 seeds each):** ~9 hours wall-time. Half a day if runs are sequential; less if Tier 0 and Tier 1 share a queue. Tier 3 adds another ~3 hours if needed.

## Deliverable structure

- Notebooks (in `experiments/experimentation_notebooks/`):
  - `exp_16_gat_gru_baseline_seeds.ipynb` (Tier 0)
  - `exp_17_gat_gru_multitask.ipynb` (Tier 1)
  - `exp_18_gat_gru_tdc_pretrain.ipynb` (Tier 2) — uses a new `models/gat_gru_pretrain.py` for the projection-head variant.
  - `exp_19_gat_gru_mae_pretrain.ipynb` (Tier 3, conditional).
- Result CSVs in matching `experiments/results/exp_<N>_.../seed_<S>/` directories.
- New file `docs/track_3_rl_reuse/implementation_notes.md` — created when Tier 2 implementation begins, captures pretext design decisions and what worked.
- Final comparison table appended to `plan.md` after Tier 4.

## Cross-track interaction

- **Item 1 Track A (IMD minimap) result may shift framing.** If IMD minimap shows GAT-GRU survives even on sparse observational data, the methodology-transfer paper has a stronger second leg. If IMD minimap fails, the framing tightens to "ERA5 reanalysis + SSL pretraining".
- **Item 2 (NP + Himalayan) result may extend the SSL evaluation.** If time permits, re-run Tier 2's best condition on the regional datasets to show SSL generalizes. Not required for the dissertation, but a nice supplement.

## Literature anchors (verified via web search this session)

- **STEP** — [Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting](https://arxiv.org/pdf/2206.09113), Shao et al., KDD 2022. Direct precedent for SSL pretraining of spatio-temporal GNN.
- **TNC** — [Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding](https://arxiv.org/abs/2106.00750), Tonekaboni et al., ICLR 2021. Direct TDC analog for time series.
- **TS2Vec** — [Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466), Yue et al., AAAI 2022. Hierarchical contrastive learning recipe.
- **Adaptive Spatio-Temporal Graphs SSL** — [arxiv 2511.00049](https://arxiv.org/abs/2511.00049), 2025. Most recent SSL-pretraining-for-weather paper.
- **Multi-task rainfall** — [Short-term rainfall forecasting using multi-task learning and Weibull post-processing](https://link.springer.com/article/10.1007/s13762-025-06690-0), Springer 2025. Path C anchor.
- **CAMT** — Channel Attention Enhanced Multi-task Learning for precipitation forecasting (referenced in [arxiv 2511.11152](https://arxiv.org/abs/2511.11152)).

## Empirical justification for the 14-day V_DIST cap (v3, 30-station probe)

This is the load-bearing evidence behind the Tier 2A short-lag cap. Reproduced from `train_distance.py` sampling logic adapted for our pivots. Code: in-line in the v3 README session transcript; saved snapshot in `docs/track_3_rl_reuse/probe_lag_distance.py` (to be written when Tier 2 begins).

**Setup.**
- 30 stations randomly sampled from the 291 ERA5 WB stations (`seed=42`).
- 1500 pairs per station per max-lag bin; each pair = two 7-day windows from one station's continuous train-period series.
- Train period: 1970-01-01 → 2007-10-19 (70% split).
- Distance metric: L2 norm on flattened `(window, var)` vectors.
- Two feature treatments compared: raw z-scored values, and aux-anomaly + rain-raw (the v3 feature-treatment lock).
- Correlation: Spearman ρ between distance and `log1p(actual_lag)`.

**Results (mean ± std across 30 stations).**

| max_lag | raw mean ± std | anomaly mean ± std | Interpretation |
|---|---|---|---|
| 7 | 0.311 ± 0.049 | **0.412 ± 0.036** | V_DIST workable here |
| 14 | 0.256 ± 0.035 | **0.326 ± 0.025** | V_DIST workable here (the cap) |
| 30 | 0.251 ± 0.034 | 0.224 ± 0.026 | Marginal |
| 90 | 0.421 ± 0.037 | 0.134 ± 0.027 | Raw shows seasonality leak; anomaly collapses |
| 180 | 0.507 ± 0.062 | 0.114 ± 0.026 | Raw mostly = "what season is it"; anomaly honest |
| 365 | 0.015 ± 0.025 | 0.043 ± 0.037 | Annual cycle folds back; no signal |

**Conclusion.**
- **Anomaly normalization is essential** at long lags — it kills the seasonality shortcut by ~40 percentage points at lag 180 (0.51 → 0.11). Confirms Codex pass 2's design point.
- **V_DIST signal lives at short lag only.** With anomalies, useful signal exists at lag ≤14 (ρ ≈ 0.33–0.41). Past lag 30 signal collapses to <0.25.
- **The 14-day cap is empirically justified, not arbitrary.** The source recipe used no cap because Atari trajectories are visually monotone in lag; rainfall is not.
- Std across stations is ~0.03–0.05 — pattern is robust, not 3-station noise.
- For dissertation: report this table as the empirical evidence for the domain-constrained sampling departure from `train_distance.py`'s uniform-over-all-lags recipe.

## Source-grounded summary of corrections (v2, post Codex pass 1; still binding in v3)

For audit speed in the next session. The "v2 verified spec" column has been re-grounded against source files after the Codex review on 2026-05-29 v1.

| Topic | v1 draft | v2 verified spec | Source line citation |
|-------|----------|------------------|----------------------|
| **Canonical pretext loss** | "V2 ordinal CE preferred, V3 hybrid optional" | **V_DIST = distance-only SmoothL1 on log1p(lag) IS the canonical s13_run7 recipe.** V2 and V1 are predecessor recipes. | `runs/encoder/s13_run7/metadata.json` (loss=`smooth_l1_log1p_lag`, ordinal=false); `experiments/distance_loss/train_distance.py:14-17` |
| Close threshold | Δt ≤ 7 days | `close = 4` (V1) or `(1,1),(2,2),(3,4)` bins (V2). **V_DIST has no buckets** — uniform lag. | `encoder_tdc.py:664`, `train_distance.py:83-100` |
| Far threshold | Δt > 30 days | `far > 50` (V1) or `(21,200)` + `>200` very-far (V2). **V_DIST has no buckets.** | `encoder_tdc.py:665, 237` |
| Cross-station role | "always-far class" | **V_DIST uses NO cross-station pairs** (source's `_sample_cross_seq` is in TripletSampler, not the distance-loss path). Cross-station appears only in V2 ablation, where the rainfall analog is weak (radius-graph stations may share storms). | `train_distance.py:74-100` (demo-only), `encoder_tdc.py:757` (TripletSampler-side only) |
| Class-balance constraint | "rainy-day balance ≥50%" | **`noop_balanced_frac` is action-class balance (action ≠ NOOP), not a rainfall analog.** For V_DIST no balance constraint applies (uniform). For V2 ablation, use wet/dry stratification with rain > 1 mm/day + over-represent heavy windows. | `encoder_tdc.py:644 is_noop`, `:842-854 balance loop` |
| Augmentation analogs | "±1 day jitter + ±5% rain noise" as if source | **Source augmentation is ±2px image translate + ±5% brightness** (pixel-level Atari ops). **Rainfall bridge: ±5% multiplicative rain noise YES; ±1 day jitter NO (corrupts the lag label).** Optional: cutout/dropout. | `encoder_tdc.py:156-175 augment_frame_uint8` |
| Optimizer / LR | "Adam lr 3e-4 pretext; 1e-3 finetune" | Same, with source attribution corrected: lr=3e-4 is V_DIST source; lr=1e-3 finetune is dissertation choice (`exp_13_gat_gru.ipynb:171`). | `train_distance.py:155`, `exp_13_gat_gru.ipynb:171` |
| Batch / epochs | "batch 256, 40 epochs, 800 steps/epoch" | **V_DIST = batch 256, 20 epochs, 800 steps/epoch.** 40 was the train_encoder.py default for V2 ordinal CE, not what s13_run7 actually ran. | `train_distance.py:152-154`, `s13_run7/metadata.json:14-16` |
| Projection head dim | 128 | **128 — confirmed.** [v4 corrected from a v2/v3 stale remnant: rainfall analog is per-`(station, window)` 128-d via the **selected station node's final GRU hidden state** after the full graph window is run through GAT + GRU. NOT mean-pool. Codex pass 3 caught this stale wording.] | `encoder_tdc.py:279, 287`, plus rainfall-side design choice |
| Finetune freeze recipe | "default unfrozen from epoch 1" | Same, with provenance clarified: source CLI default freezes; deployed recipe used `--no-freeze-encoder`. Our default mirrors deployment, not CLI. | `train_bc.py:723-724`, `bc_teacher.py:497` |
| Aug budget across phases | "source rule: must match" | Narrowed: source rule is encoder-aug ↔ BC-aug must match. Our pretext↔finetune is the analogous derived bridge, not a verbatim source rule. | `bc_teacher.py:503-508` comment |
| Multi-seed | "do 3 seeds" | Confirmed. Source headline is single-seed (seed 0); 3-seed is more rigorous than source. Multi-life completion caveat noted. | `RESULTS_5M_ABLATION_TRACK_C.md:1-3, 157-163, 213` |
| **NEW: imputation gap pitfall** | not in v1 | **Track imputed cells; reject pairs spanning >10% imputed in their lag segment.** Otherwise lag labels are fake. | `data_helper_utils.py:169-181 scale_pivots` (ffill→bfill→0), `cds_data_fetch.ipynb:447-453` |
| **NEW: seasonality leak via aux variables** | not in v1 | Excluding cyclic time features is necessary but insufficient: temp/dew/pressure encode seasonality. Mitigation: day-of-year anomaly normalization (default), or same-season hard negatives, or rain-only pretext input. | `gnn_dataset.py:9-33, 37-38` |
| **NEW: preprobe** | not in v1 | Before committing 6 hours of pretraining, do a 30-min "preprobe": freeze Tier 0 encoder, train projection head only, see if lag signal already fits cheaply. Analog of `probe_projfit`. | `experiments/distance_loss/train_distance.py:7-11` (design comment), `experiments/probe_projfit/` |

## Codex verification prompt

Paste the block below into Codex to get an independent review of this Track 3 plan. The prompt names the specific source files Codex should re-read so it can verify the corrections above against the actual code, not against my paraphrase.

```
You are reviewing a plan, NOT writing files. Output your review as a chat response only. You do not have permission to edit anything in the user's project; another assistant (Claude) is the only writer and will apply your flags after the user pastes your response back.

VERSION CONTRACT — read this first:
- The plan below carries a version stamp at the very top: `**REVIEW VERSION: <id>**` (a kebab-case string).
- Echo the EXACT version string as the very first line of your response: `REVIEWED VERSION: <id>`.
- If you cannot find a version stamp, stop and say so — do not review without it.

I am porting an encoder-pretraining recipe from C:\Users\rishe\minor_project_rl\rnd_baseline\track_c (Track-C, HD²C on Atari Montezuma's Revenge room-1) into a different project: daily rainfall forecasting with a GAT-GRU spatio-temporal graph model.

DO NOT modify anything in minor_project_rl — read-only reference.

Your job is to verify the plan's claimed mappings against the actual source. Specifically:

1. Re-read these files in this order:
   - encoder_tdc.py — confirm TripletSampler thresholds, ordinal bins, augmentation params, projection-head dim.
   - train_encoder.py — confirm the canonical training recipe (batch, epochs, lr, distance_loss_lambda semantics, the s13_run7 Codex correction about distance pairs being demo-only).
   - bc_teacher.py — confirm BCConfig freeze_encoder default, augmentation-budget-must-match rule, embedding_dim=128 requirement.
   - RESULTS_5M_ABLATION_TRACK_C.md — confirm the headline finding (c_rnd_baseline single-seed solve; encoder is the load-bearing mechanism).
   - SESSION18_NOTES.md §1 — confirm the mean_ext_return logging bug and the multi-life completion caveat.

2. For each row in my corrections table inside the plan, verify whether my "Verified spec" column accurately reflects the source. Flag any row where I have misread the source or oversimplified. Be specific — cite the line numbers.

3. Independently judge whether the rainfall analogs make sense given the source's actual design intent:
   - Cross-station pairs = cross-sequence pairs (always-far).
   - Rainy-day balance = NOOP balance.
   - Window-jitter + multiplicative rain noise = ±2px / ±5% brightness.
   - GAT-GRU encoder + 128-d projection head = Nature-DQN trunk + 128-d projection.
   These are domain-bridge analogs, not source-text claims. Tell me where the analog is weak or wrong.

4. Specifically critique these design choices in my plan, with confidence levels:
   a. Using 50% rainy-day balance vs. source's 50% NOOP balance — is the domain analog defensible? Daily rainfall is far sparser than NOOP frames in Atari (rain > 0 happens ~30% of days in monsoon, <5% off-monsoon, vs NOOP frames ~50% of trajectory). Should the constraint be tightened?
   b. Excluding cyclic time features from the pretext input is meant to prevent shortcut-learning. Is there a residual leak through the temperature/dew channels (which also encode seasonality)? If so, suggest the cleanest mitigation.
   c. The Codex correction in s13_run7 was that distance pairs must come from one canonical sequence (demo only). For us, "one sequence" = one station's time series. Is there a corresponding pitfall I might be ignoring? (e.g., the rain pivot has ffill→bfill→0 imputation — does imputed data create artificially close pairs across genuine data-gap boundaries?)
   d. Reusing the same encoder for two contrastive variants (V2 ordinal vs V3 ordinal+SmoothL1) — should we share weights across V2 and V3 runs, or train each from scratch?

5. Do NOT just say "looks good". Find at least one technical concern I missed.

6. Output format (in this order):
   a. First line: `REVIEWED VERSION: <id>` exactly.
   b. A short verdict per corrections-table row: `<row title>: correct | oversimplified | wrong — <cite source file:line>`.
   c. A short verdict per rainfall analog in question 3.
   d. Your answer to each of 4a, 4b, 4c, 4d with a confidence level (`high | medium | low`).
   e. A single paragraph on overall plan soundness.
   f. Your 1–3 strongest technical concerns I missed, each with: concern, source-file citation, suggested mitigation.

The plan I am asking you to review is appended below this prompt — please read it in full before responding.

[PASTE FULL TRACK 3 README.md CONTENT HERE — INCLUDING THE VERSION STAMP AT THE TOP]
```

### How Claude will process Codex's response (binding workflow)

Risheek pastes Codex's response into chat. Then Claude:

1. **Reads the first line.** If `REVIEWED VERSION` does not match the current README's `REVIEW VERSION`, stop. Ask Risheek to re-send Codex the current README. Do not apply any flags.
2. **For each Codex flag, re-verifies against the source files.** The pattern is non-negotiable: Codex flags → Claude reads the cited source lines → Claude updates the README only if Codex's reading matches the source. **Codex's judgment is never the sole basis for an edit.**
3. **Applies edits.** Each edit cites the Codex flag that prompted it + the source-file line that confirmed it. Edits are made with the Edit tool, not by rewriting the file.
4. **Bumps the version stamp.** After applying flags, change `track3-readme-2026-05-29-v1` to `track3-readme-2026-05-29-v2` (or the next session date). Append a one-line entry to the Change Log section of `status.md` summarizing what Codex flagged and what Claude applied.
5. **Updates `handover.md`** with the new version stamp so future sessions know what's current.

### Boundary rules

- **Codex never writes to any project file.** Codex only produces chat responses.
- **Claude is the only writer.** Edits land via the Edit / Write tools.
- **The version stamp is the source-of-truth contract.** A response without a matching version is, by policy, applied as zero edits.
- **No partial updates from a stale review.** If even one flag in a stale review looks valuable, re-run the review on the current README; do not cherry-pick from the stale one.

## Codex review history

- **Pass 1: 2026-05-29 against v1 (formal, version-stamped).** Codex returned 12 corrections-table flags + 4 design critiques (4a–4d) + 3 missed concerns. Claude re-verified each against source files. **All 12 flags confirmed by source; no Codex flag was rejected.** Major impact: canonical recipe changed from "V2 ordinal CE primary" to "V_DIST distance-only SmoothL1 primary, V2/V1 ablation-only." Augmentation set re-specified. New guardrails for imputation-gap masking, seasonality leak, preprobe. Version bumped v1 → v2.
- **Pass 2: 2026-05-29 (informal, chat-context review of Claude's chat reasoning, not version-stamped).** Codex critiqued Claude's proposed v3 changes before they were written to the README. Claude verified the load-bearing empirical claim (V_DIST Spearman) at scale on 30 stations — matched Codex's 3-station numbers. Two literature claims (Weather2K, HiSTGNN) verified via web search. Codex's 4 pushbacks applied:
  - (1) V_DIST short-lag cap framed as "source-objective, domain-constrained" not "pure faithful transfer." Two-stage preprobe added: all-lag documentation probe + 14-day capped feasibility probe.
  - (2) Tier 1 Path C aux head targets changed to anomalies (input/output distribution match).
  - (3) Tier 0b added as a separate static-feature baseline (lat/lon/elev). All SSL methods compare against matched baseline.
  - (4) Spearman probe scaled from 3 to 30 stations and baked into README as empirical evidence.
- Path B (masked reconstruction) promoted from "Tier 3 backup" to peer primary alongside Tier 2A. Version bumped v2 → v3.
- **Pass 3: 2026-05-29 (informal, chat-context review of Claude's clarification of Path A sampling).** Codex flagged: (a) Strategy A is compute-wasteful — 1-layer GAT means only ~62 of 291 nodes (~22%) get meaningful gradient per pair; (b) v3 README still had two stale mean-pool remnants at the open-questions line and the corrections table; (c) recommended Strategy C-subset (K date-pairs × M target stations). Claude verified all technical claims against ground truth: 293 stations in coords file (CONFIRMED exactly), 100km graph avg degree 62.2 (Codex said 62, CONFIRMED), batch-256 expected unique 170.4 (Codex said 170, CONFIRMED), VRAM ~1.4 GB fits 8 GB. Codex's recommendation accepted with one pushback: Strategy B dropped entirely (over-constraining is reasoned, not worth probe compute); Strategy A retained as preprobe Stage B probe vs C-subset. Stale mean-pool remnants cleaned. Version bumped v3 → v4.
- Pass 4: (not yet run.) Next review should target v5 once implementation_notes start landing.

## Opus 4 research history (informal, separate from Codex review-loop)

- **Opus pass 1: 2026-05-29 (informal, chat-context research review).** Asked Opus to evaluate whether v4's V_DIST-as-peer-primary direction was justified given (a) the BC-missing concern, (b) the weak empirical V_DIST signal, and (c) the STGCL "joint > pretrain" literature finding. Opus delivered a TL;DR: "masked reconstruction is the better-supported SSL direction; the empirically optimal integration is single-stage joint multi-task, not pretrain-then-finetune; no published recipe matches our regime (sub-300 stations, daily, single-GPU, precipitation, irregular graph)." Major correction: **HyMePre's "2048 stations" are 5.625° grid cells, NOT real meteorological stations.** Claude independently verified by fetching the HyMePre paper PDF — verbatim quote: "The 2048 stations originate from a uniformly discretized global grid (64 longitude bins × 32 latitude bins)." HyMePre removed from `docs/references.md` as a station-level precedent.

- **Opus pass 2: 2026-05-29 (informal, chat-context follow-up).** Asked Opus to verify three load-bearing claims from pass 1: (a) GPT-ST's head-to-head numbers vs DGI/GraphCL; (b) STEP and STD-MAE training topology; (c) any joint masked-recon + forecasting precedent in weather domain. Opus delivered: **(a) GPT-ST's DGI/GraphCL comparison is a bar chart in Figure 5, qualitative ordering robust (masked > infomax/contrastive > none) but exact margins not extractable from numeric tables; tested only on METR-LA + NYC Taxi (not all PEMS benchmarks).** **(b) STEP, STD-MAE, GPT-ST are all unambiguously pretrain-then-finetune masked-reconstruction frameworks, all beat SOTA on traffic. STGCL's "joint > pretrain" finding is contrastive-specific, NOT a universal STGNN-SSL rule.** **(c) Zero published examples of single-stage joint masked-recon + forecasting in weather domain.** Claude independently verified the STD-MAE and STEP topology claims via web search — both confirmed. This pass FLIPPED Claude's pre-Opus-2 recommendation: v5 promotes pretrain-then-finetune masked as PRIMARY (was joint), demotes V_DIST to documentation experiment, makes joint masked the NOVEL COMPARISON. The mechanism: masked-recon pretext is itself a regression task, aligned with regression downstream — transfers cleanly via pretrain-finetune. Contrastive pretext (instance discrimination) misaligns with regression and needs joint loss to bridge. Different SSL families have different optimal usage topologies.

## Status

Ideation locked 2026-05-29. Three Codex review passes (pass 1 formal, passes 2 and 3 informal) plus two Opus 4 research passes (both informal). Source-verified against `minor_project_rl/rnd_baseline/track_c/`. Empirical V_DIST signal probe at 30 stations baked in as justification for V_DIST demotion to documentation experiment. STEP/STD-MAE/GPT-ST verified as pretrain-then-finetune masked-recon precedent for irregular sensor graphs; the v5 Tier 2A primary direction stands on that lineage. HyMePre removed from references after Opus + Claude verified its "stations" are 5.625° grid cells. Implementation starts when Items 1+2 are done.

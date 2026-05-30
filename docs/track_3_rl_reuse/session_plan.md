# Track 3 (track_c) — Session Plan

_Created 2026-05-30, reframed + greedy-redesigned 2026-05-30. **Execution sequencing** for the RL-reuse / SSL track. Science is in [`README.md`](README.md) (v5); this file says **what we run, in what order, in which session, what to build in parallel, and which docs to update at the end of each**._

> **Status:** planning only. No experiment code written yet. New files only; existing notebooks/models/utils stay read-only until Risheek approves a change. Risheek runs every experiment; Claude prepares code (this is what makes the parallel build-ahead in §3 work).

---

## 0. Dissertation framing (decided 2026-05-30) — read first

Retitled **away from "extreme rainfall events."** No model predicts daily extremes (exp_14: preds cap ~70 mm vs obs to 350 mm — MSE under-dispersion on zero-inflated data). **Same numbers, opposite defensibility:** "we failed to predict extremes" sinks the thesis; "we quantify and explain why standard ST-deep-learning can't resolve daily extremes here" is a sound *limitation*.

**Primary thesis (the bunker — already largely DONE):** a comparative study of spatio-temporal deep models for **daily** WB rainfall + a **gauge-vs-reanalysis (IMD vs ERA5)** data-source analysis. Stands on the architecture progression (LSTM 15.82 → ConvLSTM → transformer → GCN-GRU 8.99 → GAT-GRU 8.79) + Tracks 1A/1B.

| Piece | Role |
|-------|------|
| Comparative ST + IMD-vs-ERA5 (Items 1A/1B, exp_1–13) | **Headline / spine.** Defensible now. |
| **track_c (this plan): SSL methodology transfer + the feature/aux ablation** | **Upside chapter.** Judged on RMSE + train–val gap; null-result-safe. |
| Item 2 (regional ablation) | Optional breadth chapter; separate plan. |
| **Extreme-event skill** | **Diagnosed limitation, NOT a goal.** Computed privately (§4). |

**Budget:** the **~2–3 days is for track_c specifically** (Risheek, 2026-05-30) — other things (bunker writeup, Item 2) have their own budgets. **Priority: run track_c experiments first, write afterward**, reserving a writeup buffer (the bunker needs only completed results).

---

## 1. Scope locked with Risheek (2026-05-30)

| Decision | Choice |
|----------|--------|
| "track_c" | = `docs/track_3_rl_reuse/`. RL source folder is literally named `track_c`. |
| Thesis framing | Comparative ST + data-source (spine). track_c = upside chapter. |
| **Hard backbone (fixed trajectory)** | Session 0 harness → **Tier 0** (12-feat baseline) → **Tier 0b** (15-feat, +lat/lon/elev) → **Tier 2A** (masked-recon pretrain→finetune). |
| **Greedy add-on pool (keep everything; slot by value × cost × parallelism)** | **Tier 2B** (joint recon), **Tier 1** (multi-task aux), **Tier 2C** (V_DIST doc-probe), **Tier 3** (3-stage). Nothing thrown away. §6. |
| **SEED POLICY (overrides README v5)** | **Single seed (42) for EVERYTHING during the experimental phase.** **Multi-seed is the FINAL step before writeup**, run in parallel, **good-results-only** (the conditions that will be headlined). See §3. |
| track_c success axis | RMSE + train–val gap. Extreme skill = secondary diagnostic, never a gate (§4). |
| Item 2 | Separate / later. |
| Budget | ~2–3 days, track_c-specific. Experiments first; reserve writeup time. |

---

## 2. Deviations from README v5 (explicit — nothing hidden)

Execution simplifications/overrides under the budget, not science reversals. The README v5 still stands for the full ladder.

1. **Critical path = Tiers 0, 0b, 2A** (was the README's larger set). Tiers 1, 2B, 2C, 3 become a **greedy add-on pool** (§6), not dropped.
2. **SEED POLICY OVERRIDDEN.** README says "Tier 0 — 3 seeds always." **This plan runs single-seed (42) throughout and defers ALL multi-seed to one final parallel step before writeup** (Risheek's hard note). Single-seed numbers are *exploratory/provisional*; no final claim is made until the multi-seed confirmation step. The README's per-tier "1→3 if beats by ≥0.2" gates become "flag for the final multi-seed batch."
3. **Feature schema (revised — Tier 0b is back in):**
   - **Tier 0** = 12 feat (rain + 5 ERA5 aux + 6 cyclic time), raw z-scored. Base anchor.
   - **Tier 0b** = 15 feat (+ lat/lon/elev). vs Tier 0 → isolates the static-feature effect.
   - **All SSL/multi-task tiers (1, 2A, 2B, 3) = 15 feat, anchored to Tier 0b** → each isolates its own mechanism on the rich schema. (This *reverts* the earlier 12-vs-12 simplification — adding 0b back makes the cleaner README-style comparison affordable and extends the ablation.)
4. **Cyclic + static stay visible in the pretext input** (never masked, never reconstruction targets). Pretext masks/reconstructs only the **6 physical channels** (rain + 5 aux). This fixes `in_channels=15` across pretext and finetune (no weight surgery) and keeps the anti-shortcut intent (calendar/geography aren't prediction targets).
5. **Aux features raw z-scored** (mirrors exp_13). Anomaly-normalization is an optional refinement for Tier 2A only if reconstruction shows a seasonality artifact.

> Copy each deviation into `implementation_notes.md` and flag in the writeup.

---

## 3. Greedy execution model

**Backbone is sequential (hard); the pool is opportunistic (greedy).** Run the backbone in order; slot pool items in whenever the GPU frees and their prerequisite exists, prioritizing high story-value × low cost × parallelizable. Re-evaluate priorities at every result (reroute gates, §7).

```
BACKBONE (fixed, sequential):
  S0 harness ─→ S1 Tier 0 + Tier 0b ─→ S2 Tier 2A pretext ─→ S3 Tier 2A finetune
GREEDY POOL (slot as GPU frees, single seed each):
  Tier 2B (joint)   ← pair with 2A   [~30m]   high story value (pretrain-vs-joint)
  Tier 1  (aux)     ← any time       [~30m]   med (regularizer; also feeds Tier 3)
  Tier 2C (V_DIST)  ← any time       [~35m]   med-high narrative (faithful-RL-transfer doc)
  Tier 3  (3-stage) ← needs 2A + 1   [~1-2h]  exploratory; reuses checkpoints
FINAL (before writeup):
  Multi-seed the headline winners (parallel) ─→ ablation matrix ─→ chapter writeup
```

**Full-coverage ablation matrix (the deliverable — every row single-seed first, winners multi-seeded last):**

| Condition | Schema | Mechanism | Anchor | Priority |
|-----------|:------:|-----------|--------|----------|
| `baseline` (Tier 0) | 12 | corrected pipeline, no extras | — | backbone |
| `+static` (Tier 0b) | 15 | lat/lon/elev | Tier 0 | backbone |
| `+ssl_pretrain` (Tier 2A) | 15 | masked-recon pretrain→finetune | Tier 0b | backbone |
| `+ssl_joint` (Tier 2B) | 15 | masked-recon joint loss | Tier 0b, vs 2A | pool, high |
| `+aux` (Tier 1) | 15 | multi-task aux heads | Tier 0b | pool, med |
| `+3stage` (Tier 3) | 15 | SSL→aux→finetune | Tier 0b, vs 2A | pool, low |
| `v_dist_probe` (Tier 2C) | n/a | faithful TDC probe (doc) | — | pool, doc |

**Parallelism (the efficiency the user wants).** Claude prepares code; Risheek runs. So **while Risheek's GPU trains run N, Claude builds the notebook for run N+1.** Each session below has a **Build-ahead** line naming what to prepare during that session's idle GPU time. By the time a run finishes, the next notebook is ready.

**Multi-seed = final, parallel, winners-only.** After the matrix is filled on single seeds, take the conditions that (a) will be headlined or (b) look promising, and run seeds 123 + 7 for each — these can run back-to-back/parallel while writeup begins. Report mean ± std for those; single-seed rows stay captioned as exploratory with the baseline std as the noise floor.

---

## 4. Extreme-event diagnostics (private — decision-support, NOT the headline)

Extremes are no longer the goal. Still compute extreme skill on every run for: (a) **decision calls** (does any choice move heavy-rain detection at all? → informs the optional focal/log1p probe), and (b) the **limitation section** (characterize *where* skill ends, not a bare "CSI ≈ 0").

**Honest expectation:** modest skill at ≥35.6 mm (rather-heavy), collapsing toward little at ≥64.5 (heavy) and ~none at ≥124.5 (very-heavy) — consistent with the ~70 mm cap.

**New file (additive):** `utils/metric_utils/extreme_skill.py`. Thresholds (IMD daily classification): ≥35.6 / ≥64.5 / ≥124.5 mm; **always print `n_events`** (very-heavy is rare → noisy). Scores per τ (contingency a=hits, b=false alarms, c=misses, d=correct-neg): **POD** `a/(a+c)`, **FAR** `b/(a+b)`, **CSI** `a/(a+b+c)`, **Frequency Bias** `(a+b)/(a+c)` (diagnoses under-dispersion directly), **HSS** `2(ad−bc)/[(a+c)(c+d)+(a+b)(b+d)]`. The existing `extreme_metrics()` gives precision/recall/F1 (recall≡POD); this adds the meteorology names + CSI/HSS/freq-bias + a multi-threshold wrapper. Computed on real-space (mm) preds → `extreme_skill_real.csv`. **Verify** on a hand-built contingency case first.

---

## 5. Session-by-session plan

Each: **goal → prereqs → steps (real file refs) → gates → wall-time → Build-ahead (parallel) → doc updates → research note.** Single seed (42) everywhere. Don't advance past a failed gate.

### Session 0 — Diagnostics harness + elevation fetch  [CPU, ~2–3 hr]

**Goal.** Build/test the private extreme-skill scorer; fetch SRTM elevation for Tier 0b; create the running log.

**Steps.**
1. Create `utils/metric_utils/extreme_skill.py` per §4 (new, additive). Unit-test on a synthetic contingency table (known a/b/c/d).
2. **Elevation fetch for Tier 0b:** GEE `USGS/SRTMGL1_003` sampled at the 291 `data/wb_station_coords.csv` points → `data/wb_station_elevation.csv` `[station_id, elevation_m]`. Reuse the GEE auth/pattern from `experiments/experimentation_notebooks/gee_data_fetch.py`. (Easy continuation of prior GEE work.)
3. **(Optional, if quick)** Retrofit exp_13/exp_14 extreme scores from saved checkpoints → an immediate limitation figure.
4. Create `docs/track_3_rl_reuse/implementation_notes.md`.

**Gates.** Unit test passes; `wb_station_elevation.csv` has 291 finite rows.

**Build-ahead.** Claude drafts the Tier 0 notebook (`exp_16`) skeleton from `exp_13_gat_gru.ipynb` so Session 1 starts immediately.

**Doc updates.** §8 (Session-0 entry + handover one-liner).

**Research note.** Extreme-skill definitions verified via NWS/CAWCR/EUMETRAIN forecast-verification docs.

---

### Session 1 — Baselines: Tier 0 (12) + Tier 0b (15)  [GPU, ~2–3 hr; ~25 min each]

**Goal.** Two anchors in one session: the corrected baseline, and the static-feature variant.

**Steps.**
1. Notebook `exp_16_gat_gru_baseline_seeds.ipynb` (despite the name, **single seed 42 now** — multi-seed deferred). Copy CONFIG/train/eval from `exp_13_gat_gru.ipynb`; **don't** copy exp_12 cells 64/69 (cosmetic; issue #1).
2. **Fix #1 (scaler/split):** `train_end_date = rain_pivot.index[int(0.7*T)-1]` (≈2007-10) → `scale_pivots(train_end=...)`. Never `'2015-12-31'`.
3. **Fix #2:** evaluate from `_best.pt` (`train_model` saves it), not epoch-30.
4. **Tier 0:** 12 feat (6 ERA5 vars raw z-scored + 6 cyclic), `build_edge_index_radius(threshold_km=100)`, exp_13 hyperparams (SEQ_LEN 7, hidden 64, heads 4, batch 32, 30 ep, Adam 1e-3, `MSELoss`). Save full metric set + raw `(obs,pred)` + `extreme_skill_real.csv`.
5. **Tier 0b:** same, + 3 static features (`lat, lon, elev` from `wb_station_elevation.csv`) broadcast across time → `in_channels=15`. Separate notebook `exp_16b_gat_gru_static.ipynb` or a config flag.

**Gates.** Epoch-5 loss decreasing/not-NaN; `train_end_date` printed; Tier 0b feature count = 15.

**Wall-time.** ~25 min × 2 + setup ≈ 2–3 hr.

**Build-ahead.** While Tier 0/0b train, Claude builds `models/gat_gru_pretrain.py` + the Session-2 masked-recon pretext notebook.

**Doc updates.** §8. Add `exp_16`, `exp_16b` rows to `plan.md`; record corrected baseline as the provisional anchor (differs from historical 8.79 — expected).

**Research note.** Static lat/lon/elev precedent: MetNet-2 ([Nature Comms 2022](https://www.nature.com/articles/s41467-022-32483-x)) feeds lat/lon/elevation; Weather2K (arxiv 2302.10493, in `references.md`) uses them as station constants. **Honest nuance:** some studies find lat/lon redundant — Tier 0b settles it for our setup.

---

### Session 2 — Tier 2A Phase A: masked-recon pretraining (15-feat)  [GPU, ~2–3 hr; ~1.5 hr compute]

**Goal.** Pretrain the encoder by reconstructing masked physical-channel cells. **Highest-risk step** (STEP/STD-MAE/GPT-ST were traffic, not weather). Gate is *stability*.

**Steps.**
1. `models/gat_gru_pretrain.py` (new): reuse the encoder body (GATConv heads=4 concat → ReLU → GRU), **keep all L timesteps** `(B·N, L, hidden)`, attach a 2-layer MLP decoder → `(B, L, N, 6)` reconstructing the 6 physical channels. (Current `gat_gru.py` collapses to `out[:,-1,:]`+`Linear(hidden,1)`; pretrain must not collapse — verified.)
2. Masking/collate in-notebook: mask **15%** of cells across the 6 physical channels of input `(B,7,N,15)`; cyclic + static stay visible/unmasked; replace masked with 0; keep mask.
3. Recon loss: MSE on **masked cells only**. Aug on visible cells: rain-noise ±5% (+ optional cutout).
4. Train period only (same `train_end_date`). Adam lr **3e-4**, batch **256**, **30 epochs**, ~800 steps/epoch, **seed 42**. Save `pretext_best.pt`.

**Gates.** Recon loss decreases & stable (no NaN/divergence); **VRAM:** if batch 256 OOMs on 8 GB, drop to 128/64 + raise steps/epoch (log it); unmasked positions = 0 loss.

**Build-ahead.** During the ~1.5 hr pretext run, Claude builds: the **finetune notebook**, the **Tier 2B joint** notebook, and the **Tier 1 multi-task** notebook (all share the encoder/head code).

**Doc updates.** §8. Record mask %, batch used, recon-loss curve in `implementation_notes.md` (the novel part — document carefully).

**Research note.** Pretrain→finetune masked-recon precedent: STEP/STD-MAE/GPT-ST (in `references.md`, verified pretrain-then-finetune on irregular sensor graphs).

---

### Session 3 — SSL family: Tier 2A finetune + Tier 2B joint  [GPU, ~2 hr; 1 seed each]

**Goal.** Finetune the pretrained encoder; run the joint variant; get the **pretrain-vs-joint** comparison in one session.

**Steps.**
1. **Tier 2A finetune:** load pretrained encoder, discard decoder, fresh `Linear(64,1)`. Fully unfrozen from epoch 1 (mirrors deployed `s14_unfrozen_encoder`), `MSELoss`, Adam 1e-3, batch 32, 30 ep, best-val ckpt. 15-feat. Save metrics + `extreme_skill_real.csv`.
2. **Tier 2B joint:** same GAT-GRU, two heads off the shared encoder — `Linear(64,1)` forecast + 2-layer MLP recon decoder; single training stage, `L = MSE(rain) + λ_recon·MSE(masked recon)`, λ_recon=1.0 (decay to 0.3 if forecast underfits), Adam 1e-3, batch 32, 30 ep, 15-feat. Save metrics + extreme diagnostics.
3. Compare 2A vs 2B vs Tier 0b (all 15-feat, all single seed).

**Gate (flag for final multi-seed).** Mark any condition that beats Tier 0b by ≥0.2 RMSE OR shrinks the train–val gap by ≥0.3 as a **multi-seed candidate** (not re-run now). Extreme movement recorded, never a gate.

**Build-ahead.** During finetune/2B training, Claude builds the **Tier 2C V_DIST probe** + **Tier 3 scaffolding**.

**Doc updates.** §8. Add `exp_18` (2A) + `exp_19` (2B) rows to `plan.md`; record the pretrain-vs-joint outcome.

**Research note.** Pretrain-vs-joint is exactly STGCL's question ([arxiv 2108.11873](https://arxiv.org/abs/2108.11873), in refs): "joint > pretrain" is *contrastive-specific*; for masked-recon the joint-vs-pretrain comparison has **no published weather precedent** → this pairing is the chapter's novel mini-contribution. Defensible whichever wins.

---

### Session 4 — Greedy add-ons: Tier 1 (aux) + Tier 2C (V_DIST) [+ Tier 3 if time]  [GPU, ~2–3 hr; 1 seed each]

**Goal.** Fill the rest of the ablation matrix with the cheap, independent components.

**Steps.**
1. **Tier 1 multi-task (15-feat):** modify the head to output `(B,N,6)` (rain + 5 aux); `L = MSE(rain_raw) + λ·Σ MSE(aux_next)`, λ decay 0.1→0.01. Shared encoder, separate heads. Aux predictions are a regularization signal only. Anchor: Tier 0b. Save metrics + extreme diagnostics.
2. **Tier 2C V_DIST probe (doc-only, ~35 min):** freeze the Tier 0b encoder, train a 128-d projection head with `SmoothL1(α·‖e_i−e_j‖+β, log1p(lag))` on lag ≤14 pairs, 5 epochs; plus the all-lag Spearman documentation probe. No full pretraining. Output the Spearman curve as evidence for "faithful TDC transfer is signal-bounded in our domain."
3. **Tier 3 (only if backbone+2B+1 done AND time, ~1–2 hr):** Tier 2A encoder → multi-task supervised pretrain (reuse Tier 1's 6-output head, ~10 ep) → single-task rain finetune (30 ep). Reuses checkpoints, so cheap to add.

**Gate.** Same multi-seed-candidate flagging. Reroute per §7.

**Build-ahead.** Claude builds the **final multi-seed batch driver** + the **chapter writeup skeleton**.

**Doc updates.** §8. Add `exp_17` (Tier 1), `exp_20` (2C), `exp_21` (Tier 3) rows.

**Research notes.** Tier 1 — multi-task parameter sharing reduces overfitting ([multi-task TS-GNN, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10453913/); CAMT + Springer MTL in refs). Tier 2C — TNC/TS2Vec (refs) + the README's 30-station Spearman probe (signal-bounded past lag 14). Tier 3 — intermediate-task pretraining lineage ([intermediate task selection, arxiv 2104.08247](https://arxiv.org/pdf/2104.08247); [MTL vs intermediate fine-tuning, arxiv 2205.08124](https://arxiv.org/pdf/2205.08124)); **honest caveat:** intermediate-task benefit is task-dependent — flag Tier 3 as exploratory.

---

### Session 5 — FINAL multi-seed (parallel, winners-only) + ablation + chapter writeup  [GPU + CPU, ~2–3 hr]

**Goal.** Confirm the headline conditions with multiple seeds, then write the chapter. **This is the only multi-seed step.**

**Steps.**
1. **Multi-seed (seeds 123 + 7) for the flagged headline conditions only** — at minimum `baseline`, `+static`, and the best SSL condition; add any flagged promising row. Run back-to-back / parallel while writeup begins. Report mean ± std for these.
2. **Ablation matrix** (§3 table) filled: RMSE/MAE/bias/NRMSE (primary) + extreme diagnostics block (with `n_events`). Single-seed rows captioned "exploratory; baseline std = noise floor."
3. **Figures:** RMSE ± std bar; train–val curves (does pretraining/aux reduce overfitting?); diagnostic extreme/frequency-bias panel for the limitation section.
4. **track_c chapter draft** (`docs/track_3_rl_reuse/writeup_section.md`, new): methodology-transfer narrative (RL encoder-pretraining → rainfall) + honest results. Three outcome narratives ready (SSL wins / marginal / loses).
5. Append final table to `plan.md`. Note whether to trigger **Codex pass 4** (must echo `track3-readme-2026-05-29-v5`).

**Gate.** Every reported number traces to a saved CSV.

**Doc updates.** §8 (full pass); mark track_c core complete in `status.md`.

---

## 6. Greedy add-on pool — reasoning (why keep each, when to slot)

Nothing is thrown away; each earns its slot by **story value × cost × parallelizability**.

- **Tier 2B (joint recon) — slot right after 2A. HIGH value, ~30 min.** Only meaningful *relative to* 2A: together they answer "is the SSL signal better as pretraining or as a joint auxiliary loss?" — the STGCL question, with no masked-recon weather precedent. Risheek's instinct (a story exists via ablation with 2A) is correct.
- **Tier 1 (multi-task aux) — slot any time. MED value, ~30 min.** Tests "does forcing the encoder to also predict the physical drivers regularize the rain head / shrink the train–val gap?" Established MTL-as-regularizer. Also **builds the 6-output head reused by Tier 3.**
- **Tier 2C (V_DIST doc-probe) — slot any time. MED-HIGH narrative, ~35 min.** This is the *faithful* RL→rainfall transfer (the track's whole premise). Documenting that its lag-distance signal is bounded (ρ collapses past 14 days) is the "why we adapted the recipe instead of copying it" evidence. Cheap, frozen-encoder.
- **Tier 3 (3-stage) — last, only if time. LOW/exploratory, ~1–2 hr.** Completes the RL pipeline analog (TDC→BC→task ⇒ SSL→multi-task→task). Reuses 2A's checkpoint + Tier 1's head, so it's a composition, not new infrastructure. Honest caveat: intermediate-task benefit is task-dependent.

---

## 7. Reroute gates (room to redirect as evidence arrives)

At **every single-seed result**, classify the condition: **promising** (beats its anchor ≥0.2 RMSE or shrinks gap ≥0.3) → flag for the final multi-seed batch; **null** → document, stay single-seed. Then:

- **Tier 2A pretext unstable** (NaN/divergence) → drop mask to 10%; if still bad, deprioritize 2A/2B/3 and make **Tier 1** the main SSL-adjacent contribution; document 2A instability honestly.
- **Tier 2A shows a clear lift** → prioritize building 2B + 3 (the SSL family) and ensure they reach the multi-seed batch.
- **Tier 2A shows nothing** → still report (null methodology result); pivot the chapter's "what helped" emphasis to the Tier 0b/Tier 1 ablation; keep 2B/2C as honest negative-result documentation.
- **Tier 0b static features surprise (clear lift)** → ensure 0b is in the multi-seed batch; note in the data-source chapter too.
- **Extreme diagnostics surprise (real skill at ≥64.5)** → run the optional focal/log1p probe (reuses `utils/loss.py`); raise re-elevating an extreme angle as a decision call to Risheek.

Record the reroute decision + its trigger in that session's `implementation_notes.md` entry.

---

## 8. Per-session doc-update protocol

End of every session (Goal / Current status / Important context / Decisions):

1. **`implementation_notes.md`** — what was built/run, results, any deviation or reroute (with the *why* + research note).
2. **`handover.md`** — dated section: Goal, Current status, Important context (paths, gotchas), Decisions.
3. **`status.md`** — one-line Change Log + Research Depth Map entry for new notebooks.
4. **When numbers land:** `plan.md` comparison-table row(s).
5. **Cross-cutting framing (pending, needs Risheek's OK):** the 2026-05-30 retitle (§0) is dissertation-wide → belongs in `plan.md` + the `status.md` Goal line. **Propose, get consent, then edit.**
6. **Plan/policy change:** `README.md` only with consent; bump `REVIEW VERSION` (v5 → v6) and note in `status.md`/`handover.md` (README is under the Codex review-loop). **The seed-policy override (§2.2) is one such change — when convenient, reflect it in the README with consent.**

**Working rules:** existing code/notebooks/models/utils read-only by default — add new files; Risheek runs + approves. `minor_project_rl` read-only.

---

## 9. Risks

- **Single-seed exploration could mislead.** Mitigation: the final multi-seed step confirms every headline claim; single-seed numbers are labeled exploratory until then.
- **Masked-recon may not train on weather data** (S2 gate). Fallback: mask 10%, or reroute to Tier 1.
- **8 GB VRAM at batch 256** (S2 gate). Fallback: batch 64–128 + more steps/epoch.
- **Experiments-first eats the writeup window.** Mitigation: the bunker writeup needs only completed work (Items 1A/1B + exp progression); reserve a buffer; if S2–S4 slip, jump to writing the spine.
- **Pool sprawl.** Mitigation: backbone is the only must-do; pool items are independently skippable; Tier 3 is explicitly last.

---

## 10. Pointers

- Science / full ladder + literature: [`README.md`](README.md) (v5).
- Source-of-truth RL recipe: `C:\Users\rishe\minor_project_rl\rnd_baseline\track_c\` (read-only).
- Master plan + comparison table: [`../../plan.md`](../../plan.md).
- Session continuity: [`../../handover.md`](../../handover.md).
- Doc policy: [`../track_usages.md`](../track_usages.md).
- Running log (created Session 0): `implementation_notes.md`.

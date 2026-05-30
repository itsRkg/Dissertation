# References & Literature

_Append-only catalog of external references and internal design docs. Add an entry whenever you read a paper, blog, repo, or write a new internal doc that informs the dissertation. Keep entries terse: title, link or path, and one line on **why it matters here**._

## External literature

### Cited and verified (via web search and direct paper fetch, 2026-05-29)

These were verified during Item 3 ideation. Each is tied to a specific dissertation thread.

**Masked-reconstruction SSL pretraining lineage on irregular sensor graphs — anchor for Tier 2A PRIMARY in v5:**

- **STEP — Pre-training Enhanced Spatial-temporal GNN** ([arxiv 2206.09113](https://arxiv.org/abs/2206.09113), KDD 2022). TSFormer masked autoencoder + STGNN downstream, two-stage pretrain-then-finetune. Verified topology via Opus pass 2 + Claude search.
- **GPT-ST — Generative Pre-Training of STGNN** ([arxiv 2311.04245](https://arxiv.org/abs/2311.04245), NeurIPS 2023). Masked autoencoder, pretrain-then-finetune. Verified DGI/GraphCL comparison is a Figure 5 ablation showing ordering masked > infomax/contrastive > none (qualitative; exact margins not tabulated). Tested only on METR-LA + NYC Taxi for that ablation. **Used as principal precedent for v5 Tier 2A.**
- **STD-MAE — Spatial-Temporal-Decoupled Masked Pre-training** ([arxiv 2312.00516](https://arxiv.org/abs/2312.00516), IJCAI 2024). Two decoupled masked autoencoders, pretrain-then-finetune, six benchmarks (PEMS03/04/07/08, METR-LA, PEMS-BAY) all SOTA. Verified topology via Opus pass 2 + Claude search.
- **W-MAE — Pre-trained Weather Model with Masked Autoencoder** ([arxiv 2304.08754](https://arxiv.org/abs/2304.08754), 2023). Masked autoencoder on ERA5 reanalysis, pretrain-then-finetune. ~20% ACC improvement and ~30% precipitation forecasting improvement when applied to FourCastNet. **Out-of-regime** (gridded, planetary scale, multi-GPU) but direct precipitation precedent for the masked direction.
- **SpaT-SparK — Self-supervised Spatial-Temporal Learner for Precipitation Nowcasting** ([arxiv 2412.15917](https://arxiv.org/abs/2412.15917), 2024). Masked image modeling, pretrain-then-finetune, single-GPU (`--ngpus 1`), KNMI NL-50 radar precipitation, beats SmaAt-UNet. Closer to our compute regime than W-MAE but gridded radar rather than station graph.

**Joint contrastive + forecasting on STGNN (anchor for v5 Tier 2B reasoning, NOT for using contrastive as pretraining):**

- **STGCL — When Do Contrastive Learning Signals Help Spatio-Temporal Graph Forecasting?** ([arxiv 2108.11873](https://arxiv.org/abs/2108.11873), SIGSPATIAL 2022). Explicit finding: pretrain-then-finetune contrastive FAILS to help STGNN forecasting; joint forecast + contrastive auxiliary loss WINS. **The "joint > pretrain" finding is contrastive-specific** (verified via Opus pass 2 + Claude reasoning on mechanism). Does NOT generalize to masked reconstruction.

**Time-series SSL recipes — earlier references kept for completeness; not load-bearing for v5:**

- **TNC** ([arxiv 2106.00750](https://arxiv.org/abs/2106.00750), ICLR 2021). Temporal neighborhood coding contrastive. Originally cited as V_DIST analog; V_DIST is now demoted to documentation tier.
- **TS2Vec** ([arxiv 2106.10466](https://arxiv.org/abs/2106.10466), AAAI 2022). Hierarchical contrastive. Same status as TNC.

**Multi-task rainfall (Tier 1 reference):**

- **CAMT** ([arxiv 2511.11152](https://arxiv.org/abs/2511.11152)). Channel-attention multi-task framework.
- **Short-term rainfall MTL + Weibull post-processing** ([Springer 2025](https://link.springer.com/article/10.1007/s13762-025-06690-0)).

**Static-feature baseline (Tier 0b reference):**

- **Weather2K** ([arxiv 2302.10493](https://arxiv.org/abs/2302.10493), 2023). Includes latitude, longitude, altitude as time-invariant station constants.

**Removed from references (corrected 2026-05-29):**

- ~~HyMePre~~ — **Removed.** Opus pass 1 flagged and Claude verified by fetching the paper: HyMePre's "2048 stations" are 5.625° grid cells (64×32 lat/lon binning), NOT real meteorological stations. Verbatim from the paper: *"The 2048 stations originate from a uniformly discretized global grid (64 longitude bins × 32 latitude bins), where adjacent indices correspond to geographically proximate locations (approximately 5.625° separation per index step)."* It cannot be cited as station-level weather SSL precedent.
- ~~Adaptive Spatio-Temporal Graphs SSL Weather (arxiv 2511.00049)~~ — **Removed.** Opus pass 1 flagged quality concerns ("garbled text and implausibly strong claims") and noted it uses gridded ERA5/MERRA-2 (not stations) and tests temperature (not precipitation). Not appropriate as an SSL-on-weather-stations precedent.

### Background (not yet read in depth — to be verified before citing)

- Veličković et al. (2018), "Graph Attention Networks". Defending GAT-GRU architecture choice.
- Hersbach et al. (2020), "The ERA5 global reanalysis". ERA5 data provenance.
- Cragg (1971), hurdle / zero-inflated. Relevant if a future hurdle head is explored.
- Lusch, Kutz, Brunton (2018), deep learning Koopman embeddings. If Koopman is revisited as future work.
- Wu et al. (2020), MTGNN, adaptive graph learning. If adjacency learning is revisited.

## Datasets and external tools

- **ERA5 reanalysis via Google Earth Engine** — `ECMWF/ERA5_LAND/DAILY_AGGR` collection. Used by `gee_data_fetch.py`. Will be reused for Item 2 region downloads.
- **Copernicus CDS** — `cds_data_fetch.ipynb` is the fallback path. Slower but quota-resilient.
- **SRTM elevation** — `USGS/SRTMGL1_003` on GEE. May be used in Item 2 if Himalayan elevation mask is enabled.
- **West Bengal district shapefile** — `data/West_Bengal/District_shape_West_Bengal.shp`. Source unclear; verify provenance for dissertation citation.

## Internal design docs

- `status.md` (root) — master overview, file/folder map, change log.
- `plan.md` (root) — forward-looking master plan, links to track docs.
- `handover.md` (root) — current session → next session continuity notes.
- `docs/track_usages.md` — documentation policy (where new files go, naming + linking, token rationale).
- `docs/track_1a_imd_minimap/README.md` — Item 1 Track A specifics.
- `docs/track_1b_imd_era5_corr/README.md` — Item 1 Track B specifics.
- `docs/track_2_regional_ablation/README.md` — Item 2 specifics.
- `docs/track_3_rl_reuse/README.md` — Item 3 placeholder; awaiting research.

(Each track directory will also contain any track-local files added later — implementation notes, scratch analyses, track-local refs. Discover them by listing the directory; do not duplicate that index here.)

## Per-experiment notes

_Add here once we start writing per-experiment notes alongside saved results, e.g._

- `experiments/results/exp_12_gcn_gru_baseline/notes.md` — (not yet written) per-run reflections, hyperparameter tuning notes, qualitative observations.

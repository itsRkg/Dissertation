# Running track_c on Kaggle (free GPU) — workflow

_How code, data, and results flow between local ↔ GitHub ↔ Kaggle. Created 2026-05-30._

## The shape of it

```
  CODE      local repo  --git push-->  GitHub (itsRkg/Dissertation)  --git clone-->  Kaggle notebook
  DATA      local data/  --upload once-->  Kaggle Dataset (read-only)  --mounted at-->  /kaggle/input/<slug>
  RESULTS   Kaggle /kaggle/working  --download / kaggle API-->  local      (NOT through git)
```

- **Code** travels through git. **Data** does NOT (2 GB, gitignored) — you upload it to Kaggle once as a Dataset.
- **Results do NOT come back through git** (they're gitignored on purpose). They come back via Kaggle's Output download or the Kaggle CLI. (If results went through the code repo, we couldn't also gitignore them — that's the trade-off, and downloading from Kaggle is cleaner.)

## How much code change was needed?  Very little.

- NEW `utils/env_paths.py` — detects local vs Kaggle and returns the right paths.
- Each notebook gets a tiny **bootstrap cell** (clone repo + add to path on Kaggle; no-op locally) and uses `get_paths()` instead of a hard-coded `C:/Users/rishe/...`. **`exp_18` is already done.** For older notebooks, swap their `ROOT = 'C:/...'` cell for the same two cells (snippet at the bottom).

---

## ONE-TIME setup

### 1. Clean the repo, then push (do this once)

Right now `experiments/saved_models/` (284 MB) and `experiments/results/` are committed to git. The new `.gitignore` excludes them going forward, but already-tracked files must be untracked. From the repo root:

```bash
# stop tracking results / weights / logs / editor settings (files stay on disk)
git rm -r --cached --ignore-unmatch \
  experiments/results experiments/saved_models experiments/logs \
  experiments/experimentation_notebooks/checkpoints \
  experiments/experimentation_notebooks/logs \
  experiments/experimentation_notebooks/*.csv .vscode

git add -A                 # stages: new .gitignore, new code files, and the untracks above
git status                 # REVIEW: confirm no data/ , no *.pt , no results/ are staged to ADD
git commit -m "Add track_c code (extreme_skill, embedding_diag, gat_gru_multilayer/pretrain, exp_16/18, env_paths); ignore data/results/weights"
git push origin master
```

Notes:
- The 284 MB of old checkpoints still sit in past commits (history). That's fine — Kaggle uses a **shallow clone** (`--depth 1`, see below) so it never downloads history. If you ever want to shrink the GitHub repo itself, that needs a history rewrite (BFG / `git filter-repo`) — optional, not required.
- **Clear notebook outputs before committing** so result numbers/plots don't leak into git (Jupyter → "Clear All Outputs", or `pip install nbstripout && nbstripout --install` once).

### 2. Upload the data to Kaggle (do this once)

You don't need the full 2 GB — Session 2 only needs the ERA5 pivots + 2 CSVs (~140 MB). Create a Kaggle Dataset containing, **at the dataset root**:

```
era5_pivot_data/   (the 6 *_pivot.parquet files)
wb_station_coords.csv
wb_station_elevation.csv
```

- UI: kaggle.com → Datasets → New Dataset → drag those in → note the **slug** (e.g. `itsrkg/dissertation-era5`).
- or CLI: `kaggle datasets create -p <folder-with-those-files>`.

The dataset mounts read-only at `/kaggle/input/<slug>/`. `env_paths.get_paths()` points there when you pass the slug.

---

## PER-RUN on Kaggle

1. New Notebook → **Add Data**: attach your dataset (the one from step 2).
> **Environment note:** Kaggle ships PyTorch but **not** PyTorch Geometric (which provides `GATConv`).
> The `exp_18` bootstrap cell now auto-runs `pip install -q torch_geometric` on Kaggle. If you ever see
> `ModuleNotFoundError: No module named 'torch_geometric'`, add a cell `!pip install -q torch_geometric`
> ABOVE the imports, run it, then re-run (restart the kernel only if the import still fails). The basic
> install is enough for GATConv — the torch-scatter/torch-sparse extras are optional.

2. Right panel → **Settings**: turn **Internet = On** (needed for `git clone`) and **Accelerator = GPU**.
3. First cell of `exp_18` already clones the repo and sets the path. **Edit one line**: set
   `KAGGLE_DATA_SLUG = 'itsrkg/dissertation-era5'` (your real slug) in the bootstrap cell.
   - Easiest: in Kaggle, "Import Notebook" from GitHub (paste the `exp_18` file URL), or just `!git clone` then open it.
4. Run all. It trains, then saves to `/kaggle/working/experiments/...` and prints the stability + embedding checks.

### Get results back (interactive vs committed run — IMPORTANT)

Two ways a Kaggle notebook runs, very different for saving results:

- **Interactive / Draft session** (you click Run All): the kernel keeps computing even if your internet
  drops or the live log vanishes — that's just the browser losing the stream, NOT the run stopping
  (GPU %% busy = still training; it keeps writing to `/kaggle/working`). BUT `/kaggle/working` here is
  **ephemeral**: if the session times out or you close it without downloading, outputs (incl.
  `pretext_best.pt`) are lost. -> While the session is alive (running or just finished), open the
  right-hand **Output / Data** panel, browse `/kaggle/working/experiments/.../pretext/`, and **Download**
  `pretext_best.pt` + `pretext_loss_curve.csv/png`. You can grab `pretext_best.pt` as soon as the first
  `new best` line prints, as a safety copy.
- **Save Version -> 'Save & Run All (Commit)'** (recommended for long runs): runs **headless** on
  Kaggle servers, independent of your browser/internet, and **permanently saves** `/kaggle/working` as
  the notebook Output. Download anytime from the version's Output tab, or CLI:
  `kaggle kernels output <your-username>/<kernel-slug> -p ./from_kaggle`.

After download, drop `pretext_best.pt` into `experiments/saved_models/exp_18_gat_gru_mae_pretrain/pretext/`
locally for Session 3. (Results stay out of git by design — they return via Kaggle.)

### If the repo is PRIVATE

Plain `git clone` won't work. Add a GitHub **Personal Access Token** (repo-read scope) as a Kaggle **Secret** (Add-ons → Secrets), then in the bootstrap clone use it:

```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_PAT")
os.system(f'git clone --depth 1 https://{tok}@github.com/itsRkg/Dissertation.git ' + REPO)
```

(If the repo is public, nothing to do — the current bootstrap clones it directly.)

---

## Robustness: same notebook, both places

`exp_18` runs unchanged locally and on Kaggle because:
- the bootstrap cell only clones/sets-path when `/kaggle/working` exists (otherwise local),
- `get_paths()` resolves data to `repo/data` locally and `/kaggle/input/<slug>` on Kaggle, and outputs to `repo/experiments` locally and `/kaggle/working/experiments` on Kaggle.

You can run **Session 2 on Kaggle** while **Session 1 runs locally** — they touch different folders and different machines.

## Snippet to port an OLD notebook (e.g. exp_16) to Kaggle later

Replace its `ROOT = 'C:/Users/rishe/Dissertation'` cell with these two cells:

```python
# bootstrap
import os, sys
if os.path.exists('/kaggle/working'):
    REPO = '/kaggle/working/Dissertation'
    if not os.path.exists(REPO):
        os.system('git clone --depth 1 https://github.com/itsRkg/Dissertation.git ' + REPO)
    KAGGLE_DATA_SLUG = 'itsrkg/dissertation-era5'
else:
    REPO = 'C:/Users/rishe/Dissertation'; KAGGLE_DATA_SLUG = None
if REPO not in sys.path: sys.path.insert(0, REPO)
```
```python
from utils.env_paths import get_paths
P = get_paths(kaggle_data_slug=KAGGLE_DATA_SLUG)
# then use P['pivot_dir'], P['coords_csv'], P['elev_csv'], P['results_dir'], P['models_dir'], P['logs_dir']
```

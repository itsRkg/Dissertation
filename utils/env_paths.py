"""
env_paths.py — one place that resolves file paths for BOTH local and Kaggle runs.

Why: the notebooks used to hard-code `ROOT = 'C:/Users/rishe/Dissertation'`, which breaks
on Kaggle. `get_paths()` detects the environment and returns the right locations, so the
same notebook runs unchanged on your machine and on a Kaggle GPU.

LOCAL run:
    repo root   = this repo (resolved from this file's location)
    data        = <repo>/data
    outputs     = <repo>/experiments/{results,saved_models,logs}

KAGGLE run (detected via /kaggle/working):
    code        = wherever you `git clone` the repo (e.g. /kaggle/working/Dissertation)
    data        = /kaggle/input/<your-dataset-slug>   (read-only; you upload it once)
    outputs     = /kaggle/working/experiments/{...}    (the ONLY writable + downloadable place)

Data layout expected (local data/ OR the Kaggle dataset root):
    era5_pivot_data/{rain,temp,dew,pressure,u10,v10}_pivot.parquet
    wb_station_coords.csv
    wb_station_elevation.csv

Usage in a notebook:
    from utils.env_paths import get_paths
    P = get_paths(kaggle_data_slug='your-kaggle-dataset-slug')  # slug only used on Kaggle
    pivots = load_pivots(P['pivot_dir'])
    station_df = pd.read_csv(P['coords_csv'])
    ... save under P['results_dir'] / P['models_dir'] / P['logs_dir']
"""

import os
from pathlib import Path

# repo root = parent of the `utils/` folder that holds this file.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def is_kaggle():
    """True when running inside a Kaggle notebook."""
    return (
        os.path.exists("/kaggle/working")
        or os.path.exists("/kaggle/input")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    )


def _resolve_kaggle_data(kaggle_data_slug):
    """Find the uploaded data dir under /kaggle/input."""
    slug = kaggle_data_slug or os.environ.get("KAGGLE_DATA_SLUG")
    base = Path("/kaggle/input")
    if slug:
        return base / slug
    # Fallback: if exactly one dataset is attached, use it; else point at the input root
    # (and let the missing-file error make the problem obvious).
    if base.exists():
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
    return base


def get_paths(kaggle_data_slug=None):
    """Return a dict of resolved paths for the current environment."""
    kaggle = is_kaggle()

    if kaggle:
        data_dir = _resolve_kaggle_data(kaggle_data_slug)
        out_dir = Path("/kaggle/working")          # writable + downloadable
    else:
        data_dir = _REPO_ROOT / "data"
        out_dir = _REPO_ROOT

    paths = {
        "is_kaggle": kaggle,
        "repo": str(_REPO_ROOT),
        "data_dir": str(data_dir),
        # load_pivots() concatenates path + 'rain_pivot.parquet', so keep a trailing slash:
        "pivot_dir": str(data_dir / "era5_pivot_data") + "/",
        "coords_csv": str(data_dir / "wb_station_coords.csv"),
        "elev_csv": str(data_dir / "wb_station_elevation.csv"),
        "results_dir": str(out_dir / "experiments" / "results"),
        "models_dir": str(out_dir / "experiments" / "saved_models"),
        "logs_dir": str(out_dir / "experiments" / "logs"),
    }
    return paths


if __name__ == "__main__":
    from pprint import pprint
    pprint(get_paths())

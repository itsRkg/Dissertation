"""
env_paths.py - resolve file paths for BOTH local and Kaggle runs.

get_paths() detects the environment and returns the right locations so the same
notebook runs unchanged locally and on a Kaggle GPU.

LOCAL : data=<repo>/data ; outputs=<repo>/experiments/{results,saved_models,logs}
KAGGLE: data = the folder that actually contains rain_pivot.parquet (found by globbing
        /kaggle/input, so it works no matter how the dataset is named/nested) ;
        outputs=/kaggle/working/experiments/...

Usage:
  from utils.env_paths import get_paths
  P = get_paths()                       # slug optional; data is auto-located on Kaggle
  pivots = load_pivots(P['pivot_dir']); pd.read_csv(P['coords_csv']) ; save under P['results_dir'] ...
"""

import os
import glob
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def is_kaggle():
    return (
        os.path.exists("/kaggle/working")
        or os.path.exists("/kaggle/input")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    )


def _pivot_dir(data_dir):
    """Find ERA5 pivots whether in an era5_pivot_data/ subfolder or flat at the
    data root. Returns a path ending in '/'."""
    d = Path(data_dir)
    if (d / "era5_pivot_data" / "rain_pivot.parquet").exists():
        return str(d / "era5_pivot_data") + "/"
    if (d / "rain_pivot.parquet").exists():
        return str(d) + "/"
    return str(d / "era5_pivot_data") + "/"


def _find(name, fallback_dir):
    """Locate a data file anywhere under /kaggle/input; else fall back to <dir>/name."""
    hits = glob.glob("/kaggle/input/**/" + name, recursive=True)
    return hits[0] if hits else str(Path(fallback_dir) / name)


def _resolve_kaggle_data(kaggle_data_slug):
    base = Path("/kaggle/input")
    # Most robust: locate the folder that actually holds the pivots, wherever mounted.
    hits = glob.glob(str(base / "**" / "rain_pivot.parquet"), recursive=True)
    if hits:
        return Path(hits[0]).parent
    # Fallbacks: explicit slug, case-insensitive match, or single attached dataset.
    slug = kaggle_data_slug or os.environ.get("KAGGLE_DATA_SLUG")
    if slug and (base / slug).exists():
        return base / slug
    subdirs = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    if slug:
        want = str(slug).split("/")[-1].lower()
        for p in subdirs:
            if p.name.lower() == want:
                return p
    if len(subdirs) == 1:
        return subdirs[0]
    return (base / slug) if slug else base


def get_paths(kaggle_data_slug=None):
    kaggle = is_kaggle()
    if kaggle:
        data_dir = _resolve_kaggle_data(kaggle_data_slug)
        out_dir = Path("/kaggle/working")
    else:
        data_dir = _REPO_ROOT / "data"
        out_dir = _REPO_ROOT
    data_dir = Path(data_dir)
    if kaggle:
        coords = _find("wb_station_coords.csv", data_dir)
        elev = _find("wb_station_elevation.csv", data_dir)
    else:
        coords = str(data_dir / "wb_station_coords.csv")
        elev = str(data_dir / "wb_station_elevation.csv")
    return {
        "is_kaggle": kaggle,
        "repo": str(_REPO_ROOT),
        "data_dir": str(data_dir),
        "pivot_dir": _pivot_dir(data_dir),
        "coords_csv": coords,
        "elev_csv": elev,
        "results_dir": str(out_dir / "experiments" / "results"),
        "models_dir": str(out_dir / "experiments" / "saved_models"),
        "logs_dir": str(out_dir / "experiments" / "logs"),
    }


if __name__ == "__main__":
    from pprint import pprint
    pprint(get_paths())

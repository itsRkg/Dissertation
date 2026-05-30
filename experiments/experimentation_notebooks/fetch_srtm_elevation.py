#!/usr/bin/env python3
"""
fetch_srtm_elevation.py — one-off SRTM elevation lookup for the WB stations.

Created for track_c (docs/track_3_rl_reuse/session_plan.md, Session 0). Produces
the static `elevation` feature used by Tier 0b (+lat/lon/elev, 15-feature schema).

WHY a separate small script (not the Drive-export pattern of gee_data_fetch.py):
    we only need ONE value per station from a SINGLE image (USGS/SRTMGL1_003),
    so 293 points fit comfortably in a direct .getInfo() call — no Drive round-trip.

RUN THIS ON YOUR MACHINE (the one with Earth Engine auth + project ee-risheekghosh1):
    python experiments/experimentation_notebooks/fetch_srtm_elevation.py

OUTPUT:
    data/wb_station_elevation.csv  with columns [station_id, elevation_m]

Mirrors gee_data_fetch.py for auth/station-loading so behaviour is consistent.
"""

import os
import sys

import ee
import numpy as np
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================

# Paths resolved relative to this file so the script runs from any cwd.
_THIS = os.path.abspath(__file__)
REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))  # .../Dissertation
STATION_FILE = os.path.join(REPO_ROOT, "data", "wb_station_coords.csv")
OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "wb_station_elevation.csv")

PROJECT_ID = "ee-risheekghosh1"   # same project as gee_data_fetch.py
SRTM_ASSET = "USGS/SRTMGL1_003"   # 30 m SRTM DEM, band 'elevation'
SAMPLE_SCALE_M = 30               # native SRTM resolution; point sample
# Optional: average elevation in a small buffer to reduce single-pixel noise.
# 0 = sample the single 30 m pixel under each station (default, simplest).
BUFFER_M = 0


# ==============================
# INITIALIZATION (mirrors gee_data_fetch.py)
# ==============================

def initialize_gee():
    """Authenticate and initialize Earth Engine."""
    try:
        ee.Initialize(project=PROJECT_ID)
        print("GEE initialized successfully.")
    except Exception:
        print("Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        print("GEE authenticated and initialized.")


def load_station_df(csv_path):
    """Load the station CSV. Same required columns as gee_data_fetch.py."""
    if not os.path.exists(csv_path):
        print(f"Station file not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    required = {"station_id", "latitude", "longitude"}
    if not required.issubset(df.columns):
        print(f"CSV must contain columns: {required}; found {set(df.columns)}")
        sys.exit(1)
    return df


def to_feature_collection(df):
    feats = []
    for _, row in df.iterrows():
        pt = ee.Geometry.Point([row["longitude"], row["latitude"]])
        if BUFFER_M > 0:
            pt = pt.buffer(BUFFER_M)
        feats.append(ee.Feature(pt, {"station_id": str(row["station_id"])}))
    print(f"Loaded {len(feats)} stations.")
    return ee.FeatureCollection(feats)


# ==============================
# MAIN
# ==============================

def main():
    print("SRTM elevation fetch for WB stations...\n")
    initialize_gee()

    station_df = load_station_df(STATION_FILE)
    fc = to_feature_collection(station_df)

    # unmask(0): water-masked pixels (sea / large rivers, e.g. coastal stations
    # such as SAGAR ISLAND / SANDHEADS) become 0 m = sea level, so every station
    # gets a value. Stations that come back exactly 0 are printed for a sanity check.
    srtm = ee.Image(SRTM_ASSET).select("elevation").unmask(0)

    if BUFFER_M > 0:
        sampled = srtm.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=SAMPLE_SCALE_M
        )
        value_key = "mean"
    else:
        sampled = srtm.sampleRegions(
            collection=fc, scale=SAMPLE_SCALE_M, geometries=False
        )
        value_key = "elevation"

    print("Pulling results (.getInfo)...")
    info = sampled.getInfo()
    rows = []
    for f in info["features"]:
        props = f["properties"]
        rows.append(
            {
                "station_id": props.get("station_id"),
                "elevation_m": props.get(value_key, np.nan),
            }
        )
    elev = pd.DataFrame(rows)

    # Guarantee one row per input station (left-merge; flag any missing).
    out = station_df[["station_id"]].merge(elev, on="station_id", how="left")
    out["station_id"] = out["station_id"].astype(str)

    n_missing = int(out["elevation_m"].isna().sum())
    zeros = out.loc[out["elevation_m"] == 0, "station_id"].tolist()

    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nWrote {OUTPUT_FILE}")
    print(f"  stations: {len(out)}  | missing: {n_missing}")
    if n_missing:
        print("  MISSING (no SRTM value):", out.loc[out['elevation_m'].isna(), 'station_id'].tolist())
    print(f"  elevation_m  min={out['elevation_m'].min():.1f} "
          f"mean={out['elevation_m'].mean():.1f} max={out['elevation_m'].max():.1f}")
    if zeros:
        print(f"  exactly 0 m (likely water/coastal — sanity check): {zeros}")
    print("\nDone. Use this file as the lat/lon/elev source for Tier 0b.")


if __name__ == "__main__":
    main()

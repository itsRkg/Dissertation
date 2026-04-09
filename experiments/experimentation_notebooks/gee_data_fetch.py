#!/usr/bin/env python3
"""
ERA5-Land Daily Data Extraction from Google Earth Engine
-------------------------------------------------------

Extracts daily ERA5-Land variables at station locations and exports
them to Google Drive as CSV files (one per year).

Author: Risheek Ghosh
"""

import ee
import pandas as pd
import time
import os
import sys

# ==============================
# CONFIGURATION
# ==============================

STATION_FILE = r'C:/Users/rishe/Dissertation/data/wb_station_coords.csv'
OUTPUT_FOLDER = 'WB_Weather_Data'
START_YEAR = 1970
END_YEAR = 2023

# West Bengal expanded bounding box [W, S, E, N]
BBOX = [85.5, 20.0, 90.5, 28.0]
# Define your registered Cloud Project ID
project_id = 'ee-risheekghosh1'

# ==============================
# INITIALIZATION
# ==============================

def initialize_gee():
    """Authenticate and initialize Earth Engine."""
    try:
        ee.Initialize(project=project_id)
        print("✅ GEE initialized successfully.")
    except Exception:
        print("🔐 Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("✅ GEE authenticated and initialized.")


# ==============================
# LOAD STATIONS
# ==============================

def load_stations(csv_path):
    """Load station CSV and convert to FeatureCollection."""
    if not os.path.exists(csv_path):
        print(f"❌ Station file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_cols = {'station_id', 'latitude', 'longitude'}
    if not required_cols.issubset(df.columns):
        print(f"❌ CSV must contain columns: {required_cols}")
        sys.exit(1)

    features = []
    for _, row in df.iterrows():
        point = ee.Geometry.Point([row['longitude'], row['latitude']])
        feature = ee.Feature(point, {'station_id': str(row['station_id'])})
        features.append(feature)

    print(f"✅ Loaded {len(features)} stations.")
    return ee.FeatureCollection(features)


# ==============================
# EXPORT FUNCTION
# ==============================

def export_station_data(year, station_fc, region):
    """Create and start GEE export task for a given year."""
    print(f"📦 Queueing task for {year}...")

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start_date, end_date)
        .filterBounds(region)
    )

    def extract_to_points(image):
        date_string = image.date().format('YYYY-MM-dd')

        sampled = image.sampleRegions(
            collection=station_fc,
            scale=11132,
            geometries=False
        )

        return sampled.map(lambda f: f.set('date', date_string))

    flat_table = collection.map(extract_to_points).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=flat_table,
        description=f'ERA5_Station_Data_{year}',
        folder=OUTPUT_FOLDER,
        fileFormat='CSV',
        selectors=[
            'date',
            'station_id',
            'total_precipitation_sum',
            'temperature_2m',
            'dewpoint_temperature_2m',
            'surface_pressure',
            'u_component_of_wind_10m',
            'v_component_of_wind_10m'
        ]
    )

    task.start()
    print(f"🚀 Task submitted for {year}")


# ==============================
# MAIN PIPELINE
# ==============================

def main():
    print("🔧 Starting ERA5 GEE Extraction Pipeline...\n")

    # Initialize GEE
    initialize_gee()

    # Load stations
    station_fc = load_stations(STATION_FILE)

    # Define region
    region = ee.Geometry.BBox(*BBOX)

    print("\n📡 Submitting tasks...\n")

    for year in range(START_YEAR, END_YEAR + 1):
        try:
            export_station_data(year, station_fc, region)
            time.sleep(1)  # Prevent API throttling
        except Exception as e:
            print(f"❌ Failed for {year}: {e}")

    print("\n✅ All tasks submitted!")
    print("👉 Monitor progress at: https://code.earthengine.google.com/tasks")


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()
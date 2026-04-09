import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import datetime
import re

# -----------------------------
# Aggregation helper
# -----------------------------
def aggregate_values(values, agg):
    if agg == "max":
        return np.nanmax(values)
    elif agg == "mean":
        return np.nanmean(values)
    elif agg == "median":
        return np.nanmedian(values)
    elif agg == "sum":
        return np.nansum(values)
    else:
        raise ValueError("agg must be one of ['max', 'mean', 'median', 'sum']")


# -----------------------------
# Main plotting function
# -----------------------------
def plot_rainfall(
    df,
    station,
    year=None,
    month=None,
    agg="max",
    window=None,
    window_type="year",
    figsize=(10,5)
):
    df_st = df[df["STATION"] == station].copy()
    if df_st.empty:
        raise ValueError("No data found for this station")

    # Month ordering for date construction
    month_order = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    # Select rainfall columns
    if month:
        rain_cols = [c for c in df.columns if c.startswith(f"RAIN_{month}_")]
    else:
        rain_cols = [c for c in df.columns if c.startswith("RAIN_")]

    # =====================================================
    # CASE 1: YEAR NOT GIVEN → yearly aggregation
    # =====================================================
    if year is None:
        yearly_vals = []

        for yr, grp in df_st.groupby("YEAR"):
            values = grp[rain_cols].values.flatten()
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue

            yearly_vals.append({
                "YEAR": yr,
                "VALUE": aggregate_values(values, agg)
            })

        plot_df = pd.DataFrame(yearly_vals).sort_values("YEAR")
        plot_df.set_index("YEAR", inplace=True)

        # Rolling window over years
        if window and window_type == "year":
            plot_df["ROLLED"] = plot_df["VALUE"].rolling(
                window=window, min_periods=1
            ).mean()

        plt.figure(figsize=figsize)
        plt.plot(plot_df.index, plot_df["VALUE"], marker="o", label=agg)

        if window and window_type == "year":
            plt.plot(plot_df.index, plot_df["ROLLED"],
                     linewidth=2, label=f"{window}-year rolling mean")

        title = f"{agg.capitalize()} Rainfall vs Year\nStation: {station}"
        if month:
            title += f", Month: {month}"

        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Rainfall")
        plt.legend()
        plt.grid(True)
        plt.show()

    # =====================================================
    # CASE 2: YEAR GIVEN → daily chronological plot
    # =====================================================
    else:
        df_yr = df_st[df_st["YEAR"] == year]
        if df_yr.empty:
            raise ValueError(f"No data for year {year}")

        daily_vals = []

        for col in rain_cols:
            # col example: RAIN_JAN_05
            _, mon, day = col.split("_")
            day = int(day)
            month_num = month_order[mon]

            try:
                date = datetime.date(year, month_num, day)
            except ValueError:
                continue  # skip invalid dates like Feb 30

            val = df_yr[col].values[0]
            daily_vals.append((date, val))

        # Sort by real date
        daily_vals.sort(key=lambda x: x[0])

        dates, rainfall = zip(*daily_vals)
        series = pd.Series(rainfall, index=pd.to_datetime(dates))

        # Rolling window over days
        if window and window_type == "day":
            rolled = series.rolling(window=window, min_periods=1).mean()
        else:
            rolled = None

        plt.figure(figsize=figsize)
        plt.plot(series.index, series.values, marker="o", label="Daily rainfall")

        if rolled is not None:
            plt.plot(rolled.index, rolled.values,
                     linewidth=2, label=f"{window}-day rolling mean")

        title = f"Daily Rainfall (Chronological)\nStation: {station}, Year: {year}"
        if month:
            title += f", Month: {month}"

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Rainfall")
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_station_map(df, s_path = None):
    
    # 1. Load the West Bengal Shapefile
    # Replace with the actual path to your downloaded .shp file
    shapefile_path =s_path
    wb_map = gpd.read_file(shapefile_path)


    # 2. Load your Station Data
    # Assuming you have a CSV file with your 293 stations
    # parquet_path = f'C:/Users/rishe/Dissertation/data/processed_rain_with_coords.parquet'
    df_stations = df.copy(deep = True)  # Avoid modifying original df
    df_stations = df_stations[['station_id', 'lat', 'lon']].drop_duplicates()
    # 3. Convert the pandas DataFrame to a GeoDataFrame
    # We tell geopandas which columns contain the longitude (x) and latitude (y)
    gdf_stations = gpd.GeoDataFrame(
        df_stations, 
        geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat),
        crs="EPSG:4326" # EPSG:4326 is the standard coordinate system for raw Lat/Lon
    )

    # 4. Ensure the Coordinate Reference Systems (CRS) match
    # This is the magic step that perfectly aligns your points with the map
    if wb_map.crs != gdf_stations.crs:
        wb_map = wb_map.to_crs(gdf_stations.crs)

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot the base map of West Bengal
    # color='whitesmoke' sets the land color, edgecolor='black' draws the borders
    wb_map.plot(ax=ax, color='whitesmoke', edgecolor='black', linewidth=1)

    # Plot the 293 stations on top
    # You can change the color, marker size (s), and transparency (alpha)
    gdf_stations.plot(ax=ax, color='red', markersize=15, alpha=0.8, label='Stations')

    # Add titles and labels
    plt.title('Distribution of 293 Stations in West Bengal', fontsize=15)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()

    # Display the map
    plt.show()
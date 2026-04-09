import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.plotting_utils.plots import aggregate_values

def compute_threshold(values, threshold):
    values = values[~np.isnan(values)]

    if threshold == 0:
        return 0.0
    elif threshold == "mean":
        return np.nanmean(values)
    elif threshold == "median":
        return np.nanmedian(values)
    elif threshold.startswith("p"):
        q = float(threshold[1:]) / 100.0
        return np.nanquantile(values, q)
    else:
        raise ValueError(
            "threshold must be 0, 'mean', 'median', or percentile like 'p75'"
        )


def plot_rainfall_thresholded(
    df,
    station,
    year=None,
    month=None,
    agg="max",
    threshold=0,
    window=None,
    window_type="year",
    figsize=(10,5)
):
    df_st = df[df["STATION"] == station].copy()
    if df_st.empty:
        raise ValueError("No data found for this station")

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

        # Compute threshold
        thr = compute_threshold(plot_df["VALUE"].values, threshold)

        # Apply threshold (mask)
        plot_df["THRESH_VAL"] = plot_df["VALUE"].where(
            plot_df["VALUE"] > thr
        )

        # Rolling window
        if window and window_type == "year":
            plot_df["ROLLED"] = plot_df["THRESH_VAL"].rolling(
                window=window, min_periods=1
            ).mean()

        plt.figure(figsize=figsize)
        plt.plot(
            plot_df.index,
            plot_df["THRESH_VAL"],
            marker="o",
            linestyle="",
            label="Thresholded values"
        )

        if window and window_type == "year":
            plt.plot(
                plot_df.index,
                plot_df["ROLLED"],
                linewidth=2,
                label=f"{window}-year rolling mean"
            )

        plt.axhline(thr, linestyle="--", alpha=0.7,
                    label=f"Threshold = {thr:.2f}")

        title = f"Thresholded Rainfall vs Year\nStation: {station}"
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
            _, mon, day = col.split("_")
            day = int(day)
            month_num = month_order[mon]

            try:
                date = datetime.date(year, month_num, day)
            except ValueError:
                continue

            val = df_yr[col].values[0]
            daily_vals.append((date, val))

        daily_vals.sort(key=lambda x: x[0])
        dates, rainfall = zip(*daily_vals)

        series = pd.Series(rainfall, index=pd.to_datetime(dates))

        # Compute threshold
        thr = compute_threshold(series.values, threshold)

        # Apply threshold
        series_thr = series.where(series > thr)

        # Rolling window
        if window and window_type == "day":
            rolled = series_thr.rolling(window=window, min_periods=1).mean()
        else:
            rolled = None

        plt.figure(figsize=figsize)
        plt.plot(
            series_thr.index,
            series_thr.values,
            marker="o",
            linestyle="",
            label="Thresholded rainfall"
        )

        if rolled is not None:
            plt.plot(
                rolled.index,
                rolled.values,
                linewidth=2,
                label=f"{window}-day rolling mean"
            )

        plt.axhline(thr, linestyle="--", alpha=0.7,
                    label=f"Threshold = {thr:.2f}")

        title = f"Thresholded Daily Rainfall\nStation: {station}, Year: {year}"
        if month:
            title += f", Month: {month}"

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Rainfall")
        plt.legend()
        plt.grid(True)
        plt.show()

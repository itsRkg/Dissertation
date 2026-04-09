import numpy as np
import pandas as pd

def compute_threshold(values, threshold):
    values = values[~np.isnan(values)]

    if threshold == 0:
        return 0.0
    elif threshold == "mean":
        return np.mean(values)
    elif threshold == "median":
        return np.median(values)
    elif threshold.startswith("p"):
        q = float(threshold[1:]) / 100.0
        return np.quantile(values, q)
    else:
        raise ValueError("threshold must be 0, mean, median, or pXX")


def compute_station_yearly_and_pooled_stats(df, threshold="p90"):
    rain_cols = [c for c in df.columns if c.startswith("RAIN_")]
    records = []

    for station, df_st in df.groupby("STATION"):

        # =====================================================
        # Compute pooled threshold FIRST (station-level)
        # =====================================================
        all_vals = df_st[rain_cols].values.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        nz_all = all_vals[all_vals > 0]

        if len(nz_all) == 0:
            continue

        thr = compute_threshold(nz_all, threshold)

        # =====================================================
        # YEAR-WISE STATS (with sequence length)
        # =====================================================
        for year, df_yr in df_st.groupby("YEAR"):
            vals = df_yr[rain_cols].values.flatten()
            vals = vals[~np.isnan(vals)]

            if len(vals) == 0:
                continue

            nz = vals[vals > 0]

            if len(nz) == 0:
                mean = median = p75 = p80 = p90 = p95 = max_val = 0.0
                seq_len = 0
            else:
                mean = np.mean(nz)
                median = np.median(nz)
                p75 = np.quantile(nz, 0.75)
                p80 = np.quantile(nz, 0.80)
                p90 = np.quantile(nz, 0.90)
                p95 = np.quantile(nz, 0.95)
                max_val = np.max(nz)

                # yearly sequence length using pooled threshold
                seq_len = np.sum(nz > thr)

            records.append({
                "STATION": station,
                "YEAR": year,
                "TYPE": "yearly",

                "n_days": len(vals),
                "n_nonzero_days": len(nz),

                # NON-ZERO stats
                "mean": mean,
                "median": median,
                "p75": p75,
                "p80": p80,
                "p90": p90,
                "p95": p95,
                "max": max_val,

                # threshold info
                "threshold_type": threshold,
                "threshold_value": thr,
                "sequence_length_above_threshold": seq_len
            })

        # =====================================================
        # POOLED STATS (ALL YEARS)
        # =====================================================
        records.append({
            "STATION": station,
            "YEAR": "ALL",
            "TYPE": "pooled",

            "n_days": len(all_vals),
            "n_nonzero_days": len(nz_all),

            "mean": np.mean(nz_all),
            "median": np.median(nz_all),
            "p75": np.quantile(nz_all, 0.75),
            "p80": np.quantile(nz_all, 0.80),
            "p90": np.quantile(nz_all, 0.90),
            "p95": np.quantile(nz_all, 0.95),
            "max": np.max(nz_all),

            "threshold_type": threshold,
            "threshold_value": thr,
            "sequence_length_above_threshold": np.sum(nz_all > thr)
        })

    return pd.DataFrame(records)



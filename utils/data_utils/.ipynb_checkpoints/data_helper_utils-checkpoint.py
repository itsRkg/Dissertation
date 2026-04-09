import pandas as pd
import numpy as np

def temporal_split(df_long, train_frac=0.7, val_frac=0.15):
    train_idx, val_idx, test_idx = [], [], []

    for station_id, gdf in df_long.groupby("station_id"):
        gdf = gdf.sort_values("date")
        n = len(gdf)

        t_end = int(train_frac * n)
        v_end = int((train_frac + val_frac) * n)

        train_idx.append(gdf.iloc[:t_end].index)
        val_idx.append(gdf.iloc[t_end:v_end].index)
        test_idx.append(gdf.iloc[v_end:].index)

    return (
        df_long.loc[pd.Index(np.concatenate(train_idx))],
        df_long.loc[pd.Index(np.concatenate(val_idx))],
        df_long.loc[pd.Index(np.concatenate(test_idx))]
    )

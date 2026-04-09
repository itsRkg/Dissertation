import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiScaleRainfallDataset(Dataset):
    """
    MEMORY-SAFE multi-scale rainfall dataset.

    Key properties:
    - NO pre-stacked windows
    - Windows are sliced ON-THE-FLY in __getitem__
    - Robust scaling applied consistently
    - Scales to millions of samples

    Returns per item:
        X_3, X_7, X_14, X_30 : (T, F)
        y                   : scalar (scaled)
        month               : int
        station_id, lat, lon
    """

    def __init__(
        self,
        df_long: pd.DataFrame,
        y_median: float,
        y_iqr: float,
        horizons=(3, 7, 14, 30),
        forecast_horizon: int = 1,
        min_days_per_station: int = None,
    ):
        if y_iqr <= 0:
            raise ValueError("y_iqr must be > 0")

        self.horizons = sorted(horizons)
        self.max_horizon = max(self.horizons)
        self.forecast_horizon = forecast_horizon

        self.y_median = float(y_median)
        self.y_iqr = float(y_iqr)

        # --------------------------------------------------
        # Per-station storage (RAW, NOT WINDOWED)
        # --------------------------------------------------
        self.data = {}        # station_id -> Tensor [T, F]
        self.targets = {}     # station_id -> Tensor [T]
        self.months = {}      # station_id -> Tensor [T]
        self.meta = {}        # station_id -> (lat, lon)

        # --------------------------------------------------
        # Global index: (station_id, t)
        # --------------------------------------------------
        self.indices = []

        grouped = df_long.groupby("station_id", sort=False)

        for station_id, gdf in grouped:
            gdf = gdf.sort_values("date").reset_index(drop=True)

            if min_days_per_station is not None and len(gdf) < min_days_per_station:
                continue

            X = self._build_features(gdf)   # (T, F)
            y = gdf["rainfall"].values.astype(np.float32)
            y = (y - self.y_median) / self.y_iqr

            months = gdf["month"].values.astype(np.int64)
            lat = float(gdf["lat"].iloc[0])
            lon = float(gdf["lon"].iloc[0])

            T = len(gdf)

            self.data[station_id] = torch.tensor(X, dtype=torch.float32)
            self.targets[station_id] = torch.tensor(y, dtype=torch.float32)
            self.months[station_id] = torch.tensor(months, dtype=torch.int64)
            self.meta[station_id] = (lat, lon)

            # build valid indices
            min_t = self.max_horizon
            max_t = T - forecast_horizon

            for t in range(min_t, max_t):
                self.indices.append((station_id, t))

    # --------------------------------------------------
    # FEATURE ENGINEERING (SCALED CORRECTLY)
    # --------------------------------------------------
    def _build_features(self, gdf: pd.DataFrame) -> np.ndarray:
        rainfall = gdf["rainfall"].values.astype(np.float32)
        rainfall = (rainfall - self.y_median) / self.y_iqr

        doy = gdf["day_of_year"].values
        month = gdf["month"].values
        year = gdf["year"].values

        doy_sin = np.sin(2 * np.pi * doy / 365.0)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)

        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)

        rel_year = year - year.min()
        scale = max(rel_year.max(), 1)
        year_sin = np.sin(2 * np.pi * rel_year / scale)
        year_cos = np.cos(2 * np.pi * rel_year / scale)

        return np.stack(
            [
                rainfall,
                doy_sin, doy_cos,
                month_sin, month_cos,
                year_sin, year_cos,
            ],
            axis=1,
        ).astype(np.float32)

    # --------------------------------------------------
    # PYTORCH DATASET API
    # --------------------------------------------------
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        station_id, t = self.indices[idx]

        X = self.data[station_id]

        X3  = X[t - 3  : t]
        X7  = X[t - 7  : t]
        X14 = X[t - 14 : t]
        X30 = X[t - 30 : t]

        y = self.targets[station_id][t + self.forecast_horizon - 1]
        month = self.months[station_id][t + self.forecast_horizon - 1]
        lat, lon = self.meta[station_id]

        return X3, X7, X14, X30, y, month, station_id, lat, lon

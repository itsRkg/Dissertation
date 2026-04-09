import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RainfallWindowDataset(Dataset):
    """
    Windowed rainfall dataset with aligned metadata.

    Each item:
        X           : [window_length, feature_dim]
        y           : scalar target
        month       : int (1–12)
        station_id  : int or str
        lat         : float
        lon         : float
    """

    def __init__(
        self,
        df_long: pd.DataFrame,
        window_length: int,
        horizon: int = 1,
        min_days_per_station: int = None,
        label_fn=None,
    ):
        self.samples = []
        self.targets = []
        self.months = []
        self.station_ids = []
        self.lats = []
        self.lons = []

        grouped = df_long.groupby("station_id", sort=False)

        for station_id, gdf in grouped:
            gdf = gdf.sort_values("date").reset_index(drop=True)

            if min_days_per_station is not None and len(gdf) < min_days_per_station:
                continue

            X_station = self._build_features(gdf)
            y_station = gdf["rainfall"].values.astype(np.float32)

            months = gdf["month"].values.astype(np.int64)
            lat = gdf["lat"].iloc[0]
            lon = gdf["lon"].iloc[0]

            T = len(gdf)
            max_start = T - window_length - horizon + 1
            if max_start <= 0:
                continue

            for i in range(max_start):
                target_idx = i + window_length + horizon - 1

                self.samples.append(X_station[i : i + window_length])
                self.targets.append(y_station[target_idx])
                self.months.append(months[target_idx])
                self.station_ids.append(station_id)
                self.lats.append(lat)
                self.lons.append(lon)

        self.X = torch.tensor(np.stack(self.samples), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.targets), dtype=torch.float32)
        self.months = torch.tensor(self.months, dtype=torch.int64)

        self.station_ids = np.array(self.station_ids)
        self.lats = np.array(self.lats, dtype=np.float32)
        self.lons = np.array(self.lons, dtype=np.float32)

    def _build_features(self, gdf):
        rainfall = gdf["rainfall"].values.astype(np.float32)
        doy = gdf["day_of_year"].values
        month = gdf["month"].values
        year = gdf["year"].values

        doy_sin = np.sin(2 * np.pi * doy / 365.0)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)

        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)

        rel_year = year - year.min()
        y_scale = max(rel_year.max(), 1)
        year_sin = np.sin(2 * np.pi * rel_year / y_scale)
        year_cos = np.cos(2 * np.pi * rel_year / y_scale)

        return np.stack(
            [
                rainfall,
                doy_sin, doy_cos,
                month_sin, month_cos,
                year_sin, year_cos,
            ],
            axis=1,
        ).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
            self.months[idx],
            self.station_ids[idx],
            self.lats[idx],
            self.lons[idx],
        )

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RainfallWindowDataset(Dataset):
    """
    Windowed time-series dataset for rainfall prediction.

    Each item:
        X : Tensor [window_length, feature_dim]
        y : Tensor scalar
        month : int (1–12) corresponding to target timestep
    """

    def __init__(
        self,
        df_long: pd.DataFrame,
        window_length: int,
        horizon: int = 1,
        min_days_per_station: int = None,
        device: torch.device = torch.device("cpu"),
        label_fn=None,
    ):
        self.window_length = window_length
        self.horizon = horizon
        self.device = device
        self.label_fn = label_fn

        self.samples = []   # X windows
        self.targets = []   # y values
        self.months = []    # target months

        grouped = df_long.groupby("station_id", sort=False)

        for station_id, gdf in grouped:
            gdf = gdf.sort_values("date").reset_index(drop=True)

            if min_days_per_station is not None and len(gdf) < min_days_per_station:
                continue

            X_station = self._build_features(gdf)
            y_station = gdf["rainfall"].values.astype(np.float32)
            month_station = gdf["month"].values.astype(np.int64)

            T = len(gdf)
            max_start = T - window_length - horizon + 1
            if max_start <= 0:
                continue

            for i in range(max_start):
                target_idx = i + window_length + horizon - 1

                X_win = X_station[i : i + window_length]
                y = y_station[target_idx]
                month = month_station[target_idx]

                if self.label_fn is not None:
                    y = self.label_fn(y, station_id, gdf.iloc[target_idx])

                self.samples.append(X_win)
                self.targets.append(y)
                self.months.append(month)

        # ---- convert to tensors (CPU is correct here) ----
        self.X = torch.tensor(
            np.stack(self.samples),
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            np.array(self.targets),
            dtype=torch.float32,
        )

        self.months = torch.tensor(
            np.array(self.months),
            dtype=torch.int64,
        )

    def _build_features(self, gdf: pd.DataFrame) -> np.ndarray:
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
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.months[idx]

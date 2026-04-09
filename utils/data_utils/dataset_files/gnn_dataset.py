import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader


def build_time_features(dates):
    df = pd.DataFrame(index=dates)
    
    df["day_of_year"] = dates.dayofyear
    df["month"] = dates.month
    df["year"] = dates.year

    # cyclic encodings
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # relative year (for trend)
    rel_year = df["year"] - df["year"].min()
    y_scale = max(rel_year.max(), 1)

    df["year_sin"] = np.sin(2 * np.pi * rel_year / y_scale)
    df["year_cos"] = np.cos(2 * np.pi * rel_year / y_scale)

    return df[[
        "doy_sin", "doy_cos",
        "month_sin", "month_cos",
        "year_sin", "year_cos"
    ]].values  # (T, 6)

def build_feature_tensor(scaled_pivots, use_latent=True):
    feature_order = ['rain'] if not use_latent else [
        'rain', 'temp', 'dew', 'pressure', 'u10', 'v10'
    ]

    arrays = [scaled_pivots[f].values for f in feature_order]  # each (T, N)

    X = np.stack(arrays, axis=-1)  # (T, N, F_dynamic)

    return X, feature_order

def add_time_features(X, time_features):
    # X: (T, N, F)
    # time_features: (T, F_time)

    T, N, _ = X.shape
    F_time = time_features.shape[1]

    # expand → (T, N, F_time)
    time_expanded = np.repeat(time_features[:, None, :], N, axis=1)

    # concat
    X_final = np.concatenate([X, time_expanded], axis=-1)

    return X_final

class SpatioTemporalDataset(Dataset):
    def __init__(self, X, seq_len=7, horizon=1):
        """
        X: (T, N, F_total)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

        self.T = X.shape[0]

    def __len__(self):
        return self.T - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]  # (L, N, F)

        target_idx = idx + self.seq_len + self.horizon - 1

        # rainfall = feature 0 ALWAYS
        y = self.X[target_idx, :, 0]  # (N,)

        return x, y

def build_edge_index(lat, lon, k=5):
    coords = np.vstack([lat, lon]).T

    A = kneighbors_graph(coords, k, mode='connectivity', include_self=False)
    edge_index = np.array(A.nonzero())

    return torch.tensor(edge_index, dtype=torch.long)

# def prepare_data(pivot_path, use_latent):

#     pivots = load_pivots(pivot_path)
#     scaled, scalers = scale_pivots(pivots)

#     X = build_feature_tensor(scaled, use_latent)

#     time_feats = build_time_features(pivots['rain'].index)
#     X = add_time_features(X, time_feats)

#     X_train, X_val, X_test = temporal_split(X, pivots['rain'].index)

#     train_ds = SpatioTemporalDataset(X_train)
#     val_ds = SpatioTemporalDataset(X_val)
#     test_ds = SpatioTemporalDataset(X_test)

#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=32)
#     test_loader = DataLoader(test_ds, batch_size=32)

#     # graph
#     lat = pivots['rain'].columns.map(lambda x: x[0])  # adjust if needed
#     lon = pivots['rain'].columns.map(lambda x: x[1])

#     edge_index = build_edge_index(np.array(lat), np.array(lon))

#     return train_loader, val_loader, test_loader, edge_index
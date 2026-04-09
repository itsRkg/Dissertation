import pandas as pd
import numpy as np
import torch

def temporal_split(df_long, train_frac=0.7, val_frac=0.15):
    train_parts, val_parts, test_parts = [], [], []

    for _, gdf in df_long.groupby("station_id"):
        gdf = gdf.sort_values("date")
        n = len(gdf)

        t_end = int(train_frac * n)
        v_end = int((train_frac + val_frac) * n)

        train_parts.append(gdf.iloc[:t_end])
        val_parts.append(gdf.iloc[t_end:v_end])
        test_parts.append(gdf.iloc[v_end:])

    train_df = pd.concat(train_parts)
    val_df = pd.concat(val_parts)
    test_df = pd.concat(test_parts)

    return train_df, val_df, test_df


def compute_bin_edges(rainfall):
    # Separate positive rainfall
    positive = rainfall[rainfall > 0]

    # Percentile list
    p1 = np.arange(0, 76, 5)
    p2 = np.arange(76, 95, 1)
    p3 = np.arange(94.5, 100, 0.5)
    p3 = p3[p3 <= 99.5]

    percentiles = np.concatenate([p1, p2, p3])

    # Compute percentile values
    edges = np.percentile(positive, percentiles)

    # Remove duplicates (important!)
    edges = np.unique(edges)

    return edges

class RainfallTokenizer:
    def __init__(self, edges):
        self.edges = edges  # percentile-based positive edges
        self.zero_token = 0
        
    @property
    def vocab_size(self):
        # 1 zero bin + len(edges) bins + 1 overflow
        return 1 + len(self.edges) + 1

    def encode(self, values):
        values = np.asarray(values)
        tokens = np.zeros_like(values, dtype=np.int64)

        # Zero rainfall
        zero_mask = values == 0
        tokens[zero_mask] = self.zero_token

        # Positive rainfall
        pos_mask = values > 0
        pos_values = values[pos_mask]

        # Digitize assigns bins 1..len(edges)
        bin_ids = np.digitize(pos_values, self.edges, right=True)

        tokens[pos_mask] = bin_ids + 1  # shift by 1 because 0 reserved for zero

        return tokens

    def decode(self, tokens):
        tokens = np.asarray(tokens)
        values = np.zeros_like(tokens, dtype=np.float32)

        for i, t in enumerate(tokens):
            if t == 0:
                values[i] = 0.0
            elif t <= len(self.edges):
                values[i] = self.edges[t - 1]
            else:
                # overflow bin → use last edge
                values[i] = self.edges[-1]

        return values



def haversine(lat1, lon1, lat2, lon2):

    R = 6371  # earth radius km

    lat1, lon1, lat2, lon2 = map(np.radians,
        [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    return R*c

# def build_edge_index_radius(lat, lon, threshold_km=100):
#     N = len(lat)

#     edges = []

#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 continue

#             dist = haversine(lat[i], lon[i], lat[j], lon[j])

#             if dist <= threshold_km:
#                 edges.append([i, j])

#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

#     return edge_index

def get_lat_lon_aligned(pivot_df, station_df):
    station_df = station_df.set_index("station_id")

    lat = []
    lon = []

    for col in pivot_df.columns:
        lat.append(station_df.loc[col, "lat"])
        lon.append(station_df.loc[col, "lon"])

    return np.array(lat), np.array(lon)


def build_edge_index_radius(lat, lon, threshold_km=100):
    edges = []
    N = len(lat)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            dist = haversine(lat[i], lon[i], lat[j], lon[j])

            if dist <= threshold_km:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index



def load_pivots(path):
    return {
        'rain': pd.read_parquet(path + "rain_pivot.parquet"),
        'temp': pd.read_parquet(path + "temp_pivot.parquet"),
        'dew': pd.read_parquet(path + "dew_pivot.parquet"),
        'pressure': pd.read_parquet(path + "pressure_pivot.parquet"),
        'u10': pd.read_parquet(path + "u10_pivot.parquet"),
        'v10': pd.read_parquet(path + "v10_pivot.parquet"),
    }


def scale_pivots(pivots, train_end='2015-12-31'):
    scaled = {}
    scalers = {}

    for name, df in pivots.items():
        train = df.loc[:train_end].values

        mu = np.nanmean(train)
        sigma = np.nanstd(train)
        sigma = sigma if sigma != 0 else 1e-8

        scaled[name] = (df - mu) / sigma
        scaled[name] = scaled[name].ffill().bfill().fillna(0)

        scalers[name] = (mu, sigma)

    return scaled, scalers

def temporal_split(X, dates, train_frac=0.7, val_frac=0.15):
    T = len(dates)

    t_end = int(train_frac * T)
    v_end = int((train_frac + val_frac) * T)

    return (
        X[:t_end],
        X[t_end:v_end],
        X[v_end:]
    )
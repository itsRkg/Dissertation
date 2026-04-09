import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def bias(y, yhat):
    return np.mean(yhat - y)

def nrmse(y, yhat):
    return rmse(y, yhat) / (np.mean(y) + 1e-6)


def seasonal_metrics(df_eval, yhat, season_months):
    mask = df_eval["month"].isin(season_months)
    y = df_eval.loc[mask, "rainfall"].values
    y_pred = yhat[mask.values]

    return {
        "RMSE": rmse(y, y_pred),
        "MAE": mae(y, y_pred),
        "Bias": bias(y, y_pred)
    }


def plot_rmse_vs_window(window_lengths, rmses):
    plt.figure(figsize=(6, 4))
    plt.plot(window_lengths, rmses, marker="o")
    plt.xlabel("Window Length (days)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Window Length")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_seasonal_rmse(window_lengths, monsoon_rmse, non_monsoon_rmse):
    plt.figure(figsize=(6, 4))
    plt.plot(window_lengths, monsoon_rmse, marker="o", label="Monsoon")
    plt.plot(window_lengths, non_monsoon_rmse, marker="o", label="Non-Monsoon")
    plt.xlabel("Window Length (days)")
    plt.ylabel("RMSE")
    plt.title("Seasonal RMSE vs Window Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_monthwise_rmse_heatmap(month_rmse_matrix, window_lengths, months):
    plt.figure(figsize=(10, 4))
    sns.heatmap(
        month_rmse_matrix,
        xticklabels=months,
        yticklabels=window_lengths,
        annot=False,
        cmap="viridis"
    )
    plt.xlabel("Month")
    plt.ylabel("Window Length (days)")
    plt.title("Month-wise RMSE Heatmap")
    plt.tight_layout()
    plt.show()

def stationwise_metrics(df_eval):
    return (
        df_eval
        .groupby("station_id")
        .apply(lambda g: pd.Series({
            "RMSE": rmse(g.y.values, g.yhat.values),
            "MAE": mae(g.y.values, g.yhat.values),
            "Bias": bias(g.y.values, g.yhat.values),
            "Count": len(g)
        }))
        .reset_index()
    )


def rmse_vs_latitude(df_eval, bins=20):
    df_eval["lat_bin"] = pd.cut(df_eval["lat"], bins=bins)

    return (
        df_eval
        .groupby("lat_bin")
        .apply(lambda g: rmse(g.y.values, g.yhat.values))
        .reset_index(name="RMSE")
    )


def spatial_rmse_grid(df_eval, lat_bins=20, lon_bins=20):
    df_eval["lat_bin"] = pd.cut(df_eval["lat"], lat_bins)
    df_eval["lon_bin"] = pd.cut(df_eval["lon"], lon_bins)

    return (
        df_eval
        .groupby(["lat_bin", "lon_bin"])
        .apply(lambda g: rmse(g.y.values, g.yhat.values))
        .unstack()
    )
    
def seasonal_subset(df_eval, months):
    return df_eval[df_eval["month"].isin(months)]

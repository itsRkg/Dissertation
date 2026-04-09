import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

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


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_monthly_rmse(preds, targets, doy):
    df = pd.DataFrame({
        "pred": preds,
        "true": targets,
        "doy": doy
    })

    # Convert DOY → month
    df["date"] = pd.to_datetime(df["doy"], format="%j", errors="coerce")
    df["month"] = df["date"].dt.month

    monthly_rmse = (
        df.groupby("month")
        .apply(lambda x: np.sqrt(mean_squared_error(x["true"], x["pred"])))
    )

    plt.figure(figsize=(10, 5))
    monthly_rmse.plot(marker="o")
    plt.title("Monthly RMSE")
    plt.xlabel("Month")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()

    return monthly_rmse

def plot_true_vs_pred(preds, targets):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.3)
    plt.plot([0, max(targets)], [0, max(targets)], 'r--')
    plt.xlabel("True Rainfall")
    plt.ylabel("Predicted Rainfall")
    plt.title("True vs Predicted Rainfall")
    plt.grid(True)
    plt.show()

def plot_time_series_overlay(preds, targets, n=500):
    plt.figure(figsize=(14, 5))
    plt.plot(targets[:n], label="True")
    plt.plot(preds[:n], label="Predicted", alpha=0.7)
    plt.legend()
    plt.title("True vs Predicted (First {} Samples)".format(n))
    plt.show()

def monsoon_analysis(preds, targets, doy):
    df = pd.DataFrame({
        "pred": preds,
        "true": targets,
        "doy": doy
    })

    df["date"] = pd.to_datetime(df["doy"], format="%j", errors="coerce")
    df["month"] = df["date"].dt.month

    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9])

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    results = {}

    for label, subset in df.groupby("is_monsoon"):
        rmse = np.sqrt(mean_squared_error(subset["true"], subset["pred"]))
        mae = mean_absolute_error(subset["true"], subset["pred"])
        bias = np.mean(subset["pred"] - subset["true"])

        key = "Monsoon" if label else "Non-Monsoon"
        results[key] = {
            "RMSE": rmse,
            "MAE": mae,
            "Bias": bias
        }

    return results


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





import numpy as np
import torch


@torch.no_grad()
def collect_regression_predictions(model, loader, tokenizer, device):
    model.eval()

    all_preds = []
    all_targets = []
    all_doy = []

    for batch in loader:
        tokens = batch["tokens"].to(device)
        doy = batch["doy"].to(device)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_doy = doy[:, :-1]

        logits = model(input_tokens, input_doy)
        preds = torch.argmax(logits, dim=-1)

        preds_np = preds.cpu().numpy().reshape(-1)
        targets_np = target_tokens.cpu().numpy().reshape(-1)
        doy_np = batch["doy"][:, 1:].reshape(-1).numpy()

        preds_rain = tokenizer.decode(preds_np)
        targets_rain = tokenizer.decode(targets_np)

        all_preds.append(preds_rain)
        all_targets.append(targets_rain)
        all_doy.append(doy_np)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_doy)
    )

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

@torch.no_grad()
def evaluate_classification(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        tokens = batch["tokens"].to(device)
        doy = batch["doy"].to(device)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_doy = doy[:, :-1]

        logits = model(input_tokens, input_doy)
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu().reshape(-1))
        all_targets.append(target_tokens.cpu().reshape(-1))

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    acc = accuracy_score(all_targets, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        all_targets,
        all_preds,
        zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "report": report
    }

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@torch.no_grad()
def evaluate_regression(model, loader, tokenizer, device):
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        tokens = batch["tokens"].to(device)
        doy = batch["doy"].to(device)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_doy = doy[:, :-1]

        logits = model(input_tokens, input_doy)
        preds = torch.argmax(logits, dim=-1)

        # Convert to numpy
        preds_np = preds.cpu().numpy().reshape(-1)
        targets_np = target_tokens.cpu().numpy().reshape(-1)

        # Decode tokens → rainfall
        preds_rain = tokenizer.decode(preds_np)
        targets_rain = tokenizer.decode(targets_np)

        all_preds.append(preds_rain)
        all_targets.append(targets_rain)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    bias = np.mean(all_preds - all_targets)
    nrmse = rmse / (np.mean(all_targets) + 1e-8)
    r2 = r2_score(all_targets, all_preds)
    med_ae = np.median(np.abs(all_preds - all_targets))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "NRMSE": nrmse,
        "R2": r2,
        "MedianAE": med_ae
    }


def extreme_metrics(targets, preds, threshold):
    targets_extreme = targets >= threshold
    preds_extreme = preds >= threshold

    

    return {
        "extreme_precision": precision_score(targets_extreme, preds_extreme),
        "extreme_recall": recall_score(targets_extreme, preds_extreme),
        "extreme_f1": f1_score(targets_extreme, preds_extreme),
    }

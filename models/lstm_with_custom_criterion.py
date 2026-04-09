import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from utils.metric_utils.metrics import rmse, mae, bias, nrmse
from utils.logging_utils import setup_logger


"""
Alternative LSTM trainer that allows passing the loss criterion as user input.

Usage:
    model = RainfallLSTM(input_dim=..., hidden_dim=...)
    trained = train_model(train_ds, val_ds, model, device, criterion='mse')

`criterion` accepts either:
 - a string: 'mse', 'mae', 'huber' (SmoothL1)
 - an instance of `nn.Module` (e.g., `nn.MSELoss()`)
 - a callable loss function that takes (preds, targets) and returns a tensor

This file mirrors the API from `baseline_lstm.py` but exposes the criterion option.
"""


# ============================================================
# MODEL
# ============================================================


class RainfallLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # (B, H)
        return self.head(h_last).squeeze(-1)


# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, n = 0.0, 0

    for X, y, *_ in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(X)
            loss = criterion(preds, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    preds_all, targets_all = [], []

    for X, y, *_ in loader:
        X = X.to(device)
        y = y.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(X)
            loss = criterion(preds, y)

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

        preds_all.append(preds.cpu().numpy())
        targets_all.append(y.cpu().numpy())

    return (
        total_loss / n,
        np.concatenate(preds_all),
        np.concatenate(targets_all),
    )


# ============================================================
# TRAINING DRIVER (LOGGER + TQDM + CHECKPOINTS) - custom criterion
# ============================================================


def _resolve_criterion(criterion):
    """Return an nn.Module loss given user input."""
    if criterion is None:
        return nn.MSELoss()

    if isinstance(criterion, str):
        name = criterion.lower()
        if name in ("mse", "mse_loss"):
            return nn.MSELoss()
        if name in ("mae", "l1", "l1_loss"):
            return nn.L1Loss()
        if name in ("huber", "smoothl1", "smooth_l1"):
            return nn.SmoothL1Loss()
        raise ValueError(f"Unknown criterion string: {criterion}")

    # Already an nn.Module instance
    if isinstance(criterion, nn.Module):
        return criterion

    # Callable (user provided function)
    if callable(criterion):
        # wrap into a small callable that matches nn.Module API
        class _FuncWrapper:
            def __call__(self, preds, targets):
                return criterion(preds, targets)

        return _FuncWrapper()

    raise ValueError("criterion must be None, str, nn.Module or callable")


def train_model(
    train_ds,
    val_ds,
    model,
    device,
    batch_size=64,
    epochs=30,
    lr=1e-3,
    criterion=None,
    save_dir="./checkpoints",
    log_dir="./logs",
    experiment_name="baseline",
):
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(log_dir, experiment_name)
    logger.info("Starting training")
    logger.info(f"Device: {device}")

    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_fn = _resolve_criterion(criterion)
    # If criterion is an nn.Module, use it directly in calls; if it's a simple callable,
    # our wrapper ensures it can be invoked like criterion(preds, targets).

    scaler = GradScaler(enabled=(device.type == "cuda"))

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")

    for epoch in epoch_bar:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_fn, scaler, device
        )

        val_loss, _, _ = evaluate(
            model, val_loader, criterion_fn, device
        )

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
        )

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )

        logger.info(f"Checkpoint saved: {ckpt_path}")

    logger.info("Training completed")
    return model


# ============================================================
# TEST EVALUATION (WITH SEASONAL ANALYSIS)
# ============================================================


@torch.no_grad()
def collect_test_predictions(test_ds, model, device):
    model.to(device)
    model.eval()

    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    rows = []

    for X, y, month, station_id, lat, lon in loader:
        X = X.to(device)
        y = y.to(device)

        preds = model(X).cpu().numpy()
        y = y.cpu().numpy()

        for i in range(len(y)):
            rows.append({
                "y": y[i],
                "yhat": preds[i],
                "month": int(month[i]),
                "station_id": station_id[i],
                "lat": float(lat[i]),
                "lon": float(lon[i]),
            })

    return pd.DataFrame(rows)
    

@torch.no_grad()
def evaluate_on_test(test_ds, model, device):
    logger = setup_logger("logs", "test_evaluation")

    model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        pin_memory=False,
    )

    preds_all, targets_all, months_all = [], [], []

    for X, y, month ,*_ in test_loader:
        X = X.to(device)
        y = y.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(X)

        preds_all.append(preds.cpu().numpy())
        targets_all.append(y.cpu().numpy())
        months_all.append(month.numpy())

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    months = np.concatenate(months_all)

    results = {
        "RMSE": rmse(targets, preds),
        "MAE": mae(targets, preds),
        "Bias": bias(targets, preds),
        "NRMSE": nrmse(targets, preds),
    }

    logger.info("Overall Test Metrics")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")

    seasons = {
        "monsoon": [6, 7, 8, 9],
        "non_monsoon": [1, 2, 3, 4, 5, 10, 11, 12],
    }

    for name, season_months in seasons.items():
        mask = np.isin(months, season_months)
        season_rmse = rmse(targets[mask], preds[mask])
        results[f"{name}_RMSE"] = season_rmse
        logger.info(f"{name.upper()} RMSE: {season_rmse:.4f}")

    return results

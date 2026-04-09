import os
import inspect
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
LSTM trainer that expects dataloader items to include `month` as the third field.
It calls the criterion as: criterion(preds, targets, month).
"""


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
# TRAIN / EVAL LOOPS (supports extra-arg criteria)
# ============================================================


def _wrap_criterion(criterion):
    """Return a callable-like object and mark whether it expects extra args.

    The returned object supports being called as `criterion(preds, targets, *extra)`.
    An attribute `expects_extra_args` is set True when the underlying function
    accepts >2 parameters (preds, targets, ...).
    """

    # None -> default MSE
    if criterion is None:
        c = nn.MSELoss()
        c.expects_extra_args = False
        c.extra_arg_count = 0
        return c

    # strings -> standard losses
    if isinstance(criterion, str):
        name = criterion.lower()
        if name in ("mse", "mse_loss"):
            c = nn.MSELoss()
        elif name in ("mae", "l1", "l1_loss"):
            c = nn.L1Loss()
        elif name in ("huber", "smoothl1", "smooth_l1"):
            c = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown criterion string: {criterion}")
        c.expects_extra_args = False
        c.extra_arg_count = 0
        return c

    # nn.Module instance: inspect its forward signature
    if isinstance(criterion, nn.Module):
        sig = inspect.signature(criterion.forward)
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if has_varargs:
            criterion.expects_extra_args = True
            criterion.extra_arg_count = None  # accept all extras
        else:
            # subtract preds + targets
            extra_count = max(0, len(params) - 2)
            criterion.expects_extra_args = extra_count > 0
            criterion.extra_arg_count = extra_count
        return criterion

    # plain callable: inspect signature
    if callable(criterion):
        sig = inspect.signature(criterion)
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if has_varargs:
            expects_extra = True
            extra_count = None
        else:
            extra_count = max(0, len(params) - 2)
            expects_extra = extra_count > 0

        class _FuncWrapper:
            def __init__(self, fn, expects_extra, extra_count):
                self.fn = fn
                self.expects_extra_args = expects_extra
                self.extra_arg_count = extra_count

            def __call__(self, preds, targets, *extra):
                if self.expects_extra_args:
                    if self.extra_arg_count is None:
                        return self.fn(preds, targets, *extra)
                    if len(extra) < self.extra_arg_count:
                        raise ValueError(
                            f"Criterion expects {self.extra_arg_count} extra args, "
                            f"but got {len(extra)}."
                        )
                    return self.fn(preds, targets, *extra[: self.extra_arg_count])
                return self.fn(preds, targets)

        return _FuncWrapper(criterion, expects_extra, extra_count)

    raise ValueError("criterion must be None, str, nn.Module or callable")


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc="Train", leave=False):
        # Expect loader to yield (X, y, month, ...)
        X, y, month, *_ = batch

        X = X.to(device)
        y = y.to(device)
        month = month.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(X)
            loss = criterion(preds, y, month)

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

    for batch in tqdm(loader, desc="Eval", leave=False):
        X, y, month, *_ = batch

        X = X.to(device)
        y = y.to(device)
        month = month.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(X)
            loss = criterion(preds, y, month)

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
# TRAINING DRIVER (LOGGER + TQDM + CHECKPOINTS) - seasonal criterion
# ============================================================


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
    criterion_fn = _wrap_criterion(criterion)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_fn, scaler, device
        )

        val_loss, _, _ = evaluate(
            model, val_loader, criterion_fn, device
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
# TEST UTILITIES (same as other trainer)
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

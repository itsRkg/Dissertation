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


# ============================================================
# ATTENTION MODULES
# ============================================================

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, H):
        # H: (B, T, H)
        attn_logits = self.score(H)
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (attn_weights * H).sum(dim=1)


class CrossScaleAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, Z):
        # Z: (B, S, H)
        attn_logits = self.score(Z)
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (attn_weights * Z).sum(dim=1)


# ============================================================
# MULTI-SCALE LSTM MODEL
# ============================================================

class MultiScaleLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        scales=(3, 7, 14, 30),
        use_self_attention=False,
        use_cross_attention=False,
    ):
        super().__init__()

        self.scales = scales
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention

        self.lstms = nn.ModuleDict({
            str(s): nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            for s in scales
        })

        if use_self_attention:
            self.self_attn = nn.ModuleDict({
                str(s): TemporalSelfAttention(hidden_dim)
                for s in scales
            })

        if use_cross_attention:
            self.cross_attn = CrossScaleAttention(hidden_dim)

        out_dim = hidden_dim if use_cross_attention else hidden_dim * len(scales)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, X3, X7, X14, X30):
        Xs = {3: X3, 7: X7, 14: X14, 30: X30}
        reps = []

        for s in self.scales:
            H, (h_n, _) = self.lstms[str(s)](Xs[s])

            if self.use_self_attention:
                z = self.self_attn[str(s)](H)
            else:
                z = h_n[-1]

            reps.append(z)

        if self.use_cross_attention:
            Z = torch.stack(reps, dim=1)
            z_final = self.cross_attn(Z)
        else:
            z_final = torch.cat(reps, dim=1)

        return self.head(z_final).squeeze(-1)


# ============================================================
# CRITERION WRAPPER (UNCHANGED)
# ============================================================

def _wrap_criterion(criterion):
    if criterion is None:
        c = nn.MSELoss()
        c.expects_extra_args = False
        return c

    if isinstance(criterion, nn.Module):
        sig = inspect.signature(criterion.forward)
        params = list(sig.parameters.values())[1:]
        c = criterion
        c.expects_extra_args = len(params) > 2
        return c

    raise ValueError("Unsupported criterion type")


# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc="Train", leave=False):
        X3, X7, X14, X30, y, month, *_ = batch

        X3, X7, X14, X30 = X3.to(device), X7.to(device), X14.to(device), X30.to(device)
        y = y.to(device)
        month = month.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(X3, X7, X14, X30)
            loss = criterion(preds, y, month)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    preds_all, targets_all = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        X3, X7, X14, X30, y, month, *_ = batch

        X3, X7, X14, X30 = X3.to(device), X7.to(device), X14.to(device), X30.to(device)
        y = y.to(device)
        month = month.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(X3, X7, X14, X30)
            loss = criterion(preds, y, month)

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        preds_all.append(preds.cpu().numpy())
        targets_all.append(y.cpu().numpy())

    return total_loss / n, np.concatenate(preds_all), np.concatenate(targets_all)


# ============================================================
# TRAINING DRIVER
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
    experiment_name="multiscale_lstm",
):
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(log_dir, experiment_name)

    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,      # try 2–4 depending on CPU
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(save_dir, f"epoch_{epoch}.pt"),
        )

    return model


# ============================================================
# TEST EVALUATION (IN SCALED SPACE)
# ============================================================

@torch.no_grad()
def evaluate_on_test(test_ds, model, device):
    logger = setup_logger("logs", "test_evaluation")

    model.to(device)
    model.eval()

    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    preds_all, targets_all, months_all = [], [], []

    for X3, X7, X14, X30, y, month, *_ in loader:
        X3, X7, X14, X30 = X3.to(device), X7.to(device), X14.to(device), X30.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(X3, X7, X14, X30)

        preds_all.append(preds.cpu().numpy())
        targets_all.append(y.numpy())
        months_all.append(month.numpy())

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    months = np.concatenate(months_all)

    results = {
        "RMSE_scaled": rmse(targets, preds),
        "MAE_scaled": mae(targets, preds),
        "Bias_scaled": bias(targets, preds),
        "NRMSE_scaled": nrmse(targets, preds),
    }

    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")

    return results

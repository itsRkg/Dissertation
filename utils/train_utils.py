import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler # pyright: ignore[reportPrivateImportUsage]
from torch_geometric.nn import GCNConv

from utils.logging_utils import setup_logger

# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, edge_index, device):
    model.train()
    total_loss, n = 0.0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(x, edge_index)
            loss = criterion(preds, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, edge_index, device):
    model.eval()
    total_loss, n = 0.0, 0

    preds_all, targets_all = [], []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(x, edge_index)
            loss = criterion(preds, y)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        preds_all.append(preds.cpu().numpy())
        targets_all.append(y.cpu().numpy())

    return (
        total_loss / n,
        np.concatenate(preds_all),
        np.concatenate(targets_all),
    )


# ============================================================
# TRAINING DRIVER
# ============================================================

def train_model(
    train_loader,
    val_loader,
    model,
    edge_index,
    device,
    epochs=30,
    lr=1e-3,
    criterion=None,
    save_dir="./checkpoints",
    log_dir="./logs",
    experiment_name="gcn_gru",
):
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(log_dir, experiment_name)
    logger.info("Starting GCN+GRU training")
    logger.info(f"Device: {device}")
    logger.info("Experiment Config:")
    logger.info(f"LR: {lr}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Criterion: {criterion}")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if criterion is None:
        criterion = nn.MSELoss()

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, scaler, edge_index, device
        )

        val_loss, _, _ = evaluate(
            model, val_loader,
            criterion, edge_index, device
        )

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        # ===== SAVE EVERY EPOCH =====
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, ckpt_path)

        logger.info(f"Checkpoint saved: {ckpt_path}")

        # ===== BEST MODEL TRACKING =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = os.path.join(save_dir, f"{experiment_name}_best.pt")
            torch.save(model.state_dict(), best_path)

            logger.info(f"✅ New best model saved: {best_path}")

    logger.info("Training completed")

    return model
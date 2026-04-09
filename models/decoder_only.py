import os
import math
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from utils.logging_utils import setup_logger


# ============================================================
# TRANSFORMER BLOCK (Pre-LN GPT Style)
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal_mask):
        # Self-attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + attn_out

        # Feedforward
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x


# ============================================================
# RAINFALL GPT MODEL
# ============================================================

class RainfallGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=128,
        n_heads=4,
        n_layers=4,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.doy_emb = nn.Embedding(367, d_model)  # DOY 1-366

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, doy):
        B, T = tokens.shape
        device = tokens.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        x = (
            self.token_emb(tokens)
            + self.pos_emb(pos)
            + self.doy_emb(doy)
        )

        x = self.dropout(x)

        # Causal mask (T x T)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        ).bool()

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.ln_final(x)

        logits = self.head(x)

        return logits


# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================

def train_one_epoch(model, loader, optimizer, scaler, device, vocab_size):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        tokens = batch["tokens"].to(device)
        doy = batch["doy"].to(device)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_doy = doy[:, :-1]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(input_tokens, input_doy)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_tokens.reshape(-1)
            )

        scaler.scale(loss).backward()

        # Important for transformer stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * target_tokens.numel()
        total_tokens += target_tokens.numel()

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, loader, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        tokens = batch["tokens"].to(device)
        doy = batch["doy"].to(device)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_doy = doy[:, :-1]

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(input_tokens, input_doy)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_tokens.reshape(-1)
            )

        total_loss += loss.item() * target_tokens.numel()
        total_tokens += target_tokens.numel()

    return total_loss / total_tokens


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
    lr=3e-4,
    save_dir="./checkpoints",
    log_dir="./logs",
    experiment_name="rainfall_gpt",
):

    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(log_dir, experiment_name)

    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            model.vocab_size
        )

        val_loss = evaluate(
            model,
            val_loader,
            device,
            model.vocab_size
        )

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train CE: {train_loss:.6f} | "
            f"Val CE: {val_loss:.6f}"
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
# TEST EVALUATION (Cross-Entropy Only)
# ============================================================

@torch.no_grad()
def evaluate_on_test(test_ds, model, device, batch_size=128):

    logger = setup_logger("logs", "test_evaluation_gpt")

    model.to(device)
    model.eval()

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc="Test", leave=False):
        tokens = batch["tokens"].to(device)
        doy = batch["doy"].to(device)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_doy = doy[:, :-1]

        logits = model(input_tokens, input_doy)

        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            target_tokens.reshape(-1)
        )

        total_loss += loss.item() * target_tokens.numel()
        total_tokens += target_tokens.numel()

    test_ce = total_loss / total_tokens

    logger.info(f"Test Cross-Entropy: {test_ce:.6f}")

    return test_ce

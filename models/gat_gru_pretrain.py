"""
gat_gru_pretrain.py — masked-reconstruction pretext model for track_c Tier 2A (Session 2).

NEW, additive model. Does NOT modify models/gat_gru.py.

Design (session_plan.md §5 Session 2):
  - The ENCODER (self.gat + self.gru) is byte-for-byte the same definition as
    models.gat_gru.GAT_GRU_Model, so its trained weights transfer 1:1 to the finetune
    model in Session 3 (just `finetune.gat.load_state_dict(pre.gat.state_dict())` etc.).
  - Instead of collapsing to the last timestep + a 1-output head, we keep the FULL GRU
    output sequence (B*N, L, hidden) and attach a small 2-layer MLP decoder that
    reconstructs `n_recon` physical channels at EVERY (timestep, node):
        decoder(out) -> (B*N, L, n_recon) -> (B, L, N, n_recon)
  - Pretext task = BERT/denoising-style masked reconstruction: the caller zeroes a
    random subset of the physical-channel cells in the INPUT (cyclic-time and static
    features stay visible and are NOT reconstruction targets), feeds the whole window
    through the encoder, and the MSE loss is taken ONLY on the masked physical cells.
    (We do NOT drop tokens, so this is the low-mask-ratio BERT regime ~15%, not the
    75% ViT-MAE token-drop regime.)

The decoder is discarded at finetune; only gat + gru are transferred.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT_GRU_Pretrain(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, heads=4, n_recon=6):
        super().__init__()
        self.heads = heads
        self.n_recon = n_recon

        # --- encoder: IDENTICAL definition to GAT_GRU_Model (for clean weight transfer) ---
        self.gat = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.gru = nn.GRU(
            input_size=hidden_dim * heads,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        # --- pretext head: 2-layer MLP decoder reconstructing n_recon channels ---
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_recon),
        )

    def _expand_edge_index(self, edge_index, B, N):
        return torch.cat([edge_index + b * N for b in range(B)], dim=1)

    def encode(self, x, edge_index):
        """x: (B, L, N, F) -> per-node GRU output (B*N, L, hidden_dim) + (B, L, N)."""
        B, L, N, F = x.shape
        eib = self._expand_edge_index(edge_index, B, N)
        outs = []
        for t in range(L):
            h_t = torch.relu(self.gat(x[:, t].reshape(B * N, F), eib)).reshape(B, N, -1)
            outs.append(h_t)
        h = torch.stack(outs, dim=1)                 # (B, L, N, hidden*heads)
        h = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        out, _ = self.gru(h)                         # (B*N, L, hidden_dim)
        return out, B, L, N

    def forward(self, x, edge_index):
        """Reconstruct the n_recon physical channels at every (timestep, node).
        Returns (B, L, N, n_recon)."""
        out, B, L, N = self.encode(x, edge_index)
        rec = self.decoder(out)                      # (B*N, L, n_recon)
        rec = rec.reshape(B, N, L, self.n_recon).permute(0, 2, 1, 3)  # (B, L, N, n_recon)
        return rec

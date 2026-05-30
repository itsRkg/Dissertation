"""
gat_gru_joint.py - Tier 2B JOINT masked-reconstruction + forecast model (track_c, Session 3).

NEW, additive model. Does NOT modify gat_gru.py or gat_gru_pretrain.py.

Design (docs/track_3_rl_reuse/session_plan.md §5 Session 3, step 2 - Tier 2B):
  - ONE shared GAT-GRU encoder (self.gat + self.gru), byte-for-byte the same definition
    as models.gat_gru.GAT_GRU_Model and models.gat_gru_pretrain.GAT_GRU_Pretrain, so the
    three tiers (0b / 2A / 2B) are mechanism-comparable on an identical backbone.
  - TWO heads off the shared encoder:
      * forecast head : Linear(hidden, 1) on the LAST-timestep GRU output -> (B, N)
                        next-day rain. Identical to GAT_GRU_Model.fc.
      * recon decoder : 2-layer MLP on the FULL GRU sequence -> (B, L, N, n_recon)
                        reconstructing the n_recon physical channels at every
                        (timestep, node). Identical to GAT_GRU_Pretrain.decoder.

DISTINCT FROM TIER 2A (do NOT conflate):
  - Tier 2A = TWO stages: pretrain the encoder on masked recon (exp_18 pretext), THEN
    finetune a fresh forecast head on clean input (exp_18 finetune).
  - Tier 2B = ONE stage, trained FROM SCRATCH, with the combined objective
        L = MSE(forecast, next-day rain) + lambda_recon * MSE(recon, masked phys cells).
    This is the STGCL "pretrain vs joint" question; masked-recon has no published weather
    precedent for the joint variant, so this pairing is the chapter's novel mini-contribution.

The caller masks the physical channels of the INPUT window (see exp_19 notebook); both heads
read the SAME (masked) encoded representation in a single forward pass. At inference the input
is fed CLEAN (no masking) and only the forecast head is used - so 2B test RMSE is directly
comparable to Tier 0b and the Tier 2A finetune.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT_GRU_Joint(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, heads=4, n_recon=6):
        super().__init__()
        self.heads = heads
        self.n_recon = n_recon

        # --- shared encoder: IDENTICAL definition to GAT_GRU_Model / GAT_GRU_Pretrain ---
        self.gat = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.gru = nn.GRU(
            input_size=hidden_dim * heads,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        # --- forecast head: last timestep -> next-day rain (same as GAT_GRU_Model.fc) ---
        self.fc = nn.Linear(hidden_dim, 1)
        # --- recon head: full sequence -> n_recon channels (same as GAT_GRU_Pretrain.decoder) ---
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_recon),
        )

    def _expand_edge_index(self, edge_index, B, N):
        return torch.cat([edge_index + b * N for b in range(B)], dim=1)

    def encode(self, x, edge_index):
        """x: (B, L, N, F) -> per-node GRU output (B*N, L, hidden_dim) + dims (B, L, N)."""
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
        """Returns (forecast (B, N), recon (B, L, N, n_recon))."""
        out, B, L, N = self.encode(x, edge_index)
        # forecast head: last timestep (identical collapse to GAT_GRU_Model)
        fc = self.fc(out[:, -1, :]).reshape(B, N)                          # (B, N)
        # recon head: full sequence (identical to GAT_GRU_Pretrain)
        rec = self.decoder(out)                                            # (B*N, L, n_recon)
        rec = rec.reshape(B, N, L, self.n_recon).permute(0, 2, 1, 3)       # (B, L, N, n_recon)
        return fc, rec

"""
gat_gru_multilayer.py — depth-configurable GAT-GRU (track_c depth ablation).

NEW, additive model. Does NOT modify models/gat_gru.py.

`num_layers` stacks that many GATConv spatial message-passing layers (ReLU between
them) per timestep, then the same single-layer GRU + linear head as gat_gru.py.

IMPORTANT: with num_layers=1 this is architecturally IDENTICAL to
models/gat_gru.GAT_GRU_Model (one GATConv(in, hidden, heads, concat=True) -> ReLU ->
GRU(hidden*heads, hidden) -> Linear(hidden, 1)). So the existing exp_13/Tier-0 runs
ARE the 1-layer point of this ablation; this file only adds the >=2-layer variants.

Each GAT layer outputs hidden_dim*heads (concat=True), so the GRU input size is fixed
at hidden_dim*heads regardless of depth (matches gat_gru.py). `residual=True` adds a
skip connection between consecutive hidden GAT layers (same dim) — off by default for
a clean 1-vs-2 depth comparison; turn on when trying num_layers>=3 to fight
oversmoothing (see session_plan.md depth-ablation note).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT_GRU_MultiLayer(nn.Module):
    def __init__(self, in_channels, hidden_dim, heads=4, num_layers=1, residual=False):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.num_layers = num_layers
        self.heads = heads
        self.residual = residual

        self.gats = nn.ModuleList()
        self.gats.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True))
        for _ in range(num_layers - 1):
            # hidden GAT layers: (hidden*heads) -> (hidden*heads)
            self.gats.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))

        self.gru = nn.GRU(
            input_size=hidden_dim * heads,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def _expand_edge_index(self, edge_index, B, N):
        return torch.cat([edge_index + b * N for b in range(B)], dim=1)

    def _spatial(self, x_t, edge_index_batch):
        """Apply the GAT stack to one timestep. x_t: (B*N, F) -> (B*N, hidden*heads)."""
        h = torch.relu(self.gats[0](x_t, edge_index_batch))
        for gat in self.gats[1:]:
            h_new = torch.relu(gat(h, edge_index_batch))
            h = h + h_new if self.residual else h_new
        return h

    def forward(self, x, edge_index):
        B, L, N, F = x.shape
        edge_index_batch = self._expand_edge_index(edge_index, B, N)

        outputs = []
        for t in range(L):
            x_t = x[:, t].reshape(B * N, F)
            h_t = self._spatial(x_t, edge_index_batch).reshape(B, N, -1)
            outputs.append(h_t)

        h = torch.stack(outputs, dim=1)         # (B, L, N, hidden*heads)
        h = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)

        out, _ = self.gru(h)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.reshape(B, N)

    @torch.no_grad()
    def node_embeddings(self, x, edge_index):
        """Post-GAT-stack node embeddings at the final timestep: (B, N, hidden*heads).
        Convenience for the oversmoothing diagnostic (utils.metric_utils.embedding_diag)."""
        B, L, N, F = x.shape
        edge_index_batch = self._expand_edge_index(edge_index, B, N)
        x_t = x[:, L - 1].reshape(B * N, F)
        return self._spatial(x_t, edge_index_batch).reshape(B, N, -1)

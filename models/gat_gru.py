import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT_GRU_Model(nn.Module):
    def __init__(self, in_channels, hidden_dim, heads=4):
        super().__init__()

        self.heads = heads

        self.gat = GATConv(
            in_channels,
            hidden_dim,
            heads=heads,
            concat=True
        )

        self.gru = nn.GRU(
            input_size=hidden_dim * heads,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        B, L, N, F = x.shape

        edge_index_batch = self._expand_edge_index(edge_index, B, N)

        gat_outputs = []

        for t in range(L):
            x_t = x[:, t]  # (B, N, F)
            x_t = x_t.reshape(B * N, F)

            h_t = self.gat(x_t, edge_index_batch)
            h_t = torch.relu(h_t)

            h_t = h_t.reshape(B, N, -1)
            gat_outputs.append(h_t)

        h = torch.stack(gat_outputs, dim=1)  # (B, L, N, H*)

        h = h.permute(0, 2, 1, 3)  # (B, N, L, H*)
        h = h.reshape(B * N, L, -1)

        out, _ = self.gru(h)

        out = out[:, -1, :]
        out = self.fc(out)

        return out.reshape(B, N)

    def _expand_edge_index(self, edge_index, B, N):
        edge_indices = []
        for b in range(B):
            edge_indices.append(edge_index + b * N)
        return torch.cat(edge_indices, dim=1)
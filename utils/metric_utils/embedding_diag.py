"""
embedding_diag.py — node-embedding diversity / oversmoothing diagnostics.

NEW, additive module. Created for track_c (session_plan.md). Two uses:
  1. Depth ablation (exp_16): check that stacking GAT layers (1 -> 2 -> 3) does NOT
     collapse node embeddings into near-identical vectors (oversmoothing).
  2. Tier 2A masked-recon SSL ("plan 2"): check the pretrained encoder produces
     DIVERSE node embeddings — a collapsed encoder cannot reconstruct distinct
     per-station values, so this is a health check on the pretext.

METRICS (standard, citable oversmoothing measures):
  - Dirichlet energy  E = (1/|E|) * sum_{(i,j) in edges} || x_i - x_j ||^2
      computed on L2-row-normalised embeddings (scale-invariant, comparable across
      layers/runs). -> 0 means neighbours became identical (oversmoothed).
  - MAD (Mean Average Distance) = mean over node pairs of cosine distance (1 - cos).
      -> 0 means all nodes collapsed to one direction; ~1 means diverse/near-orthogonal.
  - effective_rank = exp(entropy of normalised singular values of the centred (N x D)
      embedding matrix). -> ~1 means dimensional collapse; higher = more spread.

HONEST CAVEAT (from the literature, e.g. Rusch et al. survey arXiv:2303.10993 and
"Are We Measuring Oversmoothing Correctly?" arXiv:2502.04591): these metrics can lag
actual performance degradation and can be fooled by biases/nonlinearities. So treat
them as a DIAGNOSTIC alongside the empirical RMSE / train-val gap, not as a verdict.

Sources (verified 2026-05-30 via web search):
  - A Survey on Oversmoothing in GNNs (arXiv:2303.10993)
  - Are We Measuring Oversmoothing in GNNs Correctly? (arXiv:2502.04591)
  - A Note on Over-Smoothing for GNNs (arXiv:2006.13318)
"""

from __future__ import annotations
import numpy as np

_EPS = 1e-12


def _row_normalize(emb):
    emb = np.asarray(emb, dtype=float)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norms + _EPS)


def dirichlet_energy(emb, edge_index, normalize=True):
    """
    Mean squared difference of (optionally unit-normalised) embeddings across graph edges.
    emb: (N, D). edge_index: (2, E) array-like of node-index pairs (numpy or torch->numpy).
    Lower => neighbours more similar => more oversmoothed.
    """
    emb = np.asarray(emb, dtype=float)
    ei = np.asarray(edge_index)
    if ei.shape[0] != 2:
        ei = ei.T
    src, dst = ei[0].astype(int), ei[1].astype(int)
    if len(src) == 0:
        return float("nan")
    x = _row_normalize(emb) if normalize else emb
    diff = x[src] - x[dst]
    return float(np.mean(np.sum(diff * diff, axis=1)))


def mad(emb, max_pairs=2_000_000):
    """
    Mean pairwise cosine DISTANCE (1 - cosine similarity) over distinct node pairs.
    ~0 => collapsed (all nodes one direction); ~1 => diverse/near-orthogonal.
    """
    x = _row_normalize(emb)
    N = x.shape[0]
    if N < 2:
        return float("nan")
    sim = x @ x.T                      # (N, N) cosine similarities (unit rows)
    iu = np.triu_indices(N, k=1)       # distinct unordered pairs
    cos = sim[iu]
    if cos.size > max_pairs:           # subsample for very large graphs
        idx = np.random.default_rng(0).choice(cos.size, max_pairs, replace=False)
        cos = cos[idx]
    return float(np.mean(1.0 - cos))


def effective_rank(emb):
    """
    exp(Shannon entropy of normalised singular values) of the column-centred matrix.
    ~1 => rank-1 collapse; higher => embeddings span more independent directions.
    """
    x = np.asarray(emb, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    tot = s.sum()
    if tot < _EPS:
        return 1.0                     # all-identical rows -> fully collapsed
    p = s / tot
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def embedding_report(emb, edge_index):
    """All three metrics + basic stats, as a flat dict (one row for a CSV)."""
    emb = np.asarray(emb, dtype=float)
    return {
        "n_nodes": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "dirichlet_energy_norm": dirichlet_energy(emb, edge_index, normalize=True),
        "MAD_cosine": mad(emb),
        "effective_rank": effective_rank(emb),
        "mean_emb_norm": float(np.mean(np.linalg.norm(emb, axis=1))),
    }


def last_gat_embeddings(model, x, edge_index, device):
    """
    Pull the LAST GATConv layer's per-node output at the final timestep, for any
    GAT-based model in this repo (gat_gru.GAT_GRU_Model, gat_gru_multilayer, or the
    Tier 2A pretrain encoder) WITHOUT modifying those models — via a forward hook.

    Returns a torch tensor (B, N, D) (last-timestep node embeddings). Convert each
    sample to numpy and pass to embedding_report(...).
    """
    import torch
    from torch_geometric.nn import GATConv

    gat_layers = [m for m in model.modules() if isinstance(m, GATConv)]
    if not gat_layers:
        raise ValueError("No GATConv layer found on this model.")
    target = gat_layers[-1]            # deepest spatial layer = where oversmoothing shows

    captured = {}

    def _hook(_m, _inp, out):
        captured["out"] = out.detach()  # hook fires once per timestep; keeps the LAST (t=L-1)

    handle = target.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        _ = model(x.to(device), edge_index)
    handle.remove()

    B, L, N, F = x.shape
    return captured["out"].reshape(B, N, -1)


def _selftest():
    rng = np.random.default_rng(0)
    N, D = 60, 16
    # simple ring edges
    ei = np.array([[i for i in range(N)], [(i + 1) % N for i in range(N)]])

    collapsed = np.ones((N, D)) * 3.0            # all identical -> fully oversmoothed
    diverse = rng.standard_normal((N, D))        # near-orthogonal high-dim rows

    rc, rd = embedding_report(collapsed, ei), embedding_report(diverse, ei)

    # collapsed: ~0 energy, ~0 MAD, ~1 rank
    assert rc["dirichlet_energy_norm"] < 1e-9, rc
    assert rc["MAD_cosine"] < 1e-9, rc
    assert rc["effective_rank"] < 1.01, rc
    # diverse: clearly larger on all three
    assert rd["dirichlet_energy_norm"] > 0.5, rd
    assert rd["MAD_cosine"] > 0.5, rd
    assert rd["effective_rank"] > 5.0, rd
    # monotonic separation
    assert rd["dirichlet_energy_norm"] > rc["dirichlet_energy_norm"]
    assert rd["MAD_cosine"] > rc["MAD_cosine"]
    assert rd["effective_rank"] > rc["effective_rank"]

    print("embedding_diag self-test: ALL PASS")
    print(" collapsed:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rc.items()})
    print(" diverse:  ", {k: round(v, 4) if isinstance(v, float) else v for k, v in rd.items()})


if __name__ == "__main__":
    _selftest()

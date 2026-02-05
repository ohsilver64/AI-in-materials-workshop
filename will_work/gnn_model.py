import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def index_add_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    out = torch.zeros((B, x.size(-1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class FastMessageLayer(nn.Module):
    """
    Node update only.
    Messages use [x_j || edge_attr] but edge_attr stays fixed.
    """
    def __init__(self, hidden_dim: int, msg_dim: int, dropout: float, use_bn: bool = False):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, hidden_dim),
        )
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else None

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            x2 = self.self_lin(x)
            if self.use_bn: x2 = self.bn(x2)
            return F.relu(x2), edge_attr

        src, dst = edge_index[0], edge_index[1]

        m_in = torch.cat([x[src], edge_attr], dim=-1)  # [E, 2H]
        m = self.msg(m_in)                             # [E, H]

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)

        x_up = agg + self.self_lin(x)
        if self.use_bn:
            x_up = self.bn(x_up)
        x_up = F.relu(x_up)
        x_up = F.dropout(x_up, p=self.dropout, training=self.training)
        return x_up, edge_attr

class DualMPNN(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden_dim: int, msg_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        self.node_emb = nn.Linear(node_in, hidden_dim)
        self.edge_emb = nn.Linear(edge_in, hidden_dim)

        self.layers = nn.ModuleList([
            FastMessageLayer(hidden_dim, msg_dim, dropout, use_bn=False)
            for _ in range(num_layers)
        ])

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.node_emb(x))

        if edge_attr.numel() > 0:
            edge_attr = F.relu(self.edge_emb(edge_attr))

        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        g_sum = index_add_pool(x, batch)
        counts = torch.bincount(batch, minlength=g_sum.size(0)).clamp_min(1).unsqueeze(-1).to(x.device)
        g_mean = g_sum / counts

        g = torch.cat([g_sum, g_mean], dim=-1)
        return self.readout_mlp(g)

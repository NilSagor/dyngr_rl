# co_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        # x: [N, in_dim]
        # adj_norm: [N, N] symmetric normalized adjacency
        return torch.mm(adj_norm, torch.mm(x, self.weight))

class CoGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        x = F.relu(self.gcn1(x, adj_norm))
        x = self.dropout(x)
        x = self.gcn2(x, adj_norm)
        return x
    

# class CoGNN(nn.Module):
#     """
#     Lightweight GCN that operates on a precomputed co-occurrence matrix.
#     """
#     def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
#         super().__init__()
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, out_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
#         # x: [num_nodes, in_dim]
#         # edge_index: [2, E] (symmetric, co-occurrence edges)
#         # edge_weight: optional, Jaccard similarity values
#         x = self.conv1(x, edge_index, edge_weight)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index, edge_weight)
#         return x   # [num_nodes, out_dim]
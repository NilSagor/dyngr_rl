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
        # Ensure adj_norm is on the same device as x
        adj_norm = adj_norm.to(x.device)
        return torch.mm(adj_norm, torch.mm(x, self.weight))

class CoGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        device = x.device
        num_nodes = x.size(0)

        # Move inputs to device
        edge_index = edge_index.to(device)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)
        else:
            edge_weight = edge_weight.to(device)

        # Build sparse adjacency matrix
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes), device=device)

        # Add self-loops
        adj = adj + torch.eye(num_nodes, device=device).to_sparse()

        # Degree and normalized Laplacian
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        # Normalized adjacency: D^{-1/2} A D^{-1/2}
        adj_norm = D_inv_sqrt @ adj.to_dense() @ D_inv_sqrt

        # GCN layers
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
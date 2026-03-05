import torch
import torch.nn as nn

# from utils.utils import MergeLayer

class MergeLayer(nn.Module):
    """
    Robust tensor merging layer for temporal attention mechanisms.
    
    Industry standard usage (TGN ICML 2020):
    - ONLY supports 2-input merging (x1, x2)
    - Total input dim = dim(x1) + dim(x2)
    - Hidden dim and output dim explicitly specified
    
    Why no 4-arg support?
    - Original TGN paper ONLY uses 2-input merging
    - 4-arg pattern causes dimension mismatches (e.g., 204 vs 208 errors)
    - Safer to enforce strict 2-input interface
    """
    
    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # CRITICAL FIX 1: Explicit dimension validation
        if input_dim1 <= 0 or input_dim2 <= 0:
            raise ValueError(f"Input dimensions must be positive: {input_dim1}, {input_dim2}")
        if hidden_dim <= 0 or output_dim <= 0:
            raise ValueError(f"Hidden/output dimensions must be positive: {hidden_dim}, {output_dim}")
        
        total_input_dim = input_dim1 + input_dim2
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        
        # Proper weight initialization (with bias handling)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)  # Explicit bias initialization
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        # Store dimensions for debugging/validation
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Merge two tensors via concatenation + MLP.
        
        Args:
            x1: Tensor of shape [..., input_dim1]
            x2: Tensor of shape [..., input_dim2]
        
        Returns:
            Tensor of shape [..., output_dim]
        
        Raises:
            ValueError: If input dimensions don't match expected sizes
        """
        # Runtime dimension validation
        if x1.size(-1) != self.input_dim1:
            raise ValueError(
                f"x1 dimension mismatch: expected {self.input_dim1}, got {x1.size(-1)}. "
                f"Check attention layer configuration."
            )
        if x2.size(-1) != self.input_dim2:
            raise ValueError(
                f"x2 dimension mismatch: expected {self.input_dim2}, got {x2.size(-1)}. "
                f"This causes attention shape errors (e.g., 204 vs 208)."
            )
        
        # Safe concatenation
        x = torch.cat([x1, x2], dim=-1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class TemporalAttentionLayer(nn.Module):
  """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
  """

  def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
               output_dimension, n_head=2,
               dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.n_head = n_head

    self.feat_dim = n_node_features
    self.time_dim = time_dim

    self.query_dim = n_node_features + time_dim
    self.key_dim = n_neighbors_features + time_dim + n_edge_features

    self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

    self.multi_head_target = nn.MultiheadAttention(
        embed_dim=self.query_dim,
        kdim=self.key_dim,
        vdim=self.key_dim,
        num_heads=n_head,
        dropout=dropout
    )

    if self.query_dim != n_node_features:
        self.attn_proj = nn.Linear(self.query_dim, n_node_features)   # or vice versa
    else:
        self.attn_proj = nn.Identity()

  def forward(self, src_node_features, src_time_features, neighbors_features,
              neighbors_time_features, edge_features, neighbors_padding_mask):
    """
    "Temporal attention model
    :param src_node_features: float Tensor of shape [batch_size, n_node_features]
    :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
    :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
    :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
    time_dim]
    :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
    :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
    :return:
    attn_output: float Tensor of shape [1, batch_size, n_node_features]
    attn_output_weights: [batch_size, 1, n_neighbors]
    """
    batch_size, n_neighbors, _ = neighbors_features.shape

    #  EARLY EXIT FOR ZERO NEIGHBORS (cold-start safe)
    if n_neighbors == 0:
        # Return zero attention output (TGN paper spec: "all zero attention output for no neighbors")
        attn_output = torch.zeros(batch_size, self.feat_dim, device=src_node_features.device)
        attn_output_weights = torch.zeros(batch_size, 0, device=src_node_features.device)
        # Skip attention computation entirely
        attn_output = self.merger(attn_output, src_node_features)
        return attn_output, attn_output_weights
    
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1)  # [batch_size]
    if invalid_neighborhood_mask.any():
        # Set the first neighbor as valid for these nodes.
        # The neighbor features are already zero (padding), so the attention will
        # produce zero output, which is safe and matches the original TGN behavior.
        neighbors_padding_mask[invalid_neighborhood_mask, 0] = False

    src_node_features_unrolled = src_node_features.unsqueeze(1)          # [batch, 1, feat]

    query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)  # [batch, 1, query_dim]
    key   = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)  # [batch, n_neighbors, key_dim]

    # print(neighbors_features.shape, edge_features.shape, neighbors_time_features.shape)
    # Reshape tensors so to expected shape by multi head attention
    query = query.permute(1, 0, 2)   # [1, batch, query_dim]
    key   = key.permute(1, 0, 2)     # [n_neighbors, batch, key_dim]

    # Multi-head attention
    attn_output, attn_output_weights = self.multi_head_target(
        query=query,
        key=key,
        value=key,
        key_padding_mask=neighbors_padding_mask
    )

    # Reshape outputs
    attn_output = attn_output.squeeze(0)          # [batch, query_dim]
    attn_output_weights = attn_output_weights.squeeze(0)  # [batch, n_neighbors]
    
    # Zero out the attention output for nodes that originally had no neighbors
    # (this is redundant but harmless; the output is already zero due to the fix)
    attn_output = attn_output.masked_fill(invalid_neighborhood_mask.unsqueeze(-1), 0)

    
    # Merge with original source features
    attn_output = self.merger(attn_output, src_node_features)

    return attn_output, attn_output_weights
  

  
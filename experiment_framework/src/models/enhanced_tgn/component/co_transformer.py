import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


from .transformer_encoder import PositionalEncoding, TransformerEncoderLayer


class IntraWalkEncoder(nn.Module):
    """
    Intra-walk Transformer: encodes each walk independently.
    Captures dependencies between nodes within a single walk.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_walk_length: int = 20
    ):
        super(IntraWalkEncoder, self).__init__()
        
        self.d_model = d_model
        self.max_walk_length = max_walk_length
        
        # Positional encoding for walk positions
        self.pos_encoder = PositionalEncoding(d_model, max_walk_length)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection for walk-level aggregation
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_proj = nn.LayerNorm(d_model)
        
    def forward(
        self,
        walk_embeddings: torch.Tensor,
        walk_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            walk_embeddings: [batch_size, num_walks, walk_length, d_model]
            walk_masks: [batch_size, num_walks, walk_length] (1 for valid, 0 for padding)
            
        Returns:
            - encoded_walks: [batch_size, num_walks, walk_length, d_model]
            - walk_summaries: [batch_size, num_walks, d_model] (pooled walk representations)
        """
        batch_size, num_walks, walk_len, d_model = walk_embeddings.shape
        
        # Store expected total walks
        # expected_total_walks = batch_size * num_walks  # e.g., 200 * 5 = 1000
        
        # Reshape to process all walks in parallel
        # Combine batch and num_walks dimensions
        # Flatten for parallel processing
        x = walk_embeddings.view(batch_size * num_walks, walk_len, d_model)
        masks = walk_masks.view(batch_size * num_walks, walk_len)
        
        # Add positional encoding        
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Create key padding mask for attention (True for padded positions)
        key_padding_mask = ~masks.bool()  # [batch*num_walks, walk_len]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        x = self.norm_proj(x)

        # Reshape back BEFORE pooling
        x = x.view(batch_size, num_walks, walk_len, d_model)  # [B, N, L, D]
        masks = masks.view(batch_size, num_walks, walk_len)    # [B, N, L]
        

        # Pool to get walk summaries (mean over valid positions)
        masks_expanded = masks.unsqueeze(-1).float()  # [batch*num_walks, walk_len, 1]
        walk_summaries = (x * masks_expanded).sum(dim=2) / (masks_expanded.sum(dim=2) + 1e-8)
        # Result: [batch*num_walks, d_model]
        
        # DEBUG: Verify shape before reshape
        # actual_total_walks = walk_summaries.size(0)
        # if actual_total_walks != expected_total_walks:
        #     raise RuntimeError(
        #         f"Walk count mismatch: expected {expected_total_walks} "
        #         f"(batch={batch_size} × walks={num_walks}), got {actual_total_walks}. "
        #         f"Input shape: {walk_embeddings.shape}"
        #     )

         # Reshape back
        # walk_summaries = walk_summaries.view(batch_size, num_walks, d_model)
        
        # Final projection
        walk_summaries = self.output_proj(walk_summaries)
        walk_summaries = self.norm_proj(walk_summaries)
        
        # Also reshape encoded_walks for consistency
        encoded_walks = x
        
        return encoded_walks, walk_summaries


# class CooccurrenceMatrix(nn.Module):
#     """
#     Constructs co-occurrence matrix between walks based on anonymized node positions.
    
#     C_u[r,s] = Σ_i Σ_j I(a_i^(r) = a_j^(s)) · κ(i, j)
#     where κ(i,j) = exp(-|i-j|²/σ²) is a positional kernel.
#     """
#     def __init__(self, max_walk_length: int = 20, sigma: float = 2.0):
#         super(CooccurrenceMatrix, self).__init__()
        
#         self.max_walk_length = max_walk_length
#         self.sigma = sigma
        
#         # Pre-compute positional kernel matrix
#         kernel = torch.zeros(max_walk_length, max_walk_length)
#         for i in range(max_walk_length):
#             for j in range(max_walk_length):
#                 kernel[i, j] = math.exp(-((i - j) ** 2) / (sigma ** 2))
        
#         self.register_buffer('kernel', kernel)   
    
    
#     def forward(self, anonymized_nodes, walk_masks):
#         """Fully vectorized but memory-intensive version."""
#         batch_size, num_walks, walk_len = anonymized_nodes.shape
#         cooccurrence = torch.zeros(batch_size, num_walks, num_walks, 
#                                device=anonymized_nodes.device)
        
#         device = anonymized_nodes.device
#         # Ensure walk_masks is boolean for bitwise operations
#         walk_masks_bool = walk_masks.bool() if walk_masks.dtype != torch.bool else walk_masks

#         for r in range(num_walks):
#             for s in range(num_walks):
#                 # Vectorized comparison for one pair only
#                 nodes_r = anonymized_nodes[:, r, :].unsqueeze(-1)  # [B, L, 1]
#                 nodes_s = anonymized_nodes[:, s, :].unsqueeze(1)   # [B, 1, L]
                
#                 # match = (nodes_r == nodes_s) & walk_masks[:, r, :, None] & walk_masks[:, s, None, :]
#                 # Use boolean masks for bitwise AND
#                 mask_r = walk_masks_bool[:, r, :, None]  # [B, L, 1]
#                 mask_s = walk_masks_bool[:, s, None, :]  # [B, 1, L]
                
#                 match = (nodes_r == nodes_s) & mask_r & mask_s
                
#                 kernel = self.kernel[:walk_len, :walk_len].to(anonymized_nodes.device)
                
#                 # Per-sample normalization
#                 norm_factor = (walk_masks[:, r].sum(dim=-1) * walk_masks[:, s].sum(dim=-1)) + 1e-8  # [batch_size]
#                 cooccurrence[:, r, s] = (match.float() * kernel).sum(dim=[-2, -1]) / norm_factor  # Both [batch_size]
        
#         return cooccurrence
        
        
class CooccurrenceMatrix(nn.Module):
    """
    Optimized for CUDA using scatter operations when matches are sparse.
    Best when node IDs are diverse (few matches).
    """
    def __init__(self, max_walk_length: int = 20, sigma: float = 2.0):
        super().__init__()
        positions = torch.arange(max_walk_length, dtype=torch.float32)
        self.register_buffer('kernel', 
            torch.exp(-((positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2) / (sigma ** 2)))
        self.register_buffer('pos_indices', torch.arange(max_walk_length))
    
    def forward(self, anonymized_nodes, walk_masks):
        B, W, L = anonymized_nodes.shape
        device = anonymized_nodes.device
        walk_masks = walk_masks.bool()
        kernel = self.kernel[:L, :L]
        
        # Flatten to [B*W, L] for easier processing
        nodes_flat = anonymized_nodes.view(-1, L)  # [B*W, L]
        masks_flat = walk_masks.view(-1, L)        # [B*W, L]
        
        # For each batch and walk, get valid (position, node_id) pairs
        cooccurrence = torch.zeros(B, W, W, device=device)
        
        # Process per batch to avoid O(B²W²) indexing complexity
        for b in range(B):
            batch_nodes = anonymized_nodes[b]  # [W, L]
            batch_masks = walk_masks[b]        # [W, L]
            
            # Create sparse representation: list of (walk_idx, pos, node_id) for valid entries
            valid_pos = torch.where(batch_masks)  # ([walk_indices], [pos_indices])
            walk_indices, pos_indices = valid_pos
            node_ids = batch_nodes[valid_pos]
            
            # Group by node_id using sorting (efficient on GPU)
            sorted_node_ids, sort_perm = torch.sort(node_ids)
            sorted_walks = walk_indices[sort_perm]
            sorted_pos = pos_indices[sort_perm]
            
            # Find boundaries where node_id changes
            node_changes = torch.cat([
                torch.tensor([True], device=device),
                sorted_node_ids[1:] != sorted_node_ids[:-1]
            ])
            change_indices = torch.where(node_changes)[0]
            
            # For each unique node, compute contributions between all its walk pairs
            # This is O(total_matches²) but only for matching nodes
            for i in range(len(change_indices)):
                start = change_indices[i].item()
                end = change_indices[i + 1].item() if i + 1 < len(change_indices) else len(sorted_node_ids)
                
                # Walks containing this node
                walks_with_node = sorted_walks[start:end]      # [n_occurrences]
                positions_in_walks = sorted_pos[start:end]     # [n_occurrences]
                
                # All pairs of occurrences
                if len(walks_with_node) > 1:
                    # Outer product of positions to get kernel values
                    pos_i = positions_in_walks.unsqueeze(1)  # [n, 1]
                    pos_j = positions_in_walks.unsqueeze(0)  # [1, n]
                    
                    # Kernel values for all pairs
                    kernel_vals = kernel[pos_i, pos_j]  # [n, n]
                    
                    # Walk indices for all pairs
                    w_i = walks_with_node.unsqueeze(1)  # [n, 1]
                    w_j = walks_with_node.unsqueeze(0)  # [1, n]
                    
                    # Scatter add to cooccurrence matrix
                    cooccurrence[b].index_put_(
                        (w_i.expand_as(kernel_vals), w_j.expand_as(kernel_vals)),
                        kernel_vals,
                        accumulate=True
                    )
        
        # Normalize
        walk_lens = walk_masks.sum(dim=-1)  # [B, W]
        norm = walk_lens.unsqueeze(-1) * walk_lens.unsqueeze(-2) + 1e-8
        return cooccurrence / norm   


class InterWalkTransformer(nn.Module):
    """
    Inter-walk Transformer: processes interactions between different walks.
    Uses co-occurrence matrix as attention bias.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        cooccurrence_gamma: float = 0.5
    ):
        super(InterWalkTransformer, self).__init__()
        
        self.d_model = d_model
        self.cooccurrence_gamma = cooccurrence_gamma  # γ in the paper
        self.nhead = nhead
        
        # Learnable gamma (optional)
        self.gamma = nn.Parameter(torch.tensor(cooccurrence_gamma))
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Positional encoding for walks (walks are ordered but we use co-occurrence bias instead)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        walk_summaries: torch.Tensor,      # [batch_size, num_walks, d_model]
        cooccurrence: torch.Tensor,         # [batch_size, num_walks, num_walks]
        walk_masks: Optional[torch.Tensor] = None  # [batch_size, num_walks] (1 for valid walks)
    ) -> torch.Tensor:
        """
        Args:
            walk_summaries: Walk-level representations from intra-walk encoder
            cooccurrence: Co-occurrence matrix C_u[r,s]
            walk_masks: Mask for valid walks (some walks may be truncated)
            
        Returns:
            refined_walks: [batch_size, num_walks, d_model] with inter-walk context
        """
        batch_size, num_walks, _ = walk_summaries.shape
        
        # DEBUG: Verify shapes match
        # Dynamic check with helpful error message
        if cooccurrence.shape != (batch_size, num_walks, num_walks):
            raise ValueError(
                f"Shape mismatch in InterWalkTransformer: "
                f"walk_summaries={walk_summaries.shape}, "
                f"cooccurrence={cooccurrence.shape}. "
                f"Expected cooccurrence to have shape (batch_size, num_walks, num_walks) "
                f"where num_walks={num_walks} from walk_summaries."
            )
        
        # Create attention bias from co-occurrence matrix
        # Expand to multi-head format: [batch_size, nhead, num_walks, num_walks]
        # cooccurrence_bias = cooccurrence.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        if cooccurrence is not None:
            # Create attention bias from co-occurrence matrix
            cooccurrence_bias = cooccurrence.unsqueeze(1).expand(-1, self.nhead, -1, -1)

            # Scale bias to match attention score magnitude
            bias_scale = self.d_model ** -0.5  # Same as attention scaling         
            
            cooccurrence_bias = self.gamma * cooccurrence_bias * bias_scale
            # Reshape for multi-head attention: [B, H, N, N] -> [B*H, N, N]
            # cooccurrence_bias = cooccurrence_bias.view(batch_size * self.nhead, num_walks, num_walks)
        else:
            # Default: no bias
            cooccurrence_bias = None
        
        
        # Create key padding mask for walks
        key_padding_mask = None
        if walk_masks is not None:
            key_padding_mask = ~walk_masks.bool()  # [batch_size, num_walks]
        
        x = walk_summaries
        
        # Apply transformer layers with co-occurrence bias
        for layer in self.layers:
            # Optional debug print
            # if self.training and torch.rand(1).item() < 0.01:
            #     print(f"InterWalk: x={x.shape}, bias={cooccurrence_bias.shape}")
            
            x = layer(x, attn_bias=cooccurrence_bias, key_padding_mask=key_padding_mask)
        
        x = self.norm(x)
        
        return x
    
class HierarchicalCooccurrenceTransformer(nn.Module):
    """
    Hierarchical Co-occurrence Transformer (HCT) for HYDRA.
    
    Processes multi-scale walks through:
    1. Intra-walk Transformer (local context)
    2. Co-occurrence matrix construction (structural patterns)
    3. Inter-walk Transformer (global context)
    """
    def __init__(
        self,
        d_model: int = 128,
        memory_dim: int = 172,
        nhead: int = 4,
        num_intra_layers: int = 2,
        num_inter_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_walk_length: int = 20,
        max_num_walks: int = 20,  # Short + Long + TAWR
        cooccurrence_sigma: float = 2.0,
        cooccurrence_gamma: float = 0.5,
        use_walk_type_embedding: bool = True
    ):
        super(HierarchicalCooccurrenceTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_walk_length = max_walk_length
        self.max_num_walks = max_num_walks
        self.use_walk_type_embedding = use_walk_type_embedding
        
        # Intra-walk encoder (shared across all walk types)
        self.intra_walk_encoder = IntraWalkEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_intra_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_walk_length=max_walk_length
        )
        
        # Co-occurrence matrix constructor
        self.cooccurrence_matrix = CooccurrenceMatrix(
            max_walk_length=max_walk_length,
            sigma=cooccurrence_sigma
        )
        
        # Inter-walk transformer
        self.inter_walk_transformer = InterWalkTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_inter_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            cooccurrence_gamma=cooccurrence_gamma
        )
        
        # Optional walk type embeddings (short/long/tawr)
        if use_walk_type_embedding:
            self.walk_type_embed = nn.Embedding(3, d_model)  # 3 types
            nn.init.normal_(self.walk_type_embed.weight, std=0.02)
        
        # Final pooling layer (attention-based)
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.memory_proj = nn.Linear(memory_dim, d_model) if memory_dim != d_model else nn.Identity()

        self.restart_embed = nn.Embedding(2, d_model)
        
    def process_walk_type(
        self,
        node_embeddings: torch.Tensor,      # [batch_size, num_walks, walk_len, d_model]
        anonymized_nodes: torch.Tensor,      # [batch_size, num_walks, walk_len] anonymized IDs
        walk_masks: torch.Tensor,            # [batch_size, num_walks, walk_len] (1 for valid)
        walk_type: int = 0,                  # 0: short, 1: long, 2: tawr
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single walk type.
        """
        batch_size, num_walks, walk_len, _ = node_embeddings.shape
        
        # Validate all inputs have consistent shapes
        assert anonymized_nodes.shape == (batch_size, num_walks, walk_len), \
            f"anonymized_nodes shape {anonymized_nodes.shape} doesn't match node_embeddings {node_embeddings.shape}"
        assert walk_masks.shape == (batch_size, num_walks, walk_len), \
            f"walk_masks shape {walk_masks.shape} doesn't match expected {(batch_size, num_walks, walk_len)}"
        
        # Add walk type embedding if enabled
        if self.use_walk_type_embedding:
            type_embed = self.walk_type_embed(torch.tensor([walk_type], device=node_embeddings.device))        
            node_embeddings = node_embeddings + type_embed.unsqueeze(0).unsqueeze(0)
        
        assert type_embed.unsqueeze(0).unsqueeze(0).shape == (1, 1, 1, self.d_model)
        
        # Step 1: Intra-walk encoding
        encoded_walks, walk_summaries = self.intra_walk_encoder(
            node_embeddings, walk_masks
        )
        
        # Step 2: Compute co-occurrence matrix
        # Use anonymized nodes for co-occurrence
        cooccurrence = self.cooccurrence_matrix(anonymized_nodes, walk_masks)
        
         # Validate co-occurrence shape matches walk_summaries
        assert cooccurrence.shape == (batch_size, num_walks, num_walks), \
            f"Co-occurrence shape {cooccurrence.shape} doesn't match expected {(batch_size, num_walks, num_walks)}"
        
        
        # Create walk-level mask: [B, num_walks] (2D) - CRITICAL
        walk_level_mask = (walk_masks.sum(dim=-1) > 0).float()  # [batch_size, num_walks]
        
        # Step 3: Inter-walk transformer
        refined_walks = self.inter_walk_transformer(
            walk_summaries,
            cooccurrence,
            walk_level_mask
        )
        
        result = {
            'encoded_walks': encoded_walks,
            'walk_summaries': walk_summaries,
            'refined_walks': refined_walks,
            'cooccurrence': cooccurrence,
            'walk_masks': walk_level_mask
        }
        
        return result
    
    def fuse_walk_types(
        self,
        short_output: Dict[str, torch.Tensor],
        long_output: Dict[str, torch.Tensor],
        tawr_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse representations from different walk types.
        Uses attention-based pooling across all walks from all types.
        """
        batch_size = short_output['refined_walks'].size(0)
        
        # Collect all refined walks with explicit shape tracking
        all_walks = []
        all_masks = []
        
        for name, output in [('short', short_output), ('long', long_output), ('tawr', tawr_output)]:
            refined = output['refined_walks']  # [batch_size, num_walks, d_model]
            masks = output['walk_masks']       # [batch_size, num_walks]
            
            # DEBUG: Log shapes to catch mismatches early
            # if self.training and torch.rand(1).item() < 0.01:
            #     print(f"{name}: refined={refined.shape}, masks={masks.shape}")
                
            assert refined.size(0) == masks.size(0), f"{name}: batch size mismatch"
            assert refined.size(1) == masks.size(1), f"{name}: walk count mismatch: {refined.size(1)} vs {masks.size(1)}"
            
            all_walks.append(refined)
            all_masks.append(masks)
        
        # Concatenate along walk dimension
        all_walks = torch.cat(all_walks, dim=1)  # [batch_size, total_walks, d_model]
        all_masks = torch.cat(all_masks, dim=1)  # [batch_size, total_walks]
        
        total_walks = all_walks.size(1)
        
        # Attention-based pooling
        # Compute attention scores
        attention_scores = self.pooling(all_walks).squeeze(-1)  # [batch_size, total_walks]
        
        # CRITICAL: Verify pooling output shape matches
        assert attention_scores.size(1) == total_walks, \
            f"Pooling shape mismatch: got {attention_scores.size(1)}, expected {total_walks}"
            
        
        # Mask out invalid walks
        attention_scores = attention_scores.masked_fill(all_masks == 0, float('-inf'))
        
       
        # Softmax over walks
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, total_walks]
        attention_weights = attention_weights.unsqueeze(-1)      # [batch_size, total_walks, 1]
        
              
        
        # Hybrid pooling: 50% mean, 50% attention
        mean_pool = (all_walks * all_masks.unsqueeze(-1)).sum(dim=1) / (all_masks.sum(dim=-1, keepdim=True) + 1e-8)
        attn_pool = (all_walks * attention_weights).sum(dim=1)
        fused = 0.5 * mean_pool + 0.5 * attn_pool
        
        return self.output_proj(self.norm(fused))    
    
    def _validate_anonymization(self, nodes, nodes_anon, masks):
        # Same actual node should map to same anonymized ID within batch
        for b in range(nodes.size(0)):
            unique_actual = nodes[b][masks[b].bool()].unique()
            for actual_id in unique_actual:
                anon_ids = nodes_anon[b][(nodes[b] == actual_id) & masks[b]]
                assert anon_ids.nunique() == 1, f"Node {actual_id} has multiple anon IDs: {anon_ids.unique()}"
    
    
    def forward(
        self,
        walks_dict: Dict[str, Dict[str, torch.Tensor]],
        node_memory: torch.Tensor,  # [num_nodes, memory_dim] from SAM          
        return_all: bool = False
    ) -> Union[torch.Tensor, Dict]:
        """
        Main forward pass for HCT with proper SAM memory integration.
        
        Args:
            walks_dict: From walk sampler with 'nodes' (actual IDs) and 'nodes_anon'
            node_memory: [num_nodes, memory_dim] SAM raw memory
            memory_proj: Optional projection layer
            return_all: Return intermediate outputs
        """
        batch_size = walks_dict['short']['nodes'].size(0)
        device = walks_dict['short']['nodes'].device
        
        # assert node_memory.requires_grad, "SAM memory must have gradients enabled!"

        outputs = {}
        
        for walk_type, type_name in [(0, 'short'), (1, 'long'), (2, 'tawr')]:
            data = walks_dict[type_name]
            
            # CRITICAL: Get actual node indices for memory lookup
            nodes = data['nodes']  # [B, num_walks, L] - actual node IDs
            nodes_anon = data['nodes_anon']  # [B, num_walks, L] - for co-occurrence
            masks = data['masks']
            
            num_walks = nodes.size(1)
            walk_len = nodes.size(2)
            
            # Lookup from SAM memory using actual node IDs
            flat_nodes = nodes.reshape(-1)  # [B * num_walks * L]
            walk_node_feats = node_memory[flat_nodes]  # [B*num_walks*L, memory_dim]
            
            # # Project to d_model if needed
            # if memory_proj is not None:
            #     walk_node_feats = memory_proj(walk_node_feats)
            
            # actual_dim = self.d_model if memory_proj is not None else node_memory.size(-1)
            walk_node_feats = self.memory_proj(walk_node_feats)
            # Reshape: [B, num_walks, L, d_model]
            # walk_embeddings = walk_node_feats.view(batch_size, num_walks, walk_len, actual_dim)
            walk_embeddings = walk_node_feats.view(batch_size, num_walks, walk_len, self.d_model)

            # VALIDATE: Ensure shapes are consistent
            assert walk_embeddings.shape[0] == batch_size, \
                f"Batch size mismatch: {walk_embeddings.shape[0]} vs {batch_size}"
            assert walk_embeddings.shape[1] == num_walks, \
                f"Num walks mismatch: {walk_embeddings.shape[1]} vs {num_walks}"
            assert masks.shape == (batch_size, num_walks, walk_len), \
                f"Mask shape mismatch: {masks.shape} vs {(batch_size, num_walks, walk_len)}"
            
            
            # Add restart flags for TAWR
            if type_name == 'tawr' and 'restart_flags' in data:
                restart_flags = data['restart_flags'].unsqueeze(-1).float()
                # Learnable restart embedding or simple marker                
                restart_embed = self.restart_embed(data['restart_flags'].long())
                walk_embeddings = walk_embeddings + restart_embed
            
            # Process through HCT pipeline
            output = self.process_walk_type(
                walk_embeddings,
                nodes_anon,  # Use anonymized for co-occurrence
                masks,
                walk_type=walk_type
            )
            outputs[type_name] = output
        
        # Fuse and return
        fused = self.fuse_walk_types(outputs['short'], outputs['long'], outputs['tawr'])
        
        if return_all:
            return {'fused': fused, **outputs}
        return fused
    

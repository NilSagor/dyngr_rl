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
        self.norm = nn.LayerNorm(d_model)
        
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
        
        # Reshape to process all walks in parallel
        # Combine batch and num_walks dimensions
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
        
        x = self.norm(x)
        
        # Reshape back
        encoded_walks = x.view(batch_size, num_walks, walk_len, d_model)
        
        # Pool to get walk summaries (mean over valid positions)
        # Use masks to ignore padding
        masks_expanded = masks.unsqueeze(-1).float()  # [batch*num_walks, walk_len, 1]
        walk_summaries = (x * masks_expanded).sum(dim=1) / (masks_expanded.sum(dim=1) + 1e-8)
        walk_summaries = walk_summaries.view(batch_size, num_walks, d_model)
        
        # Final projection
        walk_summaries = self.output_proj(walk_summaries)
        
        return encoded_walks, walk_summaries


class CooccurrenceMatrix(nn.Module):
    """
    Constructs co-occurrence matrix between walks based on anonymized node positions.
    
    C_u[r,s] = Σ_i Σ_j I(a_i^(r) = a_j^(s)) · κ(i, j)
    where κ(i,j) = exp(-|i-j|²/σ²) is a positional kernel.
    """
    def __init__(self, max_walk_length: int = 20, sigma: float = 2.0):
        super(CooccurrenceMatrix, self).__init__()
        
        self.max_walk_length = max_walk_length
        self.sigma = sigma
        
        # Pre-compute positional kernel matrix
        kernel = torch.zeros(max_walk_length, max_walk_length)
        for i in range(max_walk_length):
            for j in range(max_walk_length):
                kernel[i, j] = math.exp(-((i - j) ** 2) / (sigma ** 2))
        
        self.register_buffer('kernel', kernel)
    
    # def forward(
    #     self,
    #     anonymized_nodes: torch.Tensor,  # [batch_size, num_walks, walk_length] with anonymized IDs
    #     walk_masks: torch.Tensor          # [batch_size, num_walks, walk_length] (1 for valid)
    # ) -> torch.Tensor:
    #     """
    #     Compute co-occurrence matrix for each batch item.
        
    #     Returns:
    #         cooccurrence: [batch_size, num_walks, num_walks]
    #     """
    #     batch_size, num_walks, walk_len = anonymized_nodes.shape
        
    #     # Truncate kernel to actual walk length
    #     kernel = self.kernel[:walk_len, :walk_len]  # [walk_len, walk_len]
        
    #     # Create co-occurrence tensor
    #     cooccurrence = torch.zeros(batch_size, num_walks, num_walks, device=anonymized_nodes.device)
        
    #     for b in range(batch_size):
    #         for r in range(num_walks):
    #             for s in range(num_walks):
    #                 if r == s:
    #                     # Self-cooccurrence is high by default
    #                     cooccurrence[b, r, s] = 1.0
    #                     continue
                    
    #                 # Get nodes and masks for these walks
    #                 nodes_r = anonymized_nodes[b, r]  # [walk_len]
    #                 nodes_s = anonymized_nodes[b, s]  # [walk_len]
    #                 mask_r = walk_masks[b, r]  # [walk_len]
    #                 mask_s = walk_masks[b, s]  # [walk_len]
                    
    #                 # Create valid position pairs
    #                 score = 0.0
    #                 total_pairs = 0
                    
    #                 for i in range(walk_len):
    #                     if mask_r[i] == 0:
    #                         continue
    #                     for j in range(walk_len):
    #                         if mask_s[j] == 0:
    #                             continue
                            
    #                         if nodes_r[i] == nodes_s[j] and nodes_r[i] != 0:  # 0 is padding
    #                             # Same anonymized node appears at positions i and j
    #                             score += kernel[i, j]
    #                             total_pairs += 1
                    
    #                 if total_pairs > 0:
    #                     cooccurrence[b, r, s] = score / total_pairs
    #                 else:
    #                     cooccurrence[b, r, s] = 0.0
        
    #     return cooccurrence
    
    def forward(self, anonymized_nodes, walk_masks):
        """Fully vectorized but memory-intensive version."""
        batch_size, num_walks, walk_len = anonymized_nodes.shape
        device = anonymized_nodes.device
        
        # Expand for all pairs: [batch, walks, 1, len] vs [batch, 1, walks, len]
        nodes_expanded = anonymized_nodes.unsqueeze(2)  # [B, R, 1, L]
        nodes_expanded_t = anonymized_nodes.unsqueeze(1)  # [B, 1, S, L]
        
        # Match matrix: [B, R, S, L, L] - True where node IDs match
        match = (nodes_expanded.unsqueeze(-1) == nodes_expanded_t.unsqueeze(-2))  # [B, R, S, L, L]
        
        # Apply masks: [B, R, 1, L, 1] * [B, 1, S, 1, L]
        mask_r = walk_masks.unsqueeze(2).unsqueeze(-1)  # [B, R, 1, L, 1]
        mask_s = walk_masks.unsqueeze(1).unsqueeze(-2)  # [B, 1, S, 1, L]
        match = match & mask_r.bool() & mask_s.bool()
        
        # Kernel: [L, L] -> [1, 1, 1, L, L]
        kernel = self.kernel[:walk_len, :walk_len].to(device)
        
        # Weighted sum: [B, R, S, L, L] * [L, L] -> sum -> [B, R, S]
        cooccurrence = (match.float() * kernel).sum(dim=[-2, -1])
        
        # Normalize
        valid_r = walk_masks.sum(dim=-1, keepdim=True)  # [B, R, 1]
        valid_s = walk_masks.sum(dim=-1, keepdim=True)  # [B, S, 1]
        norm = valid_r * valid_s.transpose(1, 2) + 1e-8
        
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
        
        # Create attention bias from co-occurrence matrix
        # Expand to multi-head format: [batch_size, nhead, num_walks, num_walks]
        cooccurrence_bias = cooccurrence.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        
        # Scale and apply gamma
        cooccurrence_bias = self.gamma * cooccurrence_bias
        
        # Create key padding mask for walks
        key_padding_mask = None
        if walk_masks is not None:
            key_padding_mask = ~walk_masks.bool()  # [batch_size, num_walks]
        
        x = walk_summaries
        
        # Apply transformer layers with co-occurrence bias
        for layer in self.layers:
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
        
        # Add walk type embedding if enabled
        if self.use_walk_type_embedding:
            type_embed = self.walk_type_embed(torch.tensor([walk_type], device=node_embeddings.device))
            node_embeddings = node_embeddings + type_embed.unsqueeze(0).unsqueeze(0)
        
        # Step 1: Intra-walk encoding
        encoded_walks, walk_summaries = self.intra_walk_encoder(
            node_embeddings, walk_masks
        )
        
        # Step 2: Compute co-occurrence matrix
        # Use anonymized nodes for co-occurrence
        cooccurrence = self.cooccurrence_matrix(anonymized_nodes, walk_masks)
        
        # Create walk-level mask (1 if walk has at least one valid node)
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
        
        # Collect all refined walks
        all_walks = []
        all_masks = []
        
        for output in [short_output, long_output, tawr_output]:
            refined = output['refined_walks']  # [batch_size, num_walks, d_model]
            masks = output['walk_masks']       # [batch_size, num_walks]
            
            all_walks.append(refined)
            all_masks.append(masks)
        
        # Concatenate along walk dimension
        all_walks = torch.cat(all_walks, dim=1)  # [batch_size, total_walks, d_model]
        all_masks = torch.cat(all_masks, dim=1)  # [batch_size, total_walks]
        
        # Attention-based pooling
        # Compute attention scores
        attention_scores = self.pooling(all_walks).squeeze(-1)  # [batch_size, total_walks]
        
        # Mask out invalid walks
        attention_scores = attention_scores.masked_fill(all_masks == 0, float('-inf'))
        
        # Softmax over walks
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, total_walks]
        attention_weights = attention_weights.unsqueeze(-1)      # [batch_size, total_walks, 1]
        
        # Weighted sum
        fused = (all_walks * attention_weights).sum(dim=1)  # [batch_size, d_model]
        fused = self.norm(fused)
        fused = self.output_proj(fused)
        
        return fused
    
    # def forward(
    #     self,
    #     walks_dict: Dict[str, Dict[str, torch.Tensor]],
    #     node_memory_states: torch.Tensor,  # [num_nodes, d_model] - SAM memory
    #     walk_node_indices: Dict[str, torch.Tensor],  # walks_dict[type]['nodes'] (non-anonymized)
    #     return_all: bool = False
    # ) -> Union[torch.Tensor, Dict]:
    #     """
    #     Main forward pass for HCT.
        
    #     Args:
    #         walks_dict: Dictionary from walk sampler with structure:
    #             {
    #                 'short': {'nodes_anon': [B, num_short, L], 'masks': [B, num_short, L]},
    #                 'long': {'nodes_anon': [B, num_long, L], 'masks': [B, num_long, L]},
    #                 'tawr': {'nodes_anon': [B, num_tawr, L], 'masks': [B, num_tawr, L],
    #                          'restart_flags': [B, num_tawr, L]}
    #             }
    #         node_embeddings: [batch_size, d_model] base node embeddings (optional)
    #         return_all: If True, return all intermediate outputs
            
    #     Returns:
    #         fused_embedding: [batch_size, d_model] final node embedding
    #         or dictionary with all outputs if return_all=True
    #     """
    #     batch_size = walks_dict['short']['nodes_anon'].size(0)
    #     device = walks_dict['short']['nodes_anon'].device
        
    #     # Create node feature embeddings for each walk step
    #     # In a full implementation, this would use the node_embeddings lookup
    #     # Here we simulate by using the provided node_embeddings or random ones
    #     if node_embeddings is None:
    #         node_embeddings = torch.randn(batch_size, self.d_model, device=device)
        
    #     # Process each walk type
    #     outputs = {}
        
    #     for walk_type, type_name in [(0, 'short'), (1, 'long'), (2, 'tawr')]:
    #         data = walks_dict[type_name]
    #         nodes_anon = data['nodes_anon']  # [B, num_walks, L]
    #         masks = data['masks']             # [B, num_walks, L]
            
    #         num_walks = nodes_anon.size(1)
    #         walk_len = nodes_anon.size(2)
            
    #         # Create walk embeddings by looking up node_embeddings
    #         # In practice, you'd have a learned embedding for anonymized node IDs
    #         # Here we use a simple lookup for demonstration
    #         # walk_embeddings = torch.zeros(batch_size, num_walks, walk_len, self.d_model, device=device)
            
    #         # Simulate: use node_embeddings for each step, ignoring actual anonymized IDs
    #         # In real implementation, you'd have an embedding table for anonymized IDs
    #         # for b in range(batch_size):
    #         #     for w in range(num_walks):
    #         #         for step in range(walk_len):
    #         #             if masks[b, w, step] > 0:
    #         #                 # Use base node embedding (simplified)
    #         #                 walk_embeddings[b, w, step] = node_embeddings[b]
            
    #         walk_embeddings = torch.zeros(batch_size, num_walks, walk_len, self.d_model, device=device)
    #         for b in range(batch_size):
    #             for w in range(num_walks):
    #                 for step in range(walk_len):
    #                     if masks[b, w, step] > 0:
    #                         walk_embeddings[b, w, step] = node_embeddings[b]
            
            
            
    #         # Add restart flag information for TAWR walks (if available)
    #         if 'restart_flags' in data and type_name == 'tawr':
    #             restart_flags = data['restart_flags']  # [B, num_walks, L]
    #             # Could incorporate restart flags as additional features
    #             # For example, add a learned embedding for restart positions
    #             restart_embed = torch.zeros_like(walk_embeddings)
    #             restart_embed[restart_flags.bool()] = 0.1  # Simple additive signal
    #             walk_embeddings = walk_embeddings + restart_embed
            
    #         # Process this walk type
    #         output = self.process_walk_type(
    #             walk_embeddings,
    #             nodes_anon,
    #             masks,
    #             walk_type=walk_type
    #         )
    #         outputs[type_name] = output
        
    #     # Fuse all walk types
    #     fused = self.fuse_walk_types(
    #         outputs['short'],
    #         outputs['long'],
    #         outputs['tawr']
    #     )
        
    #     if return_all:
    #         return {
    #             'fused': fused,
    #             'short': outputs['short'],
    #             'long': outputs['long'],
    #             'tawr': outputs['tawr']
    #         }
        
    #     return fused

    def forward(
        self,
        walks_dict: Dict[str, Dict[str, torch.Tensor]],
        node_memory: torch.Tensor,  # [num_nodes, memory_dim] from SAM
        memory_proj: Optional[nn.Module] = None,  # Project memory_dim -> d_model
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
            
            # Project to d_model if needed
            if memory_proj is not None:
                walk_node_feats = memory_proj(walk_node_feats)
            
            # Reshape: [B, num_walks, L, d_model]
            walk_embeddings = walk_node_feats.view(batch_size, num_walks, walk_len, self.d_model)
            
            # Add restart flags for TAWR
            if type_name == 'tawr' and 'restart_flags' in data:
                restart_flags = data['restart_flags'].unsqueeze(-1).float()
                # Learnable restart embedding or simple marker
                restart_embed = restart_flags * 0.1
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
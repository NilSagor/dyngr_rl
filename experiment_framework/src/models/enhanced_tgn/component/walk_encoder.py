from typing import Dict, List, Tuple
import torch 
import torch.nn as nn

from .time_encoder import TimeEncoder

class WalkEncoder(nn.Module):
    """
    Encode multi-scale walks into fixed-size representations.
    Uses hierarchical attention over walk steps and walk types.
    """
    def __init__(
        self,
        walk_length_short: int,
        walk_length_long: int,
        walk_length_tawr: int,
        memory_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.memory_dim = memory_dim
        self.output_dim = output_dim
        
        # Walk step encoders (process each step in a walk)
        self.step_encoder_short = nn.Linear(memory_dim, memory_dim)
        self.step_encoder_long = nn.Linear(memory_dim, memory_dim)
        self.step_encoder_tawr = nn.Linear(memory_dim, memory_dim)
        
        # Temporal encoding for walk steps
        self.time_encoder = TimeEncoder(memory_dim)
        
        # Cross-step attention (within each walk)
        self.step_attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-walk attention (aggregate multiple walks of same type)
        self.walk_attention_short = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.walk_attention_long = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.walk_attention_tawr = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-scale fusion (combine short/long/tawr representations)
        self.scale_fusion = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(memory_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def encode_walk_type(
        self,
        walk_nodes: torch.Tensor,  # [batch, num_walks, walk_len]
        walk_times: torch.Tensor,  # [batch, num_walks, walk_len]
        walk_masks: torch.Tensor,  # [batch, num_walks, walk_len]
        memory_states: torch.Tensor,  # [num_nodes, memory_dim]
        step_encoder: nn.Module,
        walk_attention: nn.Module
    ) -> torch.Tensor:
        """
        Encode walks of a single type (short/long/tawr).
        
        Returns: [batch, memory_dim] aggregated representation
        """
        batch_size, num_walks, walk_len = walk_nodes.shape
        device = walk_nodes.device
        
        # Gather memory states for walk nodes
        # walk_nodes: [batch, num_walks, walk_len] -> [batch*num_walks*walk_len]
        flat_nodes = walk_nodes.reshape(-1)
        flat_node_feats = memory_states[flat_nodes]  # [batch*num_walks*walk_len, memory_dim]
        walk_feats = flat_node_feats.view(batch_size, num_walks, walk_len, self.memory_dim)
        
        # Add temporal encoding
        flat_times = walk_times.reshape(-1)
        time_enc = self.time_encoder(flat_times)  # [batch*num_walks*walk_len, memory_dim]
        time_enc = time_enc.view(batch_size, num_walks, walk_len, self.memory_dim)
        
        # Combine node features with temporal encoding
        walk_feats = walk_feats + time_enc  # [batch, num_walks, walk_len, memory_dim]
        
        # Encode each step
        walk_feats = step_encoder(walk_feats)  # [batch, num_walks, walk_len, memory_dim]
        
        # Reshape for attention: [batch*num_walks, walk_len, memory_dim]
        walk_feats_flat = walk_feats.view(batch_size * num_walks, walk_len, self.memory_dim)
        masks_flat = walk_masks.view(batch_size * num_walks, walk_len).bool()
        
        # Invert mask for attention (True = ignore)
        attn_mask = ~masks_flat  # [batch*num_walks, walk_len]
        
        # Cross-step attention (aggregate steps within each walk)
        attended_steps, _ = self.step_attention(
            query=walk_feats_flat,
            key=walk_feats_flat,
            value=walk_feats_flat,
            key_padding_mask=attn_mask
        )  # [batch*num_walks, walk_len, memory_dim]
        
        # Mean pool over valid steps
        step_masks_expanded = masks_flat.unsqueeze(-1).float()  # [batch*num_walks, walk_len, 1]
        walk_reprs = (attended_steps * step_masks_expanded).sum(dim=1) / \
                     (step_masks_expanded.sum(dim=1) + 1e-8)  # [batch*num_walks, memory_dim]
        
        # Reshape for cross-walk attention: [batch, num_walks, memory_dim]
        walk_reprs = walk_reprs.view(batch_size, num_walks, self.memory_dim)
        
        # Cross-walk attention (aggregate multiple walks)
        walked_attended, _ = walk_attention(
            query=walk_reprs,
            key=walk_reprs,
            value=walk_reprs
        )  # [batch, num_walks, memory_dim]
        
        # Mean pool over walks
        walk_repr = walked_attended.mean(dim=1)  # [batch, memory_dim]
        
        return walk_repr
    
    def forward(
        self,
        walks_source: Dict[str, Dict[str, torch.Tensor]],
        walks_target: Dict[str, Dict[str, torch.Tensor]],
        memory_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode walks for source and target nodes.
        
        Returns:
            source_emb: [batch, output_dim]
            target_emb: [batch, output_dim]
        """
        batch_size = walks_source['short']['nodes'].shape[0]
        device = walks_source['short']['nodes'].device
        
        # Process each scale for source
        src_short = self.encode_walk_type(
            walks_source['short']['nodes'],
            walks_source['short']['times'],
            walks_source['short']['masks'],
            memory_states,
            self.step_encoder_short,
            self.walk_attention_short
        )  # [batch, memory_dim]
        
        src_long = self.encode_walk_type(
            walks_source['long']['nodes'],
            walks_source['long']['times'],
            walks_source['long']['masks'],
            memory_states,
            self.step_encoder_long,
            self.walk_attention_long
        )
        
        src_tawr = self.encode_walk_type(
            walks_source['tawr']['nodes'],
            walks_source['tawr']['times'],
            walks_source['tawr']['masks'],
            memory_states,
            self.step_encoder_tawr,
            self.walk_attention_tawr
        )
        
        # Stack scales: [batch, 3, memory_dim]
        src_scales = torch.stack([src_short, src_long, src_tawr], dim=1)
        
        # Cross-scale fusion
        src_fused, _ = self.scale_fusion(
            query=src_scales,
            key=src_scales,
            value=src_scales
        )  # [batch, 3, memory_dim]
        
        # Mean pool over scales
        src_repr = src_fused.mean(dim=1)  # [batch, memory_dim]
        
        # Same for target
        tgt_short = self.encode_walk_type(
            walks_target['short']['nodes'],
            walks_target['short']['times'],
            walks_target['short']['masks'],
            memory_states,
            self.step_encoder_short,
            self.walk_attention_short
        )
        
        tgt_long = self.encode_walk_type(
            walks_target['long']['nodes'],
            walks_target['long']['times'],
            walks_target['long']['masks'],
            memory_states,
            self.step_encoder_long,
            self.walk_attention_long
        )
        
        tgt_tawr = self.encode_walk_type(
            walks_target['tawr']['nodes'],
            walks_target['tawr']['times'],
            walks_target['tawr']['masks'],
            memory_states,
            self.step_encoder_tawr,
            self.walk_attention_tawr
        )
        
        tgt_scales = torch.stack([tgt_short, tgt_long, tgt_tawr], dim=1)
        tgt_fused, _ = self.scale_fusion(
            query=tgt_scales,
            key=tgt_scales,
            value=tgt_scales
        )
        tgt_repr = tgt_fused.mean(dim=1)
        
        # Project to output dimension
        source_emb = self.output_proj(src_repr)  # [batch, output_dim]
        target_emb = self.output_proj(tgt_repr)  # [batch, output_dim]
        
        return source_emb, target_emb
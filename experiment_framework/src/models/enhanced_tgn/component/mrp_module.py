import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MutualRefineAndPooling(nn.Module):
    """
    Mutual Refine & Pooling (Fixed Version)
    
    Architecture:
      1. Bidirectional cross-attention between src and dst fused embeddings
      2. Adaptive gating for controlled information exchange
      3. Per-type walk pooling (if per-type embeddings provided)
      4. Cross-type fusion preserving type distinctions
      5. Residual FFN with layer norm
    """
    def __init__(
        self,
        d_model: int = 172,
        nhead: int = 4,
        dropout: float = 0.1,
        num_walk_types: int = 3,  # short, long, tawr
        max_walks_per_type: int = 5,  # Maximum walks per type (for pooling)
    ):
        super().__init__()
        self.d_model = d_model
        self.num_walk_types = num_walk_types
        self.max_walks_per_type = max_walks_per_type

        # ==================== Bidirectional Cross-Attention ====================
        self.cross_attn_src2dst = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_dst2src = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # ==================== Adaptive Gating ====================
        self.gate_src = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.gate_dst = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # ==================== Per-Walk Type Pooling ====================
        # This pools across walks WITHIN each type (not across types)
        self.within_type_pool = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, dropout=dropout, batch_first=True
        )
        
        # Learnable query for pooling (instead of self-attention)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.xavier_uniform_(self.pool_query)

        # ==================== Cross-Type Fusion ====================
        # Now preserves per-type information by keeping them separate
        self.type_fusion_src = nn.Sequential(
            nn.Linear(d_model * num_walk_types, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.type_fusion_dst = nn.Sequential(
            nn.Linear(d_model * num_walk_types, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # ==================== Final Processing ====================
        self.norm_src = nn.LayerNorm(d_model)
        self.norm_dst = nn.LayerNorm(d_model)
        self.ffn_src = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_dst = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        src_walk: torch.Tensor,           # [B, D] fused src embedding
        dst_walk: torch.Tensor,           # [B, D] fused dst embedding
        src_per_type: Optional[torch.Tensor] = None,  # [B, num_types, max_walks, D] (raw walks)
        dst_per_type: Optional[torch.Tensor] = None,  # [B, num_types, max_walks, D]
        src_masks: Optional[torch.Tensor] = None,     # [B, num_types, max_walks] (valid walks mask)
        dst_masks: Optional[torch.Tensor] = None,     # [B, num_types, max_walks]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = src_walk.shape[0]
        
        # ==================== 1. Bidirectional Cross-Attention ====================
        s = src_walk.unsqueeze(1)  # [B, 1, D]
        d = dst_walk.unsqueeze(1)  # [B, 1, D]
        
        src_cross, src_attn = self.cross_attn_src2dst(query=s, key=d, value=d)
        dst_cross, dst_attn = self.cross_attn_dst2src(query=d, key=s, value=s)
        
        # ==================== 2. Adaptive Gating ====================
        gate_src = self.gate_src(torch.cat([src_walk, src_cross.squeeze(1)], dim=-1))
        gate_dst = self.gate_dst(torch.cat([dst_walk, dst_cross.squeeze(1)], dim=-1))
        
        refined_src = src_walk + gate_src * src_cross.squeeze(1)
        refined_dst = dst_walk + gate_dst * dst_cross.squeeze(1)
        
        # ==================== 3. Hierarchical Pooling (if raw walks provided) ====================
        if src_per_type is not None and dst_per_type is not None:
            # Pool within each walk type (across walks, not types)
            src_type_embeds = []
            dst_type_embeds = []
            
            # Expand pool query for batch
            pool_query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
            
            for t in range(self.num_walk_types):
                # Extract walks of this type
                src_walks_t = src_per_type[:, t]  # [B, max_walks, D]
                dst_walks_t = dst_per_type[:, t]  # [B, max_walks, D]
                
                # Get masks for this type (if provided)
                src_mask_t = src_masks[:, t] if src_masks is not None else None
                dst_mask_t = dst_masks[:, t] if dst_masks is not None else None
                
                # Pool using attention with learnable query
                # This effectively does weighted average across walks
                src_pooled, _ = self.within_type_pool(
                    query=pool_query,
                    key=src_walks_t,
                    value=src_walks_t,
                    key_padding_mask=~src_mask_t if src_mask_t is not None else None
                )  # [B, 1, D]
                
                dst_pooled, _ = self.within_type_pool(
                    query=pool_query,
                    key=dst_walks_t,
                    value=dst_walks_t,
                    key_padding_mask=~dst_mask_t if dst_mask_t is not None else None
                )  # [B, 1, D]
                
                src_type_embeds.append(src_pooled.squeeze(1))  # [B, D]
                dst_type_embeds.append(dst_pooled.squeeze(1))
            
            # Stack type embeddings (preserving type information)
            src_type_stack = torch.stack(src_type_embeds, dim=1)  # [B, num_types, D]
            dst_type_stack = torch.stack(dst_type_embeds, dim=1)  # [B, num_types, D]
            
            # Fuse across types with per-type preservation
            # Flatten but keep type order: [B, num_types * D]
            src_flat = src_type_stack.reshape(B, -1)
            dst_flat = dst_type_stack.reshape(B, -1)
            
            # Project back to d_model (implicitly mixes type information)
            src_fused = self.type_fusion_src(src_flat)  # [B, D]
            dst_fused = self.type_fusion_dst(dst_flat)  # [B, D]
            
            # Add to refined embeddings
            refined_src = refined_src + src_fused
            refined_dst = refined_dst + dst_fused
        
        # ==================== 4. Final FFN + Residual ====================
        refined_src = self.norm_src(refined_src + self.ffn_src(refined_src))
        refined_dst = self.norm_dst(refined_dst + self.ffn_dst(refined_dst))
        
        return refined_src, refined_dst
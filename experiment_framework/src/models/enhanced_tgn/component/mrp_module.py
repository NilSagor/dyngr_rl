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

        # Validate nhead divides d_model
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        # Store head dimension for checks
        self.head_dim = d_model // nhead

        
        
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
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.type_fusion_dst = nn.Sequential(
            nn.Linear(d_model * num_walk_types, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # ==================== Final Processing ====================
        self.norm_src = nn.LayerNorm(d_model)
        self.norm_dst = nn.LayerNorm(d_model)
        self.ffn_src = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),  # GELU is more stable than ReLU
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_dst = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
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
        # Validate inputs
        B, D = src_walk.shape
        assert D == self.d_model, f"src_walk dim {D} != d_model {self.d_model}"
        assert dst_walk.shape == (B, D), f"dst_walk shape {dst_walk.shape} != {(B, D)}"
        
        # ==================== 1. Bidirectional Cross-Attention ====================
        s = src_walk.unsqueeze(1)  # [B, 1, D]
        d = dst_walk.unsqueeze(1)  # [B, 1, D]
        
        src_cross, _ = self.cross_attn_src2dst(query=s, key=d, value=d)
        dst_cross, _ = self.cross_attn_dst2src(query=d, key=s, value=s)

        src_cross = src_cross.squeeze(1)  # [B, D]
        dst_cross = dst_cross.squeeze(1)  # [B, D]
        
        # ==================== 2. Adaptive Gating ====================
        gate_src = self.gate_src(torch.cat([src_walk, src_cross], dim=-1))
        gate_dst = self.gate_dst(torch.cat([dst_walk, dst_cross], dim=-1))
        
        refined_src = src_walk + gate_src * src_cross
        refined_dst = dst_walk + gate_dst * dst_cross
        
        # ==================== 3. Hierarchical Pooling (if raw walks provided) ====================
        if src_per_type is not None and dst_per_type is not None:
            # Validate shapes
            expected_shape = (B, self.num_walk_types, self.max_walks_per_type, D)
            assert src_per_type.shape == expected_shape, \
                f"src_per_type shape {src_per_type.shape} != {expected_shape}"
            assert dst_per_type.shape == expected_shape, \
                f"dst_per_type shape {dst_per_type.shape} != {expected_shape}"
            
            # Ensure masks are boolean and correct shape
            if src_masks is not None:
                if src_masks.dtype != torch.bool:
                    src_masks = src_masks.bool()
                assert src_masks.shape == (B, self.num_walk_types, self.max_walks_per_type), \
                    f"src_masks shape {src_masks.shape} != {(B, self.num_walk_types, self.max_walks_per_type)}"
            
            if dst_masks is not None:
                if dst_masks.dtype != torch.bool:
                    dst_masks = dst_masks.bool()
                assert dst_masks.shape == (B, self.num_walk_types, self.max_walks_per_type)
            
            # Pool within each walk type
            src_type_embeds = []
            dst_type_embeds = []
            
           
            # Expand pool query for batch
            pool_query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
            
            for t in range(self.num_walk_types):
                # Extract walks of this type: [B, max_walks, D]
                src_walks_t = src_per_type[:, t]
                dst_walks_t = dst_per_type[:, t]
                
                # Get masks: [B, max_walks], True = valid, False = pad
                # For key_padding_mask: True = mask out (pad), so we invert
                src_mask_t = None
                dst_mask_t = None
                
                if src_masks is not None:
                    src_mask_t = src_masks[:, t]  # [B, max_walks]
                    # Check if all walks are valid (no padding needed)
                    if not src_mask_t.any():
                        # All masked - use zeros
                        src_pooled = torch.zeros(B, 1, D, device=src_walk.device)
                    else:
                        # key_padding_mask expects True for positions to MASK OUT
                        # So we pass ~src_mask_t (invert: valid becomes False, pad becomes True)
                        src_pooled, _ = self.within_type_pool(
                            query=pool_query,
                            key=src_walks_t,
                            value=src_walks_t,
                            key_padding_mask=~src_mask_t  # True = mask out
                        )
                else:
                    # No mask - assume all valid
                    src_pooled, _ = self.within_type_pool(
                        query=pool_query, key=src_walks_t, value=src_walks_t
                    )
                
                if dst_masks is not None:
                    dst_mask_t = dst_masks[:, t]
                    if not dst_mask_t.any():
                        dst_pooled = torch.zeros(B, 1, D, device=dst_walk.device)
                    else:
                        dst_pooled, _ = self.within_type_pool(
                            query=pool_query,
                            key=dst_walks_t,
                            value=dst_walks_t,
                            key_padding_mask=~dst_mask_t
                        )
                else:
                    dst_pooled, _ = self.within_type_pool(
                        query=pool_query, key=dst_walks_t, value=dst_walks_t
                    )
                
                src_type_embeds.append(src_pooled.squeeze(1))  # [B, D]
                dst_type_embeds.append(dst_pooled.squeeze(1))
            
            # Stack: [B, num_types, D]
            src_type_stack = torch.stack(src_type_embeds, dim=1)
            dst_type_stack = torch.stack(dst_type_embeds, dim=1)
            
            # Flatten and fuse across types
            src_flat = src_type_stack.reshape(B, -1)  # [B, num_types * D]
            dst_flat = dst_type_stack.reshape(B, -1)
            
            src_fused = self.type_fusion_src(src_flat)  # [B, D]
            dst_fused = self.type_fusion_dst(dst_flat)
            
            # Residual addition
            refined_src = refined_src + src_fused
            refined_dst = refined_dst + dst_fused
        
        # ==================== 4. Final FFN + Residual ====================
        refined_src = refined_src + self.ffn_src(refined_src)
        refined_src = self.norm_src(refined_src)
        
        refined_dst = refined_dst + self.ffn_dst(refined_dst)
        refined_dst = self.norm_dst(refined_dst)
        
        # Final safety check
        refined_src = torch.nan_to_num(refined_src, nan=0.0, posinf=10.0, neginf=-10.0)
        refined_dst = torch.nan_to_num(refined_dst, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return refined_src, refined_dst
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from loguru import logger

class GatedMutualRefinementPooling(nn.Module):
    """
    Gated Mutual Refinement & Pooling (Optimized & Vectorized)
    
    Features:
    1. Tri-Modal Gating (SAM + HCT + STODE)
    2. Fully Vectorized Cross-Type Attention Pooling (No Loops)
    3. Bidirectional Cross-Attention with Pre-LN
    4. Multi-Scale Output for Feedback Loops
    """
    def __init__(
        self,
        d_model: int = 172,
        nhead: int = 4,
        dropout: float = 0.1,
        num_walk_types: int = 3,
        max_walks_per_type: int = 5,
        modalities: int = 3,  # SAM, HCT, STODE
    ):
        super().__init__()
        self.d_model = d_model
        self.num_walk_types = num_walk_types
        self.max_walks_per_type = max_walks_per_type
        self.modalities = modalities
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.head_dim = d_model // nhead

        # ==================== 1. Tri-Modal Gating ====================
        # Projects concatenated modalities [SAM, HCT, STODE] -> Gate Weights
        # Input dim: d_model * modalities
        self.modality_proj = nn.Linear(d_model * modalities, d_model * modalities)
        self.modality_gate = nn.Sequential(
            nn.Linear(d_model * modalities, d_model),
            nn.Sigmoid()
        )
        
        # Output projection after gating fusion
        self.fusion_proj = nn.Linear(d_model * modalities, d_model)

        # ==================== 2. Bidirectional Cross-Attention (Pre-LN) ====================
        self.norm_src_in = nn.LayerNorm(d_model)
        self.norm_dst_in = nn.LayerNorm(d_model)
        
        self.cross_attn_src2dst = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_dst2src = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        
        self.norm_cross = nn.LayerNorm(d_model)

        # ==================== 3. Vectorized Cross-Type Attention Pooling ====================
        # Process all types in one go: [B, Types, Walks, D]
        self.norm_pool_in = nn.LayerNorm(d_model)
        
        # Single attention layer for all types (flattened B*Types as batch)
        self.type_pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, dropout=dropout, batch_first=True
        )
        
        # Learnable pool query [1, 1, D] -> expanded to [B*Types, 1, D]
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.xavier_uniform_(self.pool_query)
        
        self.norm_pool_out = nn.LayerNorm(d_model)

        # ==================== 4. Cross-Type Fusion (Vectorized) ====================
        # Input: [B, Types, D] -> Flatten to [B, Types*D]
        self.type_fusion_net = nn.Sequential(
            nn.Linear(d_model * num_walk_types, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

        # ==================== 5. Final FFN Block (Pre-LN) ====================
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm_final = nn.LayerNorm(d_model)

    def _safe_sanitize(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.isfinite(x).all():
            logger.warning(f"NaN/Inf detected in {name}, sanitizing.")
            return torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        return x

    def forward(
        self,
        # Primary Inputs (Fused from previous layers)
        src_hct: torch.Tensor,          # [B, D] - From HCT
        dst_hct: torch.Tensor,          # [B, D] - From HCT
        
        # Tri-Modal Inputs for Gating
        src_sam: Optional[torch.Tensor] = None,   # [B, D] - From SAM Memory
        dst_sam: Optional[torch.Tensor] = None,
        src_stode: Optional[torch.Tensor] = None, # [B, D] - From STODE
        dst_stode: Optional[torch.Tensor] = None,
        
        # Raw Walks for Hierarchical Pooling
        src_per_type: Optional[torch.Tensor] = None,  # [B, Types, MaxWalks, D]
        dst_per_type: Optional[torch.Tensor] = None,
        src_masks: Optional[torch.Tensor] = None,     # [B, Types, MaxWalks] (Bool: True=Valid)
        dst_masks: Optional[torch.Tensor] = None,
        
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        
        B, D = src_hct.shape
        device = src_hct.device
        
        # --- Default Missing Modalities to HCT (Self-Loop) ---
        if src_sam is None: src_sam = src_hct
        if dst_sam is None: dst_sam = dst_hct
        if src_stode is None: src_stode = src_hct
        if dst_stode is None: dst_stode = dst_hct
        
        # Sanitize inputs
        src_hct = self._safe_sanitize(src_hct, "src_hct")
        dst_hct = self._safe_sanitize(dst_hct, "dst_hct")
        src_sam = self._safe_sanitize(src_sam, "src_sam")
        dst_sam = self._safe_sanitize(dst_sam, "dst_sam")
        src_stode = self._safe_sanitize(src_stode, "src_stode")
        dst_stode = self._safe_sanitize(dst_stode, "dst_stode")

        intermediates = {}

        # ==================== 1. Tri-Modal Gating ====================
        # Stack modalities: [B, Modalities, D] -> [B, Modalities*D]
        src_modalities = torch.stack([src_sam, src_hct, src_stode], dim=1).reshape(B, -1)
        dst_modalities = torch.stack([dst_sam, dst_hct, dst_stode], dim=1).reshape(B, -1)
        
        # Compute Gates
        # Project then Sigmoid to get weights in [0, 1]
        src_gate_weights = self.modality_gate(self.modality_proj(src_modalities)) # [B, D]
        dst_gate_weights = self.modality_gate(self.modality_proj(dst_modalities))
        
        # Weighted Sum of Modalities
        # Split weights back to [B, 3, D] implicitly by multiplying stacked tensors
        src_stack = torch.stack([src_sam, src_hct, src_stode], dim=1) # [B, 3, D]
        dst_stack = torch.stack([dst_sam, dst_hct, dst_stode], dim=1)
        
        # Expand gate to [B, 3, D] for broadcasting
        src_gw_exp = src_gate_weights.unsqueeze(1) # [B, 1, D] -> broadcast to [B, 3, D]? 
        # Wait, gate is [B, D]. We need to apply specific weights to specific modalities?
        # Current design: One gate vector per modality set? 
        # Better: Project to 3*D, split into 3 gates, apply respectively.
        
        # Refined Gating: Split the projected vector into 3 distinct gates
        full_proj_src = self.modality_proj(src_modalities) # [B, 3*D]
        full_proj_dst = self.modality_proj(dst_modalities)
        
        g_src_list = torch.chunk(full_proj_src, self.modalities, dim=1) # 3x [B, D]
        g_dst_list = torch.chunk(full_proj_dst, self.modalities, dim=1)
        
        g_src = torch.stack([torch.sigmoid(g) for g in g_src_list], dim=1) # [B, 3, D]
        g_dst = torch.stack([torch.sigmoid(g) for g in g_dst_list], dim=1)
        
        # Apply Gates
        fused_src = (src_stack * g_src).sum(dim=1) # [B, D]
        fused_dst = (dst_stack * g_dst).sum(dim=1)
        
        # Projection back to D (optional, but good for mixing)
        # Actually sum already gives [B,D]. Let's add a residual from HCT
        fused_src = self.fusion_proj(torch.cat([src_sam, src_hct, src_stode], dim=-1)) # Alternative fusion
        # Let's stick to the gated sum as the primary "refined" input for next steps
        # But to match dimensions, let's project the gated sum if needed. 
        # Here gated sum is already D. Let's use it as base.
        base_src = fused_src
        base_dst = fused_dst
        
        intermediates['gated_src'] = base_src.detach()
        intermediates['gated_dst'] = base_dst.detach()

        # ==================== 2. Bidirectional Cross-Attention ====================
        # Pre-LN
        s_norm = self.norm_src_in(base_src.unsqueeze(1)) # [B, 1, D]
        d_norm = self.norm_dst_in(base_dst.unsqueeze(1))
        
        # Cross Attention
        src_ctx, _ = self.cross_attn_src2dst(query=s_norm, key=d_norm, value=d_norm)
        dst_ctx, _ = self.cross_attn_dst2src(query=d_norm, key=s_norm, value=s_norm)
        
        # Residual + Pre-LN output norm
        src_x = base_src + src_ctx.squeeze(1)
        dst_x = base_dst + dst_ctx.squeeze(1)
        
        src_x = self.norm_cross(src_x)
        dst_x = self.norm_cross(dst_x)
        
        intermediates['cross_refined_src'] = src_x.detach()
        intermediates['cross_refined_dst'] = dst_x.detach()

        # ==================== 3. Vectorized Hierarchical Pooling ====================
        if src_per_type is not None and dst_per_type is not None:
            # Input: [B, Types, Walks, D]
            # Normalize Input
            src_pt_norm = self.norm_pool_in(src_per_type)
            dst_pt_norm = self.norm_pool_in(dst_per_type)
            
            # Reshape to [B*Types, Walks, D] for batched attention
            B, T, W, D_pt = src_pt_norm.shape
            src_flat = src_pt_norm.reshape(B * T, W, D_pt)
            dst_flat = dst_pt_norm.reshape(B * T, W, D_pt)
            
            # Prepare Masks: [B, Types, Walks] -> [B*Types, Walks]
            # key_padding_mask: True = Mask Out
            src_kpm = None
            dst_kpm = None
            
            if src_masks is not None:
                # Invert: Valid=True -> Mask=False. Invalid=False -> Mask=True
                src_kpm = (~src_masks.reshape(B * T, W)).contiguous()
            if dst_masks is not None:
                dst_kpm = (~dst_masks.reshape(B * T, W)).contiguous()
            
            # Expand Query: [1, 1, D] -> [B*T, 1, D]
            q_exp = self.pool_query.expand(B * T, -1, -1)
            
            # Batched Attention (All types processed in parallel)
            src_pooled_flat, _ = self.type_pool_attn(
                query=q_exp, key=src_flat, value=src_flat, key_padding_mask=src_kpm
            )
            dst_pooled_flat, _ = self.type_pool_attn(
                query=q_exp, key=dst_flat, value=dst_flat, key_padding_mask=dst_kpm
            )
            
            # Reshape back: [B*T, 1, D] -> [B, T, D]
            src_pooled = src_pooled_flat.squeeze(1).reshape(B, T, D_pt)
            dst_pooled = dst_pooled_flat.squeeze(1).reshape(B, T, D_pt)
            
            # Post-LN
            src_pooled = self.norm_pool_out(src_pooled)
            dst_pooled = self.norm_pool_out(dst_pooled)
            
            intermediates['pooled_per_type_src'] = src_pooled.detach()
            intermediates['pooled_per_type_dst'] = dst_pooled.detach()
            
            # ==================== 4. Cross-Type Fusion ====================
            # Flatten [B, T, D] -> [B, T*D]
            src_flat_type = src_pooled.reshape(B, -1)
            dst_flat_type = dst_pooled.reshape(B, -1)
            
            type_feat_src = self.type_fusion_net(src_flat_type)
            type_feat_dst = self.type_fusion_net(dst_flat_type)
            
            # Residual Connection to Cross-Attention Output
            src_x = src_x + type_feat_src
            dst_x = dst_x + type_feat_dst

        # ==================== 5. Final FFN Block ====================
        # Pre-LN
        src_ffn_in = self.norm_ffn(src_x)
        dst_ffn_in = self.norm_ffn(dst_x)
        
        src_out = src_x + self.ffn(src_ffn_in)
        dst_out = dst_x + self.ffn(dst_ffn_in)
        
        # Final Norm
        src_out = self.norm_final(src_out)
        dst_out = self.norm_final(dst_out)
        
        # Final Sanitization
        src_out = self._safe_sanitize(src_out, "final_src")
        dst_out = self._safe_sanitize(dst_out, "final_dst")
        
        if return_intermediates:
            return src_out, dst_out, intermediates
        
        return src_out, dst_out
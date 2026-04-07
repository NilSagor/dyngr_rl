import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from loguru import logger


class GatedMutualRefinementPooling(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 2,
        dropout: float = 0.1,
        num_walk_types: int = 3,       
        modalities: int = 3,
        fusion_mode: str = "gated_plus_residual",
        pool_attn_heads: int = 1, 
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_walk_types = num_walk_types
        self.modalities = modalities
        self.fusion_mode = fusion_mode          
        self.pool_attn_heads = pool_attn_heads  
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        # Tri-Modal Gating 
        self.modality_proj = nn.Linear(d_model * modalities, d_model * modalities)
        self.modality_gate = nn.Sequential(
            nn.Linear(d_model * modalities, d_model),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Linear(d_model * modalities, d_model)

        # Bidirectional Cross-Attention
        self.norm_src_in = nn.LayerNorm(d_model)
        self.norm_dst_in = nn.LayerNorm(d_model)
        
        self.cross_attn_src2dst = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_dst2src = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        
        self.norm_cross = nn.LayerNorm(d_model)

        # Vectorized Cross-Type Attention Pooling
        self.norm_pool_in = nn.LayerNorm(d_model)
        
        self.type_pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=self.pool_attn_heads,
            dropout=dropout, 
            batch_first=True
        )
        
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm_pool_out = nn.LayerNorm(d_model)

        # Cross-Type Fusion 
        self.type_fusion_net = nn.Sequential(
            nn.Linear(d_model * num_walk_types, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

        # Final FFN Block 
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm_final = nn.LayerNorm(d_model)

    def forward(
        self,
        src_hct: torch.Tensor,
        dst_hct: torch.Tensor,
        src_sam: Optional[torch.Tensor] = None,
        dst_sam: Optional[torch.Tensor] = None,
        src_stode: Optional[torch.Tensor] = None,
        dst_stode: Optional[torch.Tensor] = None,
        src_per_type: Optional[torch.Tensor] = None,
        dst_per_type: Optional[torch.Tensor] = None,
        src_masks: Optional[torch.Tensor] = None,
        dst_masks: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
       
        B, D = src_hct.shape
        device = src_hct.device
        
        # CRITICAL: Clone ALL inputs to break any view relationships
        src_hct = src_hct.clone()
        dst_hct = dst_hct.clone()
        src_sam = src_sam.clone() if src_sam is not None else src_hct
        dst_sam = dst_sam.clone() if dst_sam is not None else dst_hct
        src_stode = src_stode.clone() if src_stode is not None else src_hct
        dst_stode = dst_stode.clone() if dst_stode is not None else dst_hct
        src_per_type = src_per_type.clone() if src_per_type is not None else None
        dst_per_type = dst_per_type.clone() if dst_per_type is not None else None

        intermediates = {} 

        # 1. Tri-Modal Gating 
        src_modalities = torch.cat([src_sam, src_hct, src_stode], dim=-1)
        dst_modalities = torch.cat([dst_sam, dst_hct, dst_stode], dim=-1)
        
        # Compute gates
        gate_src = torch.sigmoid(self.modality_gate(src_modalities))
        gate_dst = torch.sigmoid(self.modality_gate(dst_modalities))
        
        # Stack and apply gates
        src_stack = torch.stack([src_sam, src_hct, src_stode], dim=1)
        dst_stack = torch.stack([dst_sam, dst_hct, dst_stode], dim=1)
        
        # Weighted sum (no in-place)
        base_src = (src_stack * gate_src.unsqueeze(1)).sum(dim=1)
        base_dst = (dst_stack * gate_dst.unsqueeze(1)).sum(dim=1)
        
        # Add residual projection
        if self.fusion_mode == "gated_plus_residual":
            base_src = base_src + self.fusion_proj(src_modalities)
            base_dst = base_dst + self.fusion_proj(dst_modalities)
        elif self.fusion_mode == "proj_only":
            base_src = self.fusion_proj(src_modalities)
            base_dst = self.fusion_proj(dst_modalities)
        
        if return_intermediates:
            intermediates['gated_src'] = base_src.detach()
            intermediates['gated_dst'] = base_dst.detach()

        # 2. Bidirectional Cross-Attention
        s_norm = self.norm_src_in(base_src.unsqueeze(1))
        d_norm = self.norm_dst_in(base_dst.unsqueeze(1))
        
        src_ctx, _ = self.cross_attn_src2dst(query=s_norm, key=d_norm, value=d_norm)
        dst_ctx, _ = self.cross_attn_dst2src(query=d_norm, key=s_norm, value=s_norm)
        
        # NO in-place operations - create new tensors
        src_x = base_src + src_ctx.squeeze(1)
        dst_x = base_dst + dst_ctx.squeeze(1)
        
        src_x = self.norm_cross(src_x)
        dst_x = self.norm_cross(dst_x)
        
        if return_intermediates:
            intermediates['cross_refined_src'] = src_x.detach()
            intermediates['cross_refined_dst'] = dst_x.detach()

        # 3. Hierarchical Pooling 
        if src_per_type is not None and dst_per_type is not None:
            # Clone before processing
            src_pt = src_per_type.clone()
            dst_pt = dst_per_type.clone()
            
            src_pt_norm = self.norm_pool_in(src_pt)
            dst_pt_norm = self.norm_pool_in(dst_pt)
            
            B_pt, T, W, D_pt = src_pt_norm.shape
            src_flat = src_pt_norm.reshape(B_pt * T, W, D_pt)
            dst_flat = dst_pt_norm.reshape(B_pt * T, W, D_pt)
            
            src_kpm = None
            dst_kpm = None
            if src_masks is not None:
                src_kpm = (~src_masks.reshape(B_pt * T, W)).contiguous()
            if dst_masks is not None:
                dst_kpm = (~dst_masks.reshape(B_pt * T, W)).contiguous()
            
            q_exp = self.pool_query.expand(B_pt * T, -1, -1)
            
            src_pooled_flat, _ = self.type_pool_attn(
                query=q_exp, key=src_flat, value=src_flat, key_padding_mask=src_kpm
            )
            dst_pooled_flat, _ = self.type_pool_attn(
                query=q_exp, key=dst_flat, value=dst_flat, key_padding_mask=dst_kpm
            )
            
            src_pooled = src_pooled_flat.squeeze(1).reshape(B_pt, T, D_pt)
            dst_pooled = dst_pooled_flat.squeeze(1).reshape(B_pt, T, D_pt)
            
            src_pooled = self.norm_pool_out(src_pooled)
            dst_pooled = self.norm_pool_out(dst_pooled)
            
            if return_intermediates:
                intermediates['pooled_per_type_src'] = src_pooled.detach()
                intermediates['pooled_per_type_dst'] = dst_pooled.detach()
            
            src_flat_type = src_pooled.reshape(B_pt, -1)
            dst_flat_type = dst_pooled.reshape(B_pt, -1)
            
            type_feat_src = self.type_fusion_net(src_flat_type)
            type_feat_dst = self.type_fusion_net(dst_flat_type)
            
            # NO in-place addition
            src_x = src_x + type_feat_src
            dst_x = dst_x + type_feat_dst

        # 4. Final FFN Block 
        src_ffn_in = self.norm_ffn(src_x)
        dst_ffn_in = self.norm_ffn(dst_x)
        
        src_out = src_x + self.ffn(src_ffn_in)
        dst_out = dst_x + self.ffn(dst_ffn_in)
        
        src_out = self.norm_final(src_out)
        dst_out = self.norm_final(dst_out)
        
        if return_intermediates:
            return src_out, dst_out, intermediates
        
        return src_out, dst_out, {}



# class GatedMutualRefinementPooling(nn.Module):
#     """
#     Gated mutual refinement for fusing SAM, HCT, and ST-ODE embeddings.
#     Learnable gates weight each modality, with optional residual
#     projection. Cross-type attention pools walk-type embeddings.
    
#     Features:
#     Tri-Modal Gating (SAM + HCT + STODE)
#     # Tri-modal gating: ablation in exp_017 showed 2.1% AP gain vs. simple concat
#     Fully Vectorized Cross-Type Attention Pooling (No Loops)
#     Bidirectional Cross-Attention with Pre-LN
#     Bidirectional src<->dst attention: for link prediction symmetry
#     Multi-Scale Output for Feedback Loops
#     "Multi-scale" refers to pooling over walk types (short/long/TAWR),
#     not hierarchical feature maps. Feedback loops are handled upstream in HiCoSTv3.

#     """
#     def __init__(
#         self,
#         d_model: int = 128,
#         nhead: int = 2,
#         dropout: float = 0.1,
#         num_walk_types: int = 3,       
#         modalities: int = 3,
#         fusion_mode: str = "gated_plus_residual",
#         pool_attn_heads: int = 1, 
#     ):
#         super().__init__()
        
#         self.d_model = d_model
#         self.num_walk_types = num_walk_types
#         self.modalities = modalities
#         self.fusion_mode = fusion_mode          
#         self.pool_attn_heads = pool_attn_heads  

           

        
#         if d_model % nhead != 0:
#             raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
#         self.head_dim = d_model // nhead

#         #  1. Tri-Modal Gating 
#         self.modality_proj = nn.Linear(d_model * modalities, d_model * modalities)
#         self.modality_gate = nn.Sequential(
#             nn.Linear(d_model * modalities, d_model),
#             nn.Sigmoid()
#         )
        
#         # Output projection after gating fusion (used)
#         self.fusion_proj = nn.Linear(d_model * modalities, d_model)

#         #  2. Bidirectional Cross-Attention
#         self.norm_src_in = nn.LayerNorm(d_model)
#         self.norm_dst_in = nn.LayerNorm(d_model)
        
#         self.cross_attn_src2dst = nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
#         )
#         self.cross_attn_dst2src = nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
#         )
        
#         self.norm_cross = nn.LayerNorm(d_model)

#         #  3. Vectorized Cross-Type Attention Pooling
#         self.norm_pool_in = nn.LayerNorm(d_model)
        
#         self.type_pool_attn = nn.MultiheadAttention(
#             embed_dim=d_model, 
#             num_heads=self.pool_attn_heads,
#             dropout=dropout, 
#             batch_first=True
#         )
        
               
#         self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

#         self.norm_pool_out = nn.LayerNorm(d_model)

#         # 4. Cross-Type Fusion 
#         self.type_fusion_net = nn.Sequential(
#             nn.Linear(d_model * num_walk_types, d_model * 2),
#             nn.LayerNorm(d_model * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 2, d_model),
#             nn.Dropout(dropout)
#         )

#         # 5. Final FFN Block 
#         self.norm_ffn = nn.LayerNorm(d_model)
        

#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 2, d_model),
#             nn.Dropout(dropout)
#         )
        
#         self.norm_final = nn.LayerNorm(d_model)

#     def _safe_sanitize(self, x: torch.Tensor, name: str, training:bool=True) -> torch.Tensor:
#         if not torch.isfinite(x).all():
#             if training:
#                 raise RuntimeError(f"NaN/Inf in {name} during training — check upstream")
#             logger.warning(f"NaN in {name} at inference; replacing with zeros")
#             return torch.nan_to_num(x, nan=0.0)
#         return x

#     def _apply_fusion(
#             self, 
#             stack: torch.Tensor, 
#             gates: torch.Tensor, 
#             modalities_flat: torch.Tensor
#         ) -> torch.Tensor:
#         fused = (stack * gates).sum(dim=1)
#         if self.fusion_mode == "gated_plus_residual":
#             fused = fused + self.fusion_proj(modalities_flat)
#         elif self.fusion_mode == "proj_only":
#             fused = self.fusion_proj(modalities_flat)
#         # "gated_only": just return the gated sum (no change needed)
#         return fused
    
    
#     # def forward(
#     #     self,
#     #     src_hct: torch.Tensor,
#     #     dst_hct: torch.Tensor,
#     #     src_sam: Optional[torch.Tensor] = None,
#     #     dst_sam: Optional[torch.Tensor] = None,
#     #     src_stode: Optional[torch.Tensor] = None,
#     #     dst_stode: Optional[torch.Tensor] = None,
#     #     src_per_type: Optional[torch.Tensor] = None,
#     #     dst_per_type: Optional[torch.Tensor] = None,
#     #     src_masks: Optional[torch.Tensor] = None,
#     #     dst_masks: Optional[torch.Tensor] = None,
#     #     return_intermediates: bool = False
#     # ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
#     #    # mask: 1.0 = valid position, 0.0 = masked/invalid 
       
#     #     B, D = src_hct.shape
#     #     device = src_hct.device
        
#     #     if src_sam is None: src_sam = src_hct
#     #     if dst_sam is None: dst_sam = dst_hct
#     #     if src_stode is None: src_stode = src_hct
#     #     if dst_stode is None: dst_stode = dst_hct
        
#     #     src_hct = self._safe_sanitize(src_hct, "src_hct")
#     #     dst_hct = self._safe_sanitize(dst_hct, "dst_hct")
#     #     src_sam = self._safe_sanitize(src_sam, "src_sam")
#     #     dst_sam = self._safe_sanitize(dst_sam, "dst_sam")
#     #     src_stode = self._safe_sanitize(src_stode, "src_stode")
#     #     dst_stode = self._safe_sanitize(dst_stode, "dst_stode")


#     #     # DEBUG: Check for in-place modifications
#     #     if torch.is_grad_enabled():
#     #         for name, tensor in [
#     #             ('src_hct', src_hct), ('dst_hct', dst_hct),
#     #             ('src_sam', src_sam), ('dst_sam', dst_sam),
#     #             ('src_stode', src_stode), ('dst_stode', dst_stode),
#     #             ('src_per_type', src_per_type), ('dst_per_type', dst_per_type),
#     #         ]:
#     #             if tensor is not None and tensor.requires_grad:
#     #                 logger.debug(f"{name}: version={tensor._version}, requires_grad={tensor.requires_grad}")



#     #     # Always use a dict, empty if not requested
#     #     # intermediates = {} if return_intermediates else None
#     #     intermediates = {} if return_intermediates else {} 

#     #     #  1. Tri-Modal Gating 
#     #     src_modalities = torch.stack([src_sam, src_hct, src_stode], dim=1).reshape(B, -1)
#     #     dst_modalities = torch.stack([dst_sam, dst_hct, dst_stode], dim=1).reshape(B, -1)
        
#     #     full_proj_src = self.modality_proj(src_modalities)
#     #     full_proj_dst = self.modality_proj(dst_modalities)
        
#     #     g_src_list = torch.chunk(full_proj_src, self.modalities, dim=1)
#     #     g_dst_list = torch.chunk(full_proj_dst, self.modalities, dim=1)
        
#     #     g_src = torch.stack([torch.sigmoid(g) for g in g_src_list], dim=1)
#     #     g_dst = torch.stack([torch.sigmoid(g) for g in g_dst_list], dim=1)
        
#     #     src_stack = torch.stack([src_sam, src_hct, src_stode], dim=1)
#     #     dst_stack = torch.stack([dst_sam, dst_hct, dst_stode], dim=1)
        
            

#     #     # # Use fusion_proj to combine all modalities as an alternative (residual)
#     #     base_src = self._apply_fusion(src_stack, g_src, src_modalities)
#     #     base_dst = self._apply_fusion(dst_stack, g_dst, dst_modalities)
        
               
#     #     intermediates['gated_src'] = base_src.detach()
#     #     intermediates['gated_dst'] = base_dst.detach()

#     #     #  2. Bidirectional Cross-Attention
#     #     s_norm = self.norm_src_in(base_src.unsqueeze(1))
#     #     d_norm = self.norm_dst_in(base_dst.unsqueeze(1))
        
#     #     src_ctx, _ = self.cross_attn_src2dst(query=s_norm, key=d_norm, value=d_norm)
#     #     dst_ctx, _ = self.cross_attn_dst2src(query=d_norm, key=s_norm, value=s_norm)
        
#     #     src_x = base_src + src_ctx.squeeze(1)
#     #     dst_x = base_dst + dst_ctx.squeeze(1)
        
#     #     src_x = self.norm_cross(src_x)
#     #     dst_x = self.norm_cross(dst_x)
        
#     #     intermediates['cross_refined_src'] = src_x.detach()
#     #     intermediates['cross_refined_dst'] = dst_x.detach()

#     #     #  3.Hierarchical Pooling 
#     #     if src_per_type is not None and dst_per_type is not None:
#     #         src_pt_norm = self.norm_pool_in(src_per_type.clone() if src_per_type.requires_grad else src_per_type)
#     #         dst_pt_norm = self.norm_pool_in(dst_per_type.clone() if dst_per_type.requires_grad else dst_per_type)
            
#     #         B, T, W, D_pt = src_pt_norm.shape
#     #         src_flat = src_pt_norm.reshape(B * T, W, D_pt)
#     #         dst_flat = dst_pt_norm.reshape(B * T, W, D_pt)
            
#     #         src_kpm = None
#     #         dst_kpm = None
#     #         if src_masks is not None:
#     #             src_kpm = (~src_masks.reshape(B * T, W)).contiguous()
#     #         if dst_masks is not None:
#     #             dst_kpm = (~dst_masks.reshape(B * T, W)).contiguous()
            
#     #         q_exp = self.pool_query.expand(B * T, -1, -1)
            
#     #         src_pooled_flat, _ = self.type_pool_attn(
#     #             query=q_exp, key=src_flat, value=src_flat, key_padding_mask=src_kpm
#     #         )
#     #         dst_pooled_flat, _ = self.type_pool_attn(
#     #             query=q_exp, key=dst_flat, value=dst_flat, key_padding_mask=dst_kpm
#     #         )
            
#     #         src_pooled = src_pooled_flat.squeeze(1).reshape(B, T, D_pt)
#     #         dst_pooled = dst_pooled_flat.squeeze(1).reshape(B, T, D_pt)
            
#     #         src_pooled = self.norm_pool_out(src_pooled)
#     #         dst_pooled = self.norm_pool_out(dst_pooled)
            
#     #         intermediates['pooled_per_type_src'] = src_pooled.detach()
#     #         intermediates['pooled_per_type_dst'] = dst_pooled.detach()
            
#     #         src_flat_type = src_pooled.reshape(B, -1)
#     #         dst_flat_type = dst_pooled.reshape(B, -1)
            
#     #         type_feat_src = self.type_fusion_net(src_flat_type)
#     #         type_feat_dst = self.type_fusion_net(dst_flat_type)
            
#     #         # src_x = src_x.clone() + type_feat_src if src_x.requires_grad else src_x + type_feat_src
#     #         # dst_x = dst_x.clone() + type_feat_dst if dst_x.requires_grad else dst_x + type_feat_dst
#     #         src_x = src_x + type_feat_src
#     #         dst_x = dst_x + type_feat_dst

#     #     # 4. Final FFN Block 
#     #     src_ffn_in = self.norm_ffn(src_x)
#     #     dst_ffn_in = self.norm_ffn(dst_x)
        
#     #     # src_out = src_x.clone() + self.ffn(src_ffn_in) if src_x.requires_grad else src_x + self.ffn(src_ffn_in)
#     #     # dst_out = dst_x.clone() + self.ffn(dst_ffn_in) if dst_x.requires_grad else dst_x + self.ffn(dst_ffn_in)
#     #     src_out = src_x + self.ffn(src_ffn_in)
#     #     dst_out = dst_x + self.ffn(dst_ffn_in)
        
#     #     src_out = self.norm_final(src_out)
#     #     dst_out = self.norm_final(dst_out)
        
#     #     src_out = self._safe_sanitize(src_out, "final_src")
#     #     dst_out = self._safe_sanitize(dst_out, "final_dst")
        
#     #     if return_intermediates:
#     #         return src_out, dst_out, intermediates
        
#     #     return src_out, dst_out, {}
#     def forward(
#         self,
#         src_hct: torch.Tensor,
#         dst_hct: torch.Tensor,
#         src_sam: Optional[torch.Tensor] = None,
#         dst_sam: Optional[torch.Tensor] = None,
#         src_stode: Optional[torch.Tensor] = None,
#         dst_stode: Optional[torch.Tensor] = None,
#         src_per_type: Optional[torch.Tensor] = None,
#         dst_per_type: Optional[torch.Tensor] = None,
#         src_masks: Optional[torch.Tensor] = None,
#         dst_masks: Optional[torch.Tensor] = None,
#         return_intermediates: bool = False
#     ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
    
#         B, D = src_hct.shape
#         device = src_hct.device
        
#         # Ensure inputs are independent (clone if they might share memory)
#         src_hct = src_hct.clone() if src_hct.requires_grad else src_hct
#         dst_hct = dst_hct.clone() if dst_hct.requires_grad else dst_hct
#         src_sam = src_hct.clone() if src_sam is None else src_sam.clone() if src_sam.requires_grad else src_sam
#         dst_sam = dst_hct.clone() if dst_sam is None else dst_sam.clone() if dst_sam.requires_grad else dst_sam
#         src_stode = src_hct.clone() if src_stode is None else src_stode.clone() if src_stode.requires_grad else src_stode
#         dst_stode = dst_hct.clone() if dst_stode is None else dst_stode.clone() if dst_stode.requires_grad else dst_stode

#         # Sanitize
#         src_hct = self._safe_sanitize(src_hct, "src_hct")
#         dst_hct = self._safe_sanitize(dst_hct, "dst_hct")
#         src_sam = self._safe_sanitize(src_sam, "src_sam")
#         dst_sam = self._safe_sanitize(dst_sam, "dst_sam")
#         src_stode = self._safe_sanitize(src_stode, "src_stode")
#         dst_stode = self._safe_sanitize(dst_stode, "dst_stode")

#         intermediates = {} if return_intermediates else {} 

#         # 1. Tri-Modal Gating 
#         src_modalities = torch.stack([src_sam, src_hct, src_stode], dim=1).reshape(B, -1)
#         dst_modalities = torch.stack([dst_sam, dst_hct, dst_stode], dim=1).reshape(B, -1)
        
#         full_proj_src = self.modality_proj(src_modalities)
#         full_proj_dst = self.modality_proj(dst_modalities)
        
#         g_src_list = torch.chunk(full_proj_src, self.modalities, dim=1)
#         g_dst_list = torch.chunk(full_proj_dst, self.modalities, dim=1)
        
#         g_src = torch.stack([torch.sigmoid(g) for g in g_src_list], dim=1)
#         g_dst = torch.stack([torch.sigmoid(g) for g in g_dst_list], dim=1)
        
#         src_stack = torch.stack([src_sam, src_hct, src_stode], dim=1)
#         dst_stack = torch.stack([dst_sam, dst_hct, dst_stode], dim=1)
        
#         # Fusion - NO in-place operations
#         base_src = self._apply_fusion(src_stack, g_src, src_modalities)
#         base_dst = self._apply_fusion(dst_stack, g_dst, dst_modalities)
        
#         if return_intermediates:
#             intermediates['gated_src'] = base_src.detach()
#             intermediates['gated_dst'] = base_dst.detach()

#         # 2. Bidirectional Cross-Attention
#         s_norm = self.norm_src_in(base_src.unsqueeze(1))
#         d_norm = self.norm_dst_in(base_dst.unsqueeze(1))
        
#         src_ctx, _ = self.cross_attn_src2dst(query=s_norm, key=d_norm, value=d_norm)
#         dst_ctx, _ = self.cross_attn_dst2src(query=d_norm, key=s_norm, value=s_norm)
        
#         # NO in-place addition - create new tensor
#         src_x = base_src + src_ctx.squeeze(1)
#         dst_x = base_dst + dst_ctx.squeeze(1)
        
#         src_x = self.norm_cross(src_x)
#         dst_x = self.norm_cross(dst_x)
        
#         if return_intermediates:
#             intermediates['cross_refined_src'] = src_x.detach()
#             intermediates['cross_refined_dst'] = dst_x.detach()

#         # 3. Hierarchical Pooling 
#         if src_per_type is not None and dst_per_type is not None:
#             # Clone to prevent modifying original
#             src_pt_norm = self.norm_pool_in(src_per_type.clone())
#             dst_pt_norm = self.norm_pool_in(dst_per_type.clone())
            
#             B, T, W, D_pt = src_pt_norm.shape
#             src_flat = src_pt_norm.reshape(B * T, W, D_pt)
#             dst_flat = dst_pt_norm.reshape(B * T, W, D_pt)
            
#             src_kpm = None
#             dst_kpm = None
#             if src_masks is not None:
#                 src_kpm = (~src_masks.reshape(B * T, W)).contiguous()
#             if dst_masks is not None:
#                 dst_kpm = (~dst_masks.reshape(B * T, W)).contiguous()
            
#             q_exp = self.pool_query.expand(B * T, -1, -1)
            
#             src_pooled_flat, _ = self.type_pool_attn(
#                 query=q_exp, key=src_flat, value=src_flat, key_padding_mask=src_kpm
#             )
#             dst_pooled_flat, _ = self.type_pool_attn(
#                 query=q_exp, key=dst_flat, value=dst_flat, key_padding_mask=dst_kpm
#             )
            
#             src_pooled = src_pooled_flat.squeeze(1).reshape(B, T, D_pt)
#             dst_pooled = dst_pooled_flat.squeeze(1).reshape(B, T, D_pt)
            
#             src_pooled = self.norm_pool_out(src_pooled)
#             dst_pooled = self.norm_pool_out(dst_pooled)
            
#             if return_intermediates:
#                 intermediates['pooled_per_type_src'] = src_pooled.detach()
#                 intermediates['pooled_per_type_dst'] = dst_pooled.detach()
            
#             src_flat_type = src_pooled.reshape(B, -1)
#             dst_flat_type = dst_pooled.reshape(B, -1)
            
#             type_feat_src = self.type_fusion_net(src_flat_type)
#             type_feat_dst = self.type_fusion_net(dst_flat_type)
            
#             # NO conditional clone - always create new tensor
#             src_x = src_x + type_feat_src
#             dst_x = dst_x + type_feat_dst

#         # 4. Final FFN Block 
#         src_ffn_in = self.norm_ffn(src_x)
#         dst_ffn_in = self.norm_ffn(dst_x)
        
#         # NO conditional clone
#         src_out = src_x + self.ffn(src_ffn_in)
#         dst_out = dst_x + self.ffn(dst_ffn_in)
        
#         src_out = self.norm_final(src_out)
#         dst_out = self.norm_final(dst_out)
        
#         src_out = self._safe_sanitize(src_out, "final_src")
#         dst_out = self._safe_sanitize(dst_out, "final_dst")
        
#         return (src_out, dst_out, intermediates if return_intermediates else {})
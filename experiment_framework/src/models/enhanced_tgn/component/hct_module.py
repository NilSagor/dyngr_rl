import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger

from .transformer_encoder import PositionalEncoding, TransformerEncoderLayer

# === CONFIG ===
DEBUG_SANITIZE = False
DEBUG_LOG_5D = False

# ==== Helper function =====
def _sanitize_tensor(
        x: torch.Tensor, 
        name: str = "tensor", 
        nan_val: float = 0.0, 
        inf_val: float = 10.0
    ) -> torch.Tensor:
    """Helper: sanitize tensor for NaN/Inf with consistent logging."""    
    if not torch.isfinite(x).all():
        logger.warning(f"NaN/Inf in {name}: {x.shape}")
        return torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    return x

def _sanitize_node_indices(nodes: torch.Tensor, max_nodes: int) -> torch.Tensor:
    nodes = torch.clamp(nodes, 0, max_nodes - 1)
    nodes = torch.where(torch.isfinite(nodes.float()), nodes, torch.zeros_like(nodes).long())
    return nodes.long()
def _handle_5d_tensor(tensor: torch.Tensor, is_mask: bool = False) -> torch.Tensor:
    if tensor.dim() == 5:
        if tensor.size(0) == 1:
            return tensor.squeeze(0)
        else:
            return tensor.view(tensor.size(0) * tensor.size(1), *tensor.shape[2:])
    return tensor
 

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
        self.pos_encoder = PositionalEncoding(
            d_model, 
            max_walk_length
        )
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, 
                nhead, 
                dim_feedforward, 
                dropout
            )
            for _ in range(num_layers)
        ])
        
        
        # Output projection for walk-level aggregation
        self.output_proj = nn.Linear(d_model, d_model)        
        self.dropout = nn.Dropout(dropout)
        
        self.norm_proj = nn.LayerNorm(d_model, eps=1e-6)
        
    def _normalize_5d_input(self, embeddings: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Consistently handle 5D inputs by flattening leading batch dimensions."""
        if embeddings.dim() == 4 and masks.dim() == 3:
            return embeddings, masks  # Already correct
        
        if embeddings.dim() == 5:
            # Flatten first two dims: [B1, B2, W, L, D] -> [B1*B2, W, L, D]
            B1, B2, W, L, D = embeddings.shape
            embeddings = embeddings.view(B1 * B2, W, L, D)
            
            # Match masks: [B1, B2, W, L] or [B1*B2, W, L] -> [B1*B2, W, L]
            if masks.dim() == 5:
                masks = masks.view(B1 * B2, W, L)
            elif masks.dim() == 4 and masks.shape[:2] == (B1, B2):
                masks = masks.view(B1 * B2, W, L)
            # If masks already [B1*B2, W, L], keep as-is
            
        elif embeddings.dim() == 4 and masks.dim() == 4:
            # Handle squeezed batch: [1, B, W, L] -> [B, W, L]
            if embeddings.size(0) == 1:
                embeddings = embeddings.squeeze(0)
                masks = masks.squeeze(0) if masks.size(0) == 1 else masks.view(-1, *masks.shape[2:])
        
        return embeddings, masks
    
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
         # Normalize 5D inputs consistently
        walk_embeddings, walk_masks = self._normalize_5d_input(walk_embeddings, walk_masks)
        
        # Safe dimension handling - only squeeze if dim 0 is actually 1
        # if walk_embeddings.dim() == 5:
        #     if walk_embeddings.size(0) == 1:
        #         logger.warning(f"Walk embeddings is 5D {walk_embeddings.shape}, squeezing dim 0")
        #         walk_embeddings = walk_embeddings.reshape(*walk_embeddings.shape[1:])
        #         # FIX: Also squeeze walk_masks consistently
        #         if walk_masks.dim() == 5:
        #             walk_masks = walk_masks.reshape(*walk_masks.shape[1:])
        #         elif walk_masks.dim() == 4:
        #             walk_masks = walk_masks.reshape(*walk_masks.shape[1:])
        #     else:
        #         # FIX: Properly handle 5D with batch>1 by flattening batch dimensions
        #         logger.warning(f"Walk embeddings is 5D with batch>1 {walk_embeddings.shape}, flattening")
        #         batch_dim = walk_embeddings.size(0) * walk_embeddings.size(1)
        #         walk_embeddings = walk_embeddings.view(batch_dim, *walk_embeddings.shape[2:])
        #         # FIX: Handle masks consistently
        #         if walk_masks.dim() == 5:
        #             walk_masks = walk_masks.view(batch_dim, *walk_masks.shape[2:])
        #         elif walk_masks.dim() == 4:
        #             walk_masks = walk_masks.view(batch_dim, *walk_masks.shape[2:])
        
        if walk_embeddings.dim() != 4:
            raise ValueError(f"Expected 4D walk_embeddings, got {walk_embeddings.dim()}D: {walk_embeddings.shape}")
        if walk_masks.dim() != 3:
            raise ValueError(f"Expected 3D walk_masks, got {walk_masks.dim()}D: {walk_masks.shape}")
        
                
        # if walk_masks.dim() != 3:
        #         logger.warning(f"walk_masks has wrong dim {walk_masks.dim()}, expected 3D")
        #         # Try to reshape to match
        #         if walk_masks.numel() == walk_embeddings.size(0) * walk_embeddings.size(1) * walk_embeddings.size(2):
        #             walk_masks = walk_masks.view(walk_embeddings.size(0), walk_embeddings.size(1), walk_embeddings.size(2))
            
               
        walk_embeddings = _sanitize_tensor(walk_embeddings, "walk_embeddings")
       
          
        batch_size, num_walks, walk_len, d_model = walk_embeddings.shape
               
        
        # Save original masks for pooling
        original_masks = walk_masks
        
        # Flatten for transformer
        x = walk_embeddings.view(batch_size * num_walks, walk_len, d_model)
        flat_masks = walk_masks.view(batch_size * num_walks, walk_len)
        key_padding_mask = ~flat_masks.bool()
        
        # Reshape to process all walks in parallel
        # x = walk_embeddings.view(batch_size * num_walks, walk_len, d_model)
        # masks = walk_masks.view(batch_size * num_walks, walk_len)
                
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
       
        # Create key padding mask for attention (True for padded positions)
        # key_padding_mask = ~masks.bool()
    
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            x = _sanitize_tensor(x, "transformer_layer_output")
            
        x = self.norm_proj(x)

        # Reshape using original batch/walk structure
        x = x.view(batch_size, num_walks, walk_len, d_model)
        
        # Reshape back BEFORE pooling        
        masks = masks.view(batch_size, num_walks, walk_len)

        # Pool to get walk summaries (mean over valid positions)        
        # Pool using ORIGINAL masks (not flattened)
        masks_expanded = original_masks.unsqueeze(-1).float()
        sum_feats = (x * masks_expanded).sum(dim=2)
        count = masks_expanded.sum(dim=2) + 1e-6
        count_safe = torch.where(count > 1.0, count, torch.ones_like(count))
        walk_summaries = sum_feats / count_safe
        
             
        # Handle all-padded walks (count would be ~1e-8)
        # Check for all-padded walks in the original shape        
        all_padded = (count.squeeze(-1) < 1e-6)
        walk_summaries[all_padded] = 0.0        
        
        # Final projection
        walk_summaries = self.output_proj(walk_summaries)
        walk_summaries = self.norm_proj(walk_summaries)        
        
        # Final sanitization
        walk_summaries = _sanitize_tensor(walk_summaries, "walk_summaries")
        
        # Also reshape encoded_walks for consistency
        encoded_walks = x
        
        return encoded_walks, walk_summaries # [B,W,L,D], [B,W,D]
      
class CooccurrenceMatrix(nn.Module):
    """
    Optimized for CUDA using scatter operations when matches are sparse.
    Best when node IDs are diverse (few matches).
    """
    def __init__(self, max_walk_length: int = 20, sigma: float = 2.0):
        super().__init__()
        positions = torch.arange(max_walk_length, dtype=torch.float32)
        kernel = torch.exp(-((positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2) / (sigma ** 2))
        self.register_buffer('kernel', kernel)
        self.max_walk_length = max_walk_length
    
    def forward(self, anonymized_nodes, walk_masks):
        B, W, L = anonymized_nodes.shape
        device = anonymized_nodes.device
        
             
        # Handle empty walks
        valid_walks = walk_masks.any(dim=-1)  # [B, W]
        if not valid_walks.any():
            logger.warning("All walks are empty in CooccurrenceMatrix")
            return torch.zeros(B, W, W, device=device)
        
        # For each batch and walk, get valid (position, node_id) pairs
        cooccurrence = torch.zeros(B, W, W, device=device)
   
        # Process per batch to avoid O(B²W²) indexing complexity
        # Process per batch
        for b in range(B):
            batch_nodes = anonymized_nodes[b]  # [W, L]
            batch_masks = walk_masks[b]        # [W, L]
            
            if not batch_masks.any():
                continue
            
            # Get valid entries
            valid_mask = batch_masks.bool()
            valid_indices = torch.where(valid_mask)
            walk_idx, pos_idx = valid_indices
            node_ids = batch_nodes[valid_mask]
            
            if len(node_ids) == 0:
                continue
            
            # Group by node_id using sorting (deterministic)
            sorted_node_ids, sort_perm = torch.sort(node_ids)
            sorted_walks = walk_idx[sort_perm]
            sorted_pos = pos_idx[sort_perm]
            
            # Find segment boundaries where node_id changes
            if len(sorted_node_ids) > 1:
                node_changes = torch.cat([
                    torch.tensor([True], device=device),
                    sorted_node_ids[1:] != sorted_node_ids[:-1]
                ])
            else:
                node_changes = torch.tensor([True], device=device)
                
            change_indices = torch.where(node_changes)[0]

            # Process each segment (same node_id)
            for i in range(len(change_indices)):
                start = change_indices[i].item()
                end = change_indices[i + 1].item() if i + 1 < len(change_indices) else len(sorted_node_ids)
                
                if end - start < 2:
                    continue
                
                seg_walks = sorted_walks[start:end]
                seg_pos = sorted_pos[start:end]
                
                # Compute all pairwise contributions for this segment
                self._add_segment_deterministic(
                    cooccurrence[b], 
                    seg_walks, 
                    seg_pos, 
                    self.kernel[:L, :L]
                )

        # Normalize   
        walk_lens = walk_masks.sum(dim=-1).float()  # [B, W]
        norm = walk_lens.unsqueeze(-1) * walk_lens.unsqueeze(-2)  # [B, W, W]
        norm = torch.clamp(norm, min=1e-6)
        
        valid_pairs = (walk_lens.unsqueeze(-1) > 0) & (walk_lens.unsqueeze(-2) > 0)
        cooccurrence = torch.where(valid_pairs, cooccurrence / norm, torch.zeros_like(cooccurrence))
        
        # Clamp and apply tanh for stability
        cooccurrence = torch.clamp(cooccurrence, -10.0, 10.0)
        cooccurrence = torch.tanh(cooccurrence)
        
        # Final sanitization
        cooccurrence = _sanitize_tensor(cooccurrence, "cooccurrence", inf_val=1.0)

        return cooccurrence
    
    def _add_segment_deterministic(self, cooccurrence_b, walks, positions, kernel):
        """Helper to add contributions for a group of same‑node occurrences."""
        if len(walks) == 0 or len(positions) == 0:
            return
            
        L = kernel.size(0)
        W = cooccurrence_b.size(0)

        # Clamp positions to valid range
        positions = torch.clamp(positions, 0, L - 1)
        walks = torch.clamp(walks, 0, W - 1)
        
        # Create pairwise indices
        pos_i = positions.unsqueeze(1)  # [n, 1]
        pos_j = positions.unsqueeze(0)  # [1, n]
        kernel_vals = torch.clamp(kernel[pos_i, pos_j], min=-10.0, max=10.0)  # [n, n]
        
        # w_i = walks.unsqueeze(1).expand(-1, len(walks))  # [n, n]
        # w_j = walks.unsqueeze(0).expand(len(walks), -1)  # [n, n]
        w_i = walks.unsqueeze(1)
        w_j = walks.unsqueeze(0)
        
        # Safe index computation with overflow check
        flat_indices = w_i * W + w_j
        max_valid_idx = W * W - 1
        if flat_indices.max() > max_valid_idx:
            logger.warning(f"Cooccurrence index overflow: max={flat_indices.max()}, clamping")
            flat_indices = torch.clamp(flat_indices, 0, max_valid_idx)

        flat_kernel = kernel_vals.view(-1)
        flat_idx = flat_indices.view(-1).long()
        
        # Use bincount for deterministic accumulation
        accum = torch.bincount(flat_idx, weights=flat_kernel, minlength=W * W)
        accum = accum.view(W, W)
        
        # Add to cooccurrence matrix
        cooccurrence_b.add_(accum) 


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
        self.cooccurrence_gamma = cooccurrence_gamma
        self.nhead = nhead
        
        # Learnable gamma (optional)        
        self.gamma = nn.Parameter(torch.tensor(cooccurrence_gamma))
        
        # Transformer encoder layers                
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Positional encoding for walks (walks are ordered but we use co-occurrence bias instead)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self._supports_attn_bias = self._supports_attn_bias(self.layers[0])
        
    def _supports_attn_bias(self, layer: nn.Module) -> bool:
        """Check if layer.forward() accepts attn_bias argument."""
        import inspect
        sig = inspect.signature(layer.forward)
        return 'attn_bias' in sig.parameters
    
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
                
        # Proper shape validation - raise error instead of silent truncation
        if cooccurrence.shape != (batch_size, num_walks, num_walks):
            logger.error(f"Shape mismatch: walk_summaries={walk_summaries.shape}, cooccurrence={cooccurrence.shape}")
            raise ValueError(f"Cooccurrence shape {cooccurrence.shape} doesn't match expected ({batch_size}, {num_walks}, {num_walks})")
        
        # Sanitize cooccurrence BEFORE using as attention bias
        cooccurrence = _sanitize_tensor(cooccurrence, "cooccurrence", inf_val=1.0)
        cooccurrence = torch.clamp(cooccurrence, -5.0, 5.0)
        
        # Create attention bias from co-occurrence matrix
        cooccurrence_bias = cooccurrence.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        bias_scale = 0.1 / (self.d_model ** 0.5)
        cooccurrence_bias = self.gamma * cooccurrence_bias * bias_scale
        
        
        # Create key padding mask for walks        
        key_padding_mask = None
        if walk_masks is not None:
            key_padding_mask = ~walk_masks.bool()
        
        x = walk_summaries
        # Sanitize input        
        x = _sanitize_tensor(x, "walk_summaries_input")
                
        # Apply transformer layers
        for layer_idx, layer in enumerate(self.layers):
            if self._supports_attn_bias:
                x = layer(x, attn_bias=cooccurrence_bias, key_padding_mask=key_padding_mask)
            else:
                # Explicit warning + manual bias injection as fallback
                logger.warning(f"Layer {layer_idx} doesn't support attn_bias; applying bias manually")
                x = layer(x, key_padding_mask=key_padding_mask)
                # Manual bias: add scaled cooccurrence to output (less ideal but explicit)
                bias_residual = cooccurrence_bias.mean(dim=1).squeeze(1) * self.cooccurrence_gamma * 0.1
                x = x + bias_residual
            x = _sanitize_tensor(x, f"Inter_transform_layer_{layer_idx}")
            
        x = self.norm(x)

        # Final sanitization               
        x = _sanitize_tensor(x, "Inter_transform_output")

        return x
    
class HierarchicalCooccurrenceTransformer(nn.Module):
    """
    Hierarchical Co-occurrence Transformer (HCT) for HiCoST.
    
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
        max_num_walks: int = 20,
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
            self.walk_type_embed = nn.Embedding(3, d_model)
            nn.init.normal_(self.walk_type_embed.weight, std=0.02)

         # Add dropout to pooling
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Memory projection if dimensions differ
        self.memory_proj = nn.Linear(memory_dim, d_model) if memory_dim != d_model else nn.Identity()        
        
        # Restart embedding for TAWR
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
        
        Args:
            node_embeddings: [B, N, L, D] actual node embeddings from memory
            anonymized_nodes: [B, N, L] anonymized IDs for co-occurrence
            walk_masks: [B, N, L] mask for valid positions
            walk_type: 0=short, 1=long, 2=tawr
        """        
        # Safe dimension handling
        if node_embeddings.dim() == 5:
            if node_embeddings.size(0) == 1:
                logger.warning(f"Walk embeddings is 5D {node_embeddings.shape}, squeezing dim 0")
                node_embeddings = node_embeddings.reshape(*node_embeddings.shape[1:])
            else:
                logger.warning(f"Walk embeddings is 5D with batch>1, flattening")
                node_embeddings = node_embeddings.view(-1, *node_embeddings.shape[2:])
        elif node_embeddings.dim() != 4:
            raise ValueError(f"Expected 4D node_embeddings, got {node_embeddings.dim()}D")
        
        # Sanitize input
        node_embeddings = _sanitize_tensor(node_embeddings, "node_embeddings")
       
        batch_size, num_walks, walk_len, _ = node_embeddings.shape
                
        # Add walk type embedding
        if self.use_walk_type_embedding:
            type_embed = self.walk_type_embed(torch.tensor([walk_type], device=node_embeddings.device))
            node_embeddings = node_embeddings + type_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Step 1: Intra-walk encoding
        encoded_walks, walk_summaries = self.intra_walk_encoder(node_embeddings, walk_masks)

        # Step 2: Compute co-occurrence matrix
        cooccurrence = self.cooccurrence_matrix(anonymized_nodes, walk_masks)
        
        # Create walk-level mask
        walk_level_mask = (walk_masks.sum(dim=-1) > 0).float()
        
        
        # Handle case where all walks are empty
        if walk_level_mask.sum() == 0:
            logger.warning("All walks are empty, returning zeros")
            return {
                'encoded_walks': torch.zeros_like(encoded_walks),
                'walk_summaries': torch.zeros_like(walk_summaries),
                'refined_walks': torch.zeros_like(walk_summaries),
                'cooccurrence': cooccurrence,
                'walk_masks': walk_level_mask
            }
        
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
        
        all_walks = []
        all_masks = []
        
        for _, output in [('short', short_output), ('long', long_output), ('tawr', tawr_output)]:
            refined = output['refined_walks']
            masks = output['walk_masks']
            all_walks.append(refined)
            all_masks.append(masks)         
        
        
        all_walks = torch.cat(all_walks, dim=1)
        all_masks = torch.cat(all_masks, dim=1)
        total_walks = all_walks.size(1)
    
        # Attention-based pooling
        attention_scores = self.pooling(all_walks).squeeze(-1)
                
        # Handle all-masked case BEFORE softmax to prevent NaN
        all_masked = (all_masks == 0).all(dim=-1)  # [batch_size]
        
        # Use large negative value instead of -inf for numerical stability
        attention_scores = attention_scores.masked_fill(all_masks == 0, -1e4)
        
        # Compute softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        
        # Apply uniform weights for all-masked batches BEFORE any NaN can form
        if all_masked.any():
            logger.warning(f"{all_masked.sum().item()} batches have all walks masked")
            attention_weights = attention_weights.clone()  # Avoid in-place on masked tensor
            attention_weights[all_masked] = 1.0 / total_walks
        
        attention_weights = attention_weights.unsqueeze(-1)       
        
        # Hybrid pooling: 50% mean, 50% attention        
        valid_mask_sum = all_masks.sum(dim=-1, keepdim=True)
        mean_pool = torch.where(
            valid_mask_sum > 1e-6,
            (all_walks * all_masks.unsqueeze(-1)).sum(dim=1) / (valid_mask_sum + 1e-8),
            torch.zeros_like(valid_mask_sum)
        )
        
        attn_pool = (all_walks * attention_weights).sum(dim=1)

        mean_pool = torch.nan_to_num(mean_pool, nan=0.0, posinf=0.0, neginf=0.0)
        attn_pool = torch.nan_to_num(attn_pool, nan=0.0, posinf=0.0, neginf=0.0)
        
        fused = 0.5 * mean_pool + 0.5 * attn_pool
        
        # Final projection and normalization
        fused = self.output_proj(self.norm(fused))
        
        # Final sanitization        
        fused = _sanitize_tensor(fused, "fused_output")
        
        return fused
    
    def _validate_anonymization(self, nodes, nodes_anon, masks):
        # Same actual node should map to same anonymized ID within batch
        for b in range(nodes.size(0)):
            unique_actual = nodes[b][masks[b].bool()].unique()
            for actual_id in unique_actual:
                anon_ids = nodes_anon[b][(nodes[b] == actual_id) & masks[b]]
                assert anon_ids.nunique() == 1, f"Node {actual_id} has multiple anon IDs: {anon_ids.unique()}"
    
    def _consistent_squeeze(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Squeeze leading dim of size 1 consistently across all tensors."""
        # Check if all tensors have leading dim == 1
        if all(t.dim() >= 1 and t.size(0) == 1 for t in tensors.values()):
            return {name: t.squeeze(0) for name, t in tensors.items()}
        return tensors
    
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
            return_all: Return intermediate outputs
        """       
        batch_size = walks_dict['short']['nodes'].size(0)
        device = walks_dict['short']['nodes'].device
        
        # Sanitize node_memory BEFORE lookup        
        node_memory = _sanitize_tensor(node_memory, "node_memory")

        # Ensure memory_proj device sync
        if isinstance(self.memory_proj, nn.Linear):
            proj_device = next(self.memory_proj.parameters()).device
            if node_memory.device != proj_device:
                self.memory_proj = self.memory_proj.to(node_memory.device)
        
        outputs = {}

        for walk_type, type_name in [(0, 'short'), (1, 'long'), (2, 'tawr')]:
            data = walks_dict[type_name]
            
            nodes = data['nodes'].clone()
            nodes_anon = data['nodes_anon'].clone()
            masks = data['masks'].clone()
            
            # Sanitize node indices
            nodes = _sanitize_node_indices(nodes, node_memory.size(0))
            data['nodes'] = nodes
            
            # Consistent 5D handling
            tensors = {'nodes': nodes, 'nodes_anon': nodes_anon, 'masks': masks}
            tensors = self._consistent_squeeze(tensors)
            nodes, nodes_anon, masks = tensors['nodes'], tensors['nodes_anon'], tensors['masks']
            
            num_walks, walk_len = nodes.size(1), nodes.size(2)
            flat_nodes = nodes.reshape(-1).long()
            
            # Safe memory lookup
            accessed_memory = node_memory[flat_nodes]
            accessed_memory = _sanitize_tensor(accessed_memory, "accessed_memory")
            
            walk_node_feats = self.memory_proj(accessed_memory)
            walk_embeddings = walk_node_feats.view(batch_size, num_walks, walk_len, self.d_model)
            walk_embeddings = _sanitize_tensor(walk_embeddings, "walk_embeddings")
            
            # TAWR restart embedding with shape validation
            if type_name == 'tawr' and 'restart_flags' in data:
                restart_flags = data['restart_flags'].clone()
                if restart_flags.dim() != walk_embeddings.dim() - 1:
                    if restart_flags.dim() == 4 and restart_flags.size(0) == 1:
                        restart_flags = restart_flags.squeeze(0)
                restart_flags = torch.clamp(restart_flags, 0, 1).long()
                restart_embed = self.restart_embed(restart_flags)
                if restart_embed.shape == walk_embeddings.shape:
                    walk_embeddings = walk_embeddings + restart_embed
                else:
                    logger.warning(f"Skipping restart embed: shape mismatch {restart_embed.shape} vs {walk_embeddings.shape}")
            
            output = self.process_walk_type(walk_embeddings, nodes_anon, masks, walk_type=walk_type)
            outputs[type_name] = output
            
            # Validate node indices BEFORE memory lookup
            # if nodes.max().item() >= node_memory.size(0) or nodes.min().item() < 0:
            #     logger.error(f"Invalid node index in walks! Max: {nodes.max().item()}, Min: {nodes.min().item()}, Memory size: {node_memory.size(0)}. Clamping...")
            #     nodes = torch.clamp(nodes, 0, node_memory.size(0) - 1)
            #     data['nodes'] = nodes

            # if not torch.isfinite(nodes.float()).all():
            #     logger.error("NaN/Inf in walk node indices! Replacing with zeros")
            #     nodes = torch.zeros_like(nodes).long()
            #     data['nodes'] = nodes

            
            # if nodes.dtype != torch.long and nodes.dtype != torch.int64:
            #     logger.warning(f"Converting nodes from {nodes.dtype} to long")
            #     nodes = nodes.long()
            #     data['nodes'] = nodes
            
            # Safe dimension handling
            # Consistent 5D handling - only squeeze leading batch dim if size==1
            # for name, tensor in [('nodes', nodes), ('nodes_anon', nodes_anon), ('masks', masks)]:
            #     if tensor.dim() == 4 and tensor.size(0) == 1:
            #         tensor = tensor.squeeze(0)
            #         data[name] = tensor
            
            # nodes = data['nodes']
            # nodes_anon = data['nodes_anon']
            # masks = data['masks']
            
            # num_walks = nodes.size(1)
            # walk_len = nodes.size(2)         
     
            # flat_nodes = nodes.reshape(-1).long()
            
  
            # # Validate flat_nodes before indexing
            # if flat_nodes.max().item() >= node_memory.size(0) or flat_nodes.min().item() < 0:
            #     logger.error(f"Invalid flat node index! Max: {flat_nodes.max().item()}, Min: {flat_nodes.min().item()}, Memory size: {node_memory.size(0)}. Clamping...")
            #     flat_nodes = torch.clamp(flat_nodes, 0, node_memory.size(0) - 1)
            
            # if not torch.isfinite(flat_nodes.float()).all():
            #     logger.error("NaN in flat_nodes! Replacing with zeros")
            #     flat_nodes = torch.zeros_like(flat_nodes)
            
            # if flat_nodes.dtype != torch.long:
            #     flat_nodes = flat_nodes.long()
            
            # accessed_memory = node_memory[flat_nodes]
            # accessed_memory = _sanitize_tensor(accessed_memory, "accessed_memory")
            
            
            # # walk_node_feats = node_memory[flat_nodes]
            # walk_node_feats = self.memory_proj(accessed_memory)
            # walk_embeddings = walk_node_feats.view(batch_size, num_walks, walk_len, self.d_model)

            # # Sanitize after memory lookup
            # walk_embeddings = _sanitize_tensor(walk_embeddings, "walk_embeddings")
            
            # # Add restart flags for TAWR
            # if type_name == 'tawr' and 'restart_flags' in data:
            #     restart_flags = data['restart_flags']
            #     if restart_flags.dim() == 4 and restart_flags.size(0) == 1:
            #         restart_flags = restart_flags.squeeze(0)
            #     restart_flags = torch.clamp(restart_flags, 0, 1).long()
            #     restart_embed = self.restart_embed(restart_flags)
            #     walk_embeddings = walk_embeddings + restart_embed
            
            # output = self.process_walk_type(
            #     walk_embeddings,
            #     nodes_anon,
            #     masks,
            #     walk_type=walk_type
            # )
            # outputs[type_name] = output
        
        # Fuse and return
        fused = self.fuse_walk_types(
            outputs['short'], 
            outputs['long'], 
            outputs['tawr']
        )
        
        # if return_all:
        #     return {'fused': fused, **outputs}
        
        return {'fused': fused, **outputs} if return_all else fused
    


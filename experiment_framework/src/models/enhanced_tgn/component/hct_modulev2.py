import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import math
import inspect

from .transformer_encoder import PositionalEncoding, TransformerEncoderLayer

DEBUG_SANITIZE = True

def _sanitize_tensor(
    x: torch.Tensor,
    name: str = "tensor",
    nan_val: float = 0.0,
    inf_val: float = 10.0
) -> torch.Tensor:
    if not torch.isfinite(x).all():
        if DEBUG_SANITIZE:
            logger.warning(f"NaN/Inf in {name}: {x.shape}")
        return torch.nan_to_num(x, nan=nan_val, posinf=inf_val, neginf=-inf_val)
    return x

def _sanitize_node_indices(nodes: torch.Tensor, max_nodes: int) -> torch.Tensor:
    nodes = torch.clamp(nodes, 0, max_nodes - 1)
    nodes = torch.where(torch.isfinite(nodes.float()), nodes, torch.zeros_like(nodes).long())
    return nodes.long()


class SpectralLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, n_power_iterations=1, eps=1e-12):
        super().__init__(in_features, out_features, bias=bias)
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        if self.weight.dim() > 1:
            nn.init.xavier_uniform_(self.weight, gain=0.1)
            self.register_buffer('weight_u', torch.empty(out_features).normal_(0, 0.02))

    def forward(self, input):
        if self.training and hasattr(self, 'weight_u'):            
            u = self.weight_u.detach()
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = torch.mv(self.weight.t(), u)
                    v = F.normalize(v, dim=-1, eps=self.eps)
                    u = torch.mv(self.weight, v)
                    u = F.normalize(u, dim=-1, eps=self.eps)
                sigma = torch.dot(u, torch.mv(self.weight, v))            
            self.weight_u.copy_(u)           
            weight_norm = self.weight / (sigma + self.eps)
            return F.linear(input, weight_norm, self.bias)
        return super().forward(input)

## ==== cause In-place operator due to large graph
# class SpectralLinear(nn.Linear):
#     """Linear layer with Spectral Normalization for gradient stability."""
#     def __init__(self, in_features, out_features, bias=True, n_power_iterations=1, eps=1e-12):
#         super().__init__(in_features, out_features, bias=bias)
#         self.n_power_iterations = n_power_iterations
#         self.eps = eps
#         if self.weight.dim() > 1:
#             nn.init.xavier_uniform_(self.weight, gain=0.1)
#             self.register_buffer('weight_u', torch.empty(out_features).normal_(0, 0.02))
#             self._update_weight_u()

#     def _update_weight_u(self):
#         if not hasattr(self, 'weight_u'): 
#             return
#         with torch.no_grad():
#             weight = self.weight.detach()
#             u = self.weight_u.detach()
#             for _ in range(self.n_power_iterations):
#                 v = torch.mv(weight.t(), u)
#                 v = F.normalize(v, dim=-1, eps=self.eps)
#                 u = torch.mv(weight, v)
#                 u = F.normalize(u, dim=-1, eps=self.eps)
#             self.weight_u.copy_(u)

#     def forward(self, input):
#         if self.training and hasattr(self, 'weight_u'):
#             with torch.no_grad():
#                 self._update_weight_u()
#             u = self.weight_u
#             v = torch.mv(self.weight.t(), u)
#             v = F.normalize(v, dim=-1, eps=self.eps)
#             sigma = torch.dot(u, torch.mv(self.weight, v))
#             weight_norm = self.weight / (sigma + self.eps)
#             return F.linear(input, weight_norm, self.bias)
#         return super().forward(input)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor =random_tensor.floor()
        output = x.div(keep_prob) * random_tensor
        return output

class CausalIntraWalkEncoder(nn.Module):
    """
    Intra-walk Transformer with Causal Temporal Masking.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_walk_length: int = 20,
        drop_path_rate: float = 0.1
    ):
        super(CausalIntraWalkEncoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_walk_length = max_walk_length
        
        effective_max_len = max_walk_length * 2 + 10
        self.pos_encoder = PositionalEncoding(d_model, effective_max_len)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])
        self.drop_paths = nn.ModuleList([DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity() for i in range(num_layers)])
        
        self.output_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.norm_proj = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        max_mask_size = max_walk_length * 2 + 10
        causal_mask = torch.triu(torch.ones(max_mask_size, max_mask_size), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, walk_embeddings: torch.Tensor, walk_masks: torch.Tensor, walk_times: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle 5D input (batched walks with extra dimension)
        original_shape = walk_embeddings.shape
        is_5d_input = walk_embeddings.dim() == 5
        
        if is_5d_input:
            B1, B2, W, L, D = walk_embeddings.shape
            walk_embeddings = walk_embeddings.view(B1 * B2, W, L, D)
            if walk_masks.dim() == 5:
                walk_masks = walk_masks.view(B1 * B2, W, L)
            elif walk_masks.dim() == 4:
                walk_masks = walk_masks.view(B1 * B2, W, L)
        
        batch_size, num_walks, walk_len, d_model = walk_embeddings.shape
        
        if walk_masks.shape != (batch_size, num_walks, walk_len):
            if walk_masks.numel() == batch_size * num_walks * walk_len:
                walk_masks = walk_masks.view(batch_size, num_walks, walk_len)
            else:
                current_len = walk_masks.shape[-1] if walk_masks.dim() >= 3 else walk_len
                if current_len > walk_len:
                    walk_masks = walk_masks[..., :walk_len]
                elif current_len < walk_len:
                    pad_size = walk_len - current_len
                    if walk_masks.dim() == 2:
                        walk_masks = walk_masks.unsqueeze(1)
                    if walk_masks.dim() == 3:
                        pad_shape = list(walk_masks.shape[:-1]) + [pad_size]
                        pad = torch.zeros(pad_shape, device=walk_masks.device, dtype=walk_masks.dtype)
                        walk_masks = torch.cat([walk_masks, pad], dim=-1)
                    if walk_masks.shape != (batch_size, num_walks, walk_len):
                        walk_masks = walk_masks.view(batch_size, num_walks, walk_len)
        
        x = walk_embeddings.reshape(-1, walk_len, d_model)
        flat_masks = walk_masks.reshape(-1, walk_len)
        
        total_sequences = x.size(0)
        actual_seq_len = x.size(1)
        
        if actual_seq_len > self.causal_mask.size(0):
            logger.warning(f"Sequence length {actual_seq_len} exceeds causal mask size {self.causal_mask.size(0)}. Truncating.")
            actual_seq_len = self.causal_mask.size(0)
            x = x[:, :actual_seq_len, :]
            flat_masks = flat_masks[:, :actual_seq_len]
        
        causal_mask_slice = self.causal_mask[:actual_seq_len, :actual_seq_len]
        padding_mask = ~flat_masks.bool()
        
        if padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)
        
        causal_expanded = causal_mask_slice.unsqueeze(0).expand(total_sequences, -1, -1)
        pad_mask_2d = padding_mask.unsqueeze(1).expand(-1, actual_seq_len, -1)
        
        if causal_expanded.shape != pad_mask_2d.shape:
            min_len = min(causal_expanded.size(-1), pad_mask_2d.size(-1))
            causal_expanded = causal_expanded[..., :min_len, :min_len]
            pad_mask_2d = pad_mask_2d[..., :min_len, :min_len]
            if x.size(1) > min_len:
                x = x[:, :min_len, :]
                actual_seq_len = min_len
        
        full_mask = causal_expanded | pad_mask_2d
        
        attn_bias = torch.zeros_like(full_mask, dtype=torch.float)
        attn_bias = attn_bias.masked_fill(full_mask, -1e9)
        
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for i, layer in enumerate(self.layers):
            attn_bias_expanded = attn_bias.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            x_res = layer(x, attn_bias=attn_bias_expanded, key_padding_mask=None)
            x = x + self.drop_paths[i](x_res)
            x = _sanitize_tensor(x, f"intra_layer_{i}")
        
        x = self.norm_proj(x)
        
        output_seq_len = x.size(1)
        
        x = x.view(batch_size, num_walks, output_seq_len, d_model)
        
        if walk_masks.size(-1) != output_seq_len:
            if walk_masks.size(-1) > output_seq_len:
                walk_masks = walk_masks[..., :output_seq_len]
            else:
                pad_size = output_seq_len - walk_masks.size(-1)
                pad_shape = list(walk_masks.shape[:-1]) + [pad_size]
                pad = torch.zeros(pad_shape, device=walk_masks.device, dtype=walk_masks.dtype)
                walk_masks = torch.cat([walk_masks, pad], dim=-1)
        
        masks_expanded = walk_masks.unsqueeze(-1).float()
        if x.shape != masks_expanded.shape:
            if masks_expanded.size(-1) == 1 and x.size(-1) != 1:
                masks_expanded = masks_expanded.expand(-1, -1, -1, x.size(-1))
        
        sum_feats = (x * masks_expanded).sum(dim=2)
        count = masks_expanded.sum(dim=2)
        if count.dim() == 2:
            count = count.unsqueeze(-1)
        
        walk_summaries = sum_feats / (count + 1e-6)
        
        zero_count_mask = (count.squeeze(-1) < 1e-6)
        if zero_count_mask.any():
            walk_summaries[zero_count_mask] = 0.0
        
        walk_summaries = self.output_proj(walk_summaries)
        walk_summaries = self.norm_proj(walk_summaries)
        
        if is_5d_input:
            x = x.view(*original_shape[:-1], d_model)
        
        return x, walk_summaries

class TemporalCooccurrenceMatrix(nn.Module):
    """Fixed version with robust clamping."""
    def __init__(self, max_walk_length: int = 20, sigma_dist: float = 2.0, sigma_time: float = 5.0):
        super().__init__()
        safe_max_len = max(max_walk_length, 50)
        positions = torch.arange(safe_max_len, dtype=torch.float32)
        dist_kernel = torch.exp(-((positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2) / (sigma_dist ** 2))
        self.register_buffer('dist_kernel', dist_kernel)
        self.sigma_time = sigma_time
        self.max_walk_length = max_walk_length
    
    def forward(self, anonymized_nodes: torch.Tensor, walk_masks: torch.Tensor, walk_times: torch.Tensor) -> torch.Tensor:
        B, W, L = anonymized_nodes.shape
        device = anonymized_nodes.device
        
        if W == 0 or L == 0:
            return torch.zeros(B, W, W, device=device)
        
        anonymized_nodes = torch.nan_to_num(anonymized_nodes, nan=0).long()
        walk_masks = torch.nan_to_num(walk_masks, nan=0)
        walk_times = torch.nan_to_num(walk_times, nan=0.0)
        
        cooccurrence = torch.zeros(B, W, W, device=device)
        kernel_limit = self.dist_kernel.size(0) - 1

        for b in range(B):
            nodes_b = anonymized_nodes[b]
            masks_b = walk_masks[b].bool()
            times_b = walk_times[b]
            
            valid_mask = masks_b
            if not valid_mask.any(): continue
            
            w_idx, p_idx = torch.where(valid_mask)
            node_ids = nodes_b[valid_mask]
            t_vals = times_b[valid_mask]
            
            N = len(node_ids)
            if N == 0: continue
            
            w_idx = torch.clamp(w_idx, 0, W-1).long()
            p_idx = torch.clamp(p_idx, 0, min(L-1, kernel_limit)).long()
            
            i = torch.arange(N, device=device).unsqueeze(1).expand(-1, N).reshape(-1)
            j = torch.arange(N, device=device).unsqueeze(0).expand(N, -1).reshape(-1)
            
            same_node = (node_ids[i] == node_ids[j])
            i = i[same_node]
            j = j[same_node]
            
            if len(i) == 0: continue
            
            w_i = torch.clamp(w_idx[i], 0, W-1).long()
            w_j = torch.clamp(w_idx[j], 0, W-1).long()
            p_i = torch.clamp(p_idx[i], 0, min(L-1, kernel_limit)).long()
            p_j = torch.clamp(p_idx[j], 0, min(L-1, kernel_limit)).long()
            t_i = t_vals[i]
            t_j = t_vals[j]
            
            if (p_i > kernel_limit).any() or (p_j > kernel_limit).any(): continue
            if (w_i >= W).any() or (w_j >= W).any(): continue

            dist_factor = self.dist_kernel[p_i, p_j]
            dt = torch.clamp(torch.abs(t_i - t_j), max=1e4)
            time_factor = torch.exp(-dt / self.sigma_time)
            
            weights = torch.clamp(dist_factor * time_factor, 0.0, 1.0)
            weights = torch.nan_to_num(weights, nan=0.0)
            
            flat_idx = w_i * W + w_j
            max_valid_flat = W * W - 1
            valid_range = (flat_idx >= 0) & (flat_idx <= max_valid_flat)
            
            if not valid_range.all():
                flat_idx = flat_idx[valid_range]
                weights = weights[valid_range]
            
            if flat_idx.numel() == 0: continue
            
            try:
                contrib = torch.bincount(flat_idx, weights=weights, minlength=W*W)
            except RuntimeError as e:
                logger.error(f"bincount failed for batch {b}: {e}")
                logger.error(f"flat_idx range: [{flat_idx.min()}, {flat_idx.max()}], expected < {W*W}")
                raise
                
            cooccurrence[b] += contrib.view(W, W).nan_to_num(nan=0.0)
        
        cooccurrence = torch.tanh(torch.clamp(cooccurrence, -10.0, 10.0))
        return _sanitize_tensor(cooccurrence, "temporal_cooccurrence")

class StabilizedInterWalkTransformer(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, num_layers: int,
        dim_feedforward: int, dropout: float, drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.gamma = nn.Parameter(torch.tensor(0.5))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.drop_paths = nn.ModuleList([DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity() for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self._supports_attn_bias = 'attn_bias' in inspect.signature(self.layers[0].forward).parameters

    def forward(self, x: torch.Tensor, cooccurrence_bias: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, W, D]
        cooccurrence_bias: [B, W, W] (Input from cooccurrence matrix)
        """
        B, W, D = x.shape
        
        if self._supports_attn_bias:
            # Expand bias to [B, H, W, W]
            cooccurrence_bias = cooccurrence_bias.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            bias_scale = 0.1 / (self.d_model ** 0.5)
            cooccurrence_bias = self.gamma * cooccurrence_bias * bias_scale
            
            for i, layer in enumerate(self.layers):
                x_res = layer(x, attn_bias=cooccurrence_bias, key_padding_mask=key_padding_mask)
                x = x + self.drop_paths[i](x_res)
                x = _sanitize_tensor(x, f"inter_layer_{i}")
        else:
            # Fallback: inject bias via addition after softmax (simplified)
            for i, layer in enumerate(self.layers):
                x_res = layer(x, key_padding_mask=key_padding_mask)
                # Add co‑occurrence bias as a residual after attention (simplified)
                bias_effect = cooccurrence_bias.mean(dim=2, keepdim=True) * 0.1  # [B, W, 1]
                x_res = x_res + bias_effect
                x = x + self.drop_paths[i](x_res)
                x = _sanitize_tensor(x, f"inter_layer_{i}")
            
        return self.norm(x)

class StabilizedHCT(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        memory_dim: int = 172,
        nhead: int = 4,
        num_intra_layers: int = 2,
        num_inter_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        max_walk_length: int = 20,
        cooccurrence_sigma_dist: float = 2.0,
        cooccurrence_sigma_time: float = 5.0,
        use_walk_type_embedding: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_walk_type_embedding = use_walk_type_embedding
        
        
        
        self.intra_walk_encoder = CausalIntraWalkEncoder(
            d_model, nhead, num_intra_layers, dim_feedforward, dropout,
            max_walk_length, drop_path_rate
        )
        
        self.cooccurrence_matrix = TemporalCooccurrenceMatrix(
            max_walk_length, cooccurrence_sigma_dist, cooccurrence_sigma_time
        )
        
        self.inter_walk_transformer = StabilizedInterWalkTransformer(
            d_model, nhead, num_inter_layers, dim_feedforward, dropout, drop_path_rate
        )
        
        if use_walk_type_embedding:
            self.walk_type_embed = nn.Embedding(3, d_model)
            nn.init.normal_(self.walk_type_embed.weight, std=0.02)
            
        self.pooling = nn.Sequential(
            SpectralLinear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            SpectralLinear(d_model, 1)
        )
        
        self.output_proj = SpectralLinear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.memory_proj = SpectralLinear(memory_dim, d_model) if memory_dim != d_model else nn.Identity()
        self.restart_embed = nn.Embedding(2, d_model)

    def get_cooccurrence_matrix(self, walk_data: Dict, node_memory: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = node_memory.device
        cooc_dict = {}
        if 'source' not in walk_data: return {}
        
        types = list(walk_data['source'].keys())
        for type_name in types:
            src_data = walk_data['source'][type_name]
            tgt_data = walk_data['target'][type_name]
            
            nodes_anon = torch.cat([src_data['nodes_anon'], tgt_data['nodes_anon']], dim=0)
            masks = torch.cat([src_data['masks'], tgt_data['masks']], dim=0)
            times = torch.cat([src_data['times'], tgt_data['times']], dim=0)
            
            if nodes_anon.dim() == 5 and nodes_anon.size(0) == 1:
                nodes_anon = nodes_anon.squeeze(0)
                masks = masks.squeeze(0)
                times = times.squeeze(0)
            
            cooc = self.cooccurrence_matrix(nodes_anon.to(device), masks.to(device), times.to(device))
            cooc_dict[type_name] = cooc
        return cooc_dict

    def process_walk_type(
        self,
        node_embeddings: torch.Tensor,
        anonymized_nodes: torch.Tensor,
        walk_masks: torch.Tensor,
        walk_times: torch.Tensor,
        walk_type: int = 0
    ) -> Dict[str, torch.Tensor]:
        if node_embeddings.dim() == 5:
            if node_embeddings.size(0) == 1:
                node_embeddings = node_embeddings.squeeze(0)
            else:
                node_embeddings = node_embeddings.view(-1, *node_embeddings.shape[2:])
                walk_masks = walk_masks.view(-1, *walk_masks.shape[2:]) if walk_masks.dim() == 5 else walk_masks
                walk_times = walk_times.view(-1, *walk_times.shape[2:]) if walk_times.dim() == 5 else walk_times

        batch_size, num_walks, walk_len, _ = node_embeddings.shape
        
        if self.use_walk_type_embedding:
            type_embed = self.walk_type_embed(torch.tensor([walk_type], device=node_embeddings.device))
            node_embeddings = node_embeddings + type_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        encoded_walks, walk_summaries = self.intra_walk_encoder(node_embeddings, walk_masks, walk_times)
        cooccurrence = self.cooccurrence_matrix(anonymized_nodes, walk_masks, walk_times)
        
        walk_level_mask = (walk_masks.sum(dim=-1) > 0).float()
        
        if walk_level_mask.sum() == 0:
            return {
                'encoded_walks': torch.zeros_like(encoded_walks),
                'walk_summaries': torch.zeros_like(walk_summaries),
                'refined_walks': torch.zeros_like(walk_summaries),
                'cooccurrence': cooccurrence
            }
        
        refined_walks = self.inter_walk_transformer(
            walk_summaries, cooccurrence, ~walk_level_mask.bool()
        )
        
        return {
            'encoded_walks': encoded_walks,
            'walk_summaries': walk_summaries,
            'refined_walks': refined_walks,
            'cooccurrence': cooccurrence,
            'walk_masks': walk_level_mask
        }

    def fuse_walk_types(self, short_out, long_out, tawr_out) -> torch.Tensor:
        all_walks = torch.cat([short_out['refined_walks'], long_out['refined_walks'], tawr_out['refined_walks']], dim=1)
        all_masks = torch.cat([short_out['walk_masks'], long_out['walk_masks'], tawr_out['walk_masks']], dim=1)
        total_walks = all_walks.size(1)
        
        scores = self.pooling(all_walks).squeeze(-1)
        all_masked = (all_masks == 0).all(dim=-1)
        
        # scores_for_softmax = scores.clone()
        scores_for_softmax = scores.masked_fill(all_masks == 0, -1e4)

        weights = F.softmax(scores_for_softmax, dim=-1)
        
        
        if all_masked.any():
            uniform_weights = torch.full_like(weights, 1.0 / total_walks)
            weights = torch.where(all_masked.unsqueeze(-1), uniform_weights, weights)
            
        weights = weights.unsqueeze(-1)
        
        valid_sum = all_masks.sum(dim=-1, keepdim=True) + 1e-8
        mean_pool = (all_walks * all_masks.unsqueeze(-1)).sum(dim=1) / valid_sum
        attn_pool = (all_walks * weights).sum(dim=1)
        
        fused = 0.5 * mean_pool + 0.5 * attn_pool
        fused = self.output_proj(self.norm(fused))
        
        return _sanitize_tensor(fused, "fused_output")

    def forward(
        self,
        walks_dict: Dict[str, Dict[str, torch.Tensor]],
        node_memory: torch.Tensor,
        return_all: bool = False
    ) -> Union[torch.Tensor, Dict]:
        batch_size = walks_dict['short']['nodes'].size(0)
        node_memory = _sanitize_tensor(node_memory, "node_memory")
        
        outputs = {}
        
        for walk_type, type_name in [(0, 'short'), (1, 'long'), (2, 'tawr')]:
            data = walks_dict[type_name]
            nodes = _sanitize_node_indices(data['nodes'].clone(), node_memory.size(0))
            nodes_anon = data['nodes_anon'].clone()
            masks = data['masks'].clone()
            walk_times = data.get('times', torch.zeros_like(nodes).float())
            
            if nodes.dim() == 5 and nodes.size(0) == 1:
                nodes = nodes.squeeze(0)
                nodes_anon = nodes_anon.squeeze(0)
                masks = masks.squeeze(0)
                walk_times = walk_times.squeeze(0) if walk_times.dim() == 5 else walk_times
            
            num_walks, walk_len = nodes.size(1), nodes.size(2)
            flat_nodes = nodes.reshape(-1).long()
            
            accessed_memory = node_memory[flat_nodes]
            walk_embeddings = self.memory_proj(accessed_memory).view(batch_size, num_walks, walk_len, self.d_model)
            
            if type_name == 'tawr' and 'restart_flags' in data:
                r_flags = data['restart_flags'].clone()
                if r_flags.dim() == 4 and r_flags.size(0) == 1: r_flags = r_flags.squeeze(0)
                r_flags = torch.clamp(r_flags, 0, 1).long()
                if r_flags.dim() == 3:
                     r_embed = self.restart_embed(r_flags)
                     walk_embeddings = walk_embeddings + r_embed
            
            out = self.process_walk_type(
                walk_embeddings, nodes_anon, masks, walk_times, walk_type
            )
            outputs[type_name] = out
            
        fused = self.fuse_walk_types(outputs['short'], outputs['long'], outputs['tawr'])
        
        return {'fused': fused, **outputs} if return_all else fused
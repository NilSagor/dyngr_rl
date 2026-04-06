from typing import Optional, Dict, Tuple, List
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from loguru import logger

from .time_encoder import TimeEncoder

# --- Helper Functions ---

def _sanitize_tensor(
    x: torch.Tensor, 
    name: str = "tensor",                    
    nan_val: float = 0.0, 
    inf_val: float = 10.0,
    neg_inf_val: float = -10.0,
    clamp_min: float = -50.0,
    clamp_max: float = 50.0
) -> torch.Tensor:
    """
    Robust sanitization: Handles NaN/Inf, clamps extreme values, and logs issues.
    """
    if not torch.isfinite(x).all():
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        # Log only occasionally to avoid spamming if this happens in a tight loop
        logger.warning(f"NaN/Inf detected in {name}: shape={x.shape}, nan={has_nan}, inf={has_inf}")
        x = torch.nan_to_num(x, nan=nan_val, posinf=inf_val, neginf=neg_inf_val)
    
    # Aggressive clamping to prevent overflow in subsequent ops (exp, softmax)
    return torch.clamp(x, min=clamp_min, max=clamp_max)

def _align_batch_size(tensor: torch.Tensor, batch_size: int, name: str = "tensor") -> torch.Tensor:
    """Align tensor batch size to target batch size efficiently."""
    if tensor.size(0) == batch_size:
        return tensor
    if tensor.size(0) == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    
    logger.warning(f"Batch size mismatch in {name}: {tensor.size(0)} vs expected {batch_size}. Adjusting.")
    if tensor.size(0) > batch_size:
        return tensor[:batch_size]
    else:
        pad_shape = list(tensor.shape)
        pad_shape[0] = batch_size - tensor.size(0)
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=0)

class SpectralLinear(nn.Linear):
    """Linear layer with Spectral Normalization for gradient stability."""
    def __init__(self, in_features, out_features, bias=True, n_power_iterations=1, eps=1e-12):        
        super().__init__(in_features, out_features, bias=bias)
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        if self.weight.dim() > 1:
            nn.init.xavier_uniform_(self.weight, gain=0.1)
            self.register_buffer('weight_u', torch.empty(out_features).normal_(0, 0.02))
            self._update_weight_u()

    def _update_weight_u(self):
        if not hasattr(self, 'weight_u'): 
            return
        with torch.no_grad():
            weight = self.weight.detach()
            u = self.weight_u.detach()
            for _ in range(self.n_power_iterations):
                v = torch.mv(weight.t(), u)
                v = F.normalize(v, dim=-1, eps=self.eps)
                u = torch.mv(weight, v)
                u = F.normalize(u, dim=-1, eps=self.eps)
            self.weight_u.copy_(u)

    def forward(self, input):
        if self.training and hasattr(self, 'weight_u'):
            with torch.no_grad():
                self._update_weight_u()
            u = self.weight_u
            v = torch.mv(self.weight.t(), u)
            v = F.normalize(v, dim=-1, eps=self.eps)
            sigma = torch.dot(u, torch.mv(self.weight, v))
            weight_norm = self.weight / (sigma + self.eps)
            return F.linear(input, weight_norm, self.bias)
        return super().forward(input)


class RobustSAMCell(nn.Module):
    def __init__(
            self, 
            memory_dim: int, 
            edge_feat_dim: int, 
            time_dim: int, 
            num_prototypes: int = 5, 
            similarity_metric: str = "cosine", 
            dropout: float = 0.1,
            residual_alpha: float = 0.8
        ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_prototypes = num_prototypes
        self.similarity_metric = similarity_metric
        self.residual_alpha = residual_alpha
        
        self.query_input_dim = memory_dim + edge_feat_dim + time_dim
        self.gate_input_dim = 2 * memory_dim + time_dim
        
        self.query_proj = SpectralLinear(self.query_input_dim, memory_dim)
        self.gate_proj = SpectralLinear(self.gate_input_dim, 1)
        
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.register_buffer("temperature_min", torch.tensor(0.5))
        self.register_buffer("temperature_max", torch.tensor(5.0))
                
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(memory_dim, eps=1e-6)

    def compute_similarity(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        query = _sanitize_tensor(query, "query", clamp_min=-20.0, clamp_max=20.0)
        prototypes = _sanitize_tensor(prototypes, "prototypes", clamp_min=-20.0, clamp_max=20.0)
    
        if self.similarity_metric == "cosine":
            query_norm = F.normalize(query, dim=-1, eps=1e-6)
            proto_norm = F.normalize(prototypes, dim=-1, eps=1e-6)
            sim = torch.bmm(query_norm.unsqueeze(1), proto_norm.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "dot":
            sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "scaled_dot":
            d_k = query.size(-1)
            scale = math.sqrt(max(d_k, 1))
            sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1) / scale
        else:
            raise ValueError(f"Unknown metric: {self.similarity_metric}")
        
        sim = torch.clamp(sim, min=-15.0, max=15.0)
        temp = torch.clamp(self.temperature, self.temperature_min, self.temperature_max)
        return sim / (temp + 1e-4)

    def forward(self, raw_memory, node_features, edge_features, time_encoding, prototypes, node_mask=None):
        raw_memory = _sanitize_tensor(raw_memory, "raw_memory_in")
        batch_size = raw_memory.size(0)         
        
        time_encoding = _align_batch_size(time_encoding, batch_size, "time_enc")
        if edge_features is not None:
            edge_features = _align_batch_size(edge_features, batch_size, "edge_feat")
        if node_features is not None:
            node_features = _align_batch_size(node_features, batch_size, "node_feat")
        
        # Build query
        query_inputs = [raw_memory]
        if edge_features is not None:
            query_inputs.append(_sanitize_tensor(edge_features, "edge_feat_q"))
        if time_encoding is not None:
            query_inputs.append(_sanitize_tensor(time_encoding, "time_enc_q"))
        
        query_input_cat = torch.cat(query_inputs, dim=-1)
        query = self.query_proj(query_input_cat)
        query = self.layer_norm(query)
        query = torch.tanh(query)
        
        # Similarity & Attention
        similarity = self.compute_similarity(query, prototypes)
        
        if node_mask is not None:
            all_masked = (~node_mask).all(dim=-1, keepdim=True)
            masked_sim = similarity.masked_fill(~node_mask, -1e9)
            if all_masked.any():
                uniform = torch.ones_like(similarity) / self.num_prototypes
                attention_weights = torch.where(all_masked, uniform, F.softmax(masked_sim, dim=-1))
            else:
                attention_weights = F.softmax(masked_sim, dim=-1)
        else:
            attention_weights = F.softmax(similarity, dim=-1)
        
        attention_weights = _sanitize_tensor(attention_weights, "attn_w", inf_val=0.0)
        attention_weights = torch.clamp(attention_weights, 0.0, 1.0)
        attention_weights = self.dropout(attention_weights)
        
        # Candidate Memory
        candidate_memory = torch.bmm(attention_weights.unsqueeze(1), prototypes).squeeze(1)
        candidate_memory = _sanitize_tensor(candidate_memory, "cand_mem")
        candidate_memory = torch.clamp(candidate_memory, -5.0, 5.0)
        
        # Update Gate
        gate_inputs = [raw_memory, candidate_memory]
        if time_encoding is not None:
            gate_inputs.append(time_encoding)
        
        gate_cat = torch.cat(gate_inputs, dim=-1)
        gate_cat = torch.clamp(gate_cat, min=-30.0, max=30.0)
        
        gate_logits = self.gate_proj(gate_cat)
        gate_logits = torch.clamp(gate_logits, -10.0, 10.0)
        update_gate = torch.sigmoid(gate_logits)
        
        # Standard Gated Update
        gated_update = (1 - update_gate) * raw_memory + update_gate * candidate_memory
        
        # Residual Memory Update
        updated_memory = (self.residual_alpha * raw_memory) + ((1 - self.residual_alpha) * gated_update)
        
        updated_memory = self.layer_norm(updated_memory)
        updated_memory = torch.clamp(updated_memory, -10.0, 10.0)
        updated_memory = _sanitize_tensor(updated_memory, "updated_mem_out")
        
        return updated_memory, {
            "attention_weights": attention_weights.detach(),
            "update_gate": update_gate.detach(),
            "candidate_memory": candidate_memory.detach()
        }


class RobustStabilityAugmentedMemory(nn.Module):
    """
    Robust StabilityAugmented Memory (SAM)   

    - Prototype-based attention
    - Spectral Normalization on projections
    - Residual updates (prevents forgetting)
    - Time-decay on idle nodes
    - Strict NaN/Inf safeguards
    """
    def __init__(
            self,
            num_nodes: int,
            memory_dim: int = 128,
            node_feat_dim: int = 0,
            edge_feat_dim: int = 64,
            time_dim: int = 64,
            num_prototypes: int = 5,
            similarity_metric: str = "cosine",
            dropout: float = 0.1,
            residual_alpha: float = 0.8,
            time_decay_factor: float = 0.99,
        ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_dim = time_dim
        self.num_prototypes = num_prototypes
        self.time_decay_factor = time_decay_factor
        
        self.time_encoder = TimeEncoder(time_dim)        
        
        if node_feat_dim > 0:
            self.node_proj = SpectralLinear(node_feat_dim, memory_dim)
        else:
            self.node_proj = None

        self.edge_proj = nn.Linear(edge_feat_dim, memory_dim) if edge_feat_dim > 0 else nn.Identity()
        
        self.sam_cell = RobustSAMCell(
            memory_dim=memory_dim,
            edge_feat_dim=memory_dim,   
            time_dim=time_dim,
            num_prototypes=num_prototypes,
            similarity_metric=similarity_metric,
            dropout=dropout,
            residual_alpha=residual_alpha
        )     

        self.prototype_norm = nn.LayerNorm(memory_dim, eps=1e-6)
        
        
        # Buffers
        self.register_buffer("raw_memory", torch.randn(num_nodes, memory_dim) * 0.01)
        self.register_buffer("last_update_time", torch.zeros(num_nodes))
        self.register_buffer("global_max_time", torch.zeros(1))
        
        # Learnable Prototypes
        # GLOBAL prototypes (shared across all nodes)
        self.prototypes = nn.Parameter(
            torch.empty(num_prototypes, memory_dim)
        )
        nn.init.xavier_uniform_(self.prototypes, gain=0.1)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.raw_memory[node_ids]
    
    def get_prototypes(self, node_ids: torch.Tensor) -> torch.Tensor:
        prototypes = self.prototypes.unsqueeze(0).expand(len(node_ids), -1, -1)
        if not torch.isfinite(prototypes).all():
            logger.warning("NaN in prototypes detected during retrieval. Sanitizing.")
            prototypes = torch.nan_to_num(prototypes, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.prototype_norm(prototypes)

    def _apply_time_decay(self, node_ids: torch.Tensor, current_time: torch.Tensor):
        """Apply time-decay to memory of nodes that haven't been updated recently."""
        
        if node_ids.numel() == 0:
            return

        with torch.no_grad():    
            device = self.raw_memory.device
            current_time = current_time.to(device)
            
            last_times = self.last_update_time[node_ids]

            if current_time.dim() == 0 or current_time.size(0) == 1:
                dt = current_time.item() - last_times
            else:
                if current_time.size(0) == node_ids.size(0):
                    dt = current_time - last_times
                else:
                    t_val = current_time.mean()
                    dt = t_val - last_times
            
            dt = torch.clamp(dt, min=0.0)
            if dt.max() == 0:
                return
                
            decay_vals = torch.exp(-torch.abs(torch.log(torch.tensor(self.time_decay_factor))) * dt)
            decay_vals = decay_vals.unsqueeze(-1)
        
            mask = (dt > 1e-6).unsqueeze(-1)
            if mask.any():
                self.raw_memory[node_ids] = torch.where(
                    mask,
                    self.raw_memory[node_ids] * decay_vals,
                    self.raw_memory[node_ids]
                )
                self.raw_memory[node_ids] = torch.clamp(self.raw_memory[node_ids], -10.0, 10.0)
     
    def update_memory_batch(
            self,
            source_nodes: torch.Tensor,
            target_nodes: torch.Tensor,
            edge_features: torch.Tensor,
            current_time: torch.Tensor,
            node_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if source_nodes.numel() == 0:
            return {'source_attention': {}, 'target_attention': {}, 
                    'source_memory': torch.tensor([]), 'target_memory': torch.tensor([])}
        
        batch_size = source_nodes.size(0)
        device = source_nodes.device
        
        if current_time.numel() > 0:
            new_max = max(self.global_max_time.item(), current_time.max().item())
            self.global_max_time = self.global_max_time.new_full((1,), new_max)
        
        all_involved = torch.unique(torch.cat([source_nodes, target_nodes]))
        self._apply_time_decay(all_involved, current_time)
        
        if not torch.isfinite(self.raw_memory).all():
            bad_mask = ~torch.isfinite(self.raw_memory).all(dim=1)
            bad_nodes = bad_mask.nonzero(as_tuple=True)[0]
            self.raw_memory[bad_nodes] = 0.0
            # Note: prototypes are global, no per-node reset

        # Edge feature projection
        if edge_features is not None and edge_features.numel() > 0 and self.edge_feat_dim > 0:
            edge_proj = self.edge_proj(edge_features)
            edge_proj = _sanitize_tensor(edge_proj, "edge_proj")
            norm = edge_proj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            edge_proj = (edge_proj / norm) * math.sqrt(self.memory_dim)
            edge_proj = torch.clamp(edge_proj, -10.0, 10.0)
        else:
            edge_proj = torch.zeros(batch_size, self.memory_dim, device=device)
        
        time_enc = self.time_encoder(current_time)
        time_enc = _align_batch_size(time_enc, batch_size, "time_enc")
        
        src_mem = self.raw_memory[source_nodes]
        tgt_mem = self.raw_memory[target_nodes]
        src_proto = self.get_prototypes(source_nodes)   # [batch, num_proto, D]
        tgt_proto = self.get_prototypes(target_nodes)
        
        src_node_feat = self.node_proj(node_features[source_nodes]) if (self.node_proj and node_features is not None) else None
        tgt_node_feat = self.node_proj(node_features[target_nodes]) if (self.node_proj and node_features is not None) else None
        
        src_new, src_info = self.sam_cell(src_mem, src_node_feat, edge_proj, time_enc, src_proto)
        tgt_new, tgt_info = self.sam_cell(tgt_mem, tgt_node_feat, edge_proj, time_enc, tgt_proto)
        
        if not torch.isfinite(src_new).all() or not torch.isfinite(tgt_new).all():
            logger.error("SAM Cell produced NaN. Skipping write-back.")
            return {
                'source_attention': src_info, 'target_attention': tgt_info,
                'source_memory': torch.zeros_like(src_mem), 'target_memory': torch.zeros_like(tgt_mem)
            }
        
        # Vectorized overlap handling (unchanged)
        all_nodes = torch.cat([source_nodes, target_nodes])
        all_updates = torch.cat([src_new, tgt_new])
        all_times = torch.cat([current_time, current_time])

        unique_nodes, inverse = torch.unique(all_nodes, return_inverse=True)

        group_max = torch.zeros(len(unique_nodes), device=device, dtype=all_times.dtype)
        group_max.scatter_reduce_(0, inverse, all_times, reduce='amax', include_self=False)

        centered = all_times - group_max[inverse]
        exp_vals = torch.exp(centered)

        group_exp_sum = torch.zeros(len(unique_nodes), device=device, dtype=exp_vals.dtype)
        group_exp_sum.scatter_reduce_(0, inverse, exp_vals, reduce='sum', include_self=False)

        weights = exp_vals / group_exp_sum[inverse]

        weighted_updates = all_updates * weights.unsqueeze(-1)
        
        group_weighted_sum = torch.zeros(len(unique_nodes), all_updates.size(1),
                                device=device, dtype=weighted_updates.dtype).clone()
        group_weight_sum = torch.zeros(len(unique_nodes), device=device, dtype=weights.dtype).clone()
        
        
        group_weighted_sum.scatter_reduce_(
            0, inverse.unsqueeze(-1).expand(-1, all_updates.size(1)),
            weighted_updates, reduce='sum', include_self=False
        )

        # group_weight_sum = torch.zeros(len(unique_nodes), device=device, dtype=weights.dtype)
        # group_weight_sum.scatter_reduce_(0, inverse, weights, reduce='sum', include_self=False)
        group_weight_sum.scatter_reduce_(0, inverse, weights, reduce='sum', include_self=False)

        with torch.no_grad():
            final_updates = torch.nan_to_num(final_updates, nan=0.0, posinf=10.0, neginf=-10.0)
            final_updates = torch.clamp(final_updates, -10.0, 10.0)

            # self.raw_memory[unique_nodes] = final_updates
            self.raw_memory.data[unique_nodes] = final_updates

            group_max_time = torch.zeros(len(unique_nodes), device=device, dtype=all_times.dtype)
            group_max_time.scatter_reduce_(0, inverse, all_times, reduce='amax', include_self=False)
            self.last_update_time[unique_nodes] = group_max_time

            self.raw_memory.data = torch.clamp(
                torch.nan_to_num(self.raw_memory.data, nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0, 10.0
            )

        return {
            'source_attention': {k: v.detach() for k, v in src_info.items()},
            'target_attention': {k: v.detach() for k, v in tgt_info.items()},
            'source_memory': src_new.detach(),
            'target_memory': tgt_new.detach()
        }
        
        

    @torch.no_grad()
    def get_stabilized_memory(
            self,
            node_ids: torch.Tensor,
            current_time: torch.Tensor,
            edge_features: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Inference mode: Get memory with decay applied, without updating state."""
        if node_ids.numel() == 0:
            return torch.tensor([], device=self.raw_memory.device)
        
        device = self.raw_memory.device
        node_ids = node_ids.to(device)
        current_time = current_time.to(device)
        
        raw = self.raw_memory[node_ids].clone()
        
        last_times = self.last_update_time[node_ids]
        if current_time.dim() == 0 or current_time.size(0) == 1:
            dt = current_time.item() - last_times
        else:
            dt = current_time - last_times if current_time.size(0) == node_ids.size(0) else (current_time.mean() - last_times)
        
        dt = torch.clamp(dt, min=0.0)
        decay_vals = torch.exp(-torch.abs(torch.log(torch.tensor(self.time_decay_factor))) * dt).unsqueeze(-1)
        raw = raw * decay_vals
        
        raw = _sanitize_tensor(raw, "inference_raw")
        
        proto = self.get_prototypes(node_ids)   # [len(node_ids), num_proto, D]
        time_enc = self.time_encoder(current_time)
        time_enc = _align_batch_size(time_enc, node_ids.size(0), "time_enc_inf")
        
        if edge_features is not None:
            edge_proj = self.edge_proj(edge_features.to(device))
            edge_proj = _align_batch_size(edge_proj, node_ids.size(0), "edge_proj_inf")
            norm = edge_proj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            edge_proj = torch.clamp((edge_proj / norm) * math.sqrt(self.memory_dim), -10.0, 10.0)
        else:
            edge_proj = torch.zeros(node_ids.size(0), self.memory_dim, device=device)
            
        stabilized, _ = self.sam_cell(raw, None, edge_proj, time_enc, proto)
        return stabilized.detach()
        
       

    @torch.no_grad()
    def reset_prototypes_if_needed(self, node_ids: torch.Tensor, threshold: float = 0.01):
        """
        Reset prototypes for nodes where the memory has become stale or NaN.
        Called after each batch to ensure prototypes remain stable.
        """
        proto = self.prototypes
        if torch.isnan(proto).any():
            logger.warning("NaN detected in global prototypes, resetting.")
            nn.init.xavier_uniform_(self.prototypes, gain=0.1)
            return      
        
       
        
    
    def reset_memory(self, node_ids: Optional[torch.Tensor] = None):
        if node_ids is None:
            self.raw_memory = torch.zeros_like(self.raw_memory)
            self.last_update_time = torch.zeros_like(self.last_update_time)
            nn.init.xavier_uniform_(self.all_prototypes, gain=0.1)
        else:
            self.raw_memory[node_ids] = 0.0
            self.last_update_time[node_ids] = 0.0
            self.all_prototypes[node_ids] = torch.randn_like(self.all_prototypes[node_ids]) * 0.01

    
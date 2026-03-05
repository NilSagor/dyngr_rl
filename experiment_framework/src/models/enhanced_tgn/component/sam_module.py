from typing import Optional, Dict, Tuple, List
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from loguru import logger

from .time_encoder import TimeEncoder




def _sanitize_tensor(
    x: torch.Tensor, 
    name: str = "tensor",                    
    nan_val: float = 0.0, 
    inf_val: float = 10.0,
    neg_inf_val: float = -10.0
) -> torch.Tensor:
    """Helper: sanitize tensor for NaN/Inf with consistent logging."""
    if not torch.isfinite(x).all():
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        logger.warning(f"NaN/Inf in {name}: shape={x.shape}, has_nan={has_nan}, has_inf={has_inf}")
        return torch.nan_to_num(x, nan=nan_val, posinf=inf_val, neginf=neg_inf_val)
    return x

def _align_batch_size(tensor: torch.Tensor, batch_size: int, name: str = "tensor") -> torch.Tensor:
    """Helper: align tensor batch size to target batch size."""
    if tensor.size(0) == batch_size:
        return tensor
    
    if tensor.size(0) == 1:
        return tensor.expand(batch_size, -1)  # Use expand instead of repeat for efficiency
    
    logger.warning(f"{name} batch mismatch: {tensor.size(0)} vs {batch_size}")
    if tensor.size(0) > batch_size:
        return tensor[:batch_size]
    else:
        # Pad with zeros
        pad_size = batch_size - tensor.size(0)
        return F.pad(tensor, (0, 0, 0, pad_size))



class SAMCell(nn.Module):
    def __init__(
            self, 
            memory_dim: int, 
            edge_feat_dim: int, 
            time_dim: int, 
            num_prototypes: int = 5, 
            similarity_metric: str = "cosine", 
            dropout: float = 0.1
        ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_prototypes = num_prototypes
        self.similarity_metric = similarity_metric
        
        self.query_input_dim = memory_dim + edge_feat_dim + time_dim
        self.gate_input_dim = 2 * memory_dim + time_dim
        
        self.query_proj = nn.Linear(self.query_input_dim, memory_dim)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        
        self.gate_proj = nn.Linear(self.gate_input_dim, 1)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 1.0)
        
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.temperature_min = 0.5
        self.temperature_max = 5.0
                
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(memory_dim, eps=1e-6)

        # Initialize with small values for stability
        self._init_weights_small()

    def _init_weights_small(self):
        """Initialize weights to small values for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_similarity(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        
        # Sanitize inputs before bmm operations
        # Sanitize inputs
        query = _sanitize_tensor(query, "query", inf_val=5.0, neg_inf_val=-5.0)
        prototypes = _sanitize_tensor(prototypes, "prototypes", inf_val=5.0, neg_inf_val=-5.0)
    
        
        # Compute similarity based on metric
        if self.similarity_metric == "cosine":
            query_norm = F.normalize(query, dim=-1, eps=1e-6)
            proto_norm = F.normalize(prototypes, dim=-1, eps=1e-6)
            sim = torch.bmm(query_norm.unsqueeze(1), proto_norm.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "dot":
            sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "scaled_dot":
            d_k = query.size(-1)
            scale = max(math.sqrt(d_k), 1.0)  # Prevent division by small numbers
            sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1) / scale
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # More aggressive clamping before temperature scaling
        sim = torch.clamp(sim, min=-10.0, max=10.0)

        # Use larger epsilon for temperature scaling
        temp = self.temperature.clamp(self.temperature_min, self.temperature_max)
        return sim / (temp + 1e-2)  # FIX: Larger epsilon for stability
        
        
    def forward(self, raw_memory, node_features, edge_features, time_encoding, prototypes, node_mask=None):
          
        # Complete NaN/Inf sanitization at entry
        raw_memory = _sanitize_tensor(raw_memory, "raw_memory_input")
        
        # if edge_features is not None:
        #     edge_features = _sanitize_tensor(edge_features, "edge_features")
        
        # Ensure consistent batch size across inputs
        batch_size = raw_memory.size(0)         
        
        # Consistent batch size handling with validation        
        # Align batch sizes
        time_encoding = _align_batch_size(time_encoding, batch_size, "time_encoding")
        if edge_features is not None:
            edge_features = _align_batch_size(edge_features, batch_size, "edge_features")
        if node_features is not None:
            node_features = _align_batch_size(node_features, batch_size, "node_features")
        
        
        
        # None-safe concatenation
        # Build query inputs
        query_inputs = [raw_memory]
        if edge_features is not None:
            edge_features = _sanitize_tensor(edge_features, "edge_features_input")
            query_inputs.append(edge_features)
        if time_encoding is not None:
            time_encoding = _sanitize_tensor(time_encoding, "time_encoding_input")
            query_inputs.append(time_encoding)
        
        query_inputs = torch.cat(query_inputs, dim=-1)
        
         # Project and normalize query
        query = self.query_proj(query_inputs)
        query = self.layer_norm(query)
        query = torch.tanh(query)  # Bounded activation
        
        # Compute similarity and attention
        similarity = self.compute_similarity(query, prototypes)       
       
        # Handle all-masked prototypes
        # Compute attention weights with masking
        if node_mask is not None:
            # Handle all-masked case
            all_masked = (~node_mask).all(dim=-1, keepdim=True)
            if all_masked.any():
                uniform = torch.ones_like(similarity) / self.num_prototypes
                masked_similarity = similarity.masked_fill(~node_mask, float('-inf'))
                # Stable softmax with large negative mask
                attention_weights = torch.where(
                    all_masked,
                    uniform,
                    F.softmax(masked_similarity, dim=-1)
                )
            else:
                masked_similarity = similarity.masked_fill(~node_mask, float('-inf'))
                attention_weights = F.softmax(masked_similarity, dim=-1)
        else:
            attention_weights = F.softmax(similarity, dim=-1)
        
        
        # Sanitize attention weights
        attention_weights = _sanitize_tensor(attention_weights, "attention_weights", inf_val=1.0)
        attention_weights = torch.clamp(attention_weights, 0.0, 1.0)  # Ensure valid probability
        
        # Compute candidate memory
        candidate_memory = torch.bmm(attention_weights.unsqueeze(1), prototypes).squeeze(1)
        candidate_memory = _sanitize_tensor(candidate_memory, "candidate_memory")
        candidate_memory = torch.clamp(candidate_memory, -5.0, 5.0)  # Tighter bounds
                
        # Compute update gate
        gate_inputs = [raw_memory, candidate_memory]
        if time_encoding is not None:
            gate_inputs.append(time_encoding)
        
        gate_inputs = torch.cat(gate_inputs, dim=-1)
        gate_inputs = torch.clamp(gate_inputs, min=-50.0, max=50.0)  # Prevent extreme values
        
        gate_logits = self.gate_proj(gate_inputs)
        gate_logits = torch.clamp(gate_logits, -10.0, 10.0)  # Prevent sigmoid saturation
        update_gate = torch.sigmoid(gate_logits)
        
        # Gated update
        updated_memory = (1 - update_gate) * raw_memory + update_gate * candidate_memory
        updated_memory = self.layer_norm(updated_memory)
        updated_memory = torch.clamp(updated_memory, -10.0, 10.0)
        
        # Final sanity check
        updated_memory = _sanitize_tensor(updated_memory, "updated_memory_output")
        
        return updated_memory, {
            "attention_weights": attention_weights.detach(),  # Detach to prevent graph bloat
            "update_gate": update_gate.detach(),
            "candidate_memory": candidate_memory.detach(),
            "query": query.detach(),
            "similarity_scores": similarity.detach()
        }


class StabilityAugmentedMemory(nn.Module):
    """
    StabilityAugmented Memory (SAM) Module
    
    Manages prototype-based memory for all nodes in the graph.
    Each node has k learnable prototypes representing stable states.
    """
    def __init__(
            self,
            num_nodes: int,
            memory_dim: int=128,
            node_feat_dim: int=0,
            edge_feat_dim: int=64,
            time_dim: int = 64,
            num_prototypes: int=5,
            similarity_metric: str = "cosine",
            dropout: float = 0.1,
        ):
        super(StabilityAugmentedMemory, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_dim = time_dim
        self.num_prototypes = num_prototypes
        
        # Time encoder (fixed, non-learnable)
        self.time_encoder = TimeEncoder(time_dim)        
        
        # Optional node feature projection
        if node_feat_dim > 0:
            self.node_proj = nn.Linear(node_feat_dim, memory_dim)
            nn.init.xavier_uniform_(self.node_proj.weight, gain=0.1)
            nn.init.zeros_(self.node_proj.bias)
        else:
            self.node_proj = None

        # Edge feature projection with safe initialization
        self.edge_proj = nn.Linear(edge_feat_dim, memory_dim)
        nn.init.xavier_uniform_(self.edge_proj.weight, gain=0.1)
        nn.init.zeros_(self.edge_proj.bias)       
        
        # SAM Cell for updates
        self.sam_cell = SAMCell(
            memory_dim=memory_dim,
            edge_feat_dim=memory_dim,
            time_dim=time_dim,
            num_prototypes=num_prototypes,
            similarity_metric=similarity_metric,
            dropout=dropout
        )    

        # Use larger epsilon for LayerNorm
        self.prototype_norm = nn.LayerNorm(memory_dim, eps=1e-6)
        
        # Raw memory states [num_nodes, memory_dim]
        self.register_buffer("raw_memory", torch.randn(num_nodes, memory_dim) * 0.001)
        
        # Last update time for each node
        self.register_buffer("last_update", torch.zeros(num_nodes))

        # Learnable prototypes [num_nodes, num_prototypes, memory_dim]
        self.all_prototypes = nn.Parameter(
            torch.empty(num_nodes, num_prototypes, memory_dim)
        )
        nn.init.xavier_uniform_(self.all_prototypes, gain=0.1)

        self._nodes_needing_reset: Optional[torch.Tensor] = None

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Get raw memory for specified nodes."""
        return self.raw_memory[node_ids]
    
    def get_prototypes(self, node_ids):
        prototypes = self.all_prototypes[node_ids]
        
        # FIX 5: Don't modify in-place during forward pass
        # Just sanitize and return, mark for reset after batch if needed
        if not torch.isfinite(prototypes).all():
            logger.warning("Prototypes contain NaN - sanitizing for this forward pass")
            prototypes = torch.nan_to_num(prototypes, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return self.prototype_norm(prototypes)


    def reset_prototypes_if_needed(self, node_ids: torch.Tensor):
        """Reset prototypes AFTER batch completion, not during forward pass"""
        prototypes = self.all_prototypes[node_ids]
        if not torch.isfinite(prototypes).all():
            logger.warning("Resetting NaN prototypes after batch")
            with torch.no_grad():
                new_protos = torch.empty_like(prototypes).normal_(0, 0.01)
                new_protos = torch.clamp(new_protos, min=-1.0, max=1.0)
                self.all_prototypes[node_ids] = new_protos
    
 
    def update_memory_batch(
            self,
            source_nodes: torch.Tensor,
            target_nodes: torch.Tensor,
            edge_features: torch.Tensor,
            current_time: torch.Tensor,
            node_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update memory for a batch of interactions.
        
        Returns:
            Dictionary with attention info (for debugging/analysis)
        """
        # Handle empty batch
        if source_nodes.numel() == 0 and target_nodes.numel() == 0:
            return {
                'source_attention': {}, 
                'target_attention': {},
                'source_memory': torch.tensor([]), 
                'target_memory': torch.tensor([])
            }
        
        # Validate batch sizes match
        batch_size = source_nodes.size(0)
        assert target_nodes.size(0) == batch_size, f"Source/target batch mismatch: {source_nodes.size(0)} vs {target_nodes.size(0)}"
        
        # Check and fix memory state
        if not torch.isfinite(self.raw_memory).all():
            logger.warning("Memory has NaN at update start, fixing...")
            self.raw_memory.data = torch.nan_to_num(
                self.raw_memory.data, nan=0.0, posinf=0.0, neginf=0.0
            )

        
        # Validate memory state
        # if not torch.isfinite(self.raw_memory).all():
        #     logger.warning("Memory contains NaN/Inf, resetting affected nodes...")
        #     nan_mask = ~torch.isfinite(self.raw_memory).any(dim=-1)
        #     self.raw_memory[nan_mask] = 0
        #     self.last_update[nan_mask] = 0
        
        edge_proj = self._safe_edge_projection(edge_features, batch_size)
        edge_proj = _align_batch_size(edge_proj, batch_size, "edge_proj")
   
        # if edge_features is not None and edge_features.numel() > 0:
        #     edge_proj = self.edge_proj(edge_features)
        #     # FIX 8: Check for NaN BEFORE normalization
        #     edge_proj = _sanitize_tensor(edge_proj, "edge_proj")
            
        #     edge_proj_norm = edge_proj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        #     # FIX 3: Check for near-zero norms before division
        #     # near_zero_mask = edge_proj_norm < 1e-6
        #     # edge_proj_norm = torch.where(near_zero_mask, torch.ones_like(edge_proj_norm), edge_proj_norm)
        #     # edge_proj = (edge_proj / edge_proj_norm) * 10.0
        #     # edge_proj = torch.clamp(edge_proj, -10.0, 10.0)
        #     edge_proj = (edge_proj / edge_proj_norm) * 10.0
        #     edge_proj = torch.clamp(edge_proj, -10.0, 10.0)
        # else:
        #     edge_proj = torch.zeros(batch_size, self.memory_dim, device=source_nodes.device)
        
        
        
        
        # Time encoding
        time_enc = self.time_encoder(current_time)
        assert time_enc.shape[-1] == self.time_dim, "Time encoder dimension mismatch"
        
        # Ensure time_enc has correct batch size
        if time_enc.size(0) == 1 and batch_size > 1:
            time_enc = time_enc.expand(batch_size, -1)
        elif time_enc.size(0) != batch_size:
            time_enc = _align_batch_size(time_enc, batch_size, "time_enc")

        
        # Get memory and prototypes
        src_mem = self.raw_memory[source_nodes]
        tgt_mem = self.raw_memory[target_nodes]
        src_proto = self.get_prototypes(source_nodes)
        tgt_proto = self.get_prototypes(target_nodes)
        
        # Project node features if available
        if self.node_proj is not None and node_features is not None:
            src_node_feat = self.node_proj(node_features[source_nodes])
            tgt_node_feat = self.node_proj(node_features[target_nodes])
        else:
            src_node_feat = tgt_node_feat = None
        
        
        # Compute updates via SAM cell (gradients flow through prototypes)        
        src_new, src_info = self.sam_cell(
            raw_memory=src_mem,
            node_features=src_node_feat,
            edge_features=edge_proj,
            time_encoding=time_enc,
            prototypes=src_proto
        )
        
        tgt_new, tgt_info = self.sam_cell(
            raw_memory=tgt_mem,
            node_features=tgt_node_feat,
            edge_features=edge_proj,
            time_encoding=time_enc,
            prototypes=tgt_proto
        )        

        # Check for NaN in outputs
        if not torch.isfinite(src_new).all() or not torch.isfinite(tgt_new).all():
            logger.error("SAM cell produced NaN outputs, skipping update")
            # Mark prototypes for reset
            all_nodes = torch.cat([source_nodes, target_nodes]).unique()
            if self._nodes_needing_reset is None:
                self._nodes_needing_reset = all_nodes
            else:
                self._nodes_needing_reset = torch.cat([self._nodes_needing_reset, all_nodes]).unique()
            
            return {
                'source_attention': {k: v.detach() for k, v in src_info.items()},
                'target_attention': {k: v.detach() for k, v in tgt_info.items()},
                'source_memory': torch.zeros_like(src_mem),
                'target_memory': torch.zeros_like(tgt_mem)
            }
        
        # Handle overlapping source/target nodes
        with torch.no_grad():
            src_new = torch.clamp(src_new, -10.0, 10.0)
            tgt_new = torch.clamp(tgt_new, -10.0, 10.0)

            # Handle overlapping nodes by averaging updates
            all_nodes = torch.cat([source_nodes, target_nodes])
            all_updates = torch.cat([src_new, tgt_new])
            all_times = torch.cat([current_time, current_time])     
           
            sorted_indices = torch.argsort(all_times, stable=True)
            sorted_nodes = all_nodes[sorted_indices]
            sorted_updates = all_updates[sorted_indices]

            unique_nodes = torch.unique(sorted_nodes, return_inverse=False, return_counts=False)
                        
            final_updates = torch.zeros_like(self.raw_memory[unique_nodes])
            final_times = torch.zeros(len(unique_nodes), device=self.raw_memory.device)
            
            for i, node in enumerate(unique_nodes):
                mask = sorted_nodes == node
                if mask.any():
                    # Take last update (highest index in sorted order)
                    last_idx = torch.where(mask)[0][-1]
                    final_updates[i] = sorted_updates[last_idx]
                    final_times[i] = all_times[sorted_indices][last_idx]
            
            self.raw_memory[unique_nodes] = final_updates
            self.last_update[unique_nodes] = final_times
                                  
            
            self.raw_memory.data = torch.nan_to_num(
                self.raw_memory.data,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0
            ).clamp_(-10, 10)
                   
        
        # Return attention info for analysis (detach to prevent graph leakage)
        return {
            'source_attention': {k: v.detach() for k, v in src_info.items()},
            'target_attention': {k: v.detach() for k, v in tgt_info.items()},
            'source_memory': src_new.detach(),
            'target_memory': tgt_new.detach()
        }
    
    def check_gradient_flow(self) -> Dict[str, float]:
        """Debug: Check which parameters are receiving gradients."""
        stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                stats[f"{name}_grad_norm"] = grad_norm
        return stats

    @torch.no_grad()
    def get_stabilized_memory(
            self,
            node_ids: torch.Tensor,
            current_time: torch.Tensor,
            edge_features: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Get stabilized memory for nodes without updating raw_memory.
        Useful for inference when no new interaction occurs.
        """
        if node_ids.numel() == 0:
            return torch.tensor([], device=self.raw_memory.device)
        
        # Ensure device synchronization
        device = self.raw_memory.device
        node_ids = node_ids.to(device)
        current_time = current_time.to(device)

        # Ensure current_time has at least one dimension
        if current_time.dim() == 0:
            current_time = current_time.unsqueeze(0)
        
                
        # Validate time encoding shape
        # Time encoding
        time_enc = self.time_encoder(current_time)
        if time_enc.size(0) == 1 and node_ids.size(0) > 1:
            time_enc = time_enc.expand(node_ids.size(0), -1)
        elif time_enc.size(0) != node_ids.size(0):
            time_enc = _align_batch_size(time_enc, node_ids.size(0), "time_enc_stabilized")
        
               
        raw = self.raw_memory[node_ids].detach()
        proto = self.get_prototypes(node_ids)         
                
        # Handle edge features        
        if edge_features is not None:
            edge_features = edge_features.to(device)
            edge_proj = self._safe_edge_projection(edge_features, node_ids.size(0))
            edge_proj = edge_proj.to(device)
            edge_proj = _align_batch_size(edge_proj, node_ids.size(0), "edge_proj_stabilized")
        else:
            edge_proj = torch.zeros(node_ids.size(0), self.memory_dim, device=device)
        
        # Compute stabilized memory
        stabilized, _ = self.sam_cell(
            raw_memory=raw,
            node_features=None,
            edge_features=edge_proj,
            time_encoding=time_enc,
            prototypes=proto
        )
        
        return stabilized.detach()

    def _safe_edge_projection(self, edge_features: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """Safe edge feature projection with controlled normalization."""
        if edge_features is None or edge_features.numel() == 0:
            return torch.zeros(batch_size, self.memory_dim, device=self.raw_memory.device)
        
        # Project
        edge_proj = self.edge_proj(edge_features)
        edge_proj = _sanitize_tensor(edge_proj, "edge_proj_pre_norm")
        
        # Controlled normalization: prevent explosion
        edge_proj_norm = edge_proj.norm(dim=-1, keepdim=True)
        
        # Safer normalization - use clamped norm with higher minimum
        safe_norm = torch.clamp(edge_proj_norm, min=1.0, max=100.0)
        
        # Normalize and scale moderately
        edge_proj = (edge_proj / safe_norm) * math.sqrt(self.memory_dim)
        
        # Aggressive clamping
        edge_proj = torch.clamp(edge_proj, -10.0, 10.0)
        
        return _sanitize_tensor(edge_proj, "edge_proj_final")
    
    
    def reset_memory(self, node_ids: Optional[torch.Tensor] = None):
        """Reset memory to zero."""
        if node_ids is None:
            self.raw_memory.zero_()
            self.last_update.zero_()
        else:
            self.raw_memory[node_ids] = 0.0
            self.last_update[node_ids] = 0.0

    def extra_repr(self) -> str:
        return (f"num_nodes={self.num_nodes}, memory_dim={self.memory_dim}, "
                f"num_prototypes={self.num_prototypes}, "
                f"similarity_metric={self.sam_cell.similarity_metric}")
    
    


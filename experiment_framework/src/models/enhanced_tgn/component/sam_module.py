from typing import Optional, Dict, Tuple, List
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from loguru import logger

from .time_encoder import TimeEncoder


class SAMCell(nn.Module):
    def __init__(self, memory_dim, edge_feat_dim, time_dim, 
                 num_prototypes=5, similarity_metric="cosine", dropout=0.1):
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
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        self.temperature_min = 0.05
        self.temperature_max = 2.0
                
        self.dropout = nn.Dropout(dropout)
        # Use larger epsilon for numerical stability
        self.layer_norm = nn.LayerNorm(memory_dim, eps=1e-4)

    def compute_similarity(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        
        # Sanitize inputs before bmm operations
        if not torch.isfinite(query).all():
            query = torch.nan_to_num(query, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(prototypes).all():
            prototypes = torch.nan_to_num(prototypes, nan=0.0, posinf=10.0, neginf=-10.0)

        # Handle all similarity metrics
        if self.similarity_metric == "cosine":
            # FIX: Use larger epsilon for normalize
            query_norm = F.normalize(query, dim=-1, eps=1e-6)
            proto_norm = F.normalize(prototypes, dim=-1, eps=1e-6)
            sim = torch.bmm(query_norm.unsqueeze(1), proto_norm.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "dot":
            sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "scaled_dot":
            d_k = query.size(-1)
            sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1) / math.sqrt(d_k)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        
        # More aggressive clamping before temperature scaling
        sim = torch.clamp(sim, min=-30.0, max=30.0)

        # Use larger epsilon for temperature scaling
        temp = self.temperature.clamp(self.temperature_min, self.temperature_max)
        return sim / (temp + 1e-4)  # FIX: Larger epsilon for stability
        
        
    def forward(self, raw_memory, node_features, edge_features, time_encoding, prototypes, node_mask=None):
          
        # Complete NaN/Inf sanitization at entry
        if not torch.isfinite(raw_memory).all():
            raw_memory = torch.nan_to_num(raw_memory, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if edge_features is not None and not torch.isfinite(edge_features).all():
            edge_features = torch.nan_to_num(edge_features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Ensure consistent batch size across inputs
        batch_size = raw_memory.size(0) 
        
        
        # Consistent batch size handling with validation
        if time_encoding.size(0) != batch_size:
            if time_encoding.size(0) == 1:
                time_encoding = time_encoding.repeat(batch_size, 1)
            else:
                logger.warning(f"Time encoding batch mismatch: {time_encoding.size(0)} vs {batch_size}")
                time_encoding = time_encoding[:batch_size] if time_encoding.size(0) > batch_size else \
                               F.pad(time_encoding, (0, 0, 0, batch_size - time_encoding.size(0)))
        
        if edge_features is not None and edge_features.size(0) != batch_size:
            if edge_features.size(0) == 1:
                edge_features = edge_features.repeat(batch_size, 1)
            else:
                edge_features = edge_features[:batch_size] if edge_features.size(0) > batch_size else \
                               F.pad(edge_features, (0, 0, 0, batch_size - edge_features.size(0)))
                
        
        if node_features is not None and node_features.size(0) != batch_size:
            if node_features.size(0) == 1:
                node_features = node_features.repeat(batch_size, 1)
            else:
                node_features = node_features[:batch_size] if node_features.size(0) > batch_size else \
                               F.pad(node_features, (0, 0, 0, batch_size - node_features.size(0)))
        
        
        # None-safe concatenation
        query_inputs = [raw_memory]
        if edge_features is not None:
            query_inputs.append(edge_features)
        if time_encoding is not None:
            query_inputs.append(time_encoding)
        query_inputs = torch.cat(query_inputs, dim=-1)
        
        query = self.query_proj(query_inputs)
        query = self.layer_norm(query)
        query = torch.tanh(query)
        
        similarity = self.compute_similarity(query, prototypes)       
                       
        
        # Handle all-masked prototypes
        if node_mask is not None:
            all_masked = (~node_mask).all(dim=-1, keepdim=True)
            if all_masked.any():
                uniform = torch.ones_like(similarity) / similarity.shape[-1]
                masked_similarity = similarity.masked_fill(~node_mask, float('-inf'))
                attention_weights = torch.where(
                    all_masked,
                    uniform,
                    F.softmax(masked_similarity, dim=-1)
                )
            else:
                similarity = similarity.masked_fill(~node_mask, float('-inf'))
                attention_weights = F.softmax(similarity, dim=-1)
        else:
            attention_weights = F.softmax(similarity, dim=-1)
        
        # Sanitize attention weights before bmm
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        candidate_memory = torch.bmm(
            attention_weights.unsqueeze(1), prototypes
        ).squeeze(1)
        candidate_memory = torch.clamp(candidate_memory, -5.0, 5.0)
                
        # None-safe gate inputs
        gate_inputs = [raw_memory, candidate_memory]
        if time_encoding is not None:
            gate_inputs.append(time_encoding)
        gate_inputs = torch.cat(gate_inputs, dim=-1)
        gate_inputs = torch.clamp(gate_inputs, min=-100, max=100)
        
        gate_logits = self.gate_proj(gate_inputs)
        update_gate = torch.sigmoid(gate_logits)
        
        updated_memory = (1 - update_gate) * raw_memory + update_gate * candidate_memory
        updated_memory = self.layer_norm(updated_memory)
        # Standardize clamp ranges across module
        updated_memory = torch.clamp(updated_memory, min=-10, max=10)
        
        return updated_memory, {
            "attention_weights": attention_weights,
            "update_gate": update_gate,
            "candidate_memory": candidate_memory,
            "query": query,
            "similarity_scores": similarity
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
            prototype_init: str="xavier",
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
            nn.init.xavier_uniform_(self.node_proj.weight)
            nn.init.zeros_(self.node_proj.bias)
        else:
            self.node_proj = None

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_feat_dim, memory_dim)
        nn.init.xavier_uniform_(self.edge_proj.weight, gain=1.0)
        nn.init.zeros_(self.edge_proj.bias)        
        
        # SAM Cell for updates
        self.sam_cell = SAMCell(
            memory_dim=memory_dim,
            edge_feat_dim=memory_dim,  # After projection
            time_dim=time_dim,
            num_prototypes=num_prototypes,
            similarity_metric=similarity_metric,
            dropout=dropout
        )      
        
        
        # Use larger epsilon for LayerNorm
        self.prototype_norm = nn.LayerNorm(memory_dim, eps=1e-4)
        
        # Raw memory states [num_nodes, memory_dim]
        self.register_buffer("raw_memory", torch.randn(num_nodes, memory_dim) * 0.01)

        # Last update time for each node
        self.register_buffer("last_update", torch.zeros(num_nodes))

        
        # Learnable prototypes [num_nodes, num_prototypes, memory_dim]
        self.all_prototypes = nn.Parameter(
            torch.empty(num_nodes, num_prototypes, memory_dim)
        )
        nn.init.xavier_uniform_(self.all_prototypes)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Get raw memory for specified nodes."""
        return self.raw_memory[node_ids]
    
    def get_prototypes(self, node_ids):
        prototypes = self.all_prototypes[node_ids]
        
        # Better validation and reset logic
        if not torch.isfinite(prototypes).all():
            logger.warning("Prototypes contain NaN - resetting to small random values")
            with torch.no_grad():
                new_protos = torch.empty_like(prototypes).normal_(0, 0.01)
                # FIX: Clamp to prevent rare Inf from normal_
                new_protos = torch.clamp(new_protos, min=-1.0, max=1.0)
                self.all_prototypes[node_ids] = new_protos
                prototypes = new_protos
        
        # Validate after reset
        if not torch.isfinite(prototypes).all():
            logger.error("Prototypes still contain NaN after reset - full reinitialization")
            with torch.no_grad():
                self.all_prototypes.data.normal_(0, 0.01)
                self.all_prototypes.data.clamp_(-1.0, 1.0)
                prototypes = self.all_prototypes[node_ids]
        
        
        return self.prototype_norm(prototypes)
    
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
            return {'source_attention': {}, 'target_attention': {}, 
                   'source_memory': torch.tensor([]), 'target_memory': torch.tensor([])}
        
        # Validate batch sizes match
        batch_size = source_nodes.size(0)
        assert target_nodes.size(0) == batch_size, f"Source/target batch mismatch: {source_nodes.size(0)} vs {target_nodes.size(0)}"
        
        # Validate memory state
        if not torch.isfinite(self.raw_memory).all():
            logger.warning("Memory contains NaN/Inf, resetting affected nodes...")
            nan_mask = ~torch.isfinite(self.raw_memory).any(dim=-1)
            self.raw_memory[nan_mask] = 0
            self.last_update[nan_mask] = 0
        
        # Project and normalize edge features (with None handling)
        if edge_features is not None and edge_features.numel() > 0:
            edge_proj = self.edge_proj(edge_features)
            if not torch.isfinite(edge_proj).all():
                edge_proj = torch.nan_to_num(edge_proj, nan=0.0, posinf=10.0, neginf=-10.0)
            # FIX: Use larger epsilon for norm
            edge_proj_norm = edge_proj.norm(dim=-1, keepdim=True) + 1e-6
            edge_proj = (edge_proj / edge_proj_norm) * 10.0
            edge_proj = torch.clamp(edge_proj, -10.0, 10.0)
        else:
            batch_size = max(source_nodes.size(0), target_nodes.size(0))
            edge_proj = torch.zeros(batch_size, self.memory_dim, device=source_nodes.device)
        
        
        # Time encoding
        time_enc = self.time_encoder(current_time)
        assert time_enc.shape[-1] == self.time_dim, "Time encoder dimension mismatch"

       

        # Get node-specific data
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
        
        # Standardize clamp ranges
        src_new = torch.clamp(src_new, -5.0, 5.0)
        tgt_new = torch.clamp(tgt_new, -5.0, 5.0)

        if not torch.isfinite(src_new).all() or not torch.isfinite(tgt_new).all():
            logger.error("SAM cell produced NaN - skipping update for this batch")
            return {
                'source_attention': {}, 
                'target_attention': {},
                'source_memory': torch.zeros_like(src_mem),
                'target_memory': torch.zeros_like(tgt_mem)
            }
        
        # Handle overlapping source/target nodes
        with torch.no_grad():
            src_new = torch.nan_to_num(src_new, nan=0.0, posinf=5.0, neginf=-5.0)
            tgt_new = torch.nan_to_num(tgt_new, nan=0.0, posinf=5.0, neginf=-5.0)
            
            # Handle overlapping nodes by averaging updates
            all_nodes = torch.cat([source_nodes, target_nodes])
            all_updates = torch.cat([src_new, tgt_new])
            
            # Use index_add_ for proper accumulation of overlapping updates
            unique_nodes, inverse_indices = torch.unique(all_nodes, return_inverse=True)
            accumulated_updates = torch.zeros(len(unique_nodes), self.memory_dim, 
                                            device=src_new.device)
            accumulated_updates.index_add_(0, inverse_indices, all_updates)
            update_counts = torch.bincount(inverse_indices, minlength=len(unique_nodes)).unsqueeze(1)
            averaged_updates = accumulated_updates / (update_counts + 1e-8)

            # Apply averaged updates
            self.raw_memory[unique_nodes] = averaged_updates
            self.last_update[unique_nodes] = current_time.max()
            
            # Use nan_to_num + clamp for comprehensive protection
            self.raw_memory.data = torch.nan_to_num(
                self.raw_memory.data, 
                nan=0.0, 
                posinf=10.0, 
                neginf=-10.0
            ).clamp_(-10, 10)
                   
        
        # Return attention info for analysis (detach to prevent graph leakage)
        return {
            'source_attention': {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                for k, v in src_info.items()},
            'target_attention': {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                for k, v in tgt_info.items()},
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
        if edge_features is not None:
            edge_features = edge_features.to(device)
        
        # Ensure current_time has at least one dimension
        if current_time.dim() == 0:
            current_time = current_time.unsqueeze(0)
        
        time_enc = self.time_encoder(current_time)
        
        # FIX: Validate time encoding shape
        if time_enc.size(0) == 1 and node_ids.size(0) > 1:
            time_enc = time_enc.repeat(node_ids.size(0), 1)
        elif time_enc.size(0) != node_ids.size(0):
            logger.warning(f"Time encoding batch mismatch in get_stabilized_memory")
            time_enc = time_enc[:node_ids.size(0)] if time_enc.size(0) > node_ids.size(0) else \
                      F.pad(time_enc, (0, 0, 0, node_ids.size(0) - time_enc.size(0)))
        
        raw = self.raw_memory[node_ids]
        proto = self.get_prototypes(node_ids)        
                
        # Handle edge features
        if edge_features is not None and edge_features.numel() > 0:
            edge_proj = self.edge_proj(edge_features)
            if not torch.isfinite(edge_proj).all():
                edge_proj = torch.nan_to_num(edge_proj, nan=0.0, posinf=10.0, neginf=-10.0)
            # FIX: Use larger epsilon
            edge_proj = F.normalize(edge_proj, dim=-1, eps=1e-6) * 10.0
            edge_proj = torch.clamp(edge_proj, -10.0, 10.0)
        else:
            edge_proj = torch.zeros(node_ids.size(0), self.memory_dim, device=device)
        
        
        stabilized, _ = self.sam_cell(
            raw_memory=raw,
            node_features=None,
            edge_features=edge_proj,
            time_encoding=time_enc,
            prototypes=proto
        )
        return stabilized

    def reset_memory(self, node_ids: Optional[torch.Tensor] = None):
        """Reset memory for specified nodes (or all nodes if None)."""
        if node_ids is None:
            self.raw_memory.zero_()
            self.last_update.zero_()
        else:
            self.raw_memory[node_ids] = 0
            self.last_update[node_ids] = 0
    
    def get_attention_stats(self) -> Dict[str, float]:
        """Get statistics about attention patterns (placeholder)."""
        return {
            "mean_attention_entropy": 0.0,
            "mean_update_gate": 0.0
        }


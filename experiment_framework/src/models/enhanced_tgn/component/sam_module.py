from typing import Optional, Dict, Tuple, List
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from loguru import logger

from .time_encoder import TimeEncoder

  


# class SAMCell(nn.Module):
#     def __init__(self, 
#                  num_nodes:int,
#                  memory_dim:int,
#                  node_feat_dim:int,
#                  edge_feat_dim:int,
#                  time_dim:int,
#                  num_prototypes: int=5,
#                  similarity_metric: str="cosine",
#                  dropout: float=0.1,
#                  **kwargs                        
#                 ):
#         super(SAMCell, self).__init__()
#         self.num_nodes = num_nodes
#         self.memory_dim = memory_dim
        
#         self.time_encoder = TimeEncoder(time_dim)
#         self.edge_proj = nn.Linear(edge_feat_dim, memory_dim)

#         self.sam_cell = SAMCell(
#             memory_dim=memory_dim,
#             edge_feat_dim=memory_dim,  # Projected dimension
#             time_dim=time_dim,
#             num_prototypes=num_prototypes,
#             **kwargs
#         )

#         self.prototype_norm = nn.LayerNorm(memory_dim)
#         self.register_buffer('raw_memory', torch.randn(num_nodes, memory_dim) * 0.01)
#         self.register_buffer('last_update', torch.zeros(num_nodes))
        
#         self.all_prototypes = nn.Parameter(
#             torch.empty(num_nodes, num_prototypes, memory_dim)
#         )
#         for i in range(num_nodes):
#             nn.init.xavier_uniform_(self.all_prototypes[i])

#         # self.node_feat_dim = node_feat_dim
#         # self.edge_feat_dim = edge_feat_dim
#         # self.time_dim = time_dim
#         # self.num_prototypes = num_prototypes
#         # self.similarity_metric = similarity_metric

#         # # dimensions for combined features
#         # self.query_input_dim = memory_dim + edge_feat_dim + time_dim
#         # self.gate_input_dim = memory_dim + memory_dim + time_dim # [m_u(t-)||s_u(t)||\phi(t)]

#         # # Query projection (step 1)
#         # self.query_proj = nn.Linear(self.query_input_dim, memory_dim)
#         # nn.init.xavier_uniform_(self.query_proj.weight)
#         # nn.init.zeros_(self.query_proj.bias)
        
        
        
#         # # update gate (step 4)
#         # self.gate_proj = nn.Linear(self.gate_input_dim, 1)
#         # nn.init.xavier_uniform_(self.gate_proj.weight)
#         # nn.init.constant_(self.gate_proj.bias, 1.0)
      

        

#         # # FIX 1: Bounded temperature
#         # self.temperature = nn.Parameter(torch.ones(1) * 0.1)
#         # self.temperature_min = 0.05  # Don't let it go too low
#         # self.temperature_max = 2.0   # Don't let it go too high

#         # self.residual_scale = 0.001

#         # # self.raw_memory.data = torch.randn_like(self.raw_memory)*0.1

#         # # Dropout for regularization
#         # self.dropout = nn.Dropout(dropout)

#         # # self.residual_weight = nn.Parameter(torch.tensor(0.01))
#         # # nn.init.uniform_(self.residual_weight, 0.05, 0.2)
#         # self.residual_weight = nn.Parameter(torch.tensor(0.01))
        
#         # # Layer norm for stability Add layer norm before output
#         # self.layer_norm = nn.LayerNorm(memory_dim)

#     def compute_similarity(
#             self,
#             query: torch.Tensor,
#             prototypes: torch.Tensor
#     )->torch.Tensor:
#         """
#         compute_similarity between query and prototypes.
        
        
#         :param query: [batch_size, memory_dim]
#         :type query: torch.Tensor
#         :param prototypes: [batch_size, num_prototypes, memory_dim] prototype vectors for each node in batch
#         :type prototypes: torch.Tensor 
#         :return: similarity scores [batch_size, num_prototypes]
#         :rtype: Tensor
#          query: [B, memory_dim]
#         prototypes: [B, num_prototypes, memory_dim]
#         returns: [B, num_prototypes]
#         """
#         if self.similarity_metric == "cosine":
#             # cosine similarity
#             query_norm = F.normalize(query, dim=-1)                # [B, D]
#             proto_norm = F.normalize(prototypes, dim=-1)           # [B, K, D]
            
#             # Batched matmul: (B,1,D) @ (B,D,K) -> (B,1,K) -> squeeze -> (B,K)
#             similarity = torch.bmm(query_norm.unsqueeze(1),
#                                proto_norm.transpose(1, 2)).squeeze(1) # [B, num_prototypes]
#         elif self.similarity_metric == 'dot':
#             # Dot product
#             similarity = torch.bmm(query.unsqueeze(1),
#                                prototypes.transpose(1, 2)).squeeze(1) # [B, num_prototypes]
#         elif self.similarity_metric == "scaled_dot":
#             # Scaled dot product
#             d_k = query.size(-1)
#             similarity = torch.bmm(query.unsqueeze(1),
#                                prototypes.transpose(1, 2)).squeeze(1) / math.sqrt(d_k)
#         else:
#             raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        
        
#         # Apply temperature
#         temp = self.temperature.clamp(self.temperature_min, self.temperature_max)
#         similarity = similarity / (temp + 1e-6) # Add epsilon for safety

#         return similarity
    
    
#     def forward(self,
#                 raw_memory: torch.Tensor,
#                 node_features: torch.Tensor,
#                 edge_features: torch.Tensor,
#                 time_encoding: torch.Tensor,
#                 prototypes: torch.Tensor,
#                 node_mask: Optional[torch.Tensor]=None
#             )->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         """
#         Perform SAM update for a batch of nodes
#         returns:
#             - update_memory: [B, memory_dim] s_u(t)
#             - attention_weights: [B, num_prototypes] a_u^k(t)
#             - update_gate: [B, 1] B_u(t)
#             - candidate_memory: [B, memory_dim] \tilde(s)_u(t)
#         """
#         batch_size = raw_memory.size(0)

#         # Add epsilon to prevent division by zero
#         edge_features = edge_features + 1e-8 if edge_features is not None else None
        
        
#         # EMERGENCY: Check and sanitize inputs
#         if not torch.isfinite(raw_memory).all():
#             logger.error(f"raw_memory NaN/Inf before SAM update")
#             logger.error(f"SAM input raw_memory has NaN/Inf! Location: {torch.where(~torch.isfinite(raw_memory))}")
#             raw_memory = torch.nan_to_num(raw_memory, nan=0.0, posinf=10.0, neginf=-10.0)
        
        
#         # # Diagnostic        
#         # if not torch.isfinite(edge_features).all():
#         #     logger.error(f"SAM input edge_features has NaN/Inf! Norm: {edge_features.norm()}")
        
                
#         # if edge_features.abs().sum() < 1e-6:
#         #     logger.warning("Edge features are zero in SAM update")        
        
        
#         if edge_features is not None:
#             edge_sum = edge_features.abs().sum().item()
#             edge_mean = edge_features.abs().mean().item()
            
#             if edge_sum < 1e-6:
#                 # Use debug level since small features may be expected
#                 logger.debug(f"Edge features near-zero: sum={edge_sum:.2e}, mean={edge_mean:.2e}")

        
#         # Check for NaN inputs
#         if torch.isnan(edge_features).any():
#             logger.warning("NaN detected in edge_features, replacing with zeros")
#             edge_features = torch.nan_to_num(edge_features, nan=0.0)
        
        
#         raw_memory = raw_memory + 1e-8

#         # Step 1: form query q_u(t)
#         # combine raw memory edge features, and time encoding    
        
#         query_inputs = torch.cat([
#             raw_memory,
#             edge_features,
#             time_encoding
#         ], dim=-1) # []

#         query = self.query_proj(query_inputs) #[B, memory_dim]
#         query = self.layer_norm(query)
#         # query = self.dropout(query)
#         query = torch.tanh(query)

#         query = torch.clamp(query, -10.0, 10.0)
        
#         # step 2: compute prototype attention \alpha_u^k(t)
#         similarity = self.compute_similarity(query, prototypes) # [B, num_prototypes]
        
        
#         # Constrained temperature
#         # temp = self.temperature.clamp(self.temperature_min, self.temperature_max)
#         # similarity = similarity / temp
        
#         # apply mask if provided (for padded nodes)
#         if node_mask is not None:
#             similarity = similarity.masked_fill(~node_mask.unsqueeze(-1), float('-inf'))

#         attention_weights = F.softmax(similarity, dim=-1) #[B, num_prototypes]
        
#         # Step 3: Form candidate memory \tilde(s)_u(t)
#         candidate_memory = torch.bmm(
#             attention_weights.unsqueeze(1),  # [B, 1, K]
#             prototypes                        # [B, K, D]
#         ).squeeze(1)                          # [B, D]
        
#         candidate_memory = torch.clamp(candidate_memory, -5.0, 5.0)

#         # step 4: compute update date \beta_u(t)
#         gate_inputs = torch.cat([
#             raw_memory,
#             candidate_memory,
#             time_encoding
#         ], dim=-1)
#         gate_inputs = torch.clamp(gate_inputs, min=-1000, max=1000)

        
#         gate_logits = self.gate_proj(gate_inputs) #[B, 1]
#         update_gate = torch.sigmoid(gate_logits) #[B, 1] \beta_u(t)
        
#         if self.training and torch.rand(1).item() < 0.001:  # log rarely
#             logger.info(f"Update gate: mean={update_gate.mean().item():.4f}, "
#                         f"std={update_gate.std().item():.4f}")
        
      

#         # step 5: final memory update s_u(t)      
        
#         # updated_memory = (1 - update_gate + self.residual_weight) * raw_memory + update_gate * candidate_memory
#         # updated_memory = (1 - update_gate) * raw_memory + update_gate * candidate_memory

#         # updated_memory = updated_memory + self.residual_weight * raw_memory
#         # updated_memory = self.layer_norm(updated_memory)
        
        
        
#         # Conservative update with tiny fixed residual
#         updated_memory = (1 - update_gate) * raw_memory + update_gate * candidate_memory
#         # updated_memory = updated_memory + self.residual_scale * raw_memory  # Fixed 0.1%
        
#          # FIX 3: Output normalization + gradient clipping
#         updated_memory = self.layer_norm(updated_memory)
        
#         # Emergency clamp to prevent explosion
#         updated_memory = torch.clamp(updated_memory, min=-50, max=50)
        
#         # collect attention info for analysis
#         attention_info = {
#             "attention_weights": attention_weights,
#             "update_gate": update_gate,
#             "candidate_memory": candidate_memory,
#             "query": query,
#             "similarity_scores": similarity
#         }

                
#         return updated_memory, attention_info


# class SAMCell(nn.Module):
#     def __init__(
#             self, 
#             memory_dim,             
#             edge_feat_dim, 
#             time_dim,
#             num_prototypes=5, 
#             similarity_metric="cosine", 
#             dropout=0.1
#         ):
#         super().__init__()
#         self.memory_dim = memory_dim
#         self.num_prototypes = num_prototypes
#         self.similarity_metric = similarity_metric
        
#         self.query_input_dim = memory_dim + edge_feat_dim + time_dim
#         self.gate_input_dim = 2 * memory_dim + time_dim
        
#         self.query_proj = nn.Linear(self.query_input_dim, memory_dim)
#         nn.init.xavier_uniform_(self.query_proj.weight)
#         nn.init.zeros_(self.query_proj.bias)
        
#         self.gate_proj = nn.Linear(self.gate_input_dim, 1)
#         nn.init.xavier_uniform_(self.gate_proj.weight)
#         nn.init.constant_(self.gate_proj.bias, 1.0)
        
#         self.temperature = nn.Parameter(torch.ones(1) * 0.1)
#         self.temperature_min = 0.05
#         self.temperature_max = 2.0
        
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(memory_dim)
        

#     def compute_similarity(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
#         if self.similarity_metric == "cosine":
#             query = F.normalize(query, dim=-1)
#             prototypes = F.normalize(prototypes, dim=-1)
#             sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1)
#         # ... other cases
        
#         temp = self.temperature.clamp(self.temperature_min, self.temperature_max)
#         return sim / (temp + 1e-6)

#     def forward(self, raw_memory, node_features, edge_features, time_encoding, prototypes, node_mask=None):
#         # Sanitize inputs
#         if not torch.isfinite(raw_memory).all():
#             raw_memory = torch.nan_to_num(raw_memory, nan=0.0, posinf=10.0, neginf=-10.0)
        
#         if edge_features is not None and torch.isnan(edge_features).any():
#             edge_features = torch.nan_to_num(edge_features, nan=0.0)
        
#         # Step 1: Query
#         query_inputs = torch.cat([raw_memory, edge_features, time_encoding], dim=-1)
#         query = self.query_proj(query_inputs)
#         query = self.layer_norm(query)
#         query = torch.tanh(query)  # Already bounded [-1,1], no extra clamp needed
        
#         # Step 2: Similarity (handled in compute_similarity)
#         similarity = self.compute_similarity(query, prototypes)
        
#         if node_mask is not None:
#             similarity = similarity.masked_fill(~node_mask, float('-inf'))
        
#         attention_weights = F.softmax(similarity, dim=-1)
        
#         # Step 3: Candidate memory
#         candidate_memory = torch.bmm(
#             attention_weights.unsqueeze(1), prototypes
#         ).squeeze(1)
#         candidate_memory = torch.clamp(candidate_memory, -5.0, 5.0)
        
#         # Step 4: Update gate
#         gate_inputs = torch.cat([raw_memory, candidate_memory, time_encoding], dim=-1)
#         gate_inputs = torch.clamp(gate_inputs, min=-100, max=100)  # Tighter clamp
#         gate_logits = self.gate_proj(gate_inputs)
#         update_gate = torch.sigmoid(gate_logits)
        
#         # Step 5: Final update
#         updated_memory = (1 - update_gate) * raw_memory + update_gate * candidate_memory
#         updated_memory = self.layer_norm(updated_memory)
#         updated_memory = torch.clamp(updated_memory, min=-50, max=50)
        
#         return updated_memory, {
#             "attention_weights": attention_weights,
#             "update_gate": update_gate,
#             "candidate_memory": candidate_memory,
#             "query": query,
#             "similarity_scores": similarity
#         }
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
        self.layer_norm = nn.LayerNorm(memory_dim)

    def compute_similarity(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # ✅ Fixed: Handle all similarity metrics
        if self.similarity_metric == "cosine":
            query_norm = F.normalize(query, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            sim = torch.bmm(query_norm.unsqueeze(1), proto_norm.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "dot":
            sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "scaled_dot":
            d_k = query.size(-1)
            sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1) / math.sqrt(d_k)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        temp = self.temperature.clamp(self.temperature_min, self.temperature_max)
        return sim / (temp + 1e-6)

    def forward(self, raw_memory, node_features, edge_features, time_encoding, prototypes, node_mask=None):
        # ✅ Fixed: Complete NaN/Inf sanitization
        if not torch.isfinite(raw_memory).all():
            raw_memory = torch.nan_to_num(raw_memory, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if edge_features is not None and not torch.isfinite(edge_features).all():
            edge_features = torch.nan_to_num(edge_features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # ✅ Fixed: None-safe concatenation
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
        
        # ✅ Fixed: Handle all-masked prototypes
        if node_mask is not None:
            similarity = similarity.masked_fill(~node_mask, float('-inf'))
            all_masked = (similarity == float('-inf')).all(dim=-1, keepdim=True)
            if all_masked.any():
                uniform = torch.ones_like(similarity) / similarity.shape[-1]
                attention_weights = torch.where(all_masked, uniform, F.softmax(similarity, dim=-1))
            else:
                attention_weights = F.softmax(similarity, dim=-1)
        else:
            attention_weights = F.softmax(similarity, dim=-1)
        
        candidate_memory = torch.bmm(
            attention_weights.unsqueeze(1), prototypes
        ).squeeze(1)
        candidate_memory = torch.clamp(candidate_memory, -5.0, 5.0)
        
        # ✅ Fixed: None-safe gate inputs
        gate_inputs = [raw_memory, candidate_memory]
        if time_encoding is not None:
            gate_inputs.append(time_encoding)
        gate_inputs = torch.cat(gate_inputs, dim=-1)
        gate_inputs = torch.clamp(gate_inputs, min=-100, max=100)
        
        gate_logits = self.gate_proj(gate_inputs)
        update_gate = torch.sigmoid(gate_logits)
        
        updated_memory = (1 - update_gate) * raw_memory + update_gate * candidate_memory
        updated_memory = self.layer_norm(updated_memory)
        updated_memory = torch.clamp(updated_memory, min=-50, max=50)
        
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
            # node_feat_dim=node_feat_dim,
            edge_feat_dim=memory_dim,  # After projection
            time_dim=time_dim,
            num_prototypes=num_prototypes,
            similarity_metric=similarity_metric,
            dropout=dropout
        )
        
        self.prototype_norm = nn.LayerNorm(memory_dim)
        
        # Raw memory states [num_nodes, memory_dim]
        self.register_buffer("raw_memory", torch.randn(num_nodes, memory_dim) * 0.01)
        
        # Last update time for each node
        self.register_buffer("last_update", torch.zeros(num_nodes))

        # Learnable prototypes [num_nodes, num_prototypes, memory_dim]
        self.all_prototypes = nn.Parameter(
            torch.empty(num_nodes, num_prototypes, memory_dim)
        )
        # ✅ Vectorized initialization
        nn.init.xavier_uniform_(self.all_prototypes)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Get raw memory for specified nodes."""
        return self.raw_memory[node_ids]
    
    def get_prototypes(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Get prototype vectors for specified nodes."""
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
        
        # Validate memory state
        if not torch.isfinite(self.raw_memory).all():
            logger.warning("Memory contains NaN/Inf, resetting...")
            self.reset_memory()
        
        # Project and normalize edge features (with None handling)
        if edge_features is not None and edge_features.numel() > 0:
            edge_proj = self.edge_proj(edge_features)
            # L2 normalize and scale
            edge_proj_norm = edge_proj.norm(dim=-1, keepdim=True) + 1e-8
            edge_proj = (edge_proj / edge_proj_norm) * 10.0
            edge_proj = torch.clamp(edge_proj, -10.0, 10.0)
        else:
            # Create zero features with correct shape
            batch_size = max(source_nodes.size(0), target_nodes.size(0))
            edge_proj = torch.zeros(batch_size, self.memory_dim, 
                                  device=source_nodes.device)
        
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
        
        # Update memory buffers WITHOUT gradients (state evolution)
        with torch.no_grad():
            self.raw_memory[source_nodes] = src_new
            self.raw_memory[target_nodes] = tgt_new
            self.last_update[source_nodes] = current_time
            self.last_update[target_nodes] = current_time
            # Emergency clamp to prevent explosion
            self.raw_memory.data.clamp_(-50.0, 50.0)
        
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
        
        raw = self.raw_memory[node_ids]
        proto = self.get_prototypes(node_ids)
        time_enc = self.time_encoder(current_time)
        
        # Handle edge features
        if edge_features is not None and edge_features.numel() > 0:
            edge_proj = self.edge_proj(edge_features)
            edge_proj = F.normalize(edge_proj, dim=-1) * 10.0
            edge_proj = torch.clamp(edge_proj, -10.0, 10.0)
        else:
            edge_proj = torch.zeros(node_ids.size(0), self.memory_dim, device=raw.device)
        
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






# class StabilityAugmentedMemory(nn.Module):
#     """
#     StabilityAugmented Memory (SAM) Module
    
#     Mananges prototype-based memory for all nodes in the graph.
#     Each node has k learnable prototypes representing stable states.
#     """
#     def __init__(
#             self,
#             num_nodes: int,
#             memory_dim: int=128,
#             node_feat_dim: int=0,
#             edge_feat_dim: int=64,
#             time_dim: int = 64,
#             num_prototypes: int=5,
#             prototype_init: str="xavier",
#             similarity_metric: str = "cosine",
#             dropout: float = 0.1,
#             # device: str = "cuda"

#     ):
#         super(StabilityAugmentedMemory, self).__init__()
#         self.num_nodes = num_nodes
#         self.memory_dim = memory_dim
#         self.node_feat_dim = node_feat_dim
#         self.edge_feat_dim = edge_feat_dim
#         self.time_dim = time_dim
#         self.num_prototypes = num_prototypes
#         # self.device = device

#         # Time encoder (fixed, non-learnable)
#         self.time_encoder = TimeEncoder(time_dim)

#         # Optional node feature projection (if node feature exists)
#         if node_feat_dim>0:
#             self.node_proj = nn.Linear(node_feat_dim, memory_dim)
#             nn.init.xavier_uniform_(self.node_proj.weight)
#             nn.init.zeros_(self.node_proj.bias)
#         else:
#             self.node_proj = None
        
        
#         # Edge feature projection
#         # self.edge_proj = nn.Linear(edge_feat_dim, memory_dim)
#         # nn.init.xavier_uniform_(self.edge_proj.weight, gain=0.5)
#         # nn.init.zeros_(self.edge_proj.bias)

#         self.edge_proj = nn.Linear(edge_feat_dim, memory_dim)
#         # Use smaller gain for stability with potentially sparse inputs
#         nn.init.xavier_uniform_(self.edge_proj.weight, gain=1.0)  # Was: 0.5
#         nn.init.zeros_(self.edge_proj.bias)

#         # SAM Cell for updates
#         self.sam_cell = SAMCell(
#             memory_dim = memory_dim,
#             node_feat_dim = node_feat_dim,
#             # edge_feat_dim = edge_feat_dim,
#             edge_feat_dim = memory_dim,
#             time_dim = time_dim,
#             num_prototypes = num_prototypes,
#             similarity_metric = similarity_metric,
#             dropout = dropout
#         )
        
#         self.prototype_norm = nn.LayerNorm(memory_dim)
        
#         # initialize prototype memories for each node
#         # self.prototype_memories = nn.ModuleDict()
#         # for node_id in range(num_nodes):
#         #     self.prototype_memories[str(node_id)] = PrototypeMemory(
#         #         num_prototypes = num_prototypes,
#         #         prototype_dim = memory_dim,
#         #         node_id = node_id,
#         #         initialization = prototype_init
#         #     )
        
#         # raw memory states (current m_u(t) for each node)
#         self.register_buffer(
#             "raw_memory", 
#             torch.randn(num_nodes, memory_dim) * 0.01
#         )
        
#         # last update time for each node
#         self.register_buffer(
#             "last_update",
#             torch.zeros(num_nodes)
#         )

#         self.all_prototypes = nn.Parameter(
#             torch.empty(num_nodes, num_prototypes, memory_dim)
#         )
#         # Initialize (example for xavier):
#         for i in range(num_nodes):
#             nn.init.xavier_uniform_(self.all_prototypes[i])
        
#         # # move to device
#         # self.to(device)

#     def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
#         """
#         Get raw memory for specified nodes.
        
#         Args:
#             node_ids: [batch_size] node indices
            
#         Returns:
#             [batch_size, memory_dim] raw memory states
#         """
#         return self.raw_memory[node_ids]
    
    
#     def get_prototypes(self, node_ids: torch.Tensor) -> torch.Tensor:
#         """
#         Get prototype vectors for specified nodes.
        
#         Args:
#             node_ids: [batch_size] node indices
            
#         Returns:
#             [batch_size, num_prototypes, memory_dim] prototype vectors
#         """
#         # [batch_size, num_prototypes, memory_dim]
#         # Vectorized lookup: [B] -> [B, K, D]
#         prototypes = self.all_prototypes[node_ids]
#         return self.prototype_norm(prototypes)
    
    
    
#     def update_memory_batch(
#             self,
#             source_nodes: torch.Tensor,
#             target_nodes: torch.Tensor,
#             edge_features: torch.Tensor,
#             current_time: torch.Tensor,
#             node_features: Optional[torch.Tensor] = None
#     )->Dict[str, torch.Tensor]:
#         """
#         Update memory for a batch of interactions.

#         Returns:
#             Dictionary with updated memories and attention info        

#         """
#         # CRITICAL: Wrap EVERYTHING in no_grad for inference-time memory update
        
#         # if not torch.isfinite(self.raw_memory).all():
#         #     logger.error("Memory already NaN before update! Resetting...")
#         #     self.reset_memory()
            
#         # batch_size = source_nodes.size(0)

#         #     # get raw memories for source and target
#         #     # Get raw memories (no gradients needed)
#         #     src_raw_memory = self.raw_memory[source_nodes]
#         #     tgt_raw_memory = self.raw_memory[target_nodes]
        
#         #     # Get prototypes (still need to track these for later gradient computation?
#         #     # NO—if this is just for memory state update, no gradients needed)
#         #     src_prototypes = self.get_prototypes(source_nodes)
#         #     tgt_prototypes = self.get_prototypes(target_nodes)
        
#         #     if edge_features.abs().sum() < 1e-6:
#         #         logger.debug(f"Raw edge features are near-zero: sum={edge_features.abs().sum().item():.2e}")
            
            
            
            
#         #     # Project features (no grad)
#         #     edge_features_proj = self.edge_proj(edge_features)

#         #     if edge_features_proj is not None:
#         #         edge_proj_norm = edge_features_proj.norm(dim=-1, keepdim=True)
#         #         edge_features_proj = edge_features_proj / (edge_proj_norm + 1e-8) * 10.0
                
#         #         # Safety clamp after normalization
#         #         edge_features_proj = torch.clamp(edge_features_proj, -10.0, 10.0)


#         #     if edge_features_proj.abs().sum() < 1e-6:
#         #         logger.warning(
#         #             f"Projected edge features near-zero! "
#         #             f"Raw sum={edge_features.abs().sum().item():.2e}, "
#         #             f"Proj sum={edge_features_proj.abs().sum().item():.2e}, "
#         #             f"edge_proj weight norm={self.edge_proj.weight.norm().item():.4f}"
#         #         )
            
            
#         #     # CRITICAL: Check if projection produced NaN
#         #     if not torch.isfinite(edge_features_proj).all():
#         #         logger.error(f"edge_proj produced NaN! Using zeros.")
#         #         edge_features_proj = torch.zeros_like(edge_features_proj)

#         #     # SAFETY CHECK
#         #     if edge_features_proj.abs().max() > 100:  # Explosion detection
#         #         logger.warning(f"Edge projection exploded (max={edge_features_proj.abs().max():.2f}), clamping")
#         #         edge_features_proj = torch.clamp(edge_features_proj, -10, 10)

            

#         #     # get time encoding
#         #     time_encoding = self.time_encoder(current_time)

#         #     # Check time encoding
#         #     if not torch.isfinite(time_encoding).all():
#         #         logger.error(f"time_encoder produced NaN! Using zeros.")
#         #         time_encoding = torch.zeros_like(time_encoding) 

#         #     # Process node features if needed
#         #     if self.node_proj is not None and node_features is not None:
#         #         src_node_feats = node_features[source_nodes]
#         #         tgt_node_feats = node_features[target_nodes]
#         #         src_node_feats_proj = self.node_proj(src_node_feats)
#         #         tgt_node_feats_proj = self.node_proj(tgt_node_feats)
#         #     else:
#         #         src_node_feats_proj = tgt_node_feats_proj = None
     
            
#             # Update via SAM cell (NO gradients for this update!)
#             # src_updated, src_attention = self.sam_cell(
#             #     raw_memory=src_raw_memory,
#             #     node_features=src_node_feats_proj,
#             #     edge_features=edge_features_proj,
#             #     time_encoding=time_encoding,
#             #     prototypes=src_prototypes
#             # )
            
#             # tgt_updated, tgt_attention = self.sam_cell(
#             #     raw_memory=tgt_raw_memory,
#             #     node_features=tgt_node_feats_proj,
#             #     edge_features=edge_features_proj,
#             #     time_encoding=time_encoding,
#             #     prototypes=tgt_prototypes
#             # )

#             # Update stored memories (already in no_grad context)
#             # self.raw_memory[source_nodes] = src_updated
#             # self.raw_memory[target_nodes] = tgt_updated
#             # self.last_update[source_nodes] = current_time
#             # self.last_update[target_nodes] = current_time

#             # self.raw_memory.data.clamp_(-50, 50)
        
#         # Validate inputs
#         if not torch.isfinite(self.raw_memory).all():
#             self.reset_memory()
        
#         # Project edge features once
#         edge_proj = self.edge_proj(edge_features)
#         edge_proj = F.normalize(edge_proj, dim=-1) * 10
#         edge_proj = torch.clamp(edge_proj, -10, 10)
        
#         time_enc = self.time_encoder(current_time)
        
#         # Get node data
#         src_mem, tgt_mem = self.raw_memory[source_nodes], self.raw_memory[target_nodes]
#         src_proto, tgt_proto = self.get_prototypes(source_nodes), self.get_prototypes(target_nodes)
        
#         # Compute updates (WITH gradients for prototypes)
#         src_new, src_info = self.sam_cell(src_mem, edge_proj, time_enc, src_proto)
#         tgt_new, tgt_info = self.sam_cell(tgt_mem, edge_proj, time_enc, tgt_proto)
        
#         # Update buffers (WITHOUT gradients)
#         with torch.no_grad():
#             self.raw_memory[source_nodes] = src_new
#             self.raw_memory[target_nodes] = tgt_new
#             self.last_update[source_nodes] = current_time
#             self.last_update[target_nodes] = current_time
        
        
#         # return {
#         #     'source_attention': {k: v.detach() if isinstance(v, torch.Tensor) else v 
#         #                         for k, v in src_attention.items()},
#         #     'target_attention': {k: v.detach() if isinstance(v, torch.Tensor) else v 
#         #                         for k, v in tgt_attention.items()},
#         #     'source_memory': src_updated.detach(),
#         #     'target_memory': tgt_updated.detach()
#         # }
#         return {'src': src_info, 'tgt': tgt_info}  
        
        
#     def check_gradient_flow(self) -> Dict[str, float]:
#         """Debug: Check which parameters are receiving gradients."""
#         stats = {}
#         for name, param in self.named_parameters():
#             if param.requires_grad and param.grad is not None:
#                 stats[f"{name}_grad_norm"] = param.grad.norm().item()
#             elif param.requires_grad:
#                 stats[f"{name}_grad_norm"] = 0.0  # No gradient!
#         return stats

#     @torch.no_grad()
#     def get_stabilized_memory(self, node_ids, current_time, edge_features=None):
#         raw = self.raw_memory[node_ids]
#         proto = self.get_prototypes(node_ids)
#         time_enc = self.time_encoder(current_time)
        
#         if edge_features is None:
#             edge_proj = torch.zeros(node_ids.size(0), self.memory_dim, device=raw.device)
#         else:
#             edge_proj = torch.clamp(self.edge_proj(edge_features), -10, 10)
        
#         stabilized, _ = self.sam_cell(raw, edge_proj, time_enc, proto)
#         return stabilized
    
    
    
    
#     # def get_stabilized_memory(
#     #         self,
#     #         node_ids: torch.Tensor,
#     #         current_time: torch.Tensor,
#     #         edge_features: Optional[torch.Tensor] = None,
#     #         node_features: Optional[torch.Tensor] = None
#     # )->torch.Tensor:
#     #     """
#     #     Get stabilized memory for nodes without performing a full update.
#     #     Useful for inference when no new interaction occurs.
#     #     this computes s_u(t) using the current memory and prototypes,
#     #     but does not store the result back to raw_memory.


#     #     """
#     #     batch_size = node_ids.size(0)
        
#     #     # get raw memories
#     #     raw_memory = self.raw_memory[node_ids]
        
#     #     # get prototypes
#     #     prototypes = self.get_prototypes(node_ids)
        
#     #     # get time encoding
#     #     time_encoding = self.time_encoder(current_time)
        
#     #     # Edge features (zeros if not provided)
#     #     if edge_features is None:
#     #         device = next(self.edge_proj.parameters()).device
#     #         edge_features = torch.zeros(batch_size, self.edge_feat_dim, device=device)
#     #     edge_features_proj = self.edge_proj(edge_features)
        
#     #     # node features (optional)
#     #     if self.node_proj is not None and node_features is not None:
#     #         node_feats_proj = self.node_proj(node_features[node_ids])
#     #     else:
#     #         node_feats_proj = None
        
#     #     stabilized, attention = self.sam_cell(
#     #         raw_memory = raw_memory,
#     #         node_features = node_feats_proj,
#     #         edge_features = edge_features_proj,
#     #         time_encoding = time_encoding,
#     #         prototypes = prototypes
#     #     )

#     #     return stabilized


#     def reset_memory(self, node_ids:Optional[torch.Tensor]=None):
#         """
#         Reset memory for specified nodes (or all nodes if None)
#         """
#         if node_ids is None:
#             self.raw_memory.zero_()
#             self.last_update.zero_()
#         else:
#             self.raw_memory[node_ids] = 0
#             self.last_update[node_ids] =0
    
#     def get_attention_stats(self)->Dict[str, float]:
#         """
#         Get statistics about attention patterns (for debugging/analysis)
#         # this would require storing attention over many updates
#         # placeholder for now
#         """
#         return {
#             "mean_attention_entropy": 0.0,
#             "mean_update_gate": 0.0
#         }

    

        


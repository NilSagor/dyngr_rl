from typing import Optional, Dict, Tuple, List
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


from .time_encoder import TimeEncoder

class PrototypeMemory(nn.Module):
    """
        Learnable prototype vectors for a single node.
        Each node has K prototypes representing stable "modes" or "roles".
    """    
    def __init__(self, 
                 num_prototypes:int, 
                 prototype_dim:int, 
                 node_id:int, 
                 initialization:str='xavier'):
        super(PrototypeMemory, self).__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.node_id = node_id

        # 
        self.prototypes = nn.Parameter(
            torch.empty(num_prototypes, prototype_dim)
        )

        # initialize prototypes
        if initialization == 'xavier':
            nn.init.xavier_uniform_(self.prototypes)
        elif initialization == "normal":
            nn.init.normal_(self.prototypes, mean=0.0, std=0.1)
        elif initialization == "uniform":
            nn.init.uniform_(self.prototypes, -0.1, 0.1)
        else:
            raise ValueError(f"Unknown initialization: {initialization}")
        

    def forward(self)->torch.Tensor:
        """Return prototype vectors"""
        return self.prototypes
    
    def get_prototype(self, idx:int)->torch.Tensor:
        """Get a specific prototype"""
        return self.prototypes[idx]
    


class SAMCell(nn.Module):
    def __init__(self, 
                 memory_dim:int,
                 node_feat_dim:int,
                 edge_feat_dim:int,
                 time_dim:int,
                 num_prototypes: int=5,
                 similarity_metric: str="cosine",
                 dropout: float=0.1                        
                ):
        super(SAMCell, self).__init__()
        self.memory_dim = memory_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_dim = time_dim
        self.num_prototypes = num_prototypes
        self.similarity_metric = similarity_metric

        # dimensions for combined features
        self.query_input_dim = memory_dim + edge_feat_dim + time_dim
        self.gate_input_dim = memory_dim + memory_dim + time_dim # [m_u(t-)||s_u(t)||\phi(t)]

        # Query projection (step 1)
        self.query_proj = nn.Linear(self.query_input_dim, memory_dim)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        
        # update gate (step 4)
        self.gate_proj = nn.Linear(self.gate_input_dim, 1)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

        # Similarity temperature (learnable)
        self.temperature = nn.Parameter(torch.ones(1)*0.1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(memory_dim)

    def compute_similarity(
            self,
            query: torch.Tensor,
            prototypes: torch.Tensor
    )->torch.Tensor:
        """
        compute_similarity between query and prototypes.
        
        
        :param query: [batch_size, memory_dim]
        :type query: torch.Tensor
        :param prototypes: [num_prototypes, memory_dim]
        :type prototypes: torch.Tensor 
        :return: similarity scores [batch_size, num_prototypes]
        :rtype: Tensor
        """
        if self.similarity_metric == "cosine":
            # cosine similarity
            query_norm = F.normalize(query, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            similarity = torch.matmul(query_norm, proto_norm.t()) # [B, num_prototypes]
        elif self.similarity_metric == 'dot':
            # Dot product
            similarity = torch.matmul(query, prototypes.t()) # [B, num_prototypes]
        elif self.similarity_metric == "scaled_dot":
            # Scaled dot product
            d_k = query.size(-1)
            similarity = torch.matmul(query, prototypes.t())/math.sqrt(d_k)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Apply temperature
        similarity = similarity / self.temperature

        return similarity
    
    
    def forward(self,
                raw_memory: torch.Tensor,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                time_encoding: torch.Tensor,
                prototypes: torch.Tensor,
                node_mask: Optional[torch.Tensor]=None
            )->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform SAM update for a batch of nodes
        returns:
            - update_memory: [B, memory_dim] s_u(t)
            - attention_weights: [B, num_prototypes] a_u^k(t)
            - update_gate: [B, 1] B_u(t)
            - candidate_memory: [B, memory_dim] \tilde(s)_u(t)
        """
        batch_size = raw_memory.size(0)

        # Step 1: form query q_u(t)
        # combine raw memory edge features, and time encoding
        query_inputs = torch.cat([
            raw_memory,
            edge_features,
            time_encoding
        ], dim=-1) # []

        query = self.query_proj(query_inputs) #[B, memory_dim]
        query = self.layer_norm(query)
        query = self.dropout(query)
        
        # step 2: compute prototype attention \alpha_u^k(t)
        similarity = self.compute_similarity(query, prototypes) # [B, num_prototypes]
        
        # apply mask if provided (for padded nodes)
        if node_mask is not None:
            similarity = similarity.masked_fill(~node_mask.unsqueeze(-1), float('-inf'))

        attention_weights = F.softmax(similarity, dim=-1) #[B, num_prototypes]
        
        # step 3: Form candidate memory \tilde(s)_u(t)
        candidate_memory = torch.matmul(
            attention_weights.unsqueeze(1), # [B, 1, num_prototypes]
            prototypes.unsqueeze(0) #[1, num_prototypes, memory_dim]
        ).squeeze(1) # [B, memory_dim]
        
        
        # step 4: compute update date \beta_u(t)
        gate_inputs = torch.cat([
            raw_memory,
            candidate_memory,
            time_encoding
        ], dim=-1)

        
        gate_logits = self.gate_proj(gate_inputs) #[B, 1]
        update_gate = torch.sigmoid(gate_logits) #[B, 1] \beta_u(t)
        
        # step 5: final memory update s_u(t)
        updated_memory = (1 - update_gate)*raw_memory + update_gate*candidate_memory
        updated_memory = self.layer_norm(updated_memory)
        
        # collect attention info for analysis
        attention_info = {
            "attention_weights": attention_weights,
            "update_gate": update_gate,
            "candidate_memory": candidate_memory,
            "query": query,
            "similarity_scores": similarity
        }

        return updated_memory, attention_info


class StabilityAugmentedMemory(nn.Module):
    """
    StabilityAugmented Memory (SAM) Module
    
    Mananges prototype-based memory for all nodes in the graph.
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
            device: str = "cuda"

    ):
        super(StabilityAugmentedMemory, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_dim = time_dim
        self.num_prototypes = num_prototypes
        self.device = device

        # Time encoder (fixed, non-learnable)
        self.time_encoder = TimeEncoder(time_dim)

        # Optional node feature projection (if node feature exists)
        if node_feat_dim>0:
            self.node_proj = nn.Linear(node_feat_dim, memory_dim)
            nn.init.xavier_uniform_(self.node_proj.weight)
            nn.init.zeros_(self.node_proj.bias)
        else:
            self.node_proj = None
        
        
        # Edge feature projection
        self.edge_proj = nn.Linear(edge_feat_dim, memory_dim)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)

        # SAM Cell for updates
        self.sam_cell = SAMCell(
            memory_dim = memory_dim,
            node_feat_dim = node_feat_dim,
            edge_feat_dim = edge_feat_dim,
            time_dim = time_dim,
            num_prototypes = num_prototypes,
            similarity_metric = similarity_metric,
            dropout = dropout
        )
        
        
        
        # initialize prototype memories for each node
        self.prototype_memories = nn.ModuleDict()
        for node_id in range(num_nodes):
            self.prototype_memories[str(node_id)] = PrototypeMemory(
                num_prototypes = num_prototypes,
                prototype_dim = memory_dim,
                node_id = node_id,
                initialization = prototype_init
            )
        
        # raw memory states (current m_u(t) for each node)
        self.register_buffer(
            "raw_memory",
            torch.zeros(num_nodes, memory_dim)
        )
        
        # last update time for each node
        self.register_buffer(
            "last_update",
            torch.zeros(num_nodes)
        )
        
        # move to device
        self.to(device)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Get raw memory for specified nodes.
        
        Args:
            node_ids: [batch_size] node indices
            
        Returns:
            [batch_size, memory_dim] raw memory states
        """
        return self.raw_memory[node_ids]
    
    
    def get_prototypes(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Get prototype vectors for specified nodes.
        
        Args:
            node_ids: [batch_size] node indices
            
        Returns:
            [batch_size, num_prototypes, memory_dim] prototype vectors
        """
        # prototypes = []
        # for node_id in node_ids:
        #     node_id_str = str(node_id.item())
        #     if node_id_str in self.prototype_memories:
        #         prototypes.append(self.prototype_memories[node_id_str]())
        #     else:
        #         # Handle out-of-range nodes (should not happen in normal operation)
        #         prototypes.append(torch.zeros(self.num_prototypes, self.memory_dim, device=self.device))
        
        # return torch.stack(prototypes)  # [batch_size, num_prototypes, memory_dim]
        device = node_ids.device  # Use input device, not self.device
        prototypes = []
        for node_id in node_ids:
            node_id_str = str(node_id.item())
            if node_id_str in self.prototype_memories:
                proto = self.prototype_memories[node_id_str]()
                prototypes.append(proto.to(device))  # Ensure correct device
            else:
                prototypes.append(
                    torch.zeros(self.num_prototypes, self.memory_dim, device=device)
                )
        return torch.stack(prototypes)
    
    
    
    def update_memory_batch(
            self,
            source_nodes: torch.Tensor,
            target_nodes: torch.Tensor,
            edge_features: torch.Tensor,
            current_time: torch.Tensor,
            node_features: Optional[torch.Tensor] = None
    )->Dict[str, torch.Tensor]:
        """
        Update memory for a batch of interactions.

        Returns:
            Dictionary with updated memories and attention info        

        """
        batch_size = source_nodes.size(0)
        
        # get raw memories for source and target
        src_raw_memory = self.raw_memory[source_nodes]
        tgt_raw_memory = self.raw_memory[target_nodes]        


        # get prototypes
        src_prototypes = self.get_prototypes(source_nodes)
        tgt_prototypes = self.get_prototypes(target_nodes)
        
        # Project edge features
        edge_features_proj = self.edge_proj(edge_features)

        # get time encoding
        time_encoding = self.time_encoder(current_time)
        
        # Optional node features
        if self.node_proj is not None and node_features is not None:
            src_node_feats = node_features[source_nodes]  # [batch_size, node_feat_dim]
            tgt_node_feats = node_features[target_nodes]  # [batch_size, node_feat_dim]
            src_node_feats_proj = self.node_proj(src_node_feats)  # [batch_size, memory_dim]
            tgt_node_feats_proj = self.node_proj(tgt_node_feats)  # [batch_size, memory_dim]
        else:
            src_node_feats_proj = None
            tgt_node_feats_proj = None
        
        # Update source nodes
        src_updated, src_attention = self.sam_cell(
            raw_memory=src_raw_memory,
            node_features=src_node_feats_proj,
            edge_features=edge_features_proj,
            time_encoding=time_encoding,
            prototypes=src_prototypes
        )
        
        # Update target nodes
        tgt_updated, tgt_attention = self.sam_cell(
            raw_memory=tgt_raw_memory,
            node_features=tgt_node_feats_proj,
            edge_features=edge_features_proj,
            time_encoding=time_encoding,
            prototypes=tgt_prototypes
        )

        # Update stored memories (in-place)
        self.raw_memory[source_nodes] = src_updated
        self.raw_memory[target_nodes] = tgt_updated
        self.last_update[source_nodes] = current_time
        self.last_update[target_nodes] = current_time
        
        # Return attention info for analysis
        return {
            'source_attention': src_attention,
            'target_attention': tgt_attention,
            'source_memory': src_updated,
            'target_memory': tgt_updated
        }   
        
        
    
    def get_stabilized_memory(
            self,
            node_ids: torch.Tensor,
            current_time: torch.Tensor,
            edge_features: Optional[torch.Tensor] = None,
            node_features: Optional[torch.Tensor] = None
    )->torch.Tensor:
        """
        Get stabilized memory for nodes without performing a full update.
        Useful for inference when no new interaction occurs.
        this computes s_u(t) using the current memory and prototypes,
        but does not store the result back to raw_memory.


        """
        batch_size = node_ids.size(0)
        
        # get raw memories
        raw_memory = self.raw_memory[node_ids]
        
        # get prototypes
        prototypes = self.get_prototypes(node_ids)
        
        # get time encoding
        time_encoding = self.time_encoder(current_time)
        
        # Edge features (zeros if not provided)
        if edge_features is None:
            edge_features = torch.zeros(batch_size, self.edge_feat_dim, device=self.device)
        edge_features_proj = self.edge_proj(edge_features)
        
        # node features (optional)
        if self.node_proj is not None and node_features is not None:
            node_feats_proj = self.node_proj(node_features[node_ids])
        else:
            node_feats_proj = None
        
        stabilized, attention = self.sam_cell(
            raw_memory = raw_memory,
            node_features = node_feats_proj,
            edge_features = edge_features_proj,
            time_encoding = time_encoding,
            prototypes = prototypes
        )

        return stabilized


    def reset_memory(self, node_ids:Optional[torch.Tensor]=None):
        """
        Reset memory for specified nodes (or all nodes if None)
        """
        if node_ids is None:
            self.raw_memory.zero_()
            self.last_update.zero_()
        else:
            self.raw_memory[node_ids] = 0
            self.last_update[node_ids] =0
    
    def get_attention_stats(self)->Dict[str, float]:
        """
        Get statistics about attention patterns (for debugging/analysis)
        # this would require storing attention over many updates
        # placeholder for now
        """
        return {
            "mean_attention_entropy": 0.0,
            "mean_update_gate": 0.0
        }

    

        


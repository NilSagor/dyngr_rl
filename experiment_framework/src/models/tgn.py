"""Temporal Graph Networks (TGN) implementation."""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


from .base_model import BaseDynamicGNN
from .tgn_module.temporal_attention import TemporalAttentionLayer, MergeLayer
from .tgn_module.memory import Memory
from .tgn_module.memory_updater import get_memory_updater
from .tgn_module.message_aggregator import get_message_aggregator
from .tgn_module.message_function import get_message_function
from .tgn_module.embedding_module import get_embedding_module


def scatter_mean(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = index.max().item() + 1
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.scatter_add_(dim, index, torch.ones_like(index, dtype=src.dtype))
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
    return out / count.unsqueeze(1).clamp(min=1)



class TGN(BaseDynamicGNN):
    """Temporal Graph Networks for Deep Learning on Dynamic Graphs.
    
    Reference: Rossi et al. (2020) - TGN: Temporal Graph Networks
    
    Paper: https://arxiv.org/abs/2006.10637
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_features: int,
        hidden_dim: int,
        time_encoding_dim: int = 32,
        memory_dim: int = 172,
        message_dim: int = 172,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_heads: int = 2,
        n_neighbors: int = 20,
        use_memory: bool = True,
        memory_update_at_start: bool = True,
        embedding_module_type: str = "graph_attention",
        message_function_type: str = "mlp",
        aggregator_type : str = "last", 
        memory_updater_type: str = "gru",
        neighbor_finder: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=hidden_dim,
            time_encoding_dim=time_encoding_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
        
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.n_heads = n_heads
        self.n_neighbors = n_neighbors
        self.use_memory = use_memory
        self.memory_update_at_start = memory_update_at_start
        self.embedding_module_type = embedding_module_type
        self.message_function_type = message_function_type
        self.aggregator_type = aggregator_type
        self.memory_updater_type = memory_updater_type
        self.neighbor_finder = neighbor_finder

        self.memory = torch.zeros(num_nodes, memory_dim)
        self.last_update = torch.zeros(num_nodes)
        
        # Store raw features as tensors
        self.register_buffer("node_raw_features", torch.zeros(num_nodes, node_features))
        self.register_buffer("edge_raw_features", torch.zeros(1, 1))  # Dummy
        
        # Time shift normalization
        self.register_buffer("mean_time_shift_src", torch.tensor(0.0))
        self.register_buffer("std_time_shift_src", torch.tensor(1.0))
        self.register_buffer("mean_time_shift_dst", torch.tensor(0.0))
        self.register_buffer("std_time_shift_dst", torch.tensor(1.0))

        # Initialize modules
        self._init_modules()

        # Link predictor
        self.link_predictor = MergeLayer(
            dim1=hidden_dim,
            dim2=hidden_dim,
            dim3=hidden_dim,
            dim4=1
        )

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _init_modules(self):
        """Initialize TGN modules"""
        device = self.device

        
         # Memory module
        if self.use_memory:
            self.memory = Memory(
                n_nodes=self.num_nodes,
                memory_dimension=self.memory_dim,
                input_dimension=self.message_dim + self.time_encoding_dim,
                message_dimension=self.message_dim,
                device=device
            )
        else:
            self.memory = None
        
        
        # Message function
        raw_message_dim = self.node_features * 2 + self.memory_dim * 2 + self.time_encoding_dim + 1
        self.message_fn = get_message_function(
            module_type=self.message_function_type,
            raw_message_dimension=raw_message_dim,
            message_dimension=self.message_dim
        )
        
        # Message aggregator
        self.message_aggregator = get_message_aggregator(
            aggregator_type=self.aggregator_type,
            device=device
        )
        
        # Memory updater
        if self.use_memory:
            self.memory_updater = get_memory_updater(
                module_type=self.memory_updater_type,
                memory=self.memory,
                message_dim=self.message_dim,
                memory_dim=self.memory_dim,
                device=device
            )
        else:
            self.memory_updater = None
        
        # Embedding module (requires neighbor_finder)
        if self.neighbor_finder is not None:
            self.embedding_module = get_embedding_module(
                module_type=self.embedding_module_type,
                node_features=self.node_raw_features,
                edge_features=self.edge_raw_features,
                memory=self.memory,
                neighbor_finder=self.neighbor_finder,
                time_encoder=self.time_encoder,
                n_layers=self.num_layers,
                n_node_features=self.hidden_dim,
                n_edge_features=1,  # Default edge feature dimension
                n_time_features=self.time_encoding_dim,
                embedding_dimension=self.hidden_dim,
                device=device,
                n_heads=self.n_heads,
                dropout=self.dropout,
                n_neighbors=self.n_neighbors,
                use_memory=self.use_memory
            )
        else:
            self.embedding_module = None

    def set_raw_features(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor):
        """Set raw features from dataset."""
        self.node_raw_features = node_raw_features.to(self.device)
        self.edge_raw_features = edge_raw_features.to(self.device)
        
        # Update embedding module if it exists
        if hasattr(self, 'embedding_module') and self.embedding_module is not None:
            self.embedding_module.node_features = self.node_raw_features
            self.embedding_module.edge_features = self.edge_raw_features

    def set_neighbor_finder(self, neighbor_finder):
        """Set neighbor finder after initialization."""
        self.neighbor_finder = neighbor_finder
        self._init_modules()  # Reinitialize modules with neighbor finder
       
        
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of TGN.
        
        Args:
            batch: Dictionary containing:
                - src_nodes: Source node indices [batch_size]
                - dst_nodes: Destination node indices [batch_size]
                - timestamps: Interaction timestamps [batch_size]
                - edge_indices: Edge indices for neighborhood aggregation
                - edge_times: Timestamps for neighborhood edges
                - edge_features: Optional edge features
                
        Returns:
            Link prediction logits [batch_size]
        """
        # get source and destination nodes
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']

        # Generate negative samples (random)
        negative_nodes = self._generate_negative_samples(
            src_nodes.cpu().numpy(),
            dst_nodes.cpu().numpy(),
        )

        src_nodes_np = src_nodes.cpu().numpy()
        dst_nodes_np = dst_nodes.cpu().numpy()
        timestamps_np = timestamps.cpu().numpy()

        # compute embeddings 
        source_embedding, destination_embedding, negative_embedding = self.compute_temporal_embeddings(
            source_nodes = src_nodes_np,
            destination_nodes = dst_nodes_np,
            negative_nodes = negative_nodes,
            edge_times = timestamps_np,
            edge_idxs = np.arange(len(src_nodes)), # dummy edge indice
            n_neighbors = self.n_neighbors
         )
        
        # Link Prediction
        pos_scores = self.link_predictor(
            source_embedding, source_embedding
            )        
        neg_scores = self.link_predictor(source_embedding, negative_embedding)

        # concatenate scores
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        
        return scores.squeeze()
        
    def compute_temporal_embeddings(self, source_nodes: np.ndarray, 
                                   destination_nodes: np.ndarray,
                                   negative_nodes: np.ndarray,
                                   edge_times: np.ndarray,
                                   n_neighbors: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute temporal embeddings for sources, destinations, and negatives."""
        n_samples = len(source_nodes)
        
        # Combine all nodes
        all_nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        all_timestamps = np.concatenate([edge_times, edge_times, edge_times])
        
        # Get memory if available
        memory = None
        time_diffs = None
        
        if self.use_memory and self.memory is not None:
            # Update memory at start if configured
            if self.memory_update_at_start:
                # Process any pending messages
                self._update_memory_from_messages()
                
            # Get memory for all nodes
            memory = self.memory.get_memory(torch.from_numpy(all_nodes).long().to(self.device))
            
            # Compute time differences
            last_update = self.memory.get_last_update(torch.from_numpy(all_nodes).long().to(self.device))
            edge_times_tensor = torch.from_numpy(all_timestamps).float().to(self.device)
            time_diffs = edge_times_tensor - last_update
            time_diffs = time_diffs.unsqueeze(-1)  # Add feature dimension

        # Compute embeddings
        if self.embedding_module is not None:
            node_embeddings = self.embedding_module.compute_embedding(
                memory=memory,
                source_nodes=all_nodes,
                timestamps=all_timestamps,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors,
                time_diffs=time_diffs
            )
        else:
            # Fallback to simple embedding
            if memory is not None:
                node_embeddings = memory
            elif self.node_embedding is not None:
                node_embeddings = self.node_embedding(torch.from_numpy(all_nodes).long().to(self.device))
            else:
                node_embeddings = torch.zeros(len(all_nodes), self.hidden_dim, device=self.device)
        
        # Split embeddings
        source_emb = node_embeddings[:n_samples]
        destination_emb = node_embeddings[n_samples:2*n_samples]
        negative_emb = node_embeddings[2*n_samples:]
        
        return source_emb, destination_emb, negative_emb
             
    def _update_memory_from_messages(self):
        """Update memory from stored messages."""
        if self.use_memory and self.memory is not None and self.memory_updater is not None:
            # Get unique nodes with messages
            node_ids = []
            messages = []
            timestamps = []
            
            for node_id, node_messages in self.memory.messages.items():
                if len(node_messages) > 0:
                    node_ids.append(node_id)
                    # Get the latest message
                    latest_msg = node_messages[-1]
                    messages.append(latest_msg[0])  # message tensor
                    timestamps.append(latest_msg[1])  # timestamp
            
            if node_ids:
                node_ids_tensor = torch.tensor(node_ids, device=self.device)
                messages_tensor = torch.stack(messages)
                timestamps_tensor = torch.tensor(timestamps, device=self.device)
                
                # Update memory
                self.memory_updater.update_memory(node_ids_tensor, messages_tensor, timestamps_tensor)
                
                # Clear processed messages
                self.memory.clear_messages(node_ids)
    
    def _generate_negative_samples(self, src_nodes: np.ndarray, 
                                  dst_nodes: np.ndarray) -> np.ndarray:
        """Generate negative samples for link prediction."""
        n_samples = len(src_nodes)
        negative_nodes = np.random.choice(self.num_nodes, size=n_samples, replace=True)
        
        # Ensure negative samples are not the same as destination nodes
        for i in range(n_samples):
            if negative_nodes[i] == dst_nodes[i]:
                negative_nodes[i] = (negative_nodes[i] + 1) % self.num_nodes
                
        return negative_nodes  
    
    
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute binary cross-entropy loss for link prediction."""
        logits = self.forward(batch)

        batch_size = len(batch['src_nodes'])

        labels = torch.cat([
            torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device),
        ])

        # compute loss
        loss = self.loss_fn(logits, labels)

        # Process messages and update memory after forward pass
        self._process_messages_for_batch(batch)
        
        # Update memory after forward pass
        if self.use_memory:
            self._process_messages_for_batch(batch)
        
        
        return loss
    
    # before _update_memory
    def _process_messages_for_batch(self, batch: Dict[str, torch.Tensor]):
        
        """Update node memories after interaction."""
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']

               
        # Get node features
        if self.node_features > 0:
            src_features = self.node_embedding(src_nodes) 
            dst_features = self.node_embedding(dst_nodes)
        else:
            src_features = None
            dst_features = None 
            
        
        # Get memories
        src_memory = self.memory_module.get_memory(src_nodes)
        dst_memory = self.memory_module.get_memory(dst_nodes)
        
        # Time encoding
        time_enc = self.time_encoder(timestamps)
        
        
        # Get edge features if available
        edge_features = None
        if 'edge_features' in batch:
            edge_features = batch['edge_features']
            # Expand edge features for message function
            edge_features_expanded = edge_features.unsqueeze(1).repeat(1,2,1)
            edge_features_expanded = edge_features_expanded.view(-1, edge_features.size(-1))
        else:
            edge_features_expanded = torch.zeros(len(src_nodes)*2, 1).to(self.device)

        
                
        # compute Messages for source and destination nodes/both directions
        src_messages = self.message_fn(
            src_features, dst_features, src_memory, dst_memory, time_enc
        )
        dst_messages = self.message_fn(
            dst_features, src_features, dst_memory, src_memory, time_enc
        )
        
        # # Update memories
        # self.memory_module.update_memory(src_nodes, src_messages, timestamps)
        # self.memory_module.update_memory(dst_nodes, dst_messages, timestamps)
       
        
        # combine messages
        all_messages = torch.cat([src_messages, dst_messages], dim=0)
        all_nodes = torch.cat([src_nodes, dst_nodes], dim=0)
        all_timestamps = torch.cat([timestamps, timestamps], dim=0)

        # Update memories
        self.memory_module.update_memory(all_nodes, all_messages, all_timestamps)
    
         
    def _compute_ap(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Average Precision (simplified)."""
        # Sort by probability
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels[sorted_indices]
        
        # Compute precision at each position
        cumulative_positives = torch.cumsum(sorted_labels, dim=0)
        cumulative_predictions = torch.arange(1, len(labels) + 1, device=labels.device)
        
        precisions = cumulative_positives.float() / cumulative_predictions.float()
        
        # Average precision
        ap = precisions.mean()
        return ap
    
    def _compute_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        logits = self.forward(batch)
        labels = batch['labels'].float()
        
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Compute metrics
        predictions = (probs > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        
        # Average Precision (simplified)
        ap = self._compute_ap(probs, labels)

        # compute AUC using sklearn
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels.cpu().numpy(), probs.detach().cpu().numpy())
            auc_tensor = torch.tensor(auc, device=self.device)
        except:
            auc_tensor = torch.tensor(0.0, device=self.device)
        
        loss = self.loss_fn(logits, labels)

        return {
            'accuracy': accuracy,
            'ap': ap,
            'auc': auc_tensor,
            'loss': loss
        }



    def on_train_epoch_start(self):
        """Reset memory at start of train epoch."""
        super().on_train_epoch_start()
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()

    def on_validation_epoch_start(self):
        """Reset memory at start of validation epoch."""
        super().on_validation_epoch_start()
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()

    def on_test_epoch_start(self):
        """Reset memory at start of test epoch."""
        super().on_test_epoch_start()
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()




# class LinkPredictor(nn.Module):
#     """Link prediction head."""
#     def __init__(self, hidden_dim, dropout):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#     def forward(self, src_embeddings, dst_embeddings):
#         combined = torch.cat([src_embeddings, dst_embeddings], dim=-1)
#         logits = self.mlp(combined).squeeze(-1)
#         return logits
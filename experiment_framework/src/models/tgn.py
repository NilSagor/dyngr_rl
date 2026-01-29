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
        aggregator_type: str = "last", 
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

        # Store raw features as buffers
        self.register_buffer("node_raw_features", torch.zeros(num_nodes, node_features))
        self.register_buffer("edge_raw_features", torch.zeros(1, 1))  # Will be updated
        
        # Time shift normalization (not used in basic TGN)
        self.register_buffer("mean_time_shift_src", torch.tensor(0.0))
        self.register_buffer("std_time_shift_src", torch.tensor(1.0))
        self.register_buffer("mean_time_shift_dst", torch.tensor(0.0))
        self.register_buffer("std_time_shift_dst", torch.tensor(1.0))

        # Initialize modules
        self._init_modules()

        # Ensure memory is on correct device
        if self.use_memory and self.memory is not None:
            self.memory = self.memory.to(self.device)

        # Link predictor
        self.link_predictor = MergeLayer(
            dim1=hidden_dim,
            dim2=hidden_dim,
            dim3=hidden_dim,
            dim4=1
        )

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Track messages separately from training
        self.pending_messages = []
        self._in_training_step = False

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
        
        # Embedding module
        if self.neighbor_finder is not None:
            self.embedding_module = get_embedding_module(
                module_type=self.embedding_module_type,
                node_features=self.node_raw_features,
                edge_features=self.edge_raw_features,
                memory=self.memory,
                neighbor_finder=self.neighbor_finder,
                time_encoder=self.time_encoder,
                n_layers=self.num_layers,
                n_node_features=self.node_features,
                n_edge_features=1,
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
        
        # Message function will be initialized after raw features are set
        self.message_fn = None

    def set_raw_features(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor):
        """Set raw features from dataset."""
        self.node_raw_features = node_raw_features.to(self.device)
        self.edge_raw_features = edge_raw_features.to(self.device)
        
        print(f"Node features shape: {self.node_raw_features.shape}")
        print(f"Edge features shape: {self.edge_raw_features.shape}")

        # Update embedding module if it exists
        if hasattr(self, 'embedding_module') and self.embedding_module is not None:
            self.embedding_module.node_features = self.node_raw_features
            self.embedding_module.edge_features = self.edge_raw_features
        
        # create message function after feature are known
        self._init_message_function()

    def _init_message_function(self):
        """Initialize message function with correct dimensions."""
        # Calculate actual message dimension
        # Calculate actual message dimension
        if self.node_raw_features is not None and len(self.node_raw_features.shape) > 1:
            node_feat_dim = self.node_raw_features.shape[1]
        else:
            node_feat_dim = self.node_features
            
        raw_message_dim = node_feat_dim * 2 + self.memory_dim * 2 + self.time_encoding_dim
        
        # Ensure raw_message_dim is positive
        if raw_message_dim <= 0:
            raw_message_dim = self.message_dim


        self.message_fn = get_message_function(
            module_type=self.message_function_type,
            raw_message_dimension=raw_message_dim,
            message_dimension=self.message_dim
        )
        self.message_fn = self.message_fn.to(self.device)

        # self.add_module("message_fn", message_fn)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of TGN."""
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']

        # Generate negative samples
        negative_nodes = self._generate_negative_samples(
            src_nodes.cpu().numpy(),
            dst_nodes.cpu().numpy(),
        )

        # Compute embeddings
        source_embedding, destination_embedding, negative_embedding = self.compute_temporal_embeddings(
            source_nodes=src_nodes.cpu().numpy(),
            destination_nodes=dst_nodes.cpu().numpy(),
            negative_nodes=negative_nodes,
            edge_times=timestamps.cpu().numpy(),
            n_neighbors=self.n_neighbors
        )
        
        # Link prediction
        pos_scores = self.link_predictor(source_embedding, destination_embedding)
        neg_scores = self.link_predictor(source_embedding, negative_embedding)

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
        all_nodes_tensor = torch.from_numpy(all_nodes).long().to(self.device)
        
        # Get memory if available
        memory = None
        time_diffs = None
        
        if self.use_memory and self.memory is not None:
            if self.memory_update_at_start:
                # self._update_memory_from_messages()
                self._update_memory_safely()
                
            memory = self.memory.get_memory(all_nodes_tensor)
            
            # Compute time differences
            last_update = self.memory.get_last_update(all_nodes_tensor)
            edge_times_tensor = torch.from_numpy(all_timestamps).float().to(self.device)
            time_diffs = edge_times_tensor - last_update
            time_diffs = time_diffs.unsqueeze(-1)

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
            # Fallback: use memory or node features
            if memory is not None:
                node_embeddings = memory
            elif hasattr(self, 'node_raw_features') and self.node_raw_features is not None:
                node_embeddings = self.node_raw_features[all_nodes_tensor]
            else:
                node_embeddings = torch.zeros(len(all_nodes), self.hidden_dim, device=self.device)
        
        # Split embeddings
        source_emb = node_embeddings[:n_samples]
        destination_emb = node_embeddings[n_samples:2*n_samples]
        negative_emb = node_embeddings[2*n_samples:]
        
        return source_emb, destination_emb, negative_emb
             
    # def _update_memory_from_messages(self):
    #     """Update memory from stored messages."""
    #     if self.use_memory and self.memory is not None and self.memory_updater is not None:
    #         node_ids = []
    #         messages = []
    #         timestamps = []
            
    #         for node_id, node_messages in self.memory.messages.items():
    #             if len(node_messages) > 0:
    #                 node_ids.append(node_id)
    #                 latest_msg = node_messages[-1]
    #                 messages.append(latest_msg[0])
    #                 timestamps.append(latest_msg[1])
            
    #         if node_ids:
    #             node_ids_tensor = torch.tensor(node_ids, device=self.device)
    #             messages_tensor = torch.stack(messages)
    #             timestamps_tensor = torch.tensor(timestamps, device=self.device)
                
    #             self.memory_updater.update_memory(node_ids_tensor, messages_tensor, timestamps_tensor)
    #             self.memory.clear_messages(node_ids)
    
    def _update_memory_safely(self):
        """Update memory safely without breaking gradient computation."""
        if self.use_memory and self.memory is not None and self.memory_updater is not None:
            with torch.no_grad():  # Detach from gradient computation
                self._process_pending_messages()
    
    def _process_pending_messages(self):
        """Process pending messages to update memory."""
        if not self.pending_messages:
            return
            
        # Group messages by timestamp to process in order
        self.pending_messages.sort(key=lambda x: x[2].min().item())
        
        for nodes, messages, timestamps in self.pending_messages:
            # Detach messages from computation graph
            nodes_detached = nodes.detach()
            messages_detached = messages.detach()
            timestamps_detached = timestamps.detach()
            
            # Update memory
            if self.memory_updater is not None:
                # Use the updater to update memory
                self.memory_updater.update_memory(
                    nodes_detached, 
                    messages_detached, 
                    timestamps_detached
                )
        
        # Clear pending messages
        self.pending_messages = []

    
    def _generate_negative_samples(self, src_nodes: np.ndarray, 
                                  dst_nodes: np.ndarray) -> np.ndarray:
        """Generate negative samples for link prediction."""
        n_samples = len(src_nodes)
        negative_nodes = np.random.choice(self.num_nodes, size=n_samples, replace=True)
        
        for i in range(n_samples):
            if negative_nodes[i] == dst_nodes[i]:
                negative_nodes[i] = (negative_nodes[i] + 1) % self.num_nodes
                
        return negative_nodes  
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._in_training_step = True
        try:
            # Forward pass
            logits = self.forward(batch)
            batch_size = len(batch['src_nodes'])
            
            # create labels
            labels = torch.cat([
                torch.ones(batch_size, device=self.device),
                torch.zeros(batch_size, device=self.device),
            ])

            # compute loss
            loss = self.loss_fn(logits, labels)
            
            # # Only store messages during training (for gradient safety)
            # if self.training:
            #     self._process_messages_for_batch(batch)
            # else:
            #     # Safe to update memory immediately during eval
            #     self._process_messages_for_batch(batch)
            #     # Update memory right away
            #     if hasattr(self, 'pending_messages'):
            #         for nodes, messages, timestamps in self.pending_messages:
            #             self.memory_updater.update_memory(nodes, messages, timestamps)
            #         self.pending_messages = []
            # prepare message for memory update (but not update yet) 
            self._process_messages_for_batch(batch)
            return loss
        finally:
            self._in_training_step = False
    
    
    def _process_messages_for_batch(self, batch: Dict[str, torch.Tensor]):
        """Update node memories after interaction."""
        if not self.use_memory or self.memory is None:
            return
        
        src_nodes = batch['src_nodes'].to(self.device)
        dst_nodes = batch['dst_nodes'].to(self.device)
        timestamps = batch['timestamps'].to(self.device)

        # Debug: Check node ID bounds
        max_node_id = self.node_raw_features.size(0) - 1  # 9228 for Wikipedia
        # assert src_nodes.max() <= max_node_id, f"Source node {src_nodes.max()} > {max_node_id}"
        # assert dst_nodes.max() <= max_node_id, f"Destination node {dst_nodes.max()} > {max_node_id}"
        # assert src_nodes.min() >= 1, f"Source node {src_nodes.min()} < 1"
        # assert dst_nodes.min() >= 1, f"Destination node {dst_nodes.min()} < 1"

        if src_nodes.min() < 1:
            print(f"WARNING: Invalid source node {src_nodes.min()}, filtering...")
            valid_mask = src_nodes >= 1
            if valid_mask.sum() == 0:
                return  # Skip invalid batch
            src_nodes = src_nodes[valid_mask]
            dst_nodes = dst_nodes[valid_mask]
            timestamps = timestamps[valid_mask]


        # Get node features
        if hasattr(self, 'node_raw_features') and self.node_raw_features is not None:
            src_features = self.node_raw_features[src_nodes]
            dst_features = self.node_raw_features[dst_nodes]
        else:
            src_features = torch.zeros(len(src_nodes), self.hidden_dim, device=self.device)
            dst_features = torch.zeros(len(dst_nodes), self.hidden_dim, device=self.device)
                 
        # Get memories
        if self.use_memory:
            src_memory = self.memory.get_memory(src_nodes)
            dst_memory = self.memory.get_memory(dst_nodes)
        else:
            src_memory = torch.zeros_like(src_features)
            dst_memory = torch.zeros_like(dst_features)
        
        # Time encoding - should be [batch_size, time_dim]
        # Time encoding
        time_enc = self.time_encoder(timestamps)
        
        # # Edge features (if available)
        # if 'edge_features' in batch:
        #     edge_features = batch['edge_features']
        #     edge_features_expanded = edge_features.unsqueeze(1).repeat(1, 2, 1).view(-1, edge_features.size(-1))
        # else:
        #     edge_features_expanded = torch.zeros(len(src_nodes) * 2, self.message_dim - (src_features.size(-1) * 2 + src_memory.size(-1) * 2 + time_enc.size(-1)), device=self.device)

        # Debug: Print shapes
        # print(f"src_features: {src_features.shape}")
        # print(f"dst_features: {dst_features.shape}")  
        # print(f"src_memory: {src_memory.shape}")
        # print(f"dst_memory: {dst_memory.shape}")
        # print(f"time_enc: {time_enc.shape}")
        
        
        
        # Create message inputs
        src_msg_input = torch.cat([src_features, dst_features, src_memory, dst_memory, time_enc], dim=-1)
        dst_msg_input = torch.cat([dst_features, src_features, dst_memory, src_memory, time_enc], dim=-1)
        
        # Generate messages
        src_messages = self.message_fn(src_msg_input)
        dst_messages = self.message_fn(dst_msg_input)
        
        # Combine messages
        all_messages = torch.cat([src_messages, dst_messages], dim=0)
        all_nodes = torch.cat([src_nodes, dst_nodes], dim=0)
        all_timestamps = torch.cat([timestamps, timestamps], dim=0)

        # # Update memories
        # if self.use_memory:
        #     self.memory_updater.update_memory(all_nodes, all_messages, all_timestamps)
        # Store in model state for later processing
        # if not hasattr(self, 'pending_messages'):
        #     self.pending_messages = []
        self.pending_messages.append((all_nodes, all_messages, all_timestamps))

    def _compute_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        logits = self.forward(batch)
        batch_size = len(batch['src_nodes'])
        labels = torch.cat([
            torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device),
        ]).float()
        
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        
        # Average Precision
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels[sorted_indices]
        cumulative_positives = torch.cumsum(sorted_labels, dim=0)
        cumulative_predictions = torch.arange(1, len(labels) + 1, device=labels.device, dtype=torch.float)
        precisions = cumulative_positives / cumulative_predictions
        ap = precisions.mean()
        
        # AUC
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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update memory after backward pass."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        with torch.no_grad():
            self._process_pending_messages()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Update memory after backward pass."""
        super().on_validation_batch_end(outputs, batch, batch_idx)
        with torch.no_grad():
            self._process_pending_messages()

    def on_test_batch_end(self, outputs, batch, batch_idx):
        """Update memory after backward pass."""
        super().on_test_batch_end(outputs, batch, batch_idx)
        with torch.no_grad():
            self._process_pending_messages()
    
    
    def on_train_epoch_start(self):
        """Reset memory at start of train epoch."""
        super().on_train_epoch_start()
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()
            self.pending_messages = []

    def on_validation_epoch_start(self):
        """Reset memory at start of validation epoch."""
        super().on_validation_epoch_start()
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()
            self.pending_messages = []

    def on_test_epoch_start(self):
        """Reset memory at start of test epoch."""
        super().on_test_epoch_start()
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()
            self.pending_messages = []

    


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
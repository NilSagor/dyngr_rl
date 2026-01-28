"""Temporal Graph Networks (TGN) implementation."""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch_scatter import scatter_mean, scatter_max

from .base_model import BaseDynamicGNN, MemoryModule, TimeEncoder

from .tgn_module.memory import Memory
from .tgn_module.memory_updater import get_memory_updater
from .tgn_module.message_aggregator import get_message_aggregator
from .tgn_module.message_function import get_message_function
from .tgn_module.memory_updater import get_memory_updater
from .tgn_module.embedding_module import get_embedding_module
from .tgn_module.temporal_attention import MergeLayer

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
        neighbor_sampler: Optional[nn.Module] = None,
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

        self.register_buffer("node_raw_features", None)
        self.register_buffer("edge_raw_features", None)
        
        # Memory module
        # self.memory_module = MemoryModule(
        #     num_nodes=num_nodes,
        #     memory_dim=memory_dim,
        #     message_dim=message_dim,
        #     time_encoding_dim=time_encoding_dim
        # )

        self.memory = Memory(
            n_nodes = num_nodes,
            memory_dim = memory_dim,
            input_dim = message_dim + time_encoding_dim,
            message_dimension= message_dim,
            device =self.device
        )
        self.memory_updater = get_memory_updater(

        )
        
        # Message function
        self.message_fn = MessageFunction(
            node_features=node_features,
            edge_features=0,  # Can be extended
            memory_dim=memory_dim,
            time_encoding_dim=time_encoding_dim,
            message_dim=message_dim
        )
        

        # Embedding module (Graph Neural Network) [use neighbor information from the batch]
        self.embedding_module = EmbeddingModule(
            node_features=node_features,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            time_encoding_dim=time_encoding_dim,
            num_layers=num_layers,
            n_heads = n_heads,
            dropout=dropout
        )
        
        # Link predictor
        self.link_predictor = LinkPredictor(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
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
            torch.cat([source_embedding, source_embedding], dim=0),
            torch.cat([destination_embedding, negative_embedding])
        ).squeeze(dim=0)

        return pos_scores
        
    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20):
        """Compute temporal embeddings for sources, destination, and negatives"""
        n_smaples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        time_diffs = None

        if self.use_memory and self.memory is not None:
            if self.memory_update_at_start:
                memory, last_update = self.get_update_memory(
                    list(range(self.num_nodes)),
                    self.memory.messages
                )
            else:
                memory = self.memory.get_memory(list(range(self.num_nodes)))
                last_update = self.memory.last_update

            # Compute time differences
            source_time_diffs = torch.LongTensor(edge_times).to(self.device) - \
                                last_update[source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            
            destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - \
                                    last_update[destination_nodes].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            
            negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - \
                                    last_update[negative_nodes].long()
            negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            
            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, 
                                    negative_time_diffs], dim=0)
        
        # Compute embeddings using the embedding module
        if self.embedding_module is not None:
            node_embedding = self.embedding_module.compute_embedding(
                memory=memory,
                source_nodes=nodes,
                timestamps=timestamps,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors,
                time_diffs=time_diffs
            )
        else:
            # Fallback to simple embedding
            if memory is not None:
                node_embedding = memory[nodes]
            else:
                node_embedding = self.node_embedding(
                    torch.from_numpy(nodes).long().to(self.device)
                )

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples:2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]
        
        # Update memory if needed
        if self.use_memory and self.memory_update_at_start and self.memory is not None:
            self.update_memory(positives, self.memory.messages)
            self.memory.clear_messages(positives)
        
        return source_node_embedding, destination_node_embedding, negative_node_embedding

        
        
        
        
        # # Get current memory for nodes
        # src_memory = self.memory_module.get_memory(src_nodes)
        # dst_memory = self.memory_module.get_memory(dst_nodes)
        
        # # Get node features
        # if self.node_features > 0:
        #     src_features = self.node_embedding(src_nodes)
        #     dst_features = self.node_embedding(dst_nodes)
        # else:
        #     src_features = None
        #     dst_features = None
            
        # # Time encoding
        # time_enc = self.time_encoder(timestamps)
        
        # # Compute embeddings with neighborhood aggregation
        # if 'edge_indices' in batch:
        #     # Temporal neighborhood aggregation
        #     src_embeddings = self.embedding_module(
        #         nodes=src_nodes,
        #         features=src_features,
        #         memory=src_memory,
        #         edge_indices=batch['edge_indices'],
        #         edge_times=batch['edge_times'],
        #         timestamps=timestamps
        #     )
        #     dst_embeddings = self.embedding_module(
        #         nodes=dst_nodes,
        #         features=dst_features,
        #         memory=dst_memory,
        #         edge_indices=batch['edge_indices'],
        #         edge_times=batch['edge_times'],
        #         timestamps=timestamps
        #     )
        # else:
        #     # Simple embedding without neighborhood aggregation
        #     src_embeddings = self.embedding_module.simple_embed(
        #         features=src_features,
        #         memory=src_memory,
        #         time_enc=time_enc
        #     )
        #     dst_embeddings = self.embedding_module.simple_embed(
        #         features=dst_features,
        #         memory=dst_memory,
        #         time_enc=time_enc
        #     )
            
        # # Link prediction
        # logits = self.link_predictor(src_embeddings, dst_embeddings)
        
        # return logits
        
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute binary cross-entropy loss for link prediction."""
        logits = self.forward(batch)
        labels = batch['labels'].float()
        
        # Update memory after forward pass
        if self.use_memory:
            self._update_memory(batch)
        
        return self.loss_fn(logits, labels)
        
    def _update_memory(self, batch: Dict[str, torch.Tensor]):
        
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
        
        return {
            'accuracy': accuracy,
            'ap': ap,
            'auc': auc_tensor,
            'loss': self.loss_fn(logits, labels)
        }
     
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


class MessageFunction(nn.Module):
    """Message function for computing interactions between nodes."""
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        memory_dim: int,
        time_encoding_dim: int,
        message_dim: int
    ):
        super().__init__()
        
        input_dim = (
            (node_features if node_features > 0 else memory_dim) * 2 +
            memory_dim * 2 +
            time_encoding_dim +
            edge_features
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(message_dim, message_dim)
        )
        
    def forward(
        self,
        src_features: Optional[torch.Tensor],
        dst_features: Optional[torch.Tensor],
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        time_enc: torch.Tensor
    ) -> torch.Tensor:
        """Compute message from source to destination."""
        
        # Use features if available, otherwise use memory
        if src_features is not None:
            src_input = src_features
        else:
            src_input = src_memory
            
        if dst_features is not None:
            dst_input = dst_features
        else:
            dst_input = dst_memory
            
        # Concatenate all inputs
        message_input = torch.cat([
            src_input,
            dst_input,
            src_memory,
            dst_memory,
            time_enc
        ], dim=-1)
        
        return self.mlp(message_input)


class EmbeddingModule(nn.Module):
    """Embedding module with temporal neighborhood aggregation."""
    
    def __init__(
        self,
        node_features: int,
        memory_dim: int,
        time_encoding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_features = node_features
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        input_dim = (node_features if node_features > 0 else memory_dim) + memory_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionLayer(
                hidden_dim=hidden_dim,
                time_encoding_dim=time_encoding_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        nodes: torch.Tensor,
        features: Optional[torch.Tensor],
        memory: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_times: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Compute embeddings with temporal neighborhood aggregation."""
        
        # Initial embedding
        if features is not None:
            node_input = torch.cat([features, memory], dim=-1)
        else:
            node_input = torch.cat([memory, memory], dim=-1)
            
        embeddings = self.input_proj(node_input)
        
        # Temporal attention over neighborhood
        for layer in self.temporal_layers:
            embeddings = layer(
                embeddings,
                edge_indices,
                edge_times,
                timestamps
            )
            
        return embeddings
        
    def simple_embed(
        self,
        features: Optional[torch.Tensor],
        memory: torch.Tensor,
        time_enc: torch.Tensor
    ) -> torch.Tensor:
        """Simple embedding without neighborhood aggregation."""
        
        if features is not None:
            node_input = torch.cat([features, memory, time_enc], dim=-1)
        else:
            node_input = torch.cat([memory, memory, time_enc], dim=-1)
            
        return self.input_proj(node_input)


class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer for neighborhood aggregation."""
    
    def __init__(
        self,
        hidden_dim: int,
        time_encoding_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim + time_encoding_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_times: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Apply temporal attention."""
        
        # For simplicity, we'll implement a basic version
        # In practice, this would involve more sophisticated temporal attention
        
        queries = self.query_proj(embeddings)
        keys = self.key_proj(embeddings)
        values = self.value_proj(embeddings)
        
        # Self-attention (simplified)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (embeddings.size(-1) ** 0.5)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, values)
        
        # Residual connection and layer norm
        output = self.layer_norm(embeddings + attended)
        
        return output


    

class LinkPredictor(nn.Module):
    """Link prediction head."""
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, src_embeddings, dst_embeddings):
        combined = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        logits = self.mlp(combined).squeeze(-1)
        return logits
"""Temporal Graph Networks (TGN) implementation."""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max

from .base_model import BaseDynamicGNN, MemoryModule, TimeEncoder


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
        
        # Memory module
        self.memory_module = MemoryModule(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            message_dim=message_dim,
            time_encoding_dim=time_encoding_dim
        )
        
        # Message function
        self.message_fn = MessageFunction(
            node_features=node_features,
            edge_features=0,  # Can be extended
            memory_dim=memory_dim,
            time_encoding_dim=time_encoding_dim,
            message_dim=message_dim
        )
        
        # Embedding module (Graph Neural Network)
        self.embedding_module = EmbeddingModule(
            node_features=node_features,
            memory_dim=memory_dim,
            time_encoding_dim=time_encoding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
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
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']
        
        # Get current memory for nodes
        src_memory = self.memory_module.get_memory(src_nodes)
        dst_memory = self.memory_module.get_memory(dst_nodes)
        
        # Get node features
        if self.node_features > 0:
            src_features = self.node_embedding(src_nodes)
            dst_features = self.node_embedding(dst_nodes)
        else:
            src_features = None
            dst_features = None
            
        # Time encoding
        time_enc = self.time_encoder(timestamps)
        
        # Compute embeddings with neighborhood aggregation
        if 'edge_indices' in batch:
            # Temporal neighborhood aggregation
            src_embeddings = self.embedding_module(
                nodes=src_nodes,
                features=src_features,
                memory=src_memory,
                edge_indices=batch['edge_indices'],
                edge_times=batch['edge_times'],
                timestamps=timestamps
            )
            dst_embeddings = self.embedding_module(
                nodes=dst_nodes,
                features=dst_features,
                memory=dst_memory,
                edge_indices=batch['edge_indices'],
                edge_times=batch['edge_times'],
                timestamps=timestamps
            )
        else:
            # Simple embedding without neighborhood aggregation
            src_embeddings = self.embedding_module.simple_embed(
                features=src_features,
                memory=src_memory,
                time_enc=time_enc
            )
            dst_embeddings = self.embedding_module.simple_embed(
                features=dst_features,
                memory=dst_memory,
                time_enc=time_enc
            )
            
        # Link prediction
        logits = self.link_predictor(src_embeddings, dst_embeddings)
        
        return logits
        
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute binary cross-entropy loss for link prediction."""
        logits = self.forward(batch)
        labels = batch['labels'].float()
        
        # Update memory after forward pass
        self._update_memory(batch)
        
        return self.loss_fn(logits, labels)
        
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
        
        return {
            'accuracy': accuracy,
            'ap': ap,
            'loss': self.loss_fn(logits, labels)
        }
        
    def _update_memory(self, batch: Dict[str, torch.Tensor]):
        """Update node memories after interaction."""
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']
        
        # Compute messages
        src_features = self.node_embedding(src_nodes) if self.node_features > 0 else None
        dst_features = self.node_embedding(dst_nodes) if self.node_features > 0 else None
        
        src_memory = self.memory_module.get_memory(src_nodes)
        dst_memory = self.memory_module.get_memory(dst_nodes)
        
        time_enc = self.time_encoder(timestamps)
        
        # Messages for source and destination nodes
        src_messages = self.message_fn(
            src_features, dst_features, src_memory, dst_memory, time_enc
        )
        dst_messages = self.message_fn(
            dst_features, src_features, dst_memory, src_memory, time_enc
        )
        
        # Update memories
        self.memory_module.update_memory(src_nodes, src_messages, timestamps)
        self.memory_module.update_memory(dst_nodes, dst_messages, timestamps)
        
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
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Predict link probability."""
        
        # Concatenate source and destination embeddings
        link_input = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        
        return self.mlp(link_input).squeeze(-1)
    

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
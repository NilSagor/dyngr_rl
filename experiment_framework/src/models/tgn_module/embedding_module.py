import torch
import torch.nn as nn
import numpy as np
import math
from .temporal_attention import TemporalAttentionLayer

class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.dropout = nn.Dropout(dropout)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None):
        return NotImplemented
    
    

class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None):
        return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout=0.1):
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder, time_encoder, 
                                          n_layers, n_node_features, n_edge_features, n_time_features,
                                          embedding_dimension, device, dropout)
        
        class NormalLinear(nn.Linear):
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)
        
        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None):
        time_diffs = torch.from_numpy(timestamps - self.memory.last_update[source_nodes].cpu().numpy()).float().to(self.device)
        return memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder, time_encoder, 
                                           n_layers, n_node_features, n_edge_features, n_time_features,
                                           embedding_dimension, device, dropout)
        self.use_memory = use_memory
        self.n_heads = n_heads
        self.use_memory = use_memory

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None):
        """Compute temporal embeddings using recursive neighbor sampling."""
        assert(n_layers >= 0)
        
        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.from_numpy(timestamps).float().to(self.device)
        
        # Base case: no layers, return raw features + memory
        if n_layers == 0:
            source_features = self.node_features[source_nodes_torch]
            if self.use_memory:
                source_features = torch.cat([source_features, memory[source_nodes_torch]], dim=-1)
            return source_features
        
        # Get neighbors
        neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
            source_nodes, timestamps, n_neighbors=n_neighbors)
        
        neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
        edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
        edge_times_torch = torch.from_numpy(edge_times).float().to(self.device)
        
        # Recursively compute neighbor embeddings
        neighbors_flat = neighbors.flatten()
        timestamps_repeated = np.repeat(timestamps, n_neighbors)
        neighbor_embeddings = self.compute_embedding(memory, neighbors_flat, timestamps_repeated, 
                                                  n_layers-1, n_neighbors)
        neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_neighbors, -1)
        
        # Get source embeddings
        source_embeddings = self.compute_embedding(memory, source_nodes, timestamps, n_layers-1, n_neighbors)
        
        # Edge features
        edge_features_batch = self.edge_features[edge_idxs_torch]
        
        # Time encoding
        source_time_enc = self.time_encoder(timestamps_torch)
        neighbor_time_enc = self.time_encoder(edge_times_torch)
        
        # Padding mask (0 is padding node)
        padding_mask = (neighbors_torch == 0)
        
        # Aggregate using attention or sum
        return self.aggregate(n_layers, source_embeddings, source_time_enc, 
                           neighbor_embeddings, neighbor_time_enc, edge_features_batch, padding_mask)

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphSumEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder, time_encoder, 
                                              n_layers, n_node_features, n_edge_features, n_time_features,
                                              embedding_dimension, device, n_heads, dropout, use_memory)
        
        input_dim = embedding_dimension + n_time_features + n_edge_features
        self.linear_1 = nn.ModuleList([
            nn.Linear(input_dim, embedding_dimension) for _ in range(n_layers)
        ])
        self.linear_2 = nn.ModuleList([
            nn.Linear(embedding_dimension + n_node_features + n_time_features, embedding_dimension) 
            for _ in range(n_layers)
        ])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        # Concatenate neighbor features
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features], dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbor_embeddings = torch.relu(neighbor_embeddings)
        
        # Apply mask
        neighbor_embeddings[mask] = 0
        
        # Sum neighbors
        neighbors_sum = torch.sum(neighbor_embeddings, dim=1)
        
        # Combine with source
        source_features = torch.cat([source_node_features, source_nodes_time_embedding], dim=1)
        combined = torch.cat([neighbors_sum, source_features], dim=1)
        output = self.linear_2[n_layer - 1](combined)
        
        return self.dropout(output)


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder, time_encoder, 
                                                    n_layers, n_node_features, n_edge_features, n_time_features,
                                                    embedding_dimension, device, n_heads, dropout, use_memory)
        
        # Project input features to embedding dimension
        input_dim = n_node_features + (memory.memory_dimension if use_memory else 0)
        self.feature_proj = nn.Linear(input_dim, embedding_dimension)
        
        # Attention layers
        self.attention_models = nn.ModuleList([
            TemporalAttentionLayer(
                n_node_features=embedding_dimension,
                n_neighbors_features=embedding_dimension,
                n_edge_features=n_edge_features,
                time_dim=n_time_features,
                output_dimension=embedding_dimension,
                n_head=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None):
        """Compute embeddings using temporal attention over neighbors."""
        device = next(self.parameters()).device
        
        
        source_nodes_torch = torch.from_numpy(source_nodes).long().to(device)
        timestamps_torch = torch.from_numpy(timestamps).float().to(device)
        
        
        # Ensure all features are on correct device
        if self.node_features.device != device:
            self.node_features = self.node_features.to(device)
        if self.edge_features.device != device:
            self.edge_features = self.edge_features.to(device)
        
        # DEBUG: Check node indices against memory size
        max_node_index = source_nodes_torch.max().item()
        
        # Check if memory is a Memory object or a tensor
        if hasattr(memory, 'memory'):  # It's a Memory object
            memory_size = memory.memory.size(0)
            # Get the full memory tensor from the Memory object
            full_memory = memory.memory
        else:  # It's a tensor
            memory_size = memory.size(0)
            full_memory = memory
        # memory_size = memory.size(0)
        
        if max_node_index >= memory_size:
            raise ValueError(
                f"Node index {max_node_index} is out of bounds for memory of size {memory_size}. "
                f"Expected max node index < {memory_size}. "
                f"This means TGN was initialized with wrong num_nodes parameter."
            )
        
        
        # CRITICAL: Ensure memory is on the same device as the model
        full_memory = full_memory.to(device)
        
        # Get initial features - ensure everything is on the same device
        source_features = self.node_features[source_nodes_torch].to(device)
        
        if self.use_memory:
            # Get memory for source nodes
            source_memory = full_memory[source_nodes_torch].to(device)
            source_features = torch.cat([source_features, source_memory], dim=-1)
        
        # Project to embedding dimension
        current_embeddings = self.feature_proj(source_features)
        
        # Multi-layer attention
        for layer_idx in range(n_layers):
            # Get neighbors
            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes, timestamps, n_neighbors=n_neighbors)
            
            # Convert to tensors and move to device
            neighbors_torch = torch.from_numpy(neighbors).long().to(device)
            edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(device)
            edge_times_torch = torch.from_numpy(edge_times).float().to(device)


            # Get neighbor features
            neighbor_features = self.node_features[neighbors_torch].to(device)
            
            if self.use_memory:
                # Get neighbor memory
                neighbor_memory = full_memory[neighbors_torch].to(device)
                neighbor_features = torch.cat([neighbor_features, neighbor_memory], dim=-1)
            
            # Project neighbor features
            neighbor_embeddings = self.feature_proj(neighbor_features)
            
            # Edge features
            edge_features_batch = self.edge_features[edge_idxs_torch]
            
            # Time encoding
            source_time_enc = self.time_encoder(timestamps_torch)
            neighbor_time_enc = self.time_encoder(edge_times_torch)
            
            # Padding mask
            padding_mask = (neighbors_torch == 0)
            
            # Apply attention
            current_embeddings, _ = self.attention_models[layer_idx](
                src_node_features=current_embeddings,
                src_time_features=source_time_enc.unsqueeze(1),
                neighbors_features=neighbor_embeddings,
                neighbors_time_features=neighbor_time_enc,
                edge_features=edge_features_batch,
                neighbors_padding_mask=padding_mask
            )
            
            current_embeddings = self.dropout(current_embeddings)
        
        return current_embeddings


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                        time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                        embedding_dimension, device, n_heads=2, dropout=0.1, n_neighbors=None, use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory
        )
    elif module_type == "graph_sum":
        return GraphSumEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory
        )
    elif module_type == "identity":
        return IdentityEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            dropout=dropout
        )
    elif module_type == "time":
        return TimeEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            dropout=dropout
        )
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
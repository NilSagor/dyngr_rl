import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseDynamicGNN, TimeEncoder

class DyGFormer(BaseDynamicGNN):
    def __init__(self, num_nodes, node_features, hidden_dim, time_encoding_dim=32, 
                 num_layers=2, num_heads=4, dropout=0.1, max_neighbors=20,
                 patch_size=1, max_sequence_length=256, channel_embedding_dim=64,
                 neighbor_co_occurrence=True, learning_rate=1e-4, weight_decay=1e-5, **kwargs):
        
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

        # DyGFormer specific parameters
        self.patch_size = patch_size
        self.max_sequence_length = max_sequence_length
        self.channel_embedding_dim = channel_embedding_dim
        self.max_neighbors = max_neighbors
        self.num_heads = num_heads
        self.neighbor_co_occurrence = neighbor_co_occurrence

        # Number of channels: node, edge, time, neighbor_co_occurrence
        self.num_channels = 4

        # Initialize raw features
        # In original DyGFormer, these are loaded from dataset
        
        # Initialize raw features - FIX: ensure valid dimensions
        feat_dim = max(1, node_features) if node_features > 0 else hidden_dim
        self.register_buffer('node_raw_features', torch.zeros(num_nodes + 1, feat_dim))
        self.register_buffer('edge_raw_features', torch.zeros(1, max(1, getattr(self, 'edge_feat_dim', 1))))  # Placeholder
        
        # FIX: Initialize node and edge feature dimensions properly
        self.node_feat_dim = feat_dim
        self.edge_feat_dim = 1  # Default edge feature dimension


        
        # Projection layers for each channel
        self.projection_layers = nn.ModuleDict({
            'node': nn.Linear(patch_size * feat_dim, channel_embedding_dim),
            'edge': nn.Linear(patch_size * self.edge_feat_dim, channel_embedding_dim),
            'time': nn.Linear(patch_size * time_encoding_dim, channel_embedding_dim),
            'neighbor_co_occurrence': nn.Linear(patch_size * channel_embedding_dim, 
                                               channel_embedding_dim)
        })

        # Neighbor co-occurrence encoder
        if neighbor_co_occurrence:
            self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
                neighbor_co_occurrence_feat_dim=channel_embedding_dim,
                # device=self.device
            )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(
                attention_dim=self.num_channels * channel_embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(
            self.num_channels * channel_embedding_dim,
            hidden_dim
        )

        # Link predictor
        self.link_predictor = LinkPredictor(hidden_dim, dropout)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def set_raw_features(self, node_raw_features, edge_raw_features):
        """Set raw features from dataset (should be called after initialization)"""
        self.node_raw_features = node_raw_features.to(self.device)
        self.edge_raw_features = edge_raw_features.to(self.device)
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]

        # FIX: Update projection layers with correct dimensions
        self.projection_layers['node'] = nn.Linear(
            self.patch_size * self.node_feat_dim, 
            self.channel_embedding_dim
        ).to(self.device)
        self.projection_layers['edge'] = nn.Linear(
            self.patch_size * self.edge_feat_dim, 
            self.channel_embedding_dim
        ).to(self.device)

    def forward(self, batch):
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
                if value.dtype == torch.float64:
                    batch[key] = value.float()

    # Get source and destination node embeddings with temporal information
        src_embeddings, dst_embeddings = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids=batch['src_nodes'],
            dst_node_ids=batch['dst_nodes'],
            node_interact_times=batch['timestamps'],
            src_neighbors=batch.get('src_neighbors'),
            dst_neighbors=batch.get('dst_neighbors'),
            src_neighbor_times=batch.get('src_neighbor_times'),
            dst_neighbor_times=batch.get('dst_neighbor_times')
        )

        # Link prediction
        logits = self.link_predictor(src_embeddings, dst_embeddings)
        return logits

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times,
                                                 src_neighbors=None, dst_neighbors=None,
                                                 src_neighbor_times=None, dst_neighbor_times=None):
        """
        Compute temporal embeddings for source and destination nodes
        """
        batch_size = len(src_node_ids)

        # Process source nodes
        src_padded_sequences, src_actual_lengths = self._prepare_sequences(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            neighbors=src_neighbors,
            neighbor_times=src_neighbor_times
        )

        # Process destination nodes
        dst_padded_sequences, dst_actual_lengths = self._prepare_sequences(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            neighbors=dst_neighbors,
            neighbor_times=dst_neighbor_times
        )

        # Get neighbor co-occurrence features if enabled
        if self.neighbor_co_occurrence:
            src_co_occurrence, dst_co_occurrence = self.neighbor_co_occurrence_encoder(
                src_padded_sequences['neighbor_ids'],
                dst_padded_sequences['neighbor_ids']
            )
        else:
            src_co_occurrence = torch.zeros(batch_size, self.max_sequence_length, 
                                           self.channel_embedding_dim, device=self.device)
            dst_co_occurrence = torch.zeros(batch_size, self.max_sequence_length,
                                           self.channel_embedding_dim, device=self.device)

        # Process through patches and channels
        src_embeddings = self._process_through_patches(
            padded_sequences=src_padded_sequences,
            co_occurrence_features=src_co_occurrence,
            actual_lengths=src_actual_lengths
        )

        dst_embeddings = self._process_through_patches(
            padded_sequences=dst_padded_sequences,
            co_occurrence_features=dst_co_occurrence,
            actual_lengths=dst_actual_lengths
        )

        return src_embeddings, dst_embeddings

    def _prepare_sequences(self, node_ids, node_interact_times, neighbors, neighbor_times):
        """Prepare padded sequences for nodes"""
        batch_size = len(node_ids)
        device = node_ids.device

        if neighbors is None:
            # If no neighbors provided, use empty sequences
            neighbors = torch.zeros(batch_size, self.max_neighbors, dtype=torch.long, device=device)
            neighbor_times = torch.zeros(batch_size, self.max_neighbors, dtype=torch.float, device=device)

        # Create padded sequences
        max_seq_length = min(self.max_sequence_length, self.max_neighbors + 1)
        padded_sequences = {
            'neighbor_ids': torch.zeros(batch_size, max_seq_length, dtype=torch.long, device=device),
            'node_features': torch.zeros(batch_size, max_seq_length, self.node_feat_dim, device=device),
            'time_features': torch.zeros(batch_size, max_seq_length, self.time_encoding_dim, device=device),
            'edge_features': torch.zeros(batch_size, max_seq_length, self.edge_feat_dim, device=device)
        }

        actual_lengths = []

        for i in range(batch_size):
            # Include the node itself
            seq_length = 1
            
            # Add neighbors if available
            if neighbors is not None:
                valid_mask = neighbors[i] > 0
                valid_neighbors = neighbors[i][valid_mask]
                valid_times = neighbor_times[i][valid_mask]
                num_neighbors = len(valid_neighbors)
                
                if num_neighbors > 0:
                    # Take most recent neighbors
                    seq_length += min(num_neighbors, max_seq_length - 1)
                    
                    # Fill neighbor ids
                    padded_sequences['neighbor_ids'][i, 1:seq_length] = valid_neighbors[:seq_length-1]
                    
                    # Calculate time differences
                    time_diffs = node_interact_times[i] - valid_times[:seq_length-1] # shape: [num_neighbors]

                    time_encoded = self.time_encoder(time_diffs) # add feature dim: [num_neighbors, time_dim]
                    
                    # Get time features
                    # padded_sequences['time_features'][i, 1:seq_length] = self.time_encoder(
                    #     time_diffs.unsqueeze(-1)
                    # ).squeeze(1)
                    padded_sequences['time_features'][i, 1:seq_length] = time_encoded
                    # padded_sequences['time_features'][i, 1:seq_length] = self.time_encoder(time_diffs.unsqueeze(1)).squeeze(-1)
            # Set the node itself
            padded_sequences['neighbor_ids'][i, 0] = node_ids[i]
            padded_sequences['time_features'][i, 0] = self.time_encoder(
                torch.zeros(1, device=device)
            ).squeeze(0)
            
            # Get edge features for the sequence
            # edge_ids_in_sequence = padded_sequences['edge_ids'][i, :seq_length]
            # padded_sequences['edge_features'][i, :seq_length] = self.edge_raw_features[edge_ids_in_sequence]

            # FIX: Handle potential invalid indices (skip padding node 0)
            node_ids_in_sequence = padded_sequences['neighbor_ids'][i, :seq_length]
            # Mask out padding nodes (node_id = 0)
            valid_node_mask = node_ids_in_sequence != 0
            valid_indices = node_ids_in_sequence[valid_node_mask]
            
            if len(valid_indices) > 0:
                padded_sequences['node_features'][i, :seq_length][valid_node_mask] = \
                    self.node_raw_features[valid_indices]
            
            actual_lengths.append(seq_length)
            

        return padded_sequences, actual_lengths

    def _process_through_patches(self, padded_sequences, co_occurrence_features, actual_lengths):
        """Process sequences through patches and transformer"""
        batch_size = len(actual_lengths)
        
        # Get features for each channel
        node_features = padded_sequences['node_features']
        edge_features = padded_sequences['edge_features']
        time_features = padded_sequences['time_features']
        
        # Create patches
        node_patches = self._create_patches(node_features, actual_lengths)
        edge_patches = self._create_patches(edge_features, actual_lengths)
        time_patches = self._create_patches(time_features, actual_lengths)
        co_occurrence_patches = self._create_patches(co_occurrence_features, actual_lengths)

        # Project each channel
        node_projected = self.projection_layers['node'](node_patches)
        edge_projected = self.projection_layers['edge'](edge_patches)
        time_projected = self.projection_layers['time'](time_patches)
        co_occurrence_projected = self.projection_layers['neighbor_co_occurrence'](co_occurrence_patches)

        # Stack channels
        # Shape: [batch_size, num_patches, num_channels, channel_embedding_dim]
        stacked_channels = torch.stack(
            [node_projected, edge_projected, time_projected, co_occurrence_projected],
            dim=2
        )
        
        # Reshape for transformer: [batch_size, num_patches, num_channels * channel_embedding_dim]
        transformer_input = stacked_channels.reshape(
            batch_size, 
            -1, 
            self.num_channels * self.channel_embedding_dim
        )

        # Apply transformer layers
        for transformer in self.transformer_layers:
            transformer_input = transformer(transformer_input)

        # Average over patches
        # Use mask for actual patches
        num_patches = transformer_input.shape[1]
        patch_mask = self._create_patch_mask(actual_lengths, num_patches)
        
        # Apply mask and average
        transformer_input = transformer_input * patch_mask.unsqueeze(-1)
        aggregated = transformer_input.sum(dim=1) / patch_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Final projection
        output = self.output_layer(aggregated)
        return output

    def _create_patches(self, features, actual_lengths):
        """Create patches from sequence features"""
        batch_size, max_seq_length, feat_dim = features.shape
        device = features.device
        
        # Ensure max_seq_length is valid
        if max_seq_length == 0:
            # Return empty patches
            return torch.zeros(batch_size, 0, self.patch_size * feat_dim, device=device)


        # Calculate number of patches
        num_patches = (max_seq_length + self.patch_size - 1) // self.patch_size
        
        if num_patches == 0:
            return torch.zeros(batch_size, 0, self.patch_size * feat_dim, device=device)

        # Create patches
        patches = []
        for i in range(num_patches):
            start_idx = i * self.patch_size
            end_idx = min(start_idx + self.patch_size, max_seq_length)
            
            patch = features[:, start_idx:end_idx, :]
            
            # If patch is smaller than patch_size, pad with zeros
            if patch.shape[1] < self.patch_size:
                padding = torch.zeros(batch_size, self.patch_size - patch.shape[1], 
                                     feat_dim, device=device)
                patch = torch.cat([patch, padding], dim=1)
            
            # Flatten patch: [batch_size, patch_size * feat_dim]
            patches.append(patch.reshape(batch_size, -1))
        
        # Stack patches: [batch_size, num_patches, patch_size * feat_dim]
        patches_tensor = torch.stack(patches, dim=1)
        return patches_tensor

    def _create_patch_mask(self, actual_lengths, num_patches):
        """Create mask for valid patches"""
        batch_size = len(actual_lengths)
        device = self.device
        
        mask = torch.zeros(batch_size, num_patches, device=device)
        
        for i, length in enumerate(actual_lengths):
            num_valid_patches = (length + self.patch_size - 1) // self.patch_size
            mask[i, :num_valid_patches] = 1.0
        
        return mask

    def _compute_loss(self, batch):
        logits = self.forward(batch)
        labels = batch['labels'].float()
        # DEBUG: Check label distribution
        # print(f"DEBUG: Labels - mean={labels.mean():.3f}, min={labels.min()}, max={labels.max()}")
        # Should be ~0.5 if balanced
        # If labels.mean() is 0.0 or 1.0, your negative sampling is broken.

        return self.loss_fn(logits, labels)

    def _compute_metrics(self, batch):
        logits = self.forward(batch)
        labels = batch['labels'].float()
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        ap = self._compute_ap(probs, labels)
        
        # Compute AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.cpu().numpy(), probs.detach().cpu().numpy())
        
        return {
            "accuracy": accuracy,
            "ap": ap,
            "auc": torch.tensor(auc, device=self.device),
            "loss": self.loss_fn(logits, labels)
        }

    def _compute_ap(self, probs, labels):
        """Compute Average Precision"""
        # Sort by probability
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels[sorted_indices]
        
        # Compute precision at each threshold
        cumulative_positives = torch.cumsum(sorted_labels, dim=0)
        cumulative_predictions = torch.arange(1, len(labels) + 1, device=labels.device, dtype=torch.float)
        precisions = cumulative_positives.float() / cumulative_predictions
        
        # Compute AP (only for positive predictions)
        relevant_mask = sorted_labels == 1
        if relevant_mask.sum() > 0:
            ap = precisions[relevant_mask].mean()
        else:
            ap = torch.tensor(0.0, device=labels.device)
        
        return ap






class NeighborCooccurrenceEncoder(nn.Module):
    def __init__(self, neighbor_co_occurrence_feat_dim, device='cpu'):
        super().__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        # self.device = device

        self.encode_layer = nn.Sequential(
            nn.Linear(2, neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(neighbor_co_occurrence_feat_dim, neighbor_co_occurrence_feat_dim)
        )

    def forward(self, src_neighbor_ids, dst_neighbor_ids):
        """Count co-occurrence of neighbors in source and destination sequences"""
        batch_size, seq_length = src_neighbor_ids.shape
        
        device = src_neighbor_ids.device

        # Initialize appearance tensors
        src_appearances = torch.zeros(batch_size, seq_length, 2, device=device)
        dst_appearances = torch.zeros(batch_size, seq_length, 2, device=device)
        
        for i in range(batch_size):
            # Count appearances in source sequence
            src_unique, src_counts = torch.unique(src_neighbor_ids[i], return_counts=True)
            src_dict = dict(zip(src_unique.cpu().numpy(), src_counts.cpu().numpy()))
            
            # Count appearances in destination sequence
            dst_unique, dst_counts = torch.unique(dst_neighbor_ids[i], return_counts=True)
            dst_dict = dict(zip(dst_unique.cpu().numpy(), dst_counts.cpu().numpy()))
            
            # Fill appearance counts
            for j, node_id in enumerate(src_neighbor_ids[i]):
                if node_id.item() != 0:  # Skip padding
                    src_appearances[i, j, 0] = src_dict.get(node_id.item(), 0)
                    src_appearances[i, j, 1] = dst_dict.get(node_id.item(), 0)
            
            for j, node_id in enumerate(dst_neighbor_ids[i]):
                if node_id.item() != 0:  # Skip padding
                    dst_appearances[i, j, 0] = src_dict.get(node_id.item(), 0)
                    dst_appearances[i, j, 1] = dst_dict.get(node_id.item(), 0)
        
        # Encode appearances
        src_features = self.encode_layer(src_appearances)
        dst_features = self.encode_layer(dst_appearances)
        
        return src_features, dst_features




class TransformerEncoder(nn.Module):
    def __init__(self, attention_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.linear_layers = nn.ModuleList([
            nn.Linear(attention_dim, 4 * attention_dim),
            nn.Linear(4 * attention_dim, attention_dim)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.norm_layers[0](x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](x))))
        x = self.norm_layers[1](x + self.dropout(ff_output))
        
        return x



class LinkPredictor(nn.Module):
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




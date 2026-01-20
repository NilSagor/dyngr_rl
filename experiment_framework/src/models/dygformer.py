import torch 
import torch.nn as nn

from .base_model import BaseDynamicGNN, TimeEncoder

class DyGFormer(BaseDynamicGNN):
    def __init__(self, num_nodes, node_features, hidden_dim, time_encoding_dim=32,num_layers=2,num_heads=8,dropout=0.1,max_neibors=20,neighbor_co_occurrence=True,learning_rate=1e-4,weight_decay=1e-5, **kwargs):
        super().__init__(
            num_nodes = num_nodes,
            node_features=node_features,
            hidden_dim=hidden_dim,
            time_encoding_dim=time_encoding_dim,
            num_layers=num_layers,
            dropout = dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )

        
        self.num_heads = num_heads
        self.max_neighbors = max_neibors
        self.neighbor_co_occurrence = neighbor_co_occurrence

        if node_features > 0:
            self.node_embedding = nn.Embedding(num_nodes, node_features)
            input_dim = node_features + time_encoding_dim
        else:
            self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
            input_dim = hidden_dim + time_encoding_dim
        
        if neighbor_co_occurrence:
            self.neighbor_co_occurrence = NeighborCoOccurrenceEncoder(
                max_neighbors= max_neibors,
                hidden_dim = hidden_dim
            )
            input_dim += hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.transformer_layers = nn.ModuleList([
            TemporalTransformerLayer(
                hidden_dim = hidden_dim,
                num_heads = num_heads,
                dropout = dropout
            )
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        self.link_predictor = LinkPredictor(hidden_dim, dropout)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']

        src_embeddings = self._compute_node_embeddings(
            src_nodes, timestamps, batch.get("src_neighbors"), batch.get("src_neighbor_times"), "src"
        )
        
        dst_embeddings = self._compute_node_embeddings(
            dst_nodes, timestamps, batch.get("dst_neighbors"), batch.get("dst_neighbor_times"), "dst"
        )

        logits = self.link_predictor(src_embeddings, dst_embeddings)
        return logits

    def _compute_node_embeddings(self, nodes, timestamps, neighbors, neighbor_times, node_type):
        batch_size = nodes.size(0)
        node_features = self.node_embedding(nodes)

        time_enc = self.time_encoder(timestamps)

        if self.node_features>0:
            embeddings = torch.cat([node_features, time_enc], dim=-1)
        else:
            embeddings = torch.cat([node_features, time_enc], dim=-1)

        if self.neighbor_co_occurrence and neighbors is not None:
            co_occurrence_enc = self.neighbor_co_occurrence(
                nodes, neighbors, neighbor_times, node_type
            )
            embeddings = torch.cat([embeddings, co_occurrence_enc], dim=-1)
        
        embeddings = self.input_projection(embeddings)

        for layer in self.transformer_layers:
            embeddings = layer(embeddings, neighbors, neighbor_times, timestamps)
        return embeddings
    
    def _compute_loss(self, batch):
        logits = self.forward(batch)
        labels = batch['labels'].float()
        return self.loss_fn(logits, labels)
    
    def _compute_metrics(self,batch):
        logits = self.forward(batch)
        labels = batch['labels'].float()
        probs = torch.sigmoid(logits)
        predictions = (probs>0.5).float()
        accuracy = (predictions==labels).float().mean()
        ap = self._compute_ap(probs, labels)
        return {
            "accuracy": accuracy,
            "ap":ap,
            "loss": self.loss_fn(logits, labels)
        }
    
    
    def _compute_ap(self, probs, labels):
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels[sorted_indices]

        cumulative_positives = torch.cumsum(sorted_labels, dim=0)
        cumulative_predicitions = torch.arange(1, len(labels)+1, device=labels.device)
        precisions = cumulative_positives.float()/cumulative_predicitions.float()
        ap = precisions.mean()
        return ap







class NeighborCoOccurrenceEncoder(nn.Module):
    def __init__(self, max_neighbors, hidden_dim):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.hidden_dim = hidden_dim

        self.mlp_src = nn.Sequential(
            nn.Linear(max_neighbors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2)
        )

        

        self.mlp_dst = nn.Sequential(
            nn.Linear(max_neighbors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2)
        )

        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, nodes, neighbors, neighbor_times, node_type):
        batch_size = nodes.size(0)
        co_occurrence = (neighbors>0).float()
        src_features = self.mlp_src(co_occurrence)
        dst_features = self.mlp_dst(co_occurrence)
        combined_features = torch.cat([src_features, dst_features], dim=-1)
        output = self.projection(combined_features)
        return output





class TemporalTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim= hidden_dim,
            num_heads=num_heads,
            dropout = dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, neighbors, neighbor_times, timestamps):
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        attended = self.dropout(attended)
        embeddings = self.norm1(embeddings+attended)
        ffn_output = self.ffn(embeddings)
        ffn_output = self.dropout(ffn_output)
        embeddings = self.norm2(embeddings+ffn_output)
        return embeddings





class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, src_embedding, dst_embedding):
        link_input = torch.cat([src_embedding, dst_embedding], dim=-1)
        logits = self.mlp(link_input).squeeze(-1)
        return logits
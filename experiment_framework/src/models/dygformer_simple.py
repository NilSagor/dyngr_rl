# experiment_framework/src/models/dygformer_simple.py
import torch
import torch.nn as nn
from .base_model import BaseDynamicGNN, TimeEncoder

class DyGFormerSimple(BaseDynamicGNN):
    """Drastically simplified DyGFormer for debugging"""
    
    def __init__(self, num_nodes, node_features, hidden_dim, **kwargs):
        # Call parent with minimal parameters
        super().__init__(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=hidden_dim,
            time_encoding_dim=32,
            num_layers=1,
            dropout=0.1,
            learning_rate=1e-4,
            weight_decay=1e-5
        )
        
        print(f"Initializing simplified DyGFormer")
        
        # Minimal components
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.time_encoder = TimeEncoder(32)
        self.output_layer = nn.Linear(hidden_dim * 2, 1)
        
        # Skip all complex components
        print(f"Model created with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, batch):
        # Extremely simple forward pass
        src_emb = self.node_embedding(batch['src_nodes'])
        dst_emb = self.node_embedding(batch['dst_nodes'])
        
        # Add time encoding
        time_enc = self.time_encoder(batch['timestamps'])
        src_emb = src_emb + time_enc
        dst_emb = dst_emb + time_enc
        
        # Simple link prediction
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        logits = self.output_layer(combined).squeeze(-1)
        
        return logits
    
    def _compute_loss(self, batch):
        logits = self.forward(batch)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, batch['labels']
        )
    
    def _compute_metrics(self, batch):
        logits = self.forward(batch)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == batch['labels']).float().mean()
        
        return {
            "accuracy": acc,
            "ap": torch.tensor(0.5),  # Placeholder
            "loss": self._compute_loss(batch)
        }

# Create a test script for this
if __name__ == "__main__":
    print("Testing simplified DyGFormer")
    model = DyGFormerSimple(
        num_nodes=1000,
        node_features=172,
        hidden_dim=64
    )
    
    batch = {
        'src_nodes': torch.randint(0, 1000, (4,)),
        'dst_nodes': torch.randint(0, 1000, (4,)),
        'timestamps': torch.rand(4) * 1000,
        'labels': torch.randint(0, 2, (4,)).float()
    }
    
    output = model(batch)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
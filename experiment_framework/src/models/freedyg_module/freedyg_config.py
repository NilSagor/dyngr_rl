# src/models/freedyg_module/freedyg_config.py
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class FreeDyGConfig:
    """Configuration for FreeDyG model."""
    # Runtime objects (injected by pipeline)
    node_raw_features: Any = None
    edge_raw_features: Any = None
    neighbor_sampler: Any = None
    
    # Architecture
    time_feat_dim: int = 100
    channel_embedding_dim: int = 172  # Output dim for each projected feature
    num_layers: int = 2
    dropout: float = 0.1
    max_input_sequence_length: int = 128  # Number of neighbors to sample
    
    # NIF Encoder (Neighbor Interaction Feature)
    nif_feat_dim: int = 50  # Internal NIF feature dimension
    
    # Training
    device: str = 'cpu'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
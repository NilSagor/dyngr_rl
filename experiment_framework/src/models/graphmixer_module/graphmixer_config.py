from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class GraphMixerConfig:
    time_gap: int = 2000
    node_raw_features: Any = None
    edge_raw_features: Any = None
    neighbor_sampler: Any = None
    time_feat_dim: int = 100
    num_tokens: int = 20
    num_layers: int = 2
    token_dim_expansion_factor: float = 0.5
    channel_dim_expansion_factor: float = 4.0
    dropout: float = 0.1
    device: str = 'cpu'  
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
# tawrmac_config.py
from dataclasses import dataclass, field
from typing import Optional, Any
import torch

@dataclass
class TAWRMACConfig:
    # Required runtime objects (set by pipeline)
    neighbor_finder: Any = None
    node_features: Any = None
    edge_features: Any = None
    device: Any = None

    # Architecture
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.1
    use_memory: bool = False
    memory_update_at_start: bool = True
    memory_dimension: int = 500
    n_neighbors: Optional[int] = None

    # Walk module
    enable_walk: bool = False
    enable_restart: bool = False
    pick_new_neighbors: bool = False
    walk_emb_dim: int = 172
    position_feat_dim: int = 100
    walk_length: int = 4
    num_walks: int = 10
    num_walk_heads: int = 4

    # Co-occurrence 
    enable_neighbor_cooc: bool = False
    max_input_seq_length: int = 32

    # time
    time_dim: int = 172
    fixed_time_dim: int = 20
  
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
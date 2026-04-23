# hicostdev_configdev1.py
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class HiCoSTdev2Config:
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

    memory_dim: int = 172

    # Walk module
    enable_walk: bool = False
    enable_restart: bool = False
    pick_new_neighbors: bool = False
    walk_emb_dim: int = 172
    position_feat_dim: int = 100
    walk_length: int = 4
    num_walks: int = 10
    num_walk_heads: int = 4

    # Walk Sampler Params
    debug_simple_walk: bool = False
    walk_length_short: int = 3
    walk_length_long: int = 10
    walk_length_tawr: int = 8
    num_walks_short: int = 5
    num_walks_long: int = 3
    num_walks_tawr: int = 3
    walk_temperature: float = 0.1
    walk_time_noise_std: float = 0.0
    walk_mask_prob: float = 0.05
    walk_adaptive_factor: float = 0.5
    
    # time_delta_attention_walk
    use_time_delta_attention: bool = True
    use_temporal_bias: bool = True

    # Co-occurrence 
    enable_neighbor_cooc: bool = False
    max_input_seq_length: int = 32

    # co-gnn
    use_explicit_co_gnn: bool = True

    # time
    time_dim: int = 172
    fixed_time_dim: int = 20
  
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
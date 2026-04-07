from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HiCoSTConfig:
    # Core
    num_nodes: int
    node_feat_dim: int = 0
    edge_feat_dim: int = 64
    edge_features_dim: int = 64
    hidden_dim: int = 17
    time_dim: int = 64
    memory_dim: int = 172
    
    # Learning Params
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: float = 0.1
    min_lr_ratio: float = 0.01
    
    # SAM Params
    num_prototypes: int = 5
    sam_residual_alpha: float = 0.8
    sam_time_decay: float = 0.99
    use_sam: bool = True            
    similarity_metric: str = "cosine"  
    
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
    
    # HCT Params
    use_hct: bool = True
    hct_d_model: int = 128
    hct_nhead: int = 2
    hct_num_layers: int = 1
    hct_num_intra_layers: int = 2
    hct_num_inter_layers: int = 2        
    hct_drop_path: float = 0.1
    hct_sigma_time: float = 5.0
    hct_cooccurrence_sigma: float = 2.0  
    
    # ST-ODE Params
    use_st_ode: bool = True
    ode_method: str = "dopri5"
    ode_mu: float = 0.1,                
    mu: float = 0.1                      
    adaptive_mu: bool = True            
    ode_num_eig: int = 10
    ode_velocity_gate_thresh: float = 5.0
    st_ode_update_interval: int = 10
    ode_step_size: float = 1.0          
    use_gru_ode: bool = True,          
    adjoint: bool = True                
    
    # Gate Mutual Refinement 
    mrp_fusion_mode: str = "gated_plus_residual"  # Options: "gated_plus_residual", "gated_only", "proj_only"
    mrp_pool_attn_heads: int = 1                   # Default: global pooling; ablate with 4
    mrp_nhead: int = 4                             # Attention heads for cross-attention
    mrp_modalities: int = 3                        # SAM + HCT + ST-ODE
    
    # Hard Negative Mining
    use_hard_negative_mining: bool = True
    neg_sample_ratio: int = 5
    hard_neg_threshold: float = 0.7
    label_smoothing: float = 0.1
    dropout: float = 0.1
    
    # Ablation Flags ( True = USE advanced feature)
    use_prototype_attention: bool = True
    use_hct_hierarchical: bool = True
    use_gated_refinement: bool = True
    use_multi_scale_walks: bool = True

    
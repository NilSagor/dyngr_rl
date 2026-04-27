import torch 
from typing import Dict, List, Optional, Any, Type
from loguru import logger
import inspect

# Import all models and their config classes
from src.models.dygformer import DyGFormer
from src.models.tgn import TGN
from src.models.enhanced_tgn.variants.tgn_v2 import TGNv2
from src.models.enhanced_tgn.variants.tgn_v3 import TGNv3
from src.models.enhanced_tgn.variants.tgn_v4 import TGNv4
from src.models.enhanced_tgn.variants.tgn_v5 import TGNv5
from src.models.enhanced_tgn.variants.tgn_v6 import TGNv6
from src.models.enhanced_tgn.variants.tgn_v7 import TGNv7
# from src.models.enhanced_tgn.variants.hicost import HiCoST
# from src.models.enhanced_tgn.variants.hicostv2 import HiCoSTv2
# from src.models.enhanced_tgn.variants.hicostv3 import HiCoSTv3
# from src.models.enhanced_tgn.variants.hicostv4 import HiCoSTv4

from src.models.tawrmac_module.tawrmac_variants.tawrmac_v1 import TAWRMACv1
from src.models.tawrmac_module.tawrmac_variants.tawrmac_config import TAWRMACConfig

# from src.models.hicost_dev.hicost_variants.hicostdev1 import HiCoSTdev1
# from src.models.hicost_dev.hicost_variants.hicostdev_configdev1 import HiCoSTdev1Config
# from src.models.hicost_dev.hicost_variants.hicostdev2 import HiCoSTdev2
# from src.models.hicost_dev.hicost_variants.hicostdev_configdev2 import HiCoSTdev2Config
from src.models.hicost_dev.v1.model import HiCoSTdev1
from src.models.hicost_dev.v1.config import HiCoSTConfig
# If HiCoSTdev3 exists, import its config similarly

# ============================================================================
# MODEL REGISTRY WITH CONFIG CLASS
# ============================================================================

# For models that use a configuration dataclass, map model name to (model_class, config_class)
# For models that use legacy argument passing, keep only model_class (config_class = None)
MODEL_INFO = {
    # Legacy models (no config dataclass)
    "DyGFormer": (DyGFormer, None),
    "TGN": (TGN, None),
    "TGNv2": (TGNv2, None),
    "TGNv3": (TGNv3, None),
    "TGNv4": (TGNv4, None),
    "TGNv5": (TGNv5, None),
    "TGNv6": (TGNv6, None),
    "TGNv7": (TGNv7, None),
    # "HiCoST": (HiCoST, None),
    # "HiCoSTv2": (HiCoSTv2, None),
    # TAWRMAC and HiCoSTdev variants use config dataclasses
    "TAWRMACv1": (TAWRMACv1, TAWRMACConfig),
    # "HiCoSTdev1": (HiCoSTdev1, HiCoSTdev1Config),
    # "HiCoSTdev2": (HiCoSTdev2, HiCoSTdev2Config),
    # "HiCoSTv3": (HiCoSTv3, HiCoSTConfig),   # HiCoSTv3 uses HiCoSTConfig
    # "HiCoSTv4": (HiCoSTv4, HiCoSTConfig),
    # Add HiCoSTdev3 when needed
    # "HiCoSTdev3": (HiCoSTdev3, HiCoSTdev3Config),
    "HiCoSTdev1": (HiCoSTdev1, HiCoSTConfig)
}

class ModelFactory:
    """Factory for creating and validating models using registry + config dataclasses."""

    @staticmethod
    def create(config: Dict, data_info: Dict) -> torch.nn.Module:
        """Create model with proper parameter handling."""
        model_config = config['model'].copy()
        model_name = model_config.pop('name')
        
        if model_name not in MODEL_INFO:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class, config_class = MODEL_INFO[model_name]
        
        # If model uses a config dataclass, build it and pass to constructor
        if config_class is not None:
            # Extract only the keys that exist in config_class
            sig = inspect.signature(config_class.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
            
            # Combine model_config, training config, and data_info
            training_cfg = config.get('training', {})
            # Use data_info to get node/edge feature dimensions
            base_args = {
                'node_features': data_info.get('node_features'),
                'edge_features': data_info.get('edge_features'),
                'num_nodes': data_info.get('num_nodes'),
            }
            # Add device if not already in model_config
            if 'device' not in model_config:
                base_args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Merge all sources: base_args, model_config, training_cfg
            all_args = {**base_args, **model_config, **training_cfg}
            
            # Filter to only those keys that the config class accepts
            filtered_args = {k: v for k, v in all_args.items() if k in valid_params}
            
            # Create config object
            cfg_obj = config_class(**filtered_args)
            
            # Instantiate model
            model = model_class(config=cfg_obj)
        else:
            # Legacy models: build args using signature of model class
            model_args = ModelFactory._build_legacy_args(model_class, model_config, data_info)
            logger.debug(f"Legacy model args: {model_args}")
            model = model_class(**model_args)
        
        # Validation
        ModelFactory.validate(model, data_info)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Created {model_name}: {num_params:,} parameters")
        
        return model

    @staticmethod
    def _build_legacy_args(model_class, model_config: Dict, data_info: Dict) -> Dict:
        """Build kwargs for legacy models using signature inspection."""
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        model_args = {
            'num_nodes': data_info.get('num_nodes'),
            'node_features': data_info.get('node_feat_dim', 0),
            'edge_features_dim': data_info.get('edge_feat_dim', 172),
        }
        
        for key, value in model_config.items():
            if key in valid_params:
                model_args[key] = value
            else:
                logger.debug(f"Key '{key}' ignored for {model_class.__name__}")
        
        return model_args

    @staticmethod
    def validate(model: torch.nn.Module, data_info: Dict):
        """Validate model matches data specifications."""
        num_nodes = data_info['num_nodes']
        
        # Check num_nodes consistency
        if hasattr(model, 'num_nodes'):
            assert model.num_nodes == num_nodes, \
                f"model.num_nodes mismatch: {model.num_nodes} != {num_nodes}"
        
        # Check hyperparameters preserved
        if hasattr(model, 'hparams'):
            if 'num_nodes' in model.hparams:
                assert model.hparams.num_nodes == num_nodes, \
                    f"hparams.num_nodes mismatch: {model.hparams.num_nodes} != {num_nodes}"
        
        # Check memory size for TGN
        if hasattr(model, 'memory') and model.memory is not None:
            mem_size = model.memory.memory.shape[0]
            assert mem_size == num_nodes, \
                f"Memory size mismatch: {mem_size} != {num_nodes}"
        
        logger.debug("Model validation passed")






# import torch 
# from typing import Dict, List, Optional, Any
# from loguru import logger
# import inspect


# from src.models.enhanced_tgn.component.hicost_config import HiCoSTConfig

# from src.models.dygformer import DyGFormer
# from src.models.tgn import TGN
# from src.models.enhanced_tgn.variants.tgn_v2 import TGNv2
# from src.models.enhanced_tgn.variants.tgn_v3 import TGNv3
# from src.models.enhanced_tgn.variants.tgn_v4 import TGNv4
# from src.models.enhanced_tgn.variants.tgn_v5 import TGNv5
# from src.models.enhanced_tgn.variants.tgn_v6 import TGNv6
# from src.models.enhanced_tgn.variants.tgn_v7 import TGNv7
# from src.models.enhanced_tgn.variants.hicost import HiCoST
# from src.models.enhanced_tgn.variants.hicostv2 import HiCoSTv2
# from src.models.enhanced_tgn.variants.hicostv3 import HiCoSTv3
# from src.models.enhanced_tgn.variants.hicostv4 import HiCoSTv4

# from src.models.tawrmac_module.tawrmac_variants.tawrmac_v1 import TAWRMACv1
# from src.models.tawrmac_module.tawrmac_variants.tawrmac_config import TAWRMACConfig

# from src.models.hicost_dev.hicost_variants.hicostdev1 import HiCoSTdev1
# from src.models.hicost_dev.hicost_variants.hicostdev_configdev1 import HiCoSTdev1Config
# from src.models.hicost_dev.hicost_variants.hicostdev2 import HiCoSTdev2
# from src.models.hicost_dev.hicost_variants.hicostdev_configdev2 import HiCoSTdev2Config


# # Constants
# MODEL_REGISTRY = {
#     "DyGFormer": DyGFormer,
#     "TGN": TGN,
#     "TGNv2": TGNv2,
#     "TGNv3": TGNv3,
#     "TGNv4": TGNv4,
#     "TGNv5": TGNv5,
#     "TGNv6": TGNv6,
#     "TGNv7": TGNv7,
#     "HiCoST": HiCoST,
#     "HiCoSTv2": HiCoSTv2,
#     "HiCoSTv3": HiCoSTv3,
#     "HiCoSTv4": HiCoSTv4,
#     "TAWRMACv1": TAWRMACv1,
#     "HiCoSTdev1": HiCoSTdev1,
#     "HiCoSTdev2": HiCoSTdev2,
# }



# # ============================================================================
# # MODEL
# # ============================================================================

# class ModelFactory:
#     """Factory for creating and validating models."""
            
#     @staticmethod
#     def create(config: Dict, data_info: Dict) -> torch.nn.Module:
#         """Create model with proper parameter handling."""
#         model_config = config['model'].copy()
#         model_name = model_config.pop('name')
        
#         if model_name not in MODEL_REGISTRY:
#             raise ValueError(f"Unknown model: {model_name}")
        
#         model_class = MODEL_REGISTRY[model_name]
        
#         #  HiCoSTv3 uses config dataclass
#         if model_name == "HiCoSTdev1":
#             model = ModelFactory._create_hicostdev1(model_config, data_info, config)
#         # if model_name == "HiCoSTv3":
#         #     model = ModelFactory._create_hicostv3(model_config, data_info, config)
#         # elif model_name == "HiCoSTv4":
#         #     model = ModelFactory._create_hicostv4(model_config, data_info, config)
#         elif model_name == "TAWRMACv1":
#             model = ModelFactory._create_tawrmacv1(model_config, data_info, config)
#         else:
#             # === Legacy path for all other models ===
#             model_args = ModelFactory._build_legacy_args(model_class, model_config, data_info)
#             print(f"Model args: {model_args}")
#             model = model_class(**model_args)
        
#         # Validation
#         ModelFactory.validate(model, data_info)
#         num_params = sum(p.numel() for p in model.parameters())
#         logger.info(f"Created {model_name}: {num_params:,} parameters")
        
#         return model
    
#     @staticmethod
#     def _create_hicostv4(model_config: Dict, data_info: Dict, full_config: Dict):
#         """Build HiCoSTConfig and instantiate HiCoSTv3."""
      
        
#         training_cfg = full_config.get('training', {})
        
#         # Build config object from YAML dict
#         # Only include fields that exist in HiCoSTConfig dataclass
#         hicost_config = HiCoSTConfig(
#             # Required
#             num_nodes=data_info['num_nodes'],
            
#             # Core dimensions 
#             node_feat_dim=data_info.get('node_feat_dim', data_info.get('node_features', 0)),
#             edge_feat_dim=data_info.get('edge_feat_dim', model_config.get('edge_features_dim', 64)),
#             edge_features_dim=model_config.get('edge_features_dim', 64),  # alias
#             hidden_dim=model_config.get('hidden_dim', 172),
#             memory_dim=model_config.get('memory_dim', 172),
#             time_dim=model_config.get('time_dim', 64),
            
#             # Learning params
#             learning_rate=training_cfg.get('learning_rate', 1e-4),
#             weight_decay=training_cfg.get('weight_decay', 1e-5),
#             warmup_epochs=training_cfg.get('warmup_epochs', 0.1),
#             min_lr_ratio=training_cfg.get('min_lr_ratio', 0.01),
            
#             # SAM params
#             num_prototypes=model_config.get('num_prototypes', 5),
#             sam_residual_alpha=model_config.get('sam_residual_alpha', 0.8),
#             sam_time_decay=model_config.get('sam_time_decay', 0.99),
#             use_sam=model_config.get('use_sam', True),
#             similarity_metric=model_config.get('similarity_metric', 'cosine'),
            
#             # Walk sampler params
#             debug_simple_walk=model_config.get('debug_simple_walk', False),
#             walk_length_short=model_config.get('walk_length_short', 3),
#             walk_length_long=model_config.get('walk_length_long', 10),
#             walk_length_tawr=model_config.get('walk_length_tawr', 8),
#             num_walks_short=model_config.get('num_walks_short', 5),
#             num_walks_long=model_config.get('num_walks_long', 3),
#             num_walks_tawr=model_config.get('num_walks_tawr', 3),
#             walk_temperature=model_config.get('walk_temperature', 0.1),            
#             walk_time_noise_std=model_config.get('walk_time_noise_std', 0.0),
#             walk_mask_prob=model_config.get('walk_mask_prob', 0.05),
#             walk_adaptive_factor=model_config.get('walk_adaptive_factor', 0.5),
            
#             # HCT params
#             use_hct=model_config.get('use_hct', True),
#             hct_d_model=model_config.get('hct_d_model', 128),
#             hct_nhead=model_config.get('hct_nhead', 2),
#             hct_num_layers=model_config.get('hct_num_layers', 1),
#             hct_num_intra_layers=model_config.get('hct_num_intra_layers', 2),
#             hct_num_inter_layers=model_config.get('hct_num_inter_layers', 2),
#             hct_drop_path=model_config.get('hct_drop_path', 0.1),
#             hct_sigma_time=model_config.get('hct_sigma_time', 5.0),
#             hct_cooccurrence_sigma=model_config.get('hct_cooccurrence_sigma', 2.0),
            
#             # ST-ODE params
#             use_st_ode=model_config.get('use_st_ode', True),
#             ode_method=model_config.get('ode_method', 'dopri5'),
#             ode_mu=model_config.get('ode_mu', model_config.get('mu', 0.1)),  # handle alias
#             mu=model_config.get('mu', 0.1),
#             adaptive_mu=model_config.get('adaptive_mu', True),
#             ode_num_eig=model_config.get('ode_num_eig', 10),
#             ode_velocity_gate_thresh=model_config.get('ode_velocity_gate_thresh', 5.0),
#             st_ode_update_interval=model_config.get('st_ode_update_interval', 10),
#             ode_step_size=model_config.get('ode_step_size', 1.0),
#             use_gru_ode=model_config.get('use_gru_ode', True),
#             adjoint=model_config.get('adjoint', True),
            
#             # MRP (GatedMutualRefinementPooling) params
#             mrp_fusion_mode=model_config.get('mrp_fusion_mode', 'gated_plus_residual'),
#             mrp_pool_attn_heads=model_config.get('mrp_pool_attn_heads', 1),
#             mrp_nhead=model_config.get('mrp_nhead', 4),
#             mrp_modalities=model_config.get('mrp_modalities', 3),
            
#             # Hard negative mining
#             use_hard_negative_mining=model_config.get('use_hard_negative_mining', True),
#             neg_sample_ratio=model_config.get('neg_sample_ratio', 5),
#             hard_neg_threshold=model_config.get('hard_neg_threshold', 0.7),
#             label_smoothing=model_config.get('label_smoothing', 0.1),
            
#             # Dropout & ablation flags
#             dropout=model_config.get('dropout', 0.1),
#             use_prototype_attention=model_config.get('use_prototype_attention', True),
#             use_hct_hierarchical=model_config.get('use_hct_hierarchical', True),
#             use_gated_refinement=model_config.get('use_gated_refinement', True),
#             use_multi_scale_walks=model_config.get('use_multi_scale_walks', True),
#         )
        
#         logger.info(f"HiCoSTv4 config: memory_dim={hicost_config.memory_dim}, "
#                    f"fusion_mode={hicost_config.mrp_fusion_mode}")
        
#         # Instantiate with config object
#         return MODEL_REGISTRY["HiCoSTv4"](config=hicost_config)
    
#     @staticmethod
#     def _create_hicostv3(model_config: Dict, data_info: Dict, full_config: Dict):
#         """Build HiCoSTConfig and instantiate HiCoSTv3."""
      
        
#         training_cfg = full_config.get('training', {})
        
#         # Build config object from YAML dict
#         # Only include fields that exist in HiCoSTConfig dataclass
#         hicost_config = HiCoSTConfig(
#             # Required
#             num_nodes=data_info['num_nodes'],
            
#             # Core dimensions 
#             node_feat_dim=data_info.get('node_feat_dim', data_info.get('node_features', 0)),
#             edge_feat_dim=data_info.get('edge_feat_dim', model_config.get('edge_features_dim', 64)),
#             edge_features_dim=model_config.get('edge_features_dim', 64),  # alias
#             hidden_dim=model_config.get('hidden_dim', 172),
#             memory_dim=model_config.get('memory_dim', 172),
#             time_dim=model_config.get('time_dim', 64),
            
#             # Learning params
#             learning_rate=training_cfg.get('learning_rate', 1e-4),
#             weight_decay=training_cfg.get('weight_decay', 1e-5),
#             warmup_epochs=training_cfg.get('warmup_epochs', 0.1),
#             min_lr_ratio=training_cfg.get('min_lr_ratio', 0.01),
            
#             # SAM params
#             num_prototypes=model_config.get('num_prototypes', 5),
#             sam_residual_alpha=model_config.get('sam_residual_alpha', 0.8),
#             sam_time_decay=model_config.get('sam_time_decay', 0.99),
#             use_sam=model_config.get('use_sam', True),
#             similarity_metric=model_config.get('similarity_metric', 'cosine'),
            
#             # Walk sampler params
#             debug_simple_walk=model_config.get('debug_simple_walk', False),
#             walk_length_short=model_config.get('walk_length_short', 3),
#             walk_length_long=model_config.get('walk_length_long', 10),
#             walk_length_tawr=model_config.get('walk_length_tawr', 8),
#             num_walks_short=model_config.get('num_walks_short', 5),
#             num_walks_long=model_config.get('num_walks_long', 3),
#             num_walks_tawr=model_config.get('num_walks_tawr', 3),
#             walk_temperature=model_config.get('walk_temperature', 0.1),            
#             walk_time_noise_std=model_config.get('walk_time_noise_std', 0.0),
#             walk_mask_prob=model_config.get('walk_mask_prob', 0.05),
#             walk_adaptive_factor=model_config.get('walk_adaptive_factor', 0.5),
            
#             # HCT params
#             use_hct=model_config.get('use_hct', True),
#             hct_d_model=model_config.get('hct_d_model', 128),
#             hct_nhead=model_config.get('hct_nhead', 2),
#             hct_num_layers=model_config.get('hct_num_layers', 1),
#             hct_num_intra_layers=model_config.get('hct_num_intra_layers', 2),
#             hct_num_inter_layers=model_config.get('hct_num_inter_layers', 2),
#             hct_drop_path=model_config.get('hct_drop_path', 0.1),
#             hct_sigma_time=model_config.get('hct_sigma_time', 5.0),
#             hct_cooccurrence_sigma=model_config.get('hct_cooccurrence_sigma', 2.0),
            
#             # ST-ODE params
#             use_st_ode=model_config.get('use_st_ode', True),
#             ode_method=model_config.get('ode_method', 'dopri5'),
#             ode_mu=model_config.get('ode_mu', model_config.get('mu', 0.1)),  # handle alias
#             mu=model_config.get('mu', 0.1),
#             adaptive_mu=model_config.get('adaptive_mu', True),
#             ode_num_eig=model_config.get('ode_num_eig', 10),
#             ode_velocity_gate_thresh=model_config.get('ode_velocity_gate_thresh', 5.0),
#             st_ode_update_interval=model_config.get('st_ode_update_interval', 10),
#             ode_step_size=model_config.get('ode_step_size', 1.0),
#             use_gru_ode=model_config.get('use_gru_ode', True),
#             adjoint=model_config.get('adjoint', True),
            
#             # MRP (GatedMutualRefinementPooling) params
#             mrp_fusion_mode=model_config.get('mrp_fusion_mode', 'gated_plus_residual'),
#             mrp_pool_attn_heads=model_config.get('mrp_pool_attn_heads', 1),
#             mrp_nhead=model_config.get('mrp_nhead', 4),
#             mrp_modalities=model_config.get('mrp_modalities', 3),
            
#             # Hard negative mining
#             use_hard_negative_mining=model_config.get('use_hard_negative_mining', True),
#             neg_sample_ratio=model_config.get('neg_sample_ratio', 5),
#             hard_neg_threshold=model_config.get('hard_neg_threshold', 0.7),
#             label_smoothing=model_config.get('label_smoothing', 0.1),
            
#             # Dropout & ablation flags
#             dropout=model_config.get('dropout', 0.1),
#             use_prototype_attention=model_config.get('use_prototype_attention', True),
#             use_hct_hierarchical=model_config.get('use_hct_hierarchical', True),
#             use_gated_refinement=model_config.get('use_gated_refinement', True),
#             use_multi_scale_walks=model_config.get('use_multi_scale_walks', True),
#         )
        
#         logger.info(f"HiCoSTv3 config: memory_dim={hicost_config.memory_dim}, "
#                    f"fusion_mode={hicost_config.mrp_fusion_mode}")
        
#         # Instantiate with config object
#         return MODEL_REGISTRY["HiCoSTv3"](config=hicost_config)
    
#     @staticmethod
#     def _create_tawrmacv1(model_config: Dict, data_info: Dict, full_config: Dict):     

#         training_cfg = full_config.get('training', {})
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Build config object
#         tawrmac_cfg = TAWRMACConfig(
#             neighbor_finder=None,  # Will be set later via set_neighbor_finder
#             node_features=data_info.get('node_features'),
#             edge_features=data_info.get('edge_features'),
#             device=device,
#             n_layers=model_config.get('n_layers', 2),
#             n_heads=model_config.get('n_heads', 2),
#             dropout=model_config.get('dropout', 0.1),
#             use_memory=model_config.get('use_memory', False),
#             memory_update_at_start=model_config.get('memory_update_at_start', True),
#             memory_dimension=model_config.get('memory_dim', 500),
#             n_neighbors=model_config.get('n_neighbors', None),
#             enable_walk=model_config.get('enable_walk', False),
#             enable_restart=model_config.get('enable_restart', False),
#             pick_new_neighbors=model_config.get('pick_new_neighbors', False),
#             walk_emb_dim=model_config.get('walk_emb_dim', 172),
#             position_feat_dim=model_config.get('position_feat_dim', 100),
#             walk_length=model_config.get('walk_length', 4),
#             num_walks=model_config.get('num_walks', 10),
#             num_walk_heads=model_config.get('num_walk_heads', 4),
#             enable_neighbor_cooc=model_config.get('enable_neighbor_cooc', False),
#             max_input_seq_length=model_config.get('max_input_seq_length', 32),
#             time_dim=model_config.get('time_dim', 172),
#             fixed_time_dim=model_config.get('fixed_time_dim', 20),
#             learning_rate=training_cfg.get('learning_rate', 1e-4),
#             weight_decay=training_cfg.get('weight_decay', 1e-5),
#         )
#         return TAWRMACv1(config=tawrmac_cfg)
    
#     @staticmethod
#     def _create_hicostdev1(model_config: Dict, data_info: Dict, full_config: Dict):     

#         training_cfg = full_config.get('training', {})
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Build config object
#         hicostdev1_cfg = HiCoSTdev1Config(
#             neighbor_finder=None,  # Will be set later via set_neighbor_finder
#             node_features=data_info.get('node_features'),
#             edge_features=data_info.get('edge_features'),
#             device=device,
#             n_layers=model_config.get('n_layers', 2),
#             n_heads=model_config.get('n_heads', 2),
#             dropout=model_config.get('dropout', 0.1),
#             use_memory=model_config.get('use_memory', False),
#             memory_update_at_start=model_config.get('memory_update_at_start', True),
#             memory_dimension=model_config.get('memory_dim', 500),
#             n_neighbors=model_config.get('n_neighbors', None),
#             enable_walk=model_config.get('enable_walk', False),
#             enable_restart=model_config.get('enable_restart', False),
#             pick_new_neighbors=model_config.get('pick_new_neighbors', False),
#             walk_emb_dim=model_config.get('walk_emb_dim', 172),
#             position_feat_dim=model_config.get('position_feat_dim', 100),
#             walk_length=model_config.get('walk_length', 4),
#             num_walks=model_config.get('num_walks', 10),
#             num_walk_heads=model_config.get('num_walk_heads', 4),
#             enable_neighbor_cooc=model_config.get('enable_neighbor_cooc', False),
#             use_explicit_co_gnn = model_config.get('use_explicit_co_gnn', True),
#             max_input_seq_length=model_config.get('max_input_seq_length', 32),
#             time_dim=model_config.get('time_dim', 172),
#             fixed_time_dim=model_config.get('fixed_time_dim', 20),
#             learning_rate=training_cfg.get('learning_rate', 1e-4),
#             weight_decay=training_cfg.get('weight_decay', 1e-5),
#         )
#         return HiCoSTdev1(config=hicostdev1_cfg)
    
#     @staticmethod
#     def _build_legacy_args(model_class, model_config: Dict, data_info: Dict) -> Dict:
#         """Build kwargs for legacy models using signature inspection."""
#         model_args = {
#             'num_nodes': data_info['num_nodes'],
#             'node_features': data_info.get('node_feat_dim', 0),
#             'edge_features_dim': data_info.get('edge_feat_dim', 172),
#         }
        
#         sig = inspect.signature(model_class.__init__)
#         valid_params = set(sig.parameters.keys()) - {'self'}
        
#         for key, value in model_config.items():
#             if key in valid_params:
#                 model_args[key] = value
#             else:
#                 logger.debug(f"Key '{key}' ignored for {model_class.__name__}")
        
#         return model_args
 
    
#     @staticmethod
#     def validate(model: torch.nn.Module, data_info: Dict):
#         """Validate model matches data specifications."""
#         num_nodes = data_info['num_nodes']
        
#         # Check num_nodes consistency
#         if hasattr(model, 'num_nodes'):
#             assert model.num_nodes == num_nodes, \
#                 f"model.num_nodes mismatch: {model.num_nodes} != {num_nodes}"
        
#         # Check hyperparameters preserved
#         if hasattr(model, 'hparams'):
#             if 'num_nodes' in model.hparams:
#                 assert model.hparams.num_nodes == num_nodes, \
#                     f"hparams.num_nodes mismatch: {model.hparams.num_nodes} != {num_nodes}"
        
#         # Check memory size for TGN
#         if hasattr(model, 'memory') and model.memory is not None:
#             mem_size = model.memory.memory.shape[0]
#             assert mem_size == num_nodes, \
#                 f"Memory size mismatch: {mem_size} != {num_nodes}"
        
#         logger.debug("Model validation passed")





    # @staticmethod
    # def create(config: Dict, data_info: Dict) -> torch.nn.Module:
    #     """Create model with proper parameter handling."""
    #     model_config = config['model'].copy()
    #     model_name = model_config.pop('name')
    #     print()

    #     # variant = config["model"].pop('variant', None)
        
    #     if model_name not in MODEL_REGISTRY:
    #         raise ValueError(f"Unknown model: {model_name}")
        
    #     model_class = MODEL_REGISTRY[model_name]
        
  
    #     # Base arguments from data (always required)
    #     model_args = {
    #         'num_nodes': data_info['num_nodes'],
    #         'node_features': data_info.get('node_feat_dim', 0),
    #         'edge_features_dim': data_info.get('edge_feat_dim', 172),
    #     }
        
    #     # Get the model's constructor signature
    #     sig = inspect.signature(model_class.__init__)
    #     valid_params = set(sig.parameters.keys()) - {'self'}
        
        
    #     # Add any keys from model_config that are valid
    #     for key, value in model_config.items():
    #         if key in valid_params:
    #             model_args[key] = value
    #         else:
    #             logger.warning(f"Key '{key}' in config is not accepted by {model_name}.__init__ and will be ignored.")
        
        
    #     print(f"Model args: {model_args}")
    #     model = model_class(**model_args)
        
    #     # Validation
    #     ModelFactory.validate(model, data_info)
        
    #     num_params = sum(p.numel() for p in model.parameters())
    #     logger.info(f"Created {model_name}: {num_params:,} parameters")
        
    #     return model















import torch 
from typing import Dict, List, Optional, Any
from loguru import logger

import inspect


from src.models.dygformer import DyGFormer
from src.models.tgn import TGN
from src.models.enhanced_tgn.variants.tgn_v2 import TGNv2
from src.models.enhanced_tgn.variants.tgn_v3 import TGNv3
from src.models.enhanced_tgn.variants.tgn_v4 import TGNv4
from src.models.enhanced_tgn.variants.tgn_v5 import TGNv5
from src.models.enhanced_tgn.variants.tgn_v6 import TGNv6
from src.models.enhanced_tgn.variants.tgn_v7 import TGNv7
from src.models.enhanced_tgn.variants.hicost import HiCoST
from src.models.enhanced_tgn.variants.hicostv2 import HiCoSTv2
from src.models.enhanced_tgn.variants.hicostv3 import HiCoSTv3


# Constants
MODEL_REGISTRY = {
    "DyGFormer": DyGFormer,
    "TGN": TGN,
    "TGNv2": TGNv2,
    "TGNv3": TGNv3,
    "TGNv4": TGNv4,
    "TGNv5": TGNv5,
    "TGNv6": TGNv6,
    "TGNv7": TGNv7,
    "HiCoST": HiCoST,
    "HiCoSTv2": HiCoSTv2,
    "HiCoSTv3": HiCoSTv3,
}



# ============================================================================
# MODEL
# ============================================================================

class ModelFactory:
    """Factory for creating and validating models."""
    
    @staticmethod
    def create(config: Dict, data_info: Dict) -> torch.nn.Module:
        """Create model with proper parameter handling."""
        model_config = config['model'].copy()
        model_name = model_config.pop('name')
        print()

        # variant = config["model"].pop('variant', None)
        
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = MODEL_REGISTRY[model_name]
        
        # Prepare arguments
        # model_args = {
        #     'num_nodes': data_info['num_nodes'],
        #     'node_features': data_info.get('node_feat_dim', 0),
        #     'hidden_dim': model_config.get('hidden_dim', 172),
        #     'time_encoding_dim': model_config.get('time_encoding_dim', 32),
        #     'memory_dim': model_config.get('memory_dim', 172),
        #     'message_dim': model_config.get('message_dim', 172),
        #     'edge_features_dim': data_info.get('edge_feat_dim', 172),
        #     'num_layers': model_config.get('num_layers', 1),
        #     'dropout': model_config.get('dropout', 0.1),
        #     'learning_rate': model_config.get('learning_rate', 1e-4),
        #     'weight_decay': model_config.get('weight_decay', 1e-5),
        #     'n_heads': model_config.get('n_heads', 2),
        #     'n_neighbors': model_config.get('n_neighbors', 10),
        #     'use_memory': model_config.get('use_memory', True),
        #     'embedding_module_type': model_config.get('embedding_module_type', 'graph_attention'),
        # }
        # model_args.update({
        #     'use_hct': model_config.get('use_hct', True),
        #     'use_st_ode': model_config.get('use_st_ode', True),
        #     # Add any other TGNv6‑specific parameters here
        # })
        
        
        # Base arguments from data (always required)
        model_args = {
            'num_nodes': data_info['num_nodes'],
            'node_features': data_info.get('node_feat_dim', 0),
            'edge_features_dim': data_info.get('edge_feat_dim', 172),
        }
        
        # Get the model's constructor signature
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        
        # Add any keys from model_config that are valid
        for key, value in model_config.items():
            if key in valid_params:
                model_args[key] = value
            else:
                logger.warning(f"Key '{key}' in config is not accepted by {model_name}.__init__ and will be ignored.")
        
        
        print(f"Model args: {model_args}")
        model = model_class(**model_args)
        
        # Validation
        ModelFactory.validate(model, data_info)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Created {model_name}: {num_params:,} parameters")
        
        return model
    
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















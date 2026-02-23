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

# from src.models.enhanced_tgn.base_enhance_tgn import BaseEnhancedTGN




# Constants
MODEL_REGISTRY = {
    "DyGFormer": DyGFormer,
    "TGN": TGN,
    "TGNv2": TGNv2,
    "TGNv3": TGNv3,
    "TGNv4": TGNv4,
    "TGNv5": TGNv5,
}



class ModelFactory:
    """Factory for creating and validating models."""
    @staticmethod
    def create(config: Dict, data_info: Dict) -> torch.nn.Module:
        """Create model with proper parameter handling."""
        model_config = config['model'].copy()
        model_name = model_config.pop('name')
        variant = model_config.pop('variant', 'default')
        
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = MODEL_REGISTRY[model_name]
        
        # Merge data_info with model_config (prefixed to avoid collisions)
        # Assume data_info keys like 'num_nodes' are directly usable.
        combined = {**model_config, **data_info}
        
        # Filter to valid constructor arguments
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_args = {k: v for k, v in combined.items() if k in valid_params}
        
        # Apply variant overrides if defined by the model
        variant_overrides = ModelFactory._get_variant_overrides(model_class, variant)
        filtered_args.update(variant_overrides)
        
        # Create model
        model = model_class(**filtered_args)
        
        # Log creation with variant info
        variants_attr = getattr(model_class, 'VARIANTS', {})
        if variant in variants_attr:
            logger.info(f"Created {model_name} (variant: {variant}): {sum(p.numel() for p in model.parameters()):,} params")
        else:
            logger.warning(f"Created {model_name} with unknown variant '{variant}'; using base configuration.")
        
        return model
    
    @staticmethod
    def _get_variant_overrides(model_class, variant: str) -> Dict:
        """Return variant-specific parameter overrides, if any."""
        variants = getattr(model_class, 'VARIANTS', {})
        if variant not in variants and variant != 'default':
            available = list(variants.keys()) if variants else ['default']
            logger.warning(f"Variant '{variant}' not defined for {model_class.__name__}. Available: {available}. Ignoring.")
        return variants.get(variant, {})












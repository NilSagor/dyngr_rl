from typing import Dict, Optional

def _validate_and_fix_config(self):
        """Ensure config has all required sections and keys."""
        if 'experiment' not in self.base_config:
            self.base_config['experiment'] = {}
        if 'name' not in self.base_config['experiment']:
            self.base_config['experiment']['name'] = 'sensitivity_study'

def _clean_config_none_values(cfg: Dict) -> Dict:
    """Replace None values with empty dicts for nested sections."""
    if not isinstance(cfg, dict):
        return cfg
    cleaned = {}
    for k, v in cfg.items():
        if v is None:
            # Common sections that should be dicts
            if k in ['model', 'training', 'data', 'experiment', 'logging', 'hardware']:
                cleaned[k] = {}
            else:
                cleaned[k] = v  # Keep None for leaf values
        elif isinstance(v, dict):
            cleaned[k] = _clean_config_none_values(v)
        else:
            cleaned[k] = v
    return cleaned

def _ensure_dict(config: Dict, key: str, defaults: Optional[Dict] = None) -> Dict:
    """Ensure config[key] is a dict, with optional default values."""
    if key not in config or config[key] is None:
        config[key] = {}
    if not isinstance(config[key], dict):
        raise ValueError(f"config['{key}'] should be dict, got {type(config[key]).__name__}")
    
    # Apply defaults for missing keys
    if defaults:
        for k, v in defaults.items():
            if k not in config[key]:
                config[key][k] = v
    
    return config[key]



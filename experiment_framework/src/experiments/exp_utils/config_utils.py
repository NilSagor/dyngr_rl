# src/experiments/exp_utils/config_utils.py
import hashlib
import string
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_config_placeholders(config: Dict, context: Optional[Dict] = None) -> Dict:
    if context is None:
        context = config

    def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def replace_value(value):
        if isinstance(value, str) and '${' in value:
            try:
                template = string.Template(value.replace('${', '$'))
                return template.substitute(flatten_dict(context))
            except (KeyError, ValueError):
                pass
        return value

    def recurse(obj):
        if isinstance(obj, dict):
            return {k: recurse(replace_value(v)) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        return replace_value(obj)

    return recurse(config)


def apply_overrides(config: Dict, overrides: List[str]) -> Dict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value format: {override}")
        key, value = override.split("=", 1)
        keys = key.split('.')

        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        try:
            import ast
            current[keys[-1]] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            current[keys[-1]] = value

    return config


def build_experiment_config(config: Dict, seed: int) -> Dict:
    model_name = config['model']['name']
    eval_type = config['data']['evaluation_type']
    neg_sample = config['data']['negative_sampling_strategy']
    dataset = config['data']['dataset']

    config_str = str(sorted(config['model'].items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]

    exp_name = f"{model_name}_{eval_type}_{neg_sample}"

    config['experiment'].update({
        'name': exp_name,
        'description': f"{model_name} on {dataset} | {eval_type} | {neg_sample}",
        'seed': seed,
    })

    config['logging'].update({
        'log_dir': str(DEFAULT_LOG_DIR / exp_name),
        'checkpoint_dir': str(DEFAULT_CHECKPOINT_DIR / f"{exp_name}_{config_hash}"),
    })

    return config
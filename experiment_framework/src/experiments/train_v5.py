# experiments/train_v5.py
"""
Unified training entry point using runner architecture.
Supports TGN, DyGFormer, HiCoST, TAWRMAC, etc.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
import yaml


from src.experiments.exp_utils.config_utils import (
    load_config,
    resolve_config_placeholders,
    apply_overrides,
    build_experiment_config,
)
from src.experiments.runner import get_runner

# from src.experiments.runner.tgn_runner import TGNRunner
# from src.experiments.runner.hicost_runner import HiCoSTRunner
# from src.experiments.runner.tawrmac_runner import TAWRMACRunner
# from src.experiments.runner.hicostdev1_runner import HiCoSTdev1Runner



# RUNNER_REGISTRY = {
#     "TAWRMACv1": TAWRMACRunner,
#     "HiCoSTdev1": HiCoSTdev1Runner,
#     "HiCoSTdev2": HiCoSTdev1Runner,   # explicit mapping
#     "TGN": TGNRunner,
#     # ... add others
# }

# def get_runner(model_name: str):
#     if model_name not in RUNNER_REGISTRY:
#         raise ValueError(f"No runner registered for model: {model_name}")
#     return RUNNER_REGISTRY[model_name]



# def get_runner(model_name: str):
#     """Return appropriate runner class based on model name."""
#     if model_name.startswith('TAWRMAC'):
#         return TAWRMACRunner
#     # elif model_name.startswith('HiCoST'):
#     #     return HiCoSTRunner
#     elif model_name.startswith('HiCoSTdev1'):
#         return HiCoSTdev1Runner
#     elif model_name.startswith('HiCoSTdev2'):
#         return HiCoSTdev1Runner
#     elif model_name in ['TGN', 'TGNv2', 'TGNv3', 'TGNv4', 'TGNv5', 'TGNv6', 'TGNv7', 'DyGFormer']:
#         return TGNRunner
#     else:
#         raise ValueError(f"No runner defined for model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Train temporal graph models')
    parser.add_argument('--config', '-c', required=True, help='Config file path')
    parser.add_argument('--seeds', '-s', type=int, nargs='+', help='Random seeds')
    parser.add_argument('--override', '-o', nargs='*', default=[], help='Config overrides')
    args = parser.parse_args()

    # Load and prepare config
    config = load_config(args.config)
    config = resolve_config_placeholders(config)
    config = apply_overrides(config, args.override)

    seeds = args.seeds if args.seeds else [config['experiment'].get('seed', 42)]

    for seed in seeds:
        config = build_experiment_config(config, seed)

        log_path = Path(config['logging']['log_dir']) / 'train.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_path), rotation="10 MB", level="INFO")

        # Get runner class and execute
        runner_class = get_runner(config['model']['name'])        
        runner = runner_class(config)

        try:
            results = runner.run()
        except Exception as e:
            logger.exception(f"Training failed with seed {seed}")
            raise


if __name__ == "__main__":
    main()
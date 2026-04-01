# src/experiments/main_sensitivity.py
import argparse
import yaml
import sys
from pathlib import Path
from loguru import logger
from copy import deepcopy

# Add project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.sensitivity_analyzer import SensitivityAnalyzer
from src.utils.general_utils import set_seed

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _get_nested_param(config: dict, path: str):
    """Helper to get nested param value."""
    keys = path.split('.')
    curr = config
    for k in keys:
        if k not in curr: return None
        curr = curr[k]
    return curr

def main():
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to sensitivity_config.yaml")
    parser.add_argument("--study", type=str, required=True, choices=["all", "ode_method", "batch_size", "memory_dim", "walk_length", "walk_distribution"], help="Which study to run")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42], help="Seeds for averaging")
    parser.add_argument("--output", type=str, default="results/sensitivity_full", help="Output directory")
    parser.add_argument("--filter", type=str, nargs='+', default=None, 
                        help="Filter specific configs by name (e.g., --filter balanced tawr_heavy). Only applies to studies with 'configs'")

    args = parser.parse_args()
    
    full_config = load_config(args.config)

    
    # DEBUG PRINT
    print("=== CONFIG STRUCTURE DEBUG ===")
    print(f"Top keys: {list(full_config.keys())}")
    if 'model' in full_config:
        print(f"Model keys: {list(full_config['model'].keys())}")
        print(f"Model Name: {full_config['model'].get('name', 'MISSING!')}")
    else:
        print("ERROR: 'model' key not found in config!")
    print("==============================")
    
    if 'model' not in full_config or 'name' not in full_config['model']:
        raise ValueError("Config file is structurally invalid. Check indentation.")


    studies = full_config.get('sensitivity_studies', {})
    
    # Initialize Analyzer
    analyzer = SensitivityAnalyzer(full_config, output_dir=args.output)
    
    logger.info(f"Starting Sensitivity Analysis: {args.study}")
    if args.filter:
        logger.info(f"Filtering configs: {args.filter}")
    
    if args.study == "all":
        for name, spec in studies.items():
            if 'values' in spec or 'configs' in spec:
                logger.info(f"Running study: {name}")
                analyzer.run_study_from_spec(name, spec, seeds=args.seeds)
    else:
        if args.study not in studies:
            raise ValueError(f"Study '{args.study}' not found. Available: {list(studies.keys())}")
        
        spec = studies[args.study]
        analyzer.run_study_from_spec(args.study, spec, seeds=args.seeds)

    logger.info("All sensitivity studies completed.")

if __name__ == "__main__":
    logger.add("sensitivity.log", rotation="10 MB", level="INFO")
    main()
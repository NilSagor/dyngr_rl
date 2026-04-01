# src/experiments/main_sensitivity.py
import argparse
import yaml
import sys
from pathlib import Path
from loguru import logger
from copy import deepcopy
import pandas as pd

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

def get_top_configs_from_results(results_path: Path, study_name: str, metric: str = 'test_ap', top_k: int = 2):
    """
    Automatically select top-k configs from previous results.
    For explicit config studies, param_name IS the config name.
    """
    if not results_path.exists():
        logger.warning(f"No existing results found at {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    
    # DEBUG: Show what we have
    logger.info(f"CSV columns: {df.columns.tolist()}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Unique configs: {df['param_name'].unique().tolist()}")
    
    # For explicit config studies, param_name contains the config name directly
    # Filter successful runs (no error or error is NaN/empty)
    study_df = df.copy()
    
    if 'error' in study_df.columns:
        study_df = study_df[study_df['error'].isna() | (study_df['error'] == '')]
    
    # Filter valid metric values
    study_df = study_df[study_df[metric].notna()]
    
    if study_df.empty:
        logger.warning(f"No valid results with metric '{metric}'")
        return None
    
    # Group by config name (param_name) and calculate mean performance
    grouped = study_df.groupby('param_name')[metric].agg(['mean', 'std', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)
    
    logger.info(f"\nAll configurations ranked by {metric}:")
    for idx, (config_name, row) in enumerate(grouped.iterrows(), 1):
        std_val = row.get('std', 0) if pd.notna(row.get('std', 0)) else 0
        count = int(row['count'])
        marker = " ★" if idx <= top_k else ""
        logger.info(f"  {idx}. {config_name}: {row['mean']:.4f} ± {std_val:.4f} (n={count}){marker}")
    
    # Return top-k config names
    top_configs = grouped.head(top_k).index.tolist()
    logger.info(f"\nSelected top-{top_k}: {top_configs}")
    
    return top_configs

def main():
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to sensitivity_config.yaml")
    parser.add_argument("--study", type=str, 
                        required=True, 
                        choices=["all", "ode_method", "batch_size", "memory_dim", "walk_length", "walk_distribution"], 
                        help="Which study to run")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42], help="Seeds for averaging")
    parser.add_argument("--output", type=str, default="results/sensitivity_full", help="Output directory")
    # parser.add_argument("--filter", type=str, nargs='+', default=None, 
    #                     help="Filter specific configs by name (e.g., --filter balanced tawr_heavy). Only applies to studies with 'configs'")

    # Filter options
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--filter", type=str, nargs='+', default=None, 
                              help="Filter specific configs by name (e.g., --filter balanced tawr_heavy)")
    filter_group.add_argument("--top-k", type=int, default=None, 
                              help="Automatically select top-k configs from previous results")
    filter_group.add_argument("--metric", type=str, default='test_ap', 
                              help="Metric to use for --top-k ranking (default: test_ap)")
    
    # Performance-based filtering
    parser.add_argument("--min-performance", type=float, default=None,
                        help="Only run configs with performance >= threshold (e.g., 0.75)")
    parser.add_argument("--max-performance", type=float, default=None,
                        help="Only run configs with performance <= threshold")
    
    
    args = parser.parse_args()
    
    full_config = load_config(args.config)
    output_dir = Path(args.output)
    
    # Auto-determine filter if --top-k specified
    config_filter = args.filter

    if args.top_k:
        summary_path = output_dir / "sensitivity_summary.csv"
        
        auto_filter = get_top_configs_from_results(
            summary_path, 
            study_name=args.study,  # Not used for matching, just for logging
            metric=args.metric,
            top_k=args.top_k
        )
        if auto_filter:
            config_filter = auto_filter
        else:
            logger.warning("Could not auto-select top configs, running all")
    

    # Performance threshold filtering
    performance_filter = None
    if args.min_performance or args.max_performance:
        summary_path = output_dir / "sensitivity_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            study_df = df[df['param_name'] == args.study].copy()
            
            # Handle missing 'error' column
            if 'error' in study_df.columns:
                study_df = study_df[study_df['error'].isna()]
            
            study_df = study_df[study_df[args.metric].notna()]
            
            # Apply thresholds
            mask = pd.Series([True] * len(study_df), index=study_df.index)
            if args.min_performance:
                mask = mask & (study_df[args.metric] >= args.min_performance)
            if args.max_performance:
                mask = mask & (study_df[args.metric] <= args.max_performance)
            
            filtered = study_df[mask]
            if not filtered.empty:
                performance_filter = filtered['param_value'].unique().tolist()
                logger.info(f"Performance filter: {len(performance_filter)} configs meet criteria")
    
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
    if config_filter:
        logger.info(f"Name filter: {config_filter}")
    if performance_filter:
        logger.info(f"Performance filter applied: {len(performance_filter)} configs")
    
    if args.study == "all":
        for name, spec in studies.items():
            if 'values' in spec or 'configs' in spec:
                logger.info(f"Running study: {name}")
                analyzer.run_study_from_spec(
                    name, spec, 
                    seeds=args.seeds, 
                    config_filter=config_filter,
                    performance_filter=performance_filter
                )
    else:
        if args.study not in studies:
            raise ValueError(f"Study '{args.study}' not found. Available: {list(studies.keys())}")
        
        spec = studies[args.study]
        analyzer.run_study_from_spec(
            args.study, spec, 
            seeds=args.seeds,
            config_filter=config_filter,
            performance_filter=performance_filter
        )

    logger.info("All sensitivity studies completed.")


if __name__ == "__main__":
    logger.add("sensitivity.log", rotation="10 MB", level="INFO")
    main()
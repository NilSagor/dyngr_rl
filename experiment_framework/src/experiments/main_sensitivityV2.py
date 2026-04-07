# src/experiments/main_sensitivityV2.py
import argparse
import yaml
import sys
import pandas as pd
from pathlib import Path
from loguru import logger


# Add project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.experiments.sensitivity_analyzerV2 import SensitivityAnalyzerV2


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_top_configs_from_results(results_path: Path, study_name: str, metric: str = 'test_ap', top_k: int = 2):    
    if not results_path.exists():
        logger.warning(f"No existing results found at {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    
    # checking csv 
    logger.info(f"CSV columns: {df.columns.tolist()}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Unique configs: {df['param_name'].unique().tolist()}")
        
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
    parser.add_argument("--output", type=str, default="results/sensitivity_fullv2", help="Output directory")
    parser.add_argument("--filter", type=str, nargs='+', help="Only run specific configs")
    parser.add_argument("--top-k", type=int, help="Auto-select top-k from previous results")
    parser.add_argument("--metric", type=str, default='test_ap')
    
    
    args = parser.parse_args()
    
    full_config = load_config(args.config)
    output_dir = Path(args.output)
    
    # Auto-determine filter if --top-k specified
    config_filter = args.filter

    if args.top_k:
        summary_path = output_dir / "sensitivity_summary.csv"
        
        auto_filter = get_top_configs_from_results(
            summary_path, 
            study_name=args.study,
            metric=args.metric,
            top_k=args.top_k
        )
        if auto_filter:
            config_filter = auto_filter
        else:            
            logger.warning(f"Could not auto-select top configs, running all {config_filter}")
    

    # Load studies
    studies = full_config.get('sensitivity_studies', {})
    if args.study != "all" and args.study not in studies:
        raise ValueError(f"Study '{args.study}' not found. Available: {list(studies.keys())}")
    
    # Run
    analyzer = SensitivityAnalyzerV2(full_config, output_dir)
    
    study_list = studies.items() if args.study == "all" else [(args.study, studies[args.study])]
    
    for name, spec in study_list:        
            
        if 'parameter' in spec and 'values' in spec:
            logger.info(f"Running study: {name}")
            analyzer.run_study(
                spec['parameter'], 
                spec['values'], 
                args.seeds,
                config_filter=config_filter,
                coupled_params=spec.get('coupled_params')
            )
        else:
            logger.warning(
                f"Study '{name}' missing 'parameter' or 'values'"
            )



if __name__ == "__main__":
    logger.add("sensitivity.log", rotation="10 MB", level="INFO")
    main()
"""
training Module version 0.4
Clean, modular training pipeline for temporal graph models.
"""

import os
import sys
from pathlib import Path

import argparse
import csv
import hashlib
import string
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
import yaml
import torch 
import numpy as np

# Setup path before other imports
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

from lightning.pytorch.callbacks import EarlyStopping

from src.utils.general_utils import set_seed, get_device
from src.datasets.continue_temporal.data_con_pipeline import DataPipeline
from src.experiments.exp_utils.model_factory import ModelFactory
from src.experiments.exp_utils.experiment_logger import ExperimentLogger
from src.experiments.exp_utils.trainer_setup import TrainerSetup
from src.experiments.exp_utils.analysis_callback import AnalysisCollector
from src.experiments.exp_utils.flops_calculator import FLOPsCalculator

import torch
torch.autograd.set_detect_anomaly(True)

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_config_placeholders(config: Dict, context: Optional[Dict] = None) -> Dict:
    """Resolve ${key.subkey} placeholders in config."""
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
    """Apply command-line overrides to config."""
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
    """Build final experiment configuration with derived fields."""
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


def validate_model_ready(model, pipeline):
    """Validate model has required components."""
    assert model.neighbor_finder is not None, "Neighbor finder not set"
    
    if hasattr(model, 'walk_sampler') and model.walk_sampler is not None:
        if hasattr(model.walk_sampler, 'edge_index'):
            assert model.walk_sampler.edge_index is not None, \
                "Walk sampler missing edge_index"
            assert model.walk_sampler.edge_time is not None, \
                "Walk sampler missing edge_time"
        else:
            logger.warning("Walk sampler may not be initialized - check set_graph() call")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_single_run(config: Dict) -> Dict[str, Any]:
    """Execute single training run."""
    start_time = datetime.now()
    seed = config['experiment']['seed']
    set_seed(seed)
    
    logger.info(f"{'='*50}")
    logger.info(f"Starting training: {config['experiment']['name']}")
    logger.info(f"Seed: {seed}")
    logger.info(f"{'='*50}")
    
    # Data pipeline
    pipeline = (DataPipeline(config)
        .load()
        .build_neighbor_finder()
        .build_samplers()
        .build_datasets()
        .build_loaders()
    )    

    # Extract training edges
    train_edges = pipeline.data['edges'][pipeline.data['train_mask']]
    train_times = pipeline.data['timestamps'][pipeline.data['train_mask']]

    if not isinstance(train_edges, torch.Tensor):
        train_edges = torch.tensor(train_edges)
    if not isinstance(train_times, torch.Tensor):
        train_times = torch.tensor(train_times)

    if train_edges.shape[0] != 2:
        train_edges = train_edges.T.contiguous()
        logger.info(f"Transposed train_edges to shape {train_edges.shape}")

    if train_times.dim() == 2 and train_times.shape[1] == 1:
        train_times = train_times.squeeze(1)
    
    pipeline.train_edges = train_edges
    pipeline.train_times = train_times
    
    # Model
    features = pipeline.get_features()
    model = ModelFactory.create(config, features)
    
    model.set_raw_features(features['node_features'], features['edge_features'])
    model.set_neighbor_finder(pipeline.neighbor_finder)
    model.set_graph(
        pipeline.neighbor_finder.edge_index,
        pipeline.neighbor_finder.edge_time
    )

    # Debug info
    if config['experiment'].get('debug', False):
        logger.debug(f"Embedding module: {hasattr(model, 'embedding_module') and model.embedding_module is not None}")
        logger.debug(f"Memory updater: {hasattr(model, 'memory_updater') and model.memory_updater is not None}")
        logger.debug(f"Neighbor finder: {pipeline.neighbor_finder is not None}")

    # After model creation in train_single_run
    logger.info(f"=== HiCoSTv3 Initialized Model Component Status ===")
    logger.info(f"SAM prototypes enabled: { model.hparams.get('use_prototype_attention', True)}")
    logger.info(f"HCT hierarchical enabled: {model.hparams.get('use_hct_hierarchical', True)}")
    logger.info(f"MRP gating enabled: { model.hparams.get('use_gated_refinement', True)}")
    logger.info(f"Multi-scale walks enabled: { model.hparams.get('use_multi_scale_walks', True)}")
    logger.info(f"Spectral Temporal ODE: { model.hparams.get('use_st_ode', True)}")
    logger.info(f"Hard Negative Mining: { model.hparams.get('use_hard_negative_mining', True)}")
    logger.info(f"Walk sampler initialized: {model.walk_sampler is not None}")
    if model.walk_sampler:
        logger.info(f"Walk sampler has edge_index: {hasattr(model.walk_sampler, 'edge_index')}")
    logger.info(f"================================")


    # Calculate FLOPs with dummy input
    if config['experiment'].get('debug', False):
        dummy_batch = {
            'src': torch.zeros(2, 128, dtype=torch.long),  # batch_size=128
            'dst': torch.zeros(2, 128, dtype=torch.long),
            'time': torch.zeros(128, dtype=torch.float),
            'edge_attr': torch.zeros(128, 172),  # edge_features_dim
            'n_id': torch.arange(128),
            'src_ptr': torch.arange(129),  # For batched graphs
            'dst_ptr': torch.arange(129),
        }
        
        # Move to GPU if available
        if torch.cuda.is_available():
            dummy_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                          for k, v in dummy_batch.items()}
            model = model.cuda()
            
        stats = FLOPsCalculator.print_summary(model, dummy_batch)
        
        # Log to tensorboard
        if hasattr(model, 'logger'):
            model.logger.experiment.add_scalar('model/total_gflops', stats['total_gflops'], 0)
            model.logger.experiment.add_scalar('model/total_params', 
                sum(p.numel() for p in model.parameters()), 0)              

    
    # Create analysis collector to retrieve data after training
    analysis_collector = AnalysisCollector()
    
    # Setup trainer with the callback
    trainer = TrainerSetup.create(config, callbacks=[analysis_collector])
    logger.info(f"Trainer max_epochs: {trainer.max_epochs}")
    logger.info(f"EarlyStopping patience: {[c for c in trainer.callbacks if isinstance(c, EarlyStopping)]}")
    
    # Train
    logger.info("Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=pipeline.loaders['train'],
        val_dataloaders=pipeline.loaders['val'],
    )

    # Test - Lightning handles checkpoint loading with ckpt_path='best'
    logger.info("Running evaluation with best checkpoint...")
    
    test_results = trainer.test(
        model=model,
        dataloaders=pipeline.loaders['test'],
        ckpt_path='best' if trainer.checkpoint_callback.best_model_path else None
    )
    
    # Collect co-occurrence if available
    cooccurrence_matrix = None
    if hasattr(model, 'get_cooccurrence'):
        try:
            cooccurrence_matrix = model.get_cooccurrence()
        except Exception as e:
            logger.debug(f"Could not collect co-occurrence: {e}")
    
    # Log results with analysis data
    training_time = (datetime.now() - start_time).total_seconds()
    exp_logger = ExperimentLogger(config['logging']['log_dir'])
    
    # FIX: Correct parameter names (was walk_stats repeated 4 times!)
    exp_logger.log(
        config=config, 
        test_results=test_results, 
        training_time=training_time, 
        model=model,
        walk_stats=analysis_collector.walk_stats,
        memory_trace=analysis_collector.memory_trace,
        ode_trajectory=analysis_collector.ode_trajectory,
        negative_stats=analysis_collector.negative_stats,
        cooccurrence_matrix=cooccurrence_matrix,
    )
    
    # Save checkpoint
    final_path = Path(config['logging']['checkpoint_dir']) / 'final_model.ckpt'
    final_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(final_path))
    
    logger.info(f"Training complete in {training_time:.1f}s")
    logger.info(f"Final model: {final_path}")
    
    return {
        'test_results': test_results,
        'training_time': training_time,
        'model': model,
    }


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
        
        try:
            results = train_single_run(config)
        except Exception as e:
            logger.exception(f"Training failed with seed {seed}")
            raise


if __name__ == "__main__":
    main()
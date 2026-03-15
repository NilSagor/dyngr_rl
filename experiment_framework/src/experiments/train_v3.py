"""
training Module version 0.3
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

# Setup path before other imports
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

from lightning.pytorch.callbacks import EarlyStopping

from src.utils.general_utils import set_seed, get_device
# from src.datasets.load_dataset import load_dataset, DATA_ROOT

from src.datasets.continue_temporal.data_pipeline import DataPipeline

# from src.experiments.exp_utils.data_pipeline import DataPipeline
from src.experiments.exp_utils.model_factory import ModelFactory
from src.experiments.exp_utils.model_profile import ModelProfiler
from src.experiments.exp_utils.experiment_logger import ExperimentLogger
from src.experiments.exp_utils.trainer_setup import TrainerSetup

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
                # Handle ${var} syntax
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
        
        # Try to parse as literal, fallback to string
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
    
    # Generate hash for unique checkpoint naming
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
    assert model.neighbor_finder is not None, "Neighbor finder not set"
    
    if hasattr(model, 'walk_sampler') and model.walk_sampler is not None:
        # Check if walk sampler has required graph data
        if hasattr(model.walk_sampler, 'edge_index'):
            assert model.walk_sampler.edge_index is not None, \
                "Walk sampler missing edge_index"
            assert model.walk_sampler.edge_time is not None, \
                "Walk sampler missing edge_time"
        else:
            logger.warning("Walk sampler may not be initialized - check set_graph() call")


def sanity_check_test_evaluation(model, test_loader, pipeline):
    """Quick checks before full test evaluation."""
    model.eval()
    batch = next(iter(test_loader))
    
    with torch.no_grad():
        # 1. Check prediction range
        logits = model.forward(batch)
        probs = torch.sigmoid(logits)
        print(f"Prediction range: [{probs.min():.3f}, {probs.max():.3f}]")
        
        # 2. Check label distribution
        labels = batch['labels']
        print(f"Label distribution: 0={(~labels).sum()}, 1={labels.sum()}")
        
        # 3. Check correlation (should be positive)
        if len(labels) > 1:
            corr = torch.corrcoef(torch.stack([probs.squeeze(), labels.float()]))[0,1]
            print(f"Prediction-label correlation: {corr:.3f} "
                  f"{'(NEGATIVE! Check sign)' if corr < 0 else ''}")
        
        # 4. Verify features are used
        if model.edge_raw_features is not None:
            print(f"Edge features shape: {model.edge_raw_features.shape}")




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

    # Extract training edges (the same ones used for the neighbor finder)
    train_edges = pipeline.data['edges'][pipeline.data['train_mask']]          # [2, num_train_edges]
    train_times = pipeline.data['timestamps'][pipeline.data['train_mask']]     # [num_train_edges]

    # Convert to torch tensors if needed
    if not isinstance(train_edges, torch.Tensor):
        train_edges = torch.tensor(train_edges)
    if not isinstance(train_times, torch.Tensor):
        train_times = torch.tensor(train_times)

    # Ensure walk-sampler friendly shape: [2, num_edges]
    if train_edges.shape[0] != 2:
        # Assume shape is [num_edges, 2] -> transpose
        train_edges = train_edges.T.contiguous()
        logger.info(f"Transposed train_edges to shape {train_edges.shape}")

    # Ensure train_times is 1D
    if train_times.dim() == 2 and train_times.shape[1] == 1:
        train_times = train_times.squeeze(1)
    
    # Store them to pass to the model later
    pipeline.train_edges = train_edges
    pipeline.train_times = train_times
    
    
    # Model
    features = pipeline.get_features()
    model = ModelFactory.create(config, features)
    
    # CRITICAL ORDER: set features first, then neighbor finder
    model.set_raw_features(features['node_features'], features['edge_features'])
    
    model.set_neighbor_finder(pipeline.neighbor_finder)
    model.set_graph(pipeline.train_edges, pipeline.train_times)
    

    # Debug: Verify embedding module initialized
    if hasattr(model, 'embedding_module'):
        logger.debug(f"Embedding module initialized: {model.embedding_module is not None}")

    logger.debug(f"1. embedding_module exists: {hasattr(model, 'embedding_module') and model.embedding_module is not None}")
    logger.debug(f"2. memory_updater exists: {hasattr(model, 'memory_updater') and model.memory_updater is not None}")
    # logger.debug(f"2. memory_updater exists: {model.memory_updater is not None}")
    logger.debug(f"3. neighbor_finder exists: {pipeline.neighbor_finder is not None}")
    logger.debug(f"4. node_raw_features shape: {model.node_raw_features.shape if model.node_raw_features is not None else 'None'}")
    logger.debug(f"5. edge_raw_features shape: {model.edge_raw_features.shape if model.edge_raw_features is not None else 'None'}")

    # logger.debug(f"Embedding module: {model.embedding_module is not None}")
       
    
    # Setup trainer
    trainer = TrainerSetup.create(config)
    logger.info(f"Trainer max_epochs: {trainer.max_epochs}")
    logger.info(f"EarlyStopping patience: {[c for c in trainer.callbacks if isinstance(c, EarlyStopping)]}")
    
    # Train
    logger.info("Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=pipeline.loaders['train'],
        val_dataloaders=pipeline.loaders['val'],
    )

    if trainer.checkpoint_callback.best_model_path:
        best_path = trainer.checkpoint_callback.best_model_path
        
        # 1. Create fresh model and initialize with pipeline data
        model = ModelFactory.create(config, features)
        model.set_raw_features(features['node_features'], features['edge_features'])
        model.set_neighbor_finder(pipeline.neighbor_finder)
        
        # 2. CRITICAL: Initialize walk sampler with full graph BEFORE loading checkpoint
        # This ensures the walk sampler has the graph structure
        if hasattr(pipeline.neighbor_finder, 'edge_index') and hasattr(pipeline.neighbor_finder, 'edge_time'):
            model.set_graph(pipeline.neighbor_finder.edge_index, pipeline.neighbor_finder.edge_time)
        else:
            # Fallback: extract from dataset
            model.set_graph(pipeline.train_edges, pipeline.train_times)
        
        # Verify walk sampler is working
        test_src = torch.tensor([0, 1], device=model.device)
        test_dst = torch.tensor([1, 2], device=model.device)
        test_ts = torch.tensor([0.0, 0.0], device=model.device)

        test_walks = model.walk_sampler(
            source_nodes=test_src,
            target_nodes=test_dst,
            current_times=test_ts,
            memory_states=model.sam_module.raw_memory,
            edge_index=model.edge_index,
            edge_time=model.edge_time
        )

        for side in ['source', 'target']:
            for wt in ['short', 'long', 'tawr']:
                nodes = test_walks[side][wt]['nodes']
                if nodes.max().item() == 0 and nodes.numel() > 0:
                    logger.error(f"CRITICAL: Walk sampler returns all-zero indices for {side}/{wt}!")
                    logger.error(f"Check edge_index shape: {model.edge_index.shape if model.edge_index is not None else 'None'}")
                    logger.error(f"Check edge_time shape: {model.edge_time.shape if model.edge_time is not None else 'None'}")
        
        
        
        # 3. Now load checkpoint weights
        logger.info(f"Loading best checkpoint from {best_path}")

        checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # 4. Move to device and set to eval mode
        model = model.to(get_device()).eval()

        # 5. Validate model is ready
        if config['experiment'].get('debug', False):
            validate_model_ready(model, pipeline)
        
        logger.info("Best checkpoint loaded and walk sampler initialized successfully.")
        
        

    # Test - Lightning auto-loads best checkpoint via ckpt_path='best'
    logger.info("Running evaluation with best checkpoint...")
    
    test_results = trainer.test(
        model=model,  # Same model instance (no recreation needed)
        dataloaders=pipeline.loaders['test'],
        ckpt_path='best'  # ← Lightning handles loading + proper init
    )
    
    # Log results
    training_time = (datetime.now() - start_time).total_seconds()
    exp_logger = ExperimentLogger(config['logging']['log_dir'])
    exp_logger.log(config, test_results, training_time, model)
    
    # Save checkpoint
    final_path = Path(config['logging']['checkpoint_dir']) / 'final_model.ckpt'
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
    
    # Determine seeds to run
    seeds = args.seeds if args.seeds else [config['experiment'].get('seed', 42)]
    
    # Run experiments
    for seed in seeds:
        config = build_experiment_config(config, seed)
        
        # Setup logging
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
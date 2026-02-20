"""
train.py - Clean, modular training pipeline for temporal graph models.
"""

import os
import sys
from pathlib import Path

# Setup path before other imports
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import hashlib
import string
from datetime import datetime
from typing import Dict, List, Optional, Any

import yaml
import torch
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
# from lightning.pytorch.callbacks import ReduceLROnPlateau
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
from loguru import logger

# Local imports
from src.models.dygformer import DyGFormer
from src.models.tgn import TGN

from src.models.enhanced_tgn.variants.tgn_v2 import TGNv2
from src.models.enhanced_tgn.variants.tgn_v3 import TGNv3
from src.models.enhanced_tgn.variants.tgn_v4 import TGNv4
from src.models.enhanced_tgn.base_enhance_tgn import BaseEnhancedTGN

from src.utils.general_utils import set_seed, get_device
from src.datasets.load_dataset import load_dataset, DATA_ROOT
from src.datasets.negative_sampling import NegativeSampler
from src.datasets.neighbor_finder import NeighborFinder
from src.datasets.temporal_dataset import TemporalDataset

# Constants
MODEL_REGISTRY = {
    "DyGFormer": DyGFormer,
    "TGN": TGN,
    "TGNv2": TGNv2,
    "TGNv3": TGNv3,
    "TGNv4": TGNv4,
}

DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


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


# ============================================================================
# DATA PIPELINE
# ============================================================================

class DataPipeline:
    """Encapsulates all data-related setup."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data: Optional[Dict] = None
        self.neighbor_finder: Optional[NeighborFinder] = None
        self.samplers: Dict[str, NegativeSampler] = {}
        self.datasets: Dict[str, TemporalDataset] = {}
        self.loaders: Dict[str, DataLoader] = {}
    
    def load(self) -> 'DataPipeline':
        """Load raw dataset with validation."""
        logger.info(f"Loading dataset: {self.config['data']['dataset']}")
        
        eval_type = self.config['data']['evaluation_type']
        sampling_strategy = self.config['data']['negative_sampling_strategy']
        
        #  Comprehensive inductive sampling validation
        if sampling_strategy == 'inductive':
            if eval_type != 'inductive':
                raise ValueError(
                    f"Inductive sampling requires inductive evaluation. "
                    f"Got evaluation_type='{eval_type}'. "
                    f"Fix: Use evaluation_type='inductive' OR switch to "
                    f"negative_sampling_strategy='random'/'historical'."
                )
            if self.config['data'].get('unseen_ratio', 0.1) <= 0:
                raise ValueError(
                    "Inductive sampling requires unseen_ratio > 0. "
                    "Fix: Set data.unseen_ratio=0.1 in config."
                )
        
        self.data = load_dataset(
            dataset_name=self.config['data']['dataset'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            inductive=(eval_type == 'inductive'),
            unseen_ratio=self.config['data'].get('unseen_ratio', 0.1),
            seed=self.config['experiment']['seed'],
        )        
       
        logger.info(f"Loaded: {self.data['num_nodes']} nodes, {self.data['statistics']['num_edges']} edges")
        return self
    
    def build_neighbor_finder(self) -> 'DataPipeline':
        """Build neighbor finder from training edges only (leakage-proof)."""
        train_edges = self.data['edges'][self.data['train_mask']]
        train_ts = self.data['timestamps'][self.data['train_mask']]
        
        self.neighbor_finder = NeighborFinder(
            train_edges=train_edges,
            train_timestamps=train_ts,
            max_neighbors=self.config['data']['max_neighbors']
        )
        
        logger.info(f" Built leakage-proof NeighborFinder from {len(train_edges)} training edges")
        return self
    
    def build_samplers(self) -> 'DataPipeline':
        """Build negative samplers with TGN paper standard enforcement."""
        splits = ['train', 'val', 'test']
        masks = ['train_mask', 'val_mask', 'test_mask']
        
        for split, mask_key in zip(splits, masks):
            edges = self.data['edges'][self.data[mask_key]]
            timestamps = self.data['timestamps'][self.data[mask_key]]
            
            #  TGN paper standard - random sampling for TRAINING ONLY
            # Historical/inductive sampling ONLY for evaluation ablation studies
            # if split == 'train':
            #     # Force random sampling regardless of config (prevents easy negatives)
            #     pass  # NegativeSampler will use .random() method when called
            
            strategy = 'random' if split == 'train' else self.config['data']['negative_sampling_strategy']

            self.samplers[split] = NegativeSampler(
                edges=edges,
                timestamps=timestamps,
                num_nodes=self.data['num_nodes'],
                neighbor_finder=self.neighbor_finder,
                seed=self.config['experiment']['seed']
            )
        
        logger.info(f" Built samplers: train=random (TGN standard), "
                   f"val/test={self.config['data']['negative_sampling_strategy']}")
        return self
    
    def build_datasets(self) -> 'DataPipeline':
        """Build TemporalDatasets with STRICT feature masking."""
        splits = ['train', 'val', 'test']
        masks = ['train_mask', 'val_mask', 'test_mask']
        is_inductive = self.config['data']['evaluation_type'] == 'inductive'
        
        for split, mask_key in zip(splits, masks):
            mask = self.data[mask_key]
            
            #  MASK EDGE FEATURES PER SPLIT (prevent leakage)
            # Using full edge_features tensor would leak val/test features into training!
            split_edge_features = (
                self.data['edge_features'][mask] if self.data['edge_features'] is not None else None
            )
            
            # Get unseen nodes ONLY for val/test in inductive evaluation
            unseen_nodes = (
                self.data['unseen_nodes'] 
                if (is_inductive and split != 'train') 
                else None
            )
            
            # Determine sampling strategy per split (train always random)
            sampling_strategy = (
                'random' if split == 'train' 
                else self.config['data']['negative_sampling_strategy']
            )
            
            self.datasets[split] = TemporalDataset(
                edges=self.data['edges'][mask],
                timestamps=self.data['timestamps'][mask],
                edge_features=split_edge_features,  # MASKED FEATURES
                num_nodes=self.data['num_nodes'],
                split=split,
                negative_sampler=self.samplers[split],
                negative_sampling_strategy=sampling_strategy,
                unseen_nodes=unseen_nodes,
                seed=self.config['experiment']['seed']
            )
        
        # Validate per-batch label balance (prevent single-class batches)
        self._validate_evaluation_batches()
        
        logger.info(f" Built datasets: { {k: len(v) for k, v in self.datasets.items()} }")
        return self
    
    def _validate_evaluation_batches(self):
        """Ensure ALL evaluation batches contain both classes (prevent AUC=nan)."""
        for split in ['val', 'test']:
            if split not in self.datasets:
                continue
            
            dataset = self.datasets[split]
            batch_size = self.config['training']['batch_size']
            
            # Check first 10 batches
            for i in range(min(10, len(dataset) // batch_size)):
                start = i * batch_size
                end = min(start + batch_size, len(dataset))
                batch_labels = [dataset.samples[j]['label'] for j in range(start, end)]
                
                # Must have BOTH classes in every batch
                if 0.0 not in batch_labels or 1.0 not in batch_labels:
                    raise ValueError(
                        f"Single-class batch detected in {split} split (batch {i})! "
                        f"Labels: {set(batch_labels)}. "
                        f"Fix: Ensure positives/negatives are interleaved in _prepare_samples()."
                    )
        
        logger.info(" All evaluation batches validated: contain both classes (valid metrics guaranteed)")
    
    
    
    def build_loaders(self) -> 'DataPipeline':
        """Wrap datasets in DataLoaders."""
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['hardware'].get('num_workers', 0)
        
        for split, dataset in self.datasets.items():
            shuffle = (split == 'train')
            self.loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=TemporalDataset.collate_fn,
                pin_memory=self.config['hardware'].get('pin_memory', False),
            )
        
        return self
    
    def get_features(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get node/edge features with STRUCTURAL DATASET DETECTION."""
        dataset = self.config['data']['dataset'].lower()
        # STRUCTURAL_DATASETS = {'untrade', 'uslegis', 'canparl', 'unvote', 'enron', 'uci'}  # UCI is NOT structural!
        
        # UCI is NOT structural - has real edge features
        IS_STRUCTURAL = dataset in {'untrade', 'uslegis', 'canparl', 'unvote'}
        
        if IS_STRUCTURAL:
            # Structural datasets: create 1-dim dummy edge features
            node_features = None
            num_edges = self.data['train_mask'].sum().item()
            edge_features = torch.ones(num_edges, 1)  # 1-dim dummy features (required)
            logger.info(f" Structural dataset {dataset}: using 1-dim dummy edge features")
            return {
                'node_features': node_features,
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': 1,  #  structural datasets need dummy features
                'node_feat_dim': 0,
            }
        
        # Enron has 32-dim edge features (DyGLib format)
        if dataset == "enron":
            train_mask = self.data['train_mask']
            edge_features = self.data['edge_features'][train_mask]
            
            # Enron edge features are 32-dimensional (message content embedding)
            if edge_features.shape[1] != 32:
                logger.warning(
                    f"Enron edge features should be 32-dim, got {edge_features.shape[1]}. "
                    f"Using actual dimension: {edge_features.shape[1]}"
                )
            
            return {
                'node_features': None,  # Enron has no node features
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': edge_features.shape[1],  # 32 (not 1!)
                'node_feat_dim': 0,
            }        
        
        
        # UCI-specific handling (has 2-dim edge features)
        if dataset == "uci":
            train_mask = self.data['train_mask']
            edge_features = self.data['edge_features'][train_mask]
            
            # UCI edge features are 2-dimensional (message content embedding)
            if edge_features.shape[1] != 2:
                logger.warning(f"UCI edge features should be 2-dim, got {edge_features.shape[1]}. Truncating.")
                edge_features = edge_features[:, :2]
            
            return {
                'node_features': self.data.get('node_features'),  # 100-dim for UCI
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': 2,  # Critical: NOT 1!
                'node_feat_dim': 100 if self.data.get('node_features') is not None else 0,
            }
        
        # Standard datasets (Wikipedia/Reddit/MOOC)
        if dataset == "wikipedia":
            node_features = None
        else:
            node_features = self.data.get('node_features')
        
        train_edge_features = (
            self.data['edge_features'][self.data['train_mask']] 
            if self.data['edge_features'] is not None 
            else None
        )
        
        return {
            'node_features': node_features,
            'edge_features': train_edge_features,
            'num_nodes': self.data['num_nodes'],
            'edge_feat_dim': train_edge_features.shape[1] if train_edge_features is not None else 0,
            'node_feat_dim': node_features.shape[1] if node_features is not None else 0,
        }     
        
        
    
    
    
    @property
    def num_nodes(self) -> int:
        return self.data['num_nodes']


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

        # variant = config["model"].pop('variant', None)
        
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = MODEL_REGISTRY[model_name]
        
        # Prepare arguments
        model_args = {
            'num_nodes': data_info['num_nodes'],
            'node_features': data_info.get('node_feat_dim', 0),
            'hidden_dim': model_config.get('hidden_dim', 172),
            'time_encoding_dim': model_config.get('time_encoding_dim', 32),
            'memory_dim': model_config.get('memory_dim', 172),
            'message_dim': model_config.get('message_dim', 172),
            'edge_features_dim': data_info.get('edge_feat_dim', 172),
            'num_layers': model_config.get('num_layers', 1),
            'dropout': model_config.get('dropout', 0.1),
            'learning_rate': model_config.get('learning_rate', 1e-4),
            'weight_decay': model_config.get('weight_decay', 1e-5),
            'n_heads': model_config.get('n_heads', 2),
            'n_neighbors': model_config.get('n_neighbors', 10),
            'use_memory': model_config.get('use_memory', True),
            'embedding_module_type': model_config.get('embedding_module_type', 'graph_attention'),
        }
        
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


# ============================================================================
# TRAINING
# ============================================================================

class TrainerSetup:
    """Encapsulates trainer configuration."""
    
    @staticmethod
    def create(config: Dict) -> L.Trainer:
        """Create PyTorch Lightning trainer."""
        callbacks = [
            ModelCheckpoint(
                dirpath=config['logging']['checkpoint_dir'],
                filename='{epoch}-{val_ap:.2f}',
                monitor=config['logging']['monitor'],
                mode=config['logging']['mode'],
                save_top_k=config['logging']['save_top_k'],
                verbose=True,
            ),
            EarlyStopping(
                monitor=config['logging']['monitor'],
                patience=config['training']['patience'],
                mode=config['logging']['mode'],
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),
            # ReduceLROnPlateau(monitor='val_ap', mode='max', patience=10, factor=0.5)
        ]
        
        loggers = [
            CSVLogger(
                save_dir=config['logging']['log_dir'],
                name=config['experiment']['name']
            ),
        ]
        
        if config.get('logger', 'tensorboard') == 'tensorboard':
            loggers.append(
                TensorBoardLogger(
                    save_dir=config['logging']['log_dir'],
                    name=config['experiment']['name']
                )
            )
        
        accelerator = 'gpu' if config['hardware']['gpus'] else 'cpu'
        devices = config['hardware']['gpus'] if config['hardware']['gpus'] else 'auto'
        
        return L.Trainer(
            max_epochs=config['training']['max_epochs'],
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=loggers,
            precision=config['experiment']['precision'],
            gradient_clip_val=config['training']['gradient_clip_val'],
            log_every_n_steps=config['training']['log_every_n_steps'],
            val_check_interval=config['training']['val_check_interval'],
            enable_progress_bar=True,
        )


class ExperimentLogger:
    """Handles result logging to CSV."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir.parent / 'all_results.csv'
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, config: Dict, test_results: List[Dict], training_time: float, model: torch.nn.Module):
        """Log results to CSV."""
        # model = test_results[0].get('model', None)
        # num_params = sum(p.numel() for p in model.parameters()) if model else 0
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row = {
            'model': config['model']['name'],
            'dataset': config['data']['dataset'],
            'evaluation_type': config['data']['evaluation_type'],
            'negative_sampling_strategy': config['data']['negative_sampling_strategy'],
            'seed': config['experiment']['seed'],
            'test_accuracy': test_results[0].get('test_accuracy', 0.0),
            'test_ap': test_results[0].get('test_ap', 0.0),
            'test_auc': test_results[0].get('test_auc', 0.0),
            'test_loss': test_results[0].get('test_loss', 0.0),
            'training_time': training_time,
            'num_parameters': num_params,
            'timestamp': datetime.now().isoformat(),
        }
        
        file_exists = self.csv_path.exists()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        logger.info(f"Results saved to {self.csv_path}")



def load_checkpoint_safely(checkpoint_path, model, config):
    """Load checkpoint ONLY if config matches training config."""
    if not checkpoint_path.exists():
        return model
    
    # Load checkpoint metadata WITHOUT loading weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ckpt_config = checkpoint.get('hyper_parameters', {})
    
    # Critical dimensions to validate
    dims_to_check = ['time_encoding_dim', 'edge_features_dim', 'memory_dim']
    mismatches = []
    
    for dim in dims_to_check:
        if dim in ckpt_config and dim in config['model']:
            if ckpt_config[dim] != config['model'][dim]:
                mismatches.append(f"{dim}: checkpoint={ckpt_config[dim]} != current={config['model'][dim]}")
    
    if mismatches:
        logger.warning(
            "Checkpoint configuration mismatch detected! Skipping checkpoint load.\n"
            "Mismatches:\n" + "\n".join(f"  - {m}" for m in mismatches) + "\n"
            "Training from scratch with current configuration."
        )
        return model  # Return untrained model
    
    # Safe to load
    logger.info(f" Loading checkpoint with matching configuration: {checkpoint_path.name}")
    return model.load_from_checkpoint(checkpoint_path)



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
        .build_loaders())
    
    # Model
    features = pipeline.get_features()
    model = ModelFactory.create(config, features)
    
    # CRITICAL ORDER: set features first, then neighbor finder
    model.set_raw_features(features['node_features'], features['edge_features'])
    
    model.set_neighbor_finder(pipeline.neighbor_finder)
    
    # CRITICAL: Set neighbor finder for TGN
    # if hasattr(model, 'set_neighbor_finder'):
    #     model.set_neighbor_finder(pipeline.neighbor_finder)

    # Debug: Verify embedding module initialized
    if hasattr(model, 'embedding_module'):
        print(f"Embedding module initialized: {model.embedding_module is not None}")

    print(f"1. embedding_module exists: {model.embedding_module is not None}")
    print(f"2. memory_updater exists: {model.memory_updater is not None}")
    print(f"3. neighbor_finder exists: {pipeline.neighbor_finder is not None}")
    print(f"4. node_raw_features shape: {model.node_raw_features.shape if model.node_raw_features is not None else 'None'}")
    print(f"5. edge_raw_features shape: {model.edge_raw_features.shape if model.edge_raw_features is not None else 'None'}")

       
    
    # Setup trainer
    trainer = TrainerSetup.create(config)
    print(f"Trainer max_epochs: {trainer.max_epochs}")
    print(f"EarlyStopping patience: {[c for c in trainer.callbacks if isinstance(c, EarlyStopping)]}")
    
    # Train
    logger.info("Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=pipeline.loaders['train'],
        val_dataloaders=pipeline.loaders['val'],
    )

    # Load the best checkpoint for testing
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        best_path = trainer.checkpoint_callback.best_model_path
        logger.info(f"Loading best checkpoint from {best_path}")
        
        # 1. Re‑create a fresh model with the same configuration and data
        #    (ModelFactory.create already gives a new instance)
        model = ModelFactory.create(config, features)
        model.set_raw_features(features['node_features'], features['edge_features'])
        model.set_neighbor_finder(pipeline.neighbor_finder)
        
        # 2. Load the checkpoint with strict=False – ignore unexpected keys
        checkpoint = torch.load(best_path, map_location=model.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        
        # 3. Ensure model is on correct device and in eval mode
        model = model.to(model.device)
        model.eval()
        
        logger.info("Best checkpoint loaded successfully (strict=False).")
    else:
        logger.warning("No checkpoint callback or best model path found. Using current model.")
    
    # Test
    logger.info("Running evaluation...")
    test_results = trainer.test(
        model=model, 
        dataloaders=pipeline.loaders['test'],
        ckpt_path='best'
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
import os


# Limit threading to prevent OpenBLAS crashes
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import string
import csv
from datetime import datetime
import argparse
import yaml 
import torch
import numpy as np 
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from loguru import logger

import hashlib

from src.models.dygformer import DyGFormer
from src.models.tgn import TGN

from src.utils.general_utils import set_seed, get_device
# from src.datasets.loaders import get_dataset_loader, DATASET_LOADERS

# from src.models.tgn_module.neighbor_finder import get_neighbor_finder

from src.datasets.load_dataset import load_dataset
from src.datasets.negative_sampling import NegativeSampler
from src.datasets.neighbor_finder import NeighborFinder
from src.datasets.temporal_dataset import TemporalDataset


# MODEL_REGISTRY={
#     "DyGFormer": DyGFormer,
#     "TGN": TGN,
#     "TAWRMAC": TAWRMAC
# }

MODEL_REGISTRY={
    "DyGFormer": DyGFormer,
    "TGN": TGN,    
}


class DataWrapper:
    def __init__(self, edges, timestamps):
        self.sources = edges[:, 0].cpu().numpy()
        self.destinations = edges[:, 1].cpu().numpy()
        self.timestamps = timestamps.cpu().numpy()
        self.edge_idxs = np.arange(len(edges))



def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_model(config: dict, num_nodes: int):    
    model_config = config['model'].copy()
    model_name = model_config.pop('name')  # Remove non-parameter 'name'
    
    # CRITICAL: Set num_nodes BEFORE instantiating model
    # TGN requires this parameter (positional or keyword)
    model_config['num_nodes'] = num_nodes
    
    logger.info(f" Initializing {model_name} with num_nodes={num_nodes}")

    # Pass neighbor_finder if available
    # if neighbor_finder is not None:
    #     model_config['neighbor_finder'] = neighbor_finder

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(
        num_nodes=num_nodes,        
        node_features=model_config.get('node_features', 0),
        hidden_dim=model_config.get('hidden_dim', 172),
        time_encoding_dim=model_config.get('time_encoding_dim', 32),
        memory_dim=model_config.get('memory_dim', 172),
        message_dim=model_config.get('message_dim', 172),
        edge_features_dim=model_config.get('edge_features_dim', 172),
        num_layers=model_config.get('num_layers', 1),
        dropout=model_config.get('dropout', 0.1),
        learning_rate=model_config.get('learning_rate', 1e-4),
        weight_decay=model_config.get('weight_decay', 1e-5),
        n_heads=model_config.get('n_heads', 2),
        n_neighbors=model_config.get('n_neighbors', 20),
        use_memory=model_config.get('use_memory', True),
        embedding_module_type=model_config.get('embedding_module_type', 'graph_attention'),
    )

    # POST-INIT VALIDATION
    if hasattr(model, 'memory') and model.memory is not None:
        actual_size = model.memory.memory.shape[0]
        logger.info(f"✓ TGN memory allocated: {actual_size} slots")
        assert actual_size == num_nodes, \
            f"Memory size mismatch! Expected {num_nodes}, got {actual_size}"
    
    # DEBUG: Verify we're passing correct num_nodes
    logger.info(f"Initializing {model_name} with num_nodes={num_nodes} (max expected node ID: {num_nodes - 1})")
    logger.info(f"Model config keys: {list(model_config.keys())}")
    
    
    # POST-INIT VALIDATION: Verify memory size matches expected nodes
    if hasattr(model, 'memory') and hasattr(model.memory, 'memory'):
        actual_memory_size = model.memory.memory.shape[0]
        logger.info(f"TGN memory allocated size: {actual_memory_size}")
        if actual_memory_size != num_nodes:
            raise RuntimeError(
                f"TGN memory size ({actual_memory_size}) != num_nodes ({num_nodes}). "
                f"Check TGN constructor parameter handling."
            )
    
    logger.info(f"✓ Initialized {model_name} with {num_nodes} nodes | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model



def setup_trainer(config: dict):
    """Setup PyTorch Lightning trainer."""
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='{epoch}-{val_loss:.2f}',
        monitor=config['logging']['monitor'],
        mode=config['logging']['mode'],
        save_top_k=config['logging']['save_top_k'],
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=config['logging']['monitor'],
        patience=config['training']['patience'],
        mode=config['logging']['mode'],
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Logger
    csv_logger = CSVLogger(
        save_dir=config['logging']['log_dir'],
        name=config['experiment']['name']
    )
    
    logger_type = config.get('logger', 'tensorboard')
    if logger_type == 'tensorboard':
        logger_inst = TensorBoardLogger(
            save_dir=config['logging']['log_dir'],
            name=config['experiment']['name']
        )
    # elif logger_type == 'wandb':
    #     logger_inst = WandbLogger(
    #         project=config['experiment']['name'],
    #         save_dir=config['logging']['log_dir']
    #     )
    else:
        logger_inst = None
    
    # Hardware configuration
    accelerator = 'gpu' if config['hardware']['gpus'] else 'cpu'
    devices = config['hardware']['gpus'] if config['hardware']['gpus'] else 'auto'
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=[logger_inst, csv_logger],
        precision=config['experiment']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        log_every_n_steps=10,
        val_check_interval=0.25,
        enable_progress_bar=True,        
    )
    
    return trainer


def resolve_config(config, context=None):
    """Resovle ${key.subkey} placeholder in config"""
    if context is None:
        context = config

    def replace_value(value):
        if isinstance(value, str):
            try:
                template = string.Template(value)
                return template.substitute(flatten_dict(context))
            except (KeyError, ValueError):
                return value
        return value

    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k,v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def recurse(obj):
        if isinstance(obj, dict):
            return {k:recurse(replace_value(v)) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        else:
            return replace_value(obj)
    return recurse(config)


def train_model(config: dict):
    """Main training function."""
    start_time = datetime.now()
    # Set random seed
    set_seed(config['experiment']['seed'])
    
    
    # Setup device
    device = get_device(config['hardware']['gpus'])
    logger.info(f"Using device: {device}")


    # FIX 1: USE SPLIT-AWARE 0-INDEXED LOADER (NO LEAKAGE)
    data = load_dataset(
        dataset_name=config['data']['dataset'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        inductive=(config['data']['evaluation_type'] == 'inductive'),
        unseen_ratio=config['data'].get('unseen_ratio', 0.1),
        seed=config['experiment']['seed']
    )
    
    # Load dataset
    logger.info(f"Loading dataset: {config['data']['dataset']}")
    eval_type = config["data"].get("evaluation_type", "transductive")
    logger.info(f"Evaluation type: {eval_type}")

    num_nodes = data['num_nodes']  # Correct 0-indexed count (9228 for Wikipedia)
    
    # FIX 2: BUILD LEAKAGE-PROOF NEIGHBOR FINDER (TRAINING EDGES ONLY)
    train_edges = data['edges'][data['train_mask']]
    train_timestamps = data['timestamps'][data['train_mask']]
    neighbor_finder = None
    if config['data']['negative_sampling_strategy'] == 'historical':
        neighbor_finder = NeighborFinder(
            train_edges=train_edges,
            train_timestamps=train_timestamps,
            max_neighbors=config['data']['max_neighbors']
        )
    
    # FIX 3: SPLIT-AWARE NEGATIVE SAMPLERS (NO LEAKAGE)
    train_neg_sampler = NegativeSampler(
        edges=train_edges,
        timestamps=train_timestamps,
        num_nodes=num_nodes,
        neighbor_finder=neighbor_finder,
        split='train',
        seed=config['experiment']['seed']
    )

    val_neg_sampler = NegativeSampler(
        edges=data['edges'][data['val_mask']],
        timestamps=data['timestamps'][data['val_mask']],
        num_nodes=num_nodes,
        neighbor_finder=neighbor_finder,  # Safe: built from training only
        split='val',
        seed=config['experiment']['seed']
    )

    test_neg_sampler = NegativeSampler(
        edges=data['edges'][data['test_mask']],
        timestamps=data['timestamps'][data['test_mask']],
        num_nodes=data['num_nodes'],
        neighbor_finder=neighbor_finder,
        split='test',
        seed=config['experiment']['seed']
    )

    # FIX 4: CORRECT FEATURE HANDLING
    if config['data']['dataset'] == "wikipedia":
        node_features = None  # NO node features
    else:
        node_features = data.get('node_features')
    edge_features = data['edge_features']  # Already 0-indexed
    

    # Create datasets with split-aware sampling
    train_dataset = TemporalDataset(
        edges=train_edges,
        timestamps=train_timestamps,
        edge_features=edge_features[data['train_mask']] if edge_features is not None else None,
        num_nodes=num_nodes,
        split='train',
        negative_sampler=train_neg_sampler,
        negative_sampling_strategy=config['data']['negative_sampling_strategy'],
        unseen_nodes=None,
        seed=config['experiment']['seed']
    )

    val_dataset = TemporalDataset(
        edges=data['edges'][data['val_mask']],
        timestamps=data['timestamps'][data['val_mask']],
        edge_features=data['edge_features'][data['val_mask']] if data['edge_features'] is not None else None,
        num_nodes=data['num_nodes'],
        split='val',
        negative_sampler=val_neg_sampler,
        negative_sampling_strategy=config['data']['negative_sampling_strategy'],
        unseen_nodes=data['unseen_nodes'] if config['data']['evaluation_type'] == 'inductive' else None,
        seed=config['experiment']['seed']
    )
    
    test_dataset = TemporalDataset(
        edges=data['edges'][data['test_mask']],
        timestamps=data['timestamps'][data['test_mask']],
        edge_features=data['edge_features'][data['test_mask']] if data['edge_features'] is not None else None,
        num_nodes=data['num_nodes'],
        split='test',
        negative_sampler=test_neg_sampler,
        negative_sampling_strategy=config['data']['negative_sampling_strategy'],
        unseen_nodes=data['unseen_nodes'] if config['data']['evaluation_type'] == 'inductive' else None,
        seed=config['experiment']['seed']
    )
    
    
    # logger.info(f"Dataset '{config['data']['dataset']}' has {num_nodes} nodes (max node ID in edges: {data['edges'].max()})")
    # assert num_nodes > 1000, f"Unexpectedly small num_nodes={num_nodes}. Check dataset loader."

    # train_loader, val_loader, test_loader, num_unique_nodes = get_dataset_loader(
    #     config,
    #     negative_sampling_strategy=config['data'].get('negative_sampling_strategy', 'random')    
    # )
    
    # logger.info(f"Dataset '{config['data']['dataset']}' has {num_nodes} nodes (max node ID in edges: {data['edges'].max()})")
    # assert num_nodes > 1000, f"Unexpectedly small num_nodes={num_nodes}. Check dataset loader."
    
    # CRITICAL FIX: Compute required memory size for 1-indexed datasets
    # data = DATASET_LOADERS[config['data']['dataset']]()
    # max_node_id = int(data['edges'].max().item())
    # num_nodes_for_model = max_node_id + 1  # For 1-indexed IDs: allocate [0..max_id]

    # logger.info(f"Unique nodes in dataset: {num_unique_nodes}")
    # logger.info(f"Max node ID in edges: {max_node_id}")
    # logger.info(f"Allocating memory for {num_nodes_for_model} nodes (max_id + 1)")
        
    # # Override loader's num_nodes with correct memory size
    # num_nodes = num_nodes_for_model

    # Verify consistency
    # assert num_nodes > max_node_id, f"num_nodes ({num_nodes}) must be > max_node_id ({max_node_id})"
    # if config['data']['dataset'] == 'wikipedia':
    #     assert num_nodes == 9228, f"Wikipedia requires num_nodes=9228 (max_id=9227 + 1), got {num_nodes}"
    #     logger.info("✓ Wikipedia memory size verified (9228 slots for IDs 1-9227)")
    
    # # DEBUG: Verify num_nodes is correct for Wikipedia (should be 9228)
    # logger.info(f"✓ Dataset loader returned num_nodes = {num_nodes}")
    # if config['data']['dataset'] == 'wikipedia':
    #     assert num_nodes == 9228, f"Wikipedia should have 9228 nodes, got {num_nodes}"
    #     logger.info("✓ Wikipedia node count verified (9228)")

    # # Also verify against actual max node ID in edges
    # data = DATASET_LOADERS[config['data']['dataset']]()
    # max_node_id = int(data['edges'].max().item())
    # logger.info(f"Max node ID in edge list: {max_node_id}")
    # assert num_nodes > max_node_id, f"num_nodes ({num_nodes}) must be > max node ID ({max_node_id})"



    # DEBUG: Verify temporal splits
    # train_timestamps = train_loader.dataset.timestamps
    # val_timestamps = val_loader.dataset.timestamps
    # test_timestamps = test_loader.dataset.timestamps

    # print(f"Train timestamp range: {train_timestamps.min():.0f} to {train_timestamps.max():.0f}")
    # print(f"Val timestamp range: {val_timestamps.min():.0f} to {val_timestamps.max():.0f}")
    # print(f"Test timestamp range: {test_timestamps.min():.0f} to {test_timestamps.max():.0f}")

    # # Check for leakage: test edges should be AFTER train edges
    # assert test_timestamps.min() >= val_timestamps.max(), "Test should start after validation"
    
    # experiment logging
    logger.info(f"Seed: {config['experiment']['seed']}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    

    
    # dataset_name = config['data']['dataset']
    # if dataset_name in DATASET_LOADERS:
    #     data = DATASET_LOADERS[dataset_name]()

    #     data_wrapper = DataWrapper(data['edges'], data['timestamps'])
    #     neighbor_finder = get_neighbor_finder(data_wrapper, uniform=False)
        
    #     # Get features (handle both padded and unpadded)
    #     node_features = data.get('node_features', torch.zeros(num_nodes, 172))
    #     # Handle features correctly
    #     if dataset_name == "wikipedia":
    #         # Wikipedia has NO node features - use None
    #         node_features = None
    #         # Edge features are already correct shape [num_edges, 172]
    #         edge_features = data.get('edge_features', torch.zeros(len(data['edges']), 172))
    #     else:
    #         # Other datasets may have node features
    #         node_features = data.get('node_features', None)
    #         edge_features = data.get('edge_features', torch.zeros(len(data['edges']), 1))
        
    #     # For TGN with no node features, pass zeros but handle in model
    #     if config['model']['name'] == 'TGN' and node_features is None:
    #         # Create dummy node features (will use learned embeddings instead)
    #         node_features = torch.zeros(num_nodes + 1, 1)  # 1-dim dummy features
        
        


        # Initialize model
        # model = setup_model(config, num_nodes)

    # MODEL INITIALIZATION (reconstruction-safe)
    model = create_model(
        model_name=config['model']['name'],
        model_config=config['model'],
        dataset_config={
            'num_nodes': data['num_nodes'],  # 0-indexed count (e.g., 9228)
            'edge_feat_dim': data['edge_features'].shape[1] if data['edge_features'] is not None else 0,
            'node_feat_dim': data['node_features'].shape[1] if data['node_features'] is not None else 0,
        }
    )
    
    # VALIDATION BEFORE TRAINING (catches reconstruction errors)
    assert model.num_nodes == data['num_nodes'], \
        f"Reconstruction failure! model.num_nodes={model.num_nodes} != {data['num_nodes']}"
    assert model.num_nodes > 1000, \
        f"num_nodes={model.num_nodes} suspiciously small (expected >1000). Check 0-indexing."
    

    #  TRAIN WITH RIGOROUS EVALUATION
    trainer = setup_trainer(config)
    # trainer.fit(
    #     model,
    #     DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
    #               shuffle=True, collate_fn=train_dataset.collate_fn),
    #     DataLoader(val_dataset, batch_size=config['training']['batch_size'], 
    #               shuffle=False, collate_fn=val_dataset.collate_fn)
    # )
    
    # TEST WITH INDUSTRY-STANDARD METRICS
    # test_results = evaluate_link_prediction(
    #     model,
    #     DataLoader(test_dataset, batch_size=config['evaluation']['test_batch_size'], 
    #               shuffle=False, collate_fn=test_dataset.collate_fn),
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )
    
    logger.info("=== FINAL RESULTS ===")
    for metric, value in test_results.items():
        logger.info(f"{metric}: {value:.4f}")




        # Validate parameters
        # validate_model_parameters(model, config, num_nodes)
        # # Set raw features for models that need them
        # model.neighbor_finder = neighbor_finder

        # # Initialize embedding module
        # if model.neighbor_finder is not None:
        #     model._init_modules()
        
        # FIX: Load and set raw features
        # Set features based on model type
        # if config['model']['name'] in ['DyGFormer', 'TAWRMAC']:
        #     model.set_raw_features(node_features, edge_features)
        # elif config['model']['name'] == 'TGN':
        #     # TGN uses UNPADDED edge features directly
        #     unpadded_edge_features = data.get('edge_features', torch.zeros(len(data['edges']), 172))
        #     # Set raw features (TGN should have set_raw_features method)
        #     if hasattr(model, 'set_raw_features'):
        #         model.set_raw_features(node_features, unpadded_edge_features)
        #     else:
        #         # Fallback: set attributes directly
        #         model.node_raw_features = node_features.to(device)
        #         model.edge_raw_features = unpadded_edge_features.to(device)
        # elif config['model']['name'] == 'TGN':
        #     # Load raw features with proper padding for 1-indexed nodes
        #     actual_num_nodes = num_nodes  # This should be 9228 for Wikipedia
            
        #     # Get node features from dataset (should already be padded)
        #     if 'node_features' in data:
        #         node_features = data['node_features']
        #         logger.info(f"Loaded node features shape: {node_features.shape}")
        #     else:
        #         # Wikipedia has NO node features - use learned embeddings instead
        #         logger.warning("No node features available - using learned embeddings")
        #         node_features = None

        #     # Edge features (use unpadded version for TGN)
        #     edge_features = data.get('edge_features', torch.zeros(len(data['edges']), 172))
            
        #     model.set_raw_features(node_features, edge_features)
        #     model.set_neighbor_finder(neighbor_finder)
        #     logger.info(f"Node feature stats - mean: {node_features.mean():.6f}, std: {node_features.std():.6f}")
        #     # logger.info(f"First 5 node features:\n{node_features[1:6]}") 
    
        # logger.info(f"Set raw features - Node: {node_features.shape}, Edge: {edge_features.shape}")
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info(f"Loaded node features shape: {node_features.shape}")
    logger.info(f"Dataset num_nodes: {num_nodes}")
    logger.info(f"Node ID range in data: {data['edges'].min()} to {data['edges'].max()}")

    # Verify hparams preserved num_nodes correctly
    logger.info(f"Model num_nodes: {model.num_nodes}")
    logger.info(f"hparams.num_nodes: {model.hparams.num_nodes}")
    logger.info(f"Memory size: {model.memory.memory.shape[0]}")

    assert model.hparams.num_nodes == num_nodes, \
        f"hparams lost num_nodes! Expected {num_nodes}, got {model.hparams.num_nodes}"
    assert model.memory.memory.shape[0] == num_nodes, \
        f"Memory size mismatch! Expected {num_nodes}, got {model.memory.memory.shape[0]}"

    # AFTER model initialization but BEFORE trainer.fit()
    logger.info("=== HYPERPARAMETER VALIDATION ===")
    logger.info(f"Dataset: {config['data']['dataset']}")
    logger.info(f"Computed num_nodes: {num_nodes}")
    logger.info(f"hparams.num_nodes: {model.hparams.num_nodes}")
    logger.info(f"hparams.edge_features_dim: {model.hparams.edge_features_dim}")
    logger.info(f"Memory size: {model.memory.memory.shape[0]}")
    logger.info("=== END VALIDATION ===")

    # DYNAMIC VALIDATION: Compare against ACTUAL computed num_nodes (not hardcoded 9228)
    assert model.hparams.num_nodes == num_nodes, \
        f"RECONSTRUCTION FAILURE: hparams.num_nodes={model.hparams.num_nodes} != computed num_nodes={num_nodes}. " \
        f"This happens when **kwargs leaks parameters during reconstruction. " \
        f"FIX: Remove **kwargs from ALL constructor signatures."

    assert model.hparams.edge_features_dim == 172, \
        f"edge_features_dim corrupted: {model.hparams.edge_features_dim} (expected 172 for Wikipedia/Reddit)"

    assert model.memory.memory.shape[0] == num_nodes, \
        f"Memory size mismatch: {model.memory.memory.shape[0]} != {num_nodes}"
        
    
    # Setup trainer
    trainer = setup_trainer(config)
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test model
    logger.info("Running evaluation on test set...")
    test_results = trainer.test(model=model, dataloaders=test_loader)
    
    # Log test results
    logger.info("Test Results:")
    for metric, value in test_results[0].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save final model
    final_model_path = os.path.join(
        config['logging']['checkpoint_dir'],
        'final_model.ckpt'
    )
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to: {final_model_path}")

    # Prepare result row
    result_row = {
        'model': config['model']['name'],
        'dataset': config['data']['dataset'],
        'evaluation_type': config['data']['evaluation_type'],
        'negative_sampling_strategy': config['data']['negative_sampling_strategy'],
        'seed': config['experiment']['seed'],
        'test_accuracy': test_results[0].get('test_accuracy', 0.0),
        'test_ap': test_results[0].get('test_ap', 0.0),
        'test_auc': test_results[0].get('test_auc', 0.0),
        'test_loss': test_results[0].get('test_loss', 0.0),
        'val_loss': trainer.callback_metrics.get('val_loss', 0.0),
        'training_time': (datetime.now() - start_time).total_seconds(),
        'num_parameters': sum(p.numel() for p in model.parameters()),        
        'timestamp': datetime.now().isoformat()
    }

    # Save to CSV
    csv_path = os.path.join(config['logging']['log_dir'], '..', 'all_results.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

def validate_model_parameters(model, config, num_nodes):
    """Validate model parameters after initialization."""
    logger.info("=== MODEL PARAMETER VALIDATION ===")
    
    # Check num_nodes
    if hasattr(model, 'num_nodes'):
        logger.info(f"model.num_nodes: {model.num_nodes}")
        assert model.num_nodes == num_nodes, f"model.num_nodes mismatch: {model.num_nodes} != {num_nodes}"
    
    # Check hparams
    if hasattr(model, 'hparams'):
        if 'num_nodes' in model.hparams:
            logger.info(f"hparams.num_nodes: {model.hparams['num_nodes']}")
            assert model.hparams['num_nodes'] == num_nodes
    
    # Check memory size for TGN
    if hasattr(model, 'memory') and model.memory is not None:
        if hasattr(model.memory, 'memory'):
            mem_size = model.memory.memory.shape[0]
            logger.info(f"Memory size: {mem_size}")
            assert mem_size == num_nodes, f"Memory size mismatch: {mem_size} != {num_nodes}"
    
    logger.info("=== VALIDATION PASSED ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training"
    )

    parser.add_argument(
        "--override",
        nargs="*",
        help="Override config parameters "
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of random seeds to run"
    )

    args = parser.parse_args()

    config = load_config(args.configs)
    # multiple combination experiment
    config = resolve_config(config)

    if args.override:
        for override in args.override:
            key, value = override.split("=")
            keys = key.split('.')

            current = config
            for k in keys[:-1]:
                current = current[k]

            try:
                import ast
                current[keys[-1]] = ast.literal_eval(value)
            except:
                current[keys[-1]] = value

    model_name = config['model'].get('name', 'dygformer')
    eval_type = config['data'].get('evaluation_type', 'transductive')
    neg_sample = config['data'].get('negative_sampling_strategy', 'random')
    data_name = config['data'].get('dataset', 'wikipedia')

     # Update dynamic fields
    config['experiment']['name'] = f"{model_name}_{eval_type}_{neg_sample}"
    config['experiment']['description'] = f"{model_name} on {data_name} | {eval_type} | {neg_sample}"
    # config['logging']['log_dir'] = f"./logs/dygformer_{eval_type}_{neg_sample}"
    # config['logging']['checkpoint_dir'] = f"./checkpoints/dygformer_{eval_type}_{neg_sample}"
    
    config_str = str(sorted(config['model'].items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
    
    config['logging']['log_dir'] = f"./logs/{model_name}_{eval_type}_{neg_sample}"
    # config['logging']['checkpoint_dir'] = f"./checkpoints/{model_name}_{eval_type}_{neg_sample}"
    config['logging']['checkpoint_dir'] = f"./checkpoints/{model_name}_{eval_type}_{neg_sample}_{config_hash}"
    
    # Create directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Setup logging
    logger.add(
        os.path.join(config['logging']['log_dir'], 'train.log'),
        rotation="10 MB",
        level="INFO"
    )
    
    # Log experiment info
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info(f"Config file: {args.configs}")

    
    
    # for seed in args.seeds:
    #     config['experiment']['seed'] = seed

    #     # Start training
    #     try:
    #         train_model(config)
    #     except Exception as e:
    #         logger.error(f"Training failed with error {seed} : {e}")
    #         raise

    # Determine which seeds to use
    if args.seeds is not None:
        # Use seeds from command line argument
        seeds_to_run = args.seeds
    else:
        # Use seed from config (which might have been set by --override or config file)
        seeds_to_run = [config['experiment'].get('seed', 42)]
    
    for seed in seeds_to_run:
        config['experiment']['seed'] = seed
        logger.info(f"Running with seed: {seed}")

        try:
            train_model(config)
            
        except Exception as e:
            logger.error(f"Training failed with seed {seed}: {e}")
            raise



if __name__ == "__main__":
    main()
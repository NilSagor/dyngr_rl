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

from src.models.dygformer import DyGFormer
from src.models.tgn import TGN

from src.utils.general_utils import set_seed, get_device
from src.datasets.loaders import get_dataset_loader, DATASET_LOADERS

from src.models.tgn_module.neighbor_finder import get_neighbor_finder

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
    model_config['num_nodes'] = num_nodes
    
    model_name = model_config.pop('name')
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(**model_config)
    
    logger.info(f"Initialized {model_name} model with {num_nodes} nodes")
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
    
    # Load dataset
    logger.info(f"Loading dataset: {config['data']['dataset']}")
    eval_type = config["data"].get("evaluation_type", "transductive")
    logger.info(f"Evaluation type: {eval_type}")
    
    train_loader, val_loader, test_loader, num_nodes = get_dataset_loader(
        config,
        negative_sampling_strategy=config['data'].get('negative_sampling_strategy', 'random')    
    )
    
    # DEBUG: Verify temporal splits
    train_timestamps = train_loader.dataset.timestamps
    val_timestamps = val_loader.dataset.timestamps
    test_timestamps = test_loader.dataset.timestamps

    print(f"Train timestamp range: {train_timestamps.min():.0f} to {train_timestamps.max():.0f}")
    print(f"Val timestamp range: {val_timestamps.min():.0f} to {val_timestamps.max():.0f}")
    print(f"Test timestamp range: {test_timestamps.min():.0f} to {test_timestamps.max():.0f}")

    # Check for leakage: test edges should be AFTER train edges
    assert test_timestamps.min() >= val_timestamps.max(), "Test should start after validation"
    
    # experiment logging
    logger.info(f"Seed: {config['experiment']['seed']}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    

    
    dataset_name = config['data']['dataset']
    if dataset_name in DATASET_LOADERS:
        data = DATASET_LOADERS[dataset_name]()

        data_wrapper = DataWrapper(data['edges'], data['timestamps'])
        neighbor_finder = get_neighbor_finder(data_wrapper, uniform=False)
        
        # Get features (handle both padded and unpadded)
        node_features = data.get('node_features', torch.zeros(num_nodes, 172))
        # Handle features correctly
        if dataset_name == "wikipedia":
            # Wikipedia has NO node features - use None
            node_features = None
            # Edge features are already correct shape [num_edges, 172]
            edge_features = data.get('edge_features', torch.zeros(len(data['edges']), 172))
        else:
            # Other datasets may have node features
            node_features = data.get('node_features', None)
            edge_features = data.get('edge_features', torch.zeros(len(data['edges']), 1))
        
        
        
        # Initialize model
        model = setup_model(config, num_nodes)
        # Set raw features for models that need them
        model.neighbor_finder = neighbor_finder
        
        # FIX: Load and set raw features
        # Set features based on model type
        if config['model']['name'] in ['DyGFormer', 'TAWRMAC']:
            model.set_raw_features(node_features, edge_features)
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
        elif config['model']['name'] == 'TGN':
            # Load raw features with proper padding for 1-indexed nodes
            actual_num_nodes = num_nodes  # This should be 9228 for Wikipedia
            
            # Get node features from dataset (should already be padded)
            if 'node_features' in data:
                node_features = data['node_features']
                logger.info(f"Loaded node features shape: {node_features.shape}")
            else:
                # Wikipedia has NO node features - use learned embeddings instead
                logger.warning("No node features available - using learned embeddings")
                node_features = None

            # Edge features (use unpadded version for TGN)
            edge_features = data.get('edge_features', torch.zeros(len(data['edges']), 172))
            
            model.set_raw_features(node_features, edge_features)
            logger.info(f"Node feature stats - mean: {node_features.mean():.6f}, std: {node_features.std():.6f}")
            # logger.info(f"First 5 node features:\n{node_features[1:6]}") 
    
        logger.info(f"Set raw features - Node: {node_features.shape}, Edge: {edge_features.shape}")
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info(f"Loaded node features shape: {node_features.shape}")
    logger.info(f"Dataset num_nodes: {num_nodes}")
    logger.info(f"Node ID range in data: {data['edges'].min()} to {data['edges'].max()}")

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
    config['logging']['log_dir'] = f"./logs/{model_name}_{eval_type}_{neg_sample}"
    config['logging']['checkpoint_dir'] = f"./checkpoints/{model_name}_{eval_type}_{neg_sample}"
    
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

    
    
    for seed in args.seeds:
        config['experiment']['seed'] = seed

        # Start training
        try:
            train_model(config)
        except Exception as e:
            logger.error(f"Training failed with error {seed} : {e}")
            raise



if __name__ == "__main__":
    main()
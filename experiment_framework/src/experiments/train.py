import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))



import argparse
import yaml 
import torch 
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger

from src.models.dygformer import DyGFormer
from src.utils.general_utils import set_seed, get_device
from src.datasets.loaders import get_dataset_loader


# MODEL_REGISTRY={
#     "DyGFormer": DyGFormer,
#     "TGN": TGN,
#     "TAWRMAC": TAWRMAC
# }

MODEL_REGISTRY={
    "DyGFormer": DyGFormer,    
}

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
    
    # Logger
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
        logger=logger_inst,
        precision=config['experiment']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        log_every_n_steps=10,
        val_check_interval=0.25,
        enable_progress_bar=True,
    )
    
    return trainer





def train_model(config: dict):
    """Main training function."""
    
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
    
    # experiment logging
    logger.info(f"Seed: {config['experiment']['seed']}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    # Initialize model
    model = setup_model(config, num_nodes)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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

    args = parser.parse_args()

    config = load_config(args.configs)

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
    
    # Start training
    try:
        train_model(config)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise



if __name__ == "__main__":
    main()
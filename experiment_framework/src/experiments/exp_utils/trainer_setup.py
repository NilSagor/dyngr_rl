import torch 
from typing import Dict, List, Optional, Any
from loguru import logger
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.experiments.exp_utils.model_profile import ModelProfiler

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
        
        # profiler = None
        # if config.get('profiling', {}).get('enabled', False):
        #     profiler = ModelProfiler(
        #     dirpath=config['logging']['log_dir'],
        #     filename="profile",
        #     schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(...),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        # )

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
            # profiler=profiler,
        )
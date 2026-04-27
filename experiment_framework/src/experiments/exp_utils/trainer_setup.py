import torch 
from typing import Dict, List, Optional, Any
from loguru import logger
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from src.experiments.exp_utils.analysis_callback import AnalysisCollector
# from src.experiments.exp_utils.model_profile import ModelProfiler

from lightning.pytorch.strategies import DDPStrategy

# ============================================================================
# TRAINING
# ============================================================================

class TrainerSetup:
    """Encapsulates trainer configuration."""
    
    @staticmethod
    def create(
        config: Dict, 
        callbacks: Optional[List[Callback]] = None,
        default_root_dir: Optional[str] = None,   
        fast_dev_run: bool = False                
    ) -> L.Trainer:
        """Create PyTorch Lightning trainer."""
        
        if default_root_dir:
            ckpt_dir = str(Path(default_root_dir) / "checkpoints")
            log_dir  = str(Path(default_root_dir) / "logs")
        else:
            ckpt_dir = config['logging']['checkpoint_dir']
            log_dir  = config['logging']['log_dir']
        
        # ------ callbacks ------
        base_callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename='{epoch}-{val_ap:.2f}',
                monitor=config['training']['monitor'],
                mode=config['training']['mode'],
                save_top_k=config['training']['save_top_k'],
                verbose=True,
            ),
            EarlyStopping(
                monitor=config['training']['monitor'],
                patience=config['training']['patience'],
                mode=config['training']['mode'],
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),          
            # ReduceLROnPlateau(monitor='val_ap', mode='max', patience=10, factor=0.5)
        ]

        # has_analysis_collector = any(
        #     isinstance(c, AnalysisCollector) for c in (callbacks or [])
        # )
        # if not has_analysis_collector:
        #     base_callbacks.append(AnalysisCollector())

        if callbacks:
            for cb in callbacks:
                if not any(type(cb) == type(existing) for existing in base_callbacks):
                    base_callbacks.append(cb)
        
        all_callbacks = base_callbacks
        
        # ------ loggers ------
        loggers = [
            CSVLogger(save_dir=log_dir, name=config['experiment']['name']),
        ]
        if config.get('logger', 'tensorboard') == 'tensorboard':
            loggers.append(
                TensorBoardLogger(
                    save_dir=log_dir, 
                    name=config['experiment']['name']
                )
            )


        
        accelerator = 'gpu' if config['hardware']['gpus'] else 'cpu'
        devices = config['hardware']['gpus'] if config['hardware']['gpus'] else 'auto'
        
        # Detect multi-GPU and set strategy
        strategy = "auto"  # Default for single GPU
        if isinstance(devices, int) and devices > 1:
            # For 2x RTX 5090
            strategy = DDPStrategy(
                find_unused_parameters=False,  # Set True if you have unused params in graph
                gradient_as_bucket_view=True,   # Saves memory
                static_graph=True               # Optimizes DDP for static graphs
            )
            print(f"Using DDP strategy with {devices} GPUs")
        elif isinstance(devices, list) and len(devices) > 1:
            strategy = "ddp"
        
        # Mixed precision for RTX 5090 (FP16/BF16)
        precision = config['experiment'].get('precision', 32)
        if torch.cuda.is_available():
            if "RTX 40" in torch.cuda.get_device_name(0) or "RTX 50" in torch.cuda.get_device_name(0):
                print("Detected RTX 40/50 series - enabling TF32 for faster training")
                torch.set_float32_matmul_precision('high')
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
        
       

        # ------ create the trainer ------
        trainer_args = dict(
            max_epochs=config['training']['max_epochs'],
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            callbacks=all_callbacks,
            logger=loggers,
            precision=precision,
            gradient_clip_val=config['training']['gradient_clip_val'],
            log_every_n_steps=config['training']['log_every_n_steps'],
            val_check_interval=config['training']['val_check_interval'],
            enable_progress_bar=True,
        )

        if default_root_dir:
            trainer_args['default_root_dir'] = default_root_dir

        if fast_dev_run:
            trainer_args['fast_dev_run'] = True
            trainer_args.pop('max_epochs', None)        
            trainer_args.pop('val_check_interval', None)

        return L.Trainer(**trainer_args)
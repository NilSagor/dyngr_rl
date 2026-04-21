import torch 
from typing import Dict, List, Optional, Any
from loguru import logger
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
    def create(config: Dict, callbacks: Optional[List[Callback]] = None) -> L.Trainer:
        """Create PyTorch Lightning trainer."""
        base_callbacks = [
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

        has_analysis_collector = any(
            isinstance(c, AnalysisCollector) for c in (callbacks or [])
        )
        if not has_analysis_collector:
            base_callbacks.append(AnalysisCollector())

        if callbacks:
            for cb in callbacks:
                if not any(type(cb) == type(existing) for existing in base_callbacks):
                    base_callbacks.append(cb)
        
        all_callbacks = base_callbacks
        
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
            strategy=strategy,
            callbacks=all_callbacks,
            logger=loggers,
            # precision=config['experiment']['precision'],
            precision = precision,            
            gradient_clip_val=config['training']['gradient_clip_val'],
            log_every_n_steps=config['training']['log_every_n_steps'],
            val_check_interval=config['training']['val_check_interval'],
            enable_progress_bar=True,                       
            # profiler=profiler,
        )
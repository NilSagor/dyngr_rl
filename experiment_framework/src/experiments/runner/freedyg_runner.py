# experiments/runner/freedyg_runner.py
from typing import Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger

from .base_runner import BaseRunner
from datasets.freedyg_dataloading.data_pipeline import FreeDyGDataPipeline
from models.freedyg_module.freedyg_variants.freedyg_v1 import FreeDyGLightningModule

class FreeDyGRunner(BaseRunner):
    """Runner for FreeDyG model."""

    def create_data_pipeline(self):
        """Create and return the data pipeline."""
        return FreeDyGDataPipeline(self.config)

    def setup_model(self, model: nn.Module, pipeline) -> None:
        """Inject pipeline components into the LightningModule."""
        # The model is a FreeDyGLightningModule; we need to assign its attributes
        model.node_raw_features = pipeline.node_raw_features
        model.edge_raw_features = pipeline.edge_raw_features
        model.neighbor_sampler = pipeline.train_neighbor_sampler
        
        # Also attach negative samplers if needed (they are accessed via datamodule, but we can store them)
        model.train_neg_sampler = pipeline.train_neg_sampler
        model.val_neg_sampler = pipeline.val_neg_sampler
        model.test_neg_sampler = pipeline.test_neg_sampler

        # Update the backbone's neighbor sampler
        model.dynamic_backbone.set_neighbor_sampler(pipeline.train_neighbor_sampler)
        
        logger.info("FreeDyG model setup complete.")

    def _log_model_status(self, model: nn.Module) -> None:
        """Log number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    def _profile_model(self, model: nn.Module, pipeline) -> None:
        """Optional profiling: compute FLOPs and latency."""
        try:
            from fvcore.nn import FlopCountAnalysis, flop_count_table
            # Create a dummy batch
            batch_size = self.config.batch_size
            src = torch.randint(1, pipeline.num_nodes, (batch_size,))
            dst = torch.randint(1, pipeline.num_nodes, (batch_size,))
            times = torch.rand(batch_size) * 1000
            # Forward pass
            flops = FlopCountAnalysis(model, (src, dst, times))
            logger.info(f"FLOPs per batch: {flop_count_table(flops)}")
        except ImportError:
            logger.warning("fvcore not installed, skipping FLOPs profiling")
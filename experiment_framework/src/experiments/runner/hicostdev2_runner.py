# experiments/runner/hicostdev2_runner.py
from .hicost_base_runner import HiCoSTBaseRunner
from loguru import logger
import torch

class HiCoSTdev2Runner(HiCoSTBaseRunner):
     """Runner for HiCoSTdev2 - replaces walk bias with time - delta attention."""
     def _log_model_status(self, model: torch.nn.Module) -> None:
        logger.info(f"=== HiCoSTdev2 (Causal Time-Delta Attention) Model Status ===")
        logger.info(f"Use memory: {model.use_memory}")
        logger.info(f"Enable walk: {model.enable_walk}")
        logger.info(f"Enable restart: {model.enable_restart}")
        logger.info(f"Time-delta attention enabled: {model.use_time_delta_attention}")
        # Also show temporal bias flag
        if hasattr(model, 'walk_sampler') and hasattr(model.walk_sampler, 'use_temporal_bias'):
            logger.info(f"Walk temporal bias: {model.walk_sampler.use_temporal_bias}")
        logger.info(f"Walk length: {model.walk_length if model.enable_walk else 'N/A'}")
        logger.info(f"Num walks: {model.num_walks if model.enable_walk else 'N/A'}")
        logger.info("=" * 40)
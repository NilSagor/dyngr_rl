# src/experiments/runners/tawrmac_runner.py
import torch
from loguru import logger
from src.datasets.tawrmac_dataloading.data_pipeline import TAWRMACDataPipeline
from src.experiments.runner.base_runner import BaseRunner


class TAWRMACRunner(BaseRunner):
    """Runner for TAWRMAC models."""

    def create_data_pipeline(self):
        pipeline = (TAWRMACDataPipeline(self.config)
                    .load()
                    .build_neighbor_finder()
                    .build_samplers()
                    .build_datasets()
                    .build_loaders())
        return pipeline

    def setup_model(self, model: torch.nn.Module, pipeline) -> None:
        # TAWRMAC only needs neighbor finder; features already in config
        model.set_neighbor_finder(pipeline.neighbor_finder)

    def _log_model_status(self, model: torch.nn.Module) -> None:
        logger.info(f"=== TAWRMAC Model Status ===")
        logger.info(f"Use memory: {model.use_memory}")
        logger.info(f"Enable walk: {model.enable_walk}")
        logger.info(f"Enable restart: {model.enable_restart}")
        logger.info(f"Enable co-occurrence: {model.neighbor_cooc}")
        logger.info(f"Walk length: {model.walk_length if model.enable_walk else 'N/A'}")
        logger.info(f"Num walks: {model.num_walks if model.enable_walk else 'N/A'}")
        logger.info("=" * 40)
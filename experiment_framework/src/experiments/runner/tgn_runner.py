from typing import Dict, Any
import torch
from loguru import logger

from src.datasets.continue_temporal.data_con_pipeline import DataPipeline
from src.experiments.runner.base_runner import BaseRunner


class TGNRunner(BaseRunner):
    """Runner for TGN, DyGFormer, and other models using the standard DataPipeline."""

    def create_data_pipeline(self):
        pipeline = (DataPipeline(self.config)
                    .load()
                    .build_neighbor_finder()
                    .build_samplers()
                    .build_datasets()
                    .build_loaders())

        # Optional: extract training edges for some models
        train_edges = pipeline.data['edges'][pipeline.data['train_mask']]
        train_times = pipeline.data['timestamps'][pipeline.data['train_mask']]

        if not isinstance(train_edges, torch.Tensor):
            train_edges = torch.tensor(train_edges)
        if not isinstance(train_times, torch.Tensor):
            train_times = torch.tensor(train_times)

        if train_edges.shape[0] != 2:
            train_edges = train_edges.T.contiguous()
        if train_times.dim() == 2 and train_times.shape[1] == 1:
            train_times = train_times.squeeze(1)

        pipeline.train_edges = train_edges
        pipeline.train_times = train_times

        return pipeline

    def setup_model(self, model: torch.nn.Module, pipeline) -> None:
        features = pipeline.get_features()
        model.set_raw_features(features['node_features'], features['edge_features'])
        model.set_neighbor_finder(pipeline.neighbor_finder)
        model.set_graph(
            pipeline.neighbor_finder.edge_index,
            pipeline.neighbor_finder.edge_time
        )

    def _log_model_status(self, model: torch.nn.Module) -> None:
        """Log TGN-specific component status."""
        logger.info(f"=== {self.config['model']['name']} Model Status ===")
        logger.info(f"Embedding module: {hasattr(model, 'embedding_module') and model.embedding_module is not None}")
        logger.info(f"Memory updater: {hasattr(model, 'memory_updater') and model.memory_updater is not None}")
        logger.info(f"Memory size: {model.memory.memory.shape[0] if hasattr(model, 'memory') and model.memory else 'N/A'}")
        logger.info("=" * 40)
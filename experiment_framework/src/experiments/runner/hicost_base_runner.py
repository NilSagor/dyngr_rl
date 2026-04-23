# experiments/runner/hicost_base_runner.py
from .base_runner import BaseRunner
from loguru import logger
import torch 

from src.datasets.tawrmac_dataloading.data_pipeline import TAWRMACDataPipeline

class HiCoSTBaseRunner(BaseRunner):
    def create_data_pipeline(self):
        return (TAWRMACDataPipeline(self.config)
                .load()
                .build_neighbor_finder()
                .build_samplers()
                .build_datasets()
                .build_loaders()
            )

    def setup_model(self, model, pipeline):
        model.set_neighbor_finder(pipeline.neighbor_finder)
        if hasattr(pipeline, 'train_edge_index') and pipeline.train_edge_index is not None:
            model.set_graph(pipeline.train_edge_index, pipeline.train_edge_time)
            logger.info("Co-GNN graph passed to model via set_graph")
        else:
            logger.warning("Pipeline missing train_edge_index; Co-GNN will use zero fallback") 

    def _profile_model(self, model: torch.nn.Module, pipeline) -> None:
        logger.info(f"Computing FLOPs for {self.config['model']['name']}...")
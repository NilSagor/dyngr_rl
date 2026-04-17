# src/experiments/runners/tawrmac_runner.py
import torch
from loguru import logger
from src.datasets.tawrmac_dataloading.data_pipeline import TAWRMACDataPipeline
from src.experiments.runner.base_runner import BaseRunner
from pathlib import Path
import numpy as np

from src.experiments.exp_utils.flops_calculator import FLOPsCalculator

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

    def _profile_model(self, model: torch.nn.Module, pipeline) -> None:
        """Compute FLOPs for TAWRMAC models."""
        logger.info("Computing FLOPs for TAWRMAC...")

        batch_size = min(self.config['training']['batch_size'], 32)  # Smaller to avoid OOM
        n_neighbors = getattr(model, 'n_neighbors', 20)
        device = next(model.parameters()).device

        # Create dummy NumPy arrays as expected by compute_edge_probabilities
        sources = np.random.randint(0, pipeline.num_nodes, size=batch_size)
        destinations = np.random.randint(0, pipeline.num_nodes, size=batch_size)
        timestamps = np.random.uniform(0, 1e6, size=batch_size)
        edge_idxs = np.arange(batch_size)
        neg_sources = np.random.randint(0, pipeline.num_nodes, size=batch_size)
        neg_destinations = np.random.randint(0, pipeline.num_nodes, size=batch_size)

        # Warm-up and profile
        model.eval()
        with torch.no_grad():
            # Warm-up
            _ = model.compute_edge_probabilities(
                sources, destinations, neg_sources, neg_destinations,
                timestamps, edge_idxs, n_neighbors
            )

            # Use torch.profiler for detailed analysis (FLOPsCalculator may not work directly)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else [],
                with_flops=True,
                profile_memory=True,
            ) as prof:
                _ = model.compute_edge_probabilities(
                    sources, destinations, neg_sources, neg_destinations,
                    timestamps, edge_idxs, n_neighbors
                )

            # Log summary
            key_avg = prof.key_averages()
            total_flops = sum([e.flops for e in key_avg if e.flops is not None])
            logger.info(f"TAWRMAC estimated FLOPs: {total_flops / 1e9:.3f} GFLOPs")
            logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Optionally save trace
            trace_path = Path(self.config['logging']['log_dir']) / "tawrmac_trace.json"
            prof.export_chrome_trace(str(trace_path))
            logger.info(f"Profiling trace saved to {trace_path}")
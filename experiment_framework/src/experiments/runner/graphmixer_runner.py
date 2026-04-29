# src/experiments/runners/graphmixer_runner.py
import torch
from loguru import logger
from pathlib import Path
import numpy as np




from src.datasets.tawrmac_dataloading.data_pipeline import TAWRMACDataPipeline
from src.datasets.graphmixer_dataloading.data_pipeline import GraphMixerDataPipeline
from src.experiments.runner.base_runner import BaseRunner
from src.models.graphmixer_module.components.neighbor_sampler import NeighborSampler
from utils.neighbor_utils import build_adj_list



class GraphMixerRunner(BaseRunner):
    """Runner for GraphMixer models."""

    def create_data_pipeline(self):
        """Build GraphMixer pipeline: reuse TAWRMAC loading, add NeighborSampler."""
        # Start with TAWRMAC pipeline to reuse load(), build_samplers(), etc.
        pipeline = (TAWRMACDataPipeline(self.config)
                    .load()
                    .build_samplers()      # Reuse NegativeEdgeSampler logic
                    .build_datasets()      # Reuse dataset format
                    .build_loaders())      # Reuse DataLoader setup
        
        # Inject GraphMixer-specific NeighborSampler
        pipeline = self._attach_neighbor_sampler(pipeline)
        return pipeline
    
    def _attach_neighbor_sampler(self, pipeline: TAWRMACDataPipeline) -> TAWRMACDataPipeline:
        """Attach NeighborSampler to existing pipeline."""
        train_mask = pipeline.data['train_mask']
        train_edges = pipeline.data['edges'][train_mask]
        train_src = train_edges[:, 0].cpu().numpy()
        train_dst = train_edges[:, 1].cpu().numpy()
        train_ts = pipeline.data['timestamps'][train_mask].cpu().numpy()
        
        adj_list = build_adj_list(
            src_node_ids=train_src,
            dst_node_ids=train_dst,
            edge_ids=np.arange(len(train_src)),
            timestamps=train_ts,
            max_node_id=pipeline.data['num_nodes'] - 1
        )
        
        strategy = self.config['model'].get('sample_neighbor_strategy', 'uniform')
        time_scaling = self.config['model'].get('time_scaling_factor', 0.0)
        seed = self.config['experiment'].get('seed')
        
        pipeline.neighbor_sampler = NeighborSampler(
            adj_list=adj_list,
            sample_neighbor_strategy=strategy,
            time_scaling_factor=time_scaling,
            seed=seed
        )
        return pipeline

    def setup_model(self, model: torch.nn.Module, pipeline) -> None:
        """Inject NeighborSampler into GraphMixer model."""
        if hasattr(model, 'set_neighbor_sampler'):
            model.set_neighbor_sampler(pipeline.neighbor_sampler)
        else:
            model.neighbor_sampler = pipeline.neighbor_sampler
            if hasattr(pipeline.neighbor_sampler, 'reset_random_state'):
                pipeline.neighbor_sampler.reset_random_state()


    def _log_model_status(self, model: torch.nn.Module) -> None:
        """Log GraphMixer-specific architecture configuration."""
        logger.info(f"=== GraphMixer Model Status ===")
        logger.info(f"Time feature dim: {model.time_feat_dim}")
        logger.info(f"Num tokens (neighbors): {model.num_tokens}")
        logger.info(f"MLP-Mixer layers: {model.num_layers}")
        logger.info(f"Token expansion factor: {model.token_dim_expansion_factor}")
        logger.info(f"Channel expansion factor: {model.channel_dim_expansion_factor}")
        logger.info(f"Dropout: {model.dropout}")
        logger.info(f"Time gap (node encoder window): {getattr(model.cfg, 'time_gap', 2000)}")
        logger.info(f"Sampling strategy: {getattr(pipeline.neighbor_sampler, 'sample_neighbor_strategy', 'N/A')}")
        logger.info(f"Node feat dim: {model.node_feat_dim}")
        logger.info(f"Edge feat dim: {model.edge_feat_dim}")
        logger.info("=" * 40)

    def _profile_model(self, model: torch.nn.Module, pipeline) -> None:
        """Compute FLOPs and parameters for GraphMixer."""
        logger.info("Computing FLOPs for GraphMixer...")

        batch_size = min(self.config['training']['batch_size'], 64)  # GraphMixer is lighter
        num_neighbors = getattr(model.cfg, 'num_tokens', 20)
        time_gap = getattr(model.cfg, 'time_gap', 2000)
        device = next(model.parameters()).device

        # Create dummy NumPy arrays matching GraphMixer's expected input format
        src_nodes = np.random.randint(0, pipeline.num_nodes, size=batch_size)
        dst_nodes = np.random.randint(0, pipeline.num_nodes, size=batch_size)
        timestamps = np.random.uniform(0, 1e6, size=batch_size).astype(np.float32)

        model.eval()
        with torch.no_grad():
            # Warm-up run
            _ = model(
                src_node_ids=src_nodes,
                dst_node_ids=dst_nodes,
                node_interact_times=timestamps,
                num_neighbors=num_neighbors,
                time_gap=time_gap
            )

            # Profile with torch.profiler
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
                with_flops=True,
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                _ = model(
                    src_node_ids=src_nodes,
                    dst_node_ids=dst_nodes,
                    node_interact_times=timestamps,
                    num_neighbors=num_neighbors,
                    time_gap=time_gap
                )

            # Aggregate and log FLOPs
            key_avg = prof.key_averages()
            total_flops = sum([e.flops for e in key_avg if e.flops is not None])
            total_params = sum(p.numel() for p in model.parameters())
            
            logger.info(f"GraphMixer estimated FLOPs: {total_flops / 1e9:.3f} GFLOPs")
            logger.info(f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
            logger.info(f"Batch size used for profiling: {batch_size}")

            # Save trace for visualization
            if self.config.get('logging', {}).get('log_dir'):
                trace_path = Path(self.config['logging']['log_dir']) / "graphmixer_trace.json"
                prof.export_chrome_trace(str(trace_path))
                logger.info(f"Profiling trace saved to {trace_path}")

    def _get_forward_inputs(self, batch: dict, model: torch.nn.Module, pipeline) -> dict:
        """Prepare inputs for GraphMixer forward pass from a batch."""
        
        # src = batch['sources'].cpu().numpy()
        # dst = batch['destinations'].cpu().numpy()
        # ts = batch['timestamps'].cpu().numpy()
        # num_neighbors = self.config['model']['num_tokens']
        # time_gap = self.config['model'].get('time_gap', 2000)
        
        return {
            'src_node_ids': batch['sources'].cpu().numpy(),
            'dst_node_ids': batch['destinations'].cpu().numpy(),
            'node_interact_times': batch['timestamps'].cpu().numpy(),
            'num_neighbors': self.config['model'].get('num_tokens', 20),
            'time_gap': self.config['model'].get('time_gap', 2000),
        }
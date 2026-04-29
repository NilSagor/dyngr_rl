# src/datasets/freedyg_dataloading/data_pipeline.py
import numpy as np
from src.datasets.tawrmac_dataloading.data_pipeline import TAWRMACDataPipeline
from src.models.freedyg_module.components.neighbor_sampler import NeighborSampler
from utils.neighbor_utils import build_adj_list  


class FreeDyGDataPipeline(TAWRMACDataPipeline):
    """
    FreeDyG data pipeline: reuses TAWRMAC loading logic, adds NeighborSampler.
    
    FreeDyG uses the same batch format and negative sampling as TAWRMAC,
    so we inherit most functionality and only override sampler construction.
    """
    
    def build_neighbor_sampler(self) -> 'FreeDyGDataPipeline':
        """Build NeighborSampler for FreeDyG (replaces NewNeighborFinder)."""
        train_mask = self.data['train_mask']
        train_edges = self.data['edges'][train_mask]
        train_src = train_edges[:, 0].cpu().numpy()
        train_dst = train_edges[:, 1].cpu().numpy()
        train_ts = self.data['timestamps'][train_mask].cpu().numpy()
        
        # Build adjacency list (undirected graph)
        adj_list = build_adj_list(
            src_node_ids=train_src,
            dst_node_ids=train_dst,
            edge_ids=np.arange(len(train_src)),
            timestamps=train_ts,
            max_node_id=self.data['num_nodes'] - 1
        )
        
        # Create sampler with config-driven strategy
        strategy = self.config['model'].get('sample_neighbor_strategy', 'uniform')
        time_scaling = self.config['model'].get('time_scaling_factor', 0.0)
        seed = self.config['experiment'].get('seed')
        
        self.neighbor_sampler = NeighborSampler(
            adj_list=adj_list,
            sample_neighbor_strategy=strategy,
            time_scaling_factor=time_scaling,
            seed=seed
        )
        return self
    
    @property
    def train_neighbor_sampler(self):
        """Alias for pipeline compatibility (some runners expect this name)."""
        return self.neighbor_sampler
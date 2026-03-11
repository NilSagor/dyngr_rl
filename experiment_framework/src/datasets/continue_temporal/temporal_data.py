# temporal_data.py
"""Temporal dataset class for dynamic graph learning."""

from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
import torch
import numpy as np
from torch.utils.data import Dataset
from loguru import logger

if TYPE_CHECKING:
    from src.datasets.sam_dataloading.negative_sample import NegativeSampler



BALANCE_TOLERANCE = 0.01  # 1% tolerance for 50/50 balance

class TemporalDataset(Dataset):
    """Industry-standard temporal graph dataset (TGN/DyGLib compliant)."""
    
    def __init__(
        self,
        edges: torch.Tensor,
        timestamps: torch.Tensor,
        edge_features: Optional[torch.Tensor],
        num_nodes: int,
        split: str,
        negative_sampler: 'NegativeSampler',
        negative_sampling_strategy: str = 'random',
        unseen_nodes: Optional[torch.Tensor] = None,
        seed: int = 42
    ):
        self.edges = edges
        self.timestamps = timestamps
        self.edge_features = edge_features
        self.num_nodes = num_nodes
        self.split = split
        self.negative_sampler = negative_sampler
        self.negative_sampling_strategy = negative_sampling_strategy
        self.unseen_nodes = unseen_nodes.numpy() if unseen_nodes is not None else None
        
        # Local RNG to avoid global state pollution
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        
        self._prepare_samples()
        
        # Cache temporal bounds for ODE solvers
        self._cache_temporal_bounds()
    
    def _cache_temporal_bounds(self):
        """Cache temporal bounds for efficient ODE time domain queries."""
        if len(self.samples) > 0:
            all_timestamps = [s['timestamp'] for s in self.samples]
            self.t_min = min(all_timestamps)
            self.t_max = max(all_timestamps)
            self.time_span = self.t_max - self.t_min
        else:
            self.t_min = self.t_max = self.time_span = 0.0
    
    def get_temporal_bounds(self) -> Tuple[float, float]:
        """Get the temporal bounds of this dataset (for ODE time domain)."""
        return self.t_min, self.t_max
    
    def get_time_span(self) -> float:
        """Get total time span (t_max - t_min)."""
        return self.time_span
    
    def _prepare_samples(self):
        """Prepare exactly 1 negative sample per positive edge with ZERO edge features for negatives."""
        num_positives = len(self.edges)
        
        # Create positive samples with their indices for stable sorting
        pos_samples = []
        for i in range(num_positives):
            src, dst = self.edges[i].tolist()
            ts = self.timestamps[i].item()
            edge_feat = self.edge_features[i] if self.edge_features is not None else None
            
            pos_samples.append({
                'src': src, 'dst': dst, 'timestamp': ts, 'edge_feature': edge_feat,
                'label': 1.0, 'is_positive': True, 'orig_idx': i  # Track original index
            })
        
        # Sort positives by timestamp FIRST (stable sort preserves original order for ties)
        pos_samples.sort(key=lambda x: (x['timestamp'], x['orig_idx']))
        
        # Create negative samples with SAME timestamps as their NOW-SORTED paired positives
        # This ensures negatives follow the same temporal order as positives
        sorted_timestamps = [s['timestamp'] for s in pos_samples]
        sorted_srcs = torch.tensor([s['src'] for s in pos_samples])
        sorted_timestamps_np = np.array(sorted_timestamps)
        
        # Sample negatives based on the temporally ordered sources
        neg_dsts = self._sample_negatives(sorted_srcs.numpy(), sorted_timestamps_np)
        
        neg_samples = []
        for i, pos in enumerate(pos_samples):
            neg_feat = (
                torch.zeros_like(self.edge_features[0]) 
                if self.edge_features is not None 
                else None
            )
            neg_samples.append({
                'src': pos['src'],  # Same src as paired positive
                'dst': int(neg_dsts[i]),
                'timestamp': pos['timestamp'],  # Same timestamp (guaranteed monotonic)
                'edge_feature': neg_feat,
                'label': 0.0, 'is_positive': False,
            })
        
        # INTERLEAVE: pos[0], neg[0], pos[1], neg[1], ... 
        # This maintains temporal monotonicity because pos[i].timestamp <= pos[i+1].timestamp
        # and neg[i].timestamp == pos[i].timestamp
        interleaved = []
        for i in range(num_positives):
            interleaved.append(pos_samples[i])
            interleaved.append(neg_samples[i])
        
        self.samples = interleaved
        
        # Verify temporal ordering (should always pass now)
        timestamps = [s['timestamp'] for s in self.samples]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1] - 1e-9:  # Small epsilon for float comparison
                raise ValueError(f"Temporal ordering violated at index {i}: "
                            f"{timestamps[i]} < {timestamps[i-1]}")
        
        # Validation: check label balance
        pos_ratio = sum(s['label'] for s in self.samples) / len(self.samples)
        assert abs(pos_ratio - 0.5) <= BALANCE_TOLERANCE
        logger.info(f"{self.split} split: {len(self.samples)} samples "
                    f"(pos_ratio={pos_ratio:.3f}), time_range=[{min(timestamps):.0f}, {max(timestamps):.0f}]")
        
        
    def _sample_negatives(
        self, 
        neg_srcs: np.ndarray, 
        neg_timestamps: np.ndarray
    ) -> np.ndarray:
        """Dispatch to appropriate negative sampling strategy."""
        strategy = self.negative_sampling_strategy
        
        if strategy == 'random':
            return self.negative_sampler.random(neg_srcs)
        elif strategy == 'historical':
            return self.negative_sampler.historical(neg_srcs, neg_timestamps)
        elif strategy == 'inductive':
            if self.unseen_nodes is None:
                raise ValueError("unseen_nodes required for inductive sampling")
            return self.negative_sampler.inductive(neg_srcs, self.unseen_nodes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch sample by index. Assumes idx is integer (not list)."""
        if isinstance(idx, list):
            # Handle list indexing by returning list of samples
            return [self.samples[i] for i in idx]
        
        sample = self.samples[idx]
        return {
            'src_node': torch.tensor(sample['src'], dtype=torch.long),
            'dst_node': torch.tensor(sample['dst'], dtype=torch.long),
            'timestamp': torch.tensor(sample['timestamp'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'edge_feature': sample['edge_feature']
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function that PRESERVES temporal ordering from the sampler.
        
        CRITICAL: The batch list is already in the order yielded by the sampler,
        which is temporally ordered. We must NOT sort or reorder here.
        """
        # Stack tensors preserving input order (which is temporal order)
        batch_dict = {
            'src_nodes': torch.stack([item['src_node'] for item in batch]),
            'dst_nodes': torch.stack([item['dst_node'] for item in batch]),
            'timestamps': torch.stack([item['timestamp'] for item in batch]),
            'labels': torch.stack([item['label'] for item in batch]),
            'edge_features': torch.stack([item['edge_feature'] for item in batch]),
        }
        
        # Verify temporal monotonicity (debugging check)
        times = batch_dict['timestamps']
        if len(times) > 1:
            is_monotonic = torch.all(times[1:] >= times[:-1] - 1e-5)
            if not is_monotonic:
                # Find violations for debugging
                violations = (times[1:] < times[:-1] - 1e-5).nonzero(as_tuple=True)[0]
                violation_times = [(times[i].item(), times[i+1].item()) for i in violations[:3]]
                logger.warning(f"Temporal non-monotonicity detected in batch at positions: {violations.tolist()[:5]}, "
                              f"time_pairs={violation_times}")
        
        return batch_dict
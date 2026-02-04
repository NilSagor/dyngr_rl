"""Temporal dataset class for dynamic graph learning."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
import torch
import numpy as np
from torch.utils.data import Dataset
from loguru import logger


if TYPE_CHECKING:
    from .negative_sampling import NegativeSampler

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
    
    def _prepare_samples(self):
        """Prepare exactly 1 negative sample per positive edge."""
        num_positives = len(self.edges)
        self.samples = []
        
        # Positive samples
        for i in range(num_positives):
            src, dst = self.edges[i].tolist()
            ts = self.timestamps[i].item()
            edge_feat = self.edge_features[i] if self.edge_features is not None else None
            
            self.samples.append({
                'src': src,
                'dst': dst,
                'timestamp': ts,
                'edge_feature': edge_feat,
                'label': 1.0,
                'is_positive': True
            })
        
        # Negative samples (exactly 1 per positive)
        neg_srcs = self.edges[:, 0].numpy()
        neg_timestamps = self.timestamps.numpy()
        
        neg_dsts = self._sample_negatives(neg_srcs, neg_timestamps)
        
        for i in range(num_positives):
            self.samples.append({
                'src': int(neg_srcs[i]),
                'dst': int(neg_dsts[i]),
                'timestamp': float(neg_timestamps[i]),
                'edge_feature': None,
                'label': 0.0,
                'is_positive': False
            })
        
        # Shuffle training only
        if self.split == 'train':
            self.rng.shuffle(self.samples)
        
        # Validation
        pos_ratio = sum(s['label'] for s in self.samples) / len(self.samples)
        assert abs(pos_ratio - 0.5) <= BALANCE_TOLERANCE, \
            f"Label imbalance in {self.split}: {pos_ratio:.3f}"
        logger.info(f"{self.split} split: {len(self.samples)} samples (pos_ratio={pos_ratio:.3f})")
    
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
        """Industry-standard collation (TGN compatible)."""
        batch_dict = {
            'src_nodes': torch.stack([item['src_node'] for item in batch]),
            'dst_nodes': torch.stack([item['dst_node'] for item in batch]),
            'timestamps': torch.stack([item['timestamp'] for item in batch]),
            'labels': torch.stack([item['label'] for item in batch]),
        }
        
        edge_features = [item['edge_feature'] for item in batch]
        if any(ef is not None for ef in edge_features):
            # Assumes 1D edge features; gets feature dimension from first valid feature
            first_feat = next(ef for ef in edge_features if ef is not None)
            feat_dim = first_feat.shape[0]
            batch_dict['edge_features'] = torch.stack([
                ef if ef is not None else torch.zeros(feat_dim, dtype=first_feat.dtype)
                for ef in edge_features
            ])
        
        return batch_dict









# class TemporalDataset(Dataset):
#     """Dataset for temporal graph data.
    
#     This dataset handles temporal edges with timestamps and provides
#     functionality for negative sampling and neighborhood sampling.
#     """
    
#     def __init__(
#         self,
#         edges: torch.Tensor,
#         timestamps: torch.Tensor,
#         edge_features: Optional[torch.Tensor] = None,
#         num_nodes: int = None,
#         max_neighbors: int = 10,
#         split: str = 'train',
#         negative_sampling_ratio: float = 1.0,
#         negative_sampling_strategy: str = "random",
#         unseen_nodes = None,
#         device: str = "cpu" # device fixed
#     ):
#         """Initialize temporal dataset.
        
#         Args:
#             edges: Edge indices [num_edges, 2]
#             timestamps: Edge timestamps [num_edges]
#             edge_features: Optional edge features [num_edges, feature_dim]
#             num_nodes: Number of nodes (inferred if not provided)
#             max_neighbors: Maximum number of neighbors to sample
#             split: Dataset split ('train', 'val', 'test')
#             negative_sampling_ratio: Ratio of negative to positive samples
#         """
#         super().__init__()
        
#         self.edges = edges.to(device)
#         # print("Edges shape:", self.edges.shape)
#         # assert self.edges.shape[1] == 2, f"Expected 2 columns for edges, got {self.edges.shape[1]}"
#         self.timestamps = timestamps.to(device)
#         # print("Timestamps shape:", self.timestamps.shape)
#         self.edge_features = edge_features.to(device) if edge_features is not None else None        
#         self.max_neighbors = max_neighbors
#         self.split = split
#         self.negative_sampling_ratio = negative_sampling_ratio
#         self.negative_sampling_strategy = negative_sampling_strategy
#         self.device = device

#         self.unseen_nodes = unseen_nodes if unseen_nodes is not None else torch.tensor([]) 

        
        
#         # Infer number of nodes if not provided
#         if num_nodes is None:
#             self.num_nodes = int(edges.max().item())
#         else:
#             self.num_nodes = num_nodes
            
#         self.num_edges = len(edges)
        
#         # Sort edges by timestamp before creating samples
#         sorted_indices = torch.argsort(self.timestamps)
#         self.edges = self.edges[sorted_indices]
#         self.timestamps = self.timestamps[sorted_indices]
#         if self.edge_features is not None:
#             self.edge_features = self.edge_features[sorted_indices]
        
#         # Get actual number of nodes (excluding padding)
#         actual_max_node = int(self.edges.max().item())
#         self.num_nodes = actual_max_node  # This should be 9228 for Wikipedia


#         # Build adjacency list for neighborhood sampling
#         self._build_adjacency_list()
        
#         # Create positive edge set for negative sampling
#         self.positive_edge_set = set()
#         for src, dst in self.edges.tolist():
#             self.positive_edge_set.add((src, dst))
#             self.positive_edge_set.add((dst, src))  # Undirected

#         # Prepare positive and negative samples
#         self._prepare_samples()
        
#     def _build_adjacency_list(self):
#         """Build adjacency list for efficient neighborhood sampling."""
#         self.adj_list = {i: [] for i in range(self.num_nodes+1)}
        
#         for idx, (src, dst) in enumerate(self.edges):
#             timestamp = self.timestamps[idx].item()
#             self.adj_list[int(src.item())].append((dst.item(), timestamp))
#             # For undirected graphs, add reverse edge
#             self.adj_list[int(dst.item())].append((src.item(), timestamp))
            
#         # Sort neighbors by timestamp for temporal ordering
#         for node in self.adj_list:
#             self.adj_list[node].sort(key=lambda x: x[1])
            
    
#     def _prepare_samples(self):
#         """Prepare positive and negative samples with exact 1:1 balance."""
#         self.samples = []
        
#         # Create positive samples
#         for idx in range(self.num_edges):
#             # Positive sample
#             self.samples.append({
#                 'idx': idx,
#                 'label': 1.0,
#                 'src_node': self.edges[idx, 0].item(),
#                 'dst_node': self.edges[idx, 1].item(),
#                 'timestamp': self.timestamps[idx].item(),
#                 'edge_feature': self.edge_features[idx] if self.edge_features is not None else None,
#                 'is_positive': True
#             })
            
#             # Negative sample (only for non-training splits OR balanced training)
#             if self.split != 'train':
#                 # For val/test: generate one negative per positive
#                 if self.negative_sampling_strategy == "historical":
#                     neg_src, neg_dst, neg_ts = self._generate_single_historical_negative(
#                         self.edges[idx, 0].item(), 
#                         self.edges[idx, 1].item(), 
#                         self.timestamps[idx].item()
#                     )
#                 elif self.negative_sampling_strategy == "inductive":
#                     neg_src, neg_dst, neg_ts = self._generate_single_inductive_negative(
#                         self.edges[idx, 0].item(), 
#                         self.edges[idx, 1].item(), 
#                         self.timestamps[idx].item()
#                     )
#                 else:
#                     neg_src, neg_dst, neg_ts = self._generate_single_negative_random(
#                         self.edges[idx, 0].item(), 
#                         self.edges[idx, 1].item(), 
#                         self.timestamps[idx].item()
#                     )
                    
#                 self.samples.append({
#                     'idx': -1,
#                     'label': 0.0,
#                     'src_node': neg_src,
#                     'dst_node': neg_dst,
#                     'timestamp': neg_ts,
#                     'edge_feature': None,
#                     'is_positive': False
#                 })
        
#         # For training: use exactly 1:1 ratio (same number of negatives as positives)
#         if self.split == 'train':
#             num_positives = self.num_edges
#             negative_samples = []
#             for i in range(num_positives):
#                 pos_sample = {
#                     'src_node': self.edges[i, 0].item(),
#                     'dst_node': self.edges[i, 1].item(),
#                     'timestamp': self.timestamps[i].item()
#                 }
                
#                 if self.negative_sampling_strategy == "historical":
#                     neg_src, neg_dst, neg_ts = self._generate_single_historical_negative(
#                         pos_sample['src_node'], pos_sample['dst_node'], pos_sample['timestamp']
#                     )
#                 elif self.negative_sampling_strategy == "inductive":
#                     neg_src, neg_dst, neg_ts = self._generate_single_inductive_negative(
#                         pos_sample['src_node'], pos_sample['dst_node'], pos_sample['timestamp']
#                     )
#                 else:
#                     neg_src, neg_dst, neg_ts = self._generate_single_negative_random(
#                         pos_sample['src_node'], pos_sample['dst_node'], pos_sample['timestamp']
#                     )
                
#                 negative_samples.append({
#                     'idx': -1,
#                     'label': 0.0,
#                     'src_node': neg_src,
#                     'dst_node': neg_dst,
#                     'timestamp': neg_ts,
#                     'edge_feature': None,
#                     'is_positive': False
#                 })
            
#             self.samples.extend(negative_samples)
        
#         # Shuffle only for training
#         if self.split == 'train':
#             indices = torch.randperm(len(self.samples))
#             self.samples = [self.samples[i] for i in indices]

#         # Debug validation
#         if len(self.samples) > 0:
#             labels = [s['label'] for s in self.samples]
#             pos_ratio = sum(labels) / len(labels)
#             print(f"Split {self.split}: {len(self.samples)} samples, positive ratio = {pos_ratio:.3f}")
            
#             # Ensure we have both classes
#             assert 0.4 <= pos_ratio <= 0.6, f"Label imbalance: {pos_ratio:.3f}"
            
#     def _generate_single_historical_negative(self, pos_src, pos_dst, pos_timestamp):
#         """Generate historical negative: (pos_src, w, pos_timestamp) where w is a past neighbor of pos_src ≠ pos_dst"""
#         # Get all neighbors of pos_src BEFORE pos_timestamp
#         mask = (self.edges[:, 0] == pos_src) & (self.timestamps < pos_timestamp)
#         candidate_dsts = self.edges[mask, 1]
        
#         # Filter out the positive destination
#         valid_candidates = candidate_dsts[candidate_dsts != pos_dst]
        
#         if len(valid_candidates) > 0:
#             neg_dst = valid_candidates[torch.randint(0, len(valid_candidates), (1,))].item()
#             return pos_src, neg_dst, pos_timestamp
        
#         # Fallback to random if no historical neighbors
#         return self._generate_single_negative_random(pos_src, pos_dst, pos_timestamp)

#     def _generate_single_inductive_negative(self, pos_src, pos_dst, pos_timestamp):
#         """Generate inductive negative: sample from future nodes"""
#         """Generate inductive negative: (pos_src, unseen_node, pos_timestamp)"""
#         if hasattr(self, 'unseen_nodes') and len(self.unseen_nodes) > 0:
#             # Sample from truly unseen nodes
#             unseen_idx = torch.randint(0, len(self.unseen_nodes), (1,)).item()
#             neg_dst = self.unseen_nodes[unseen_idx].item()
#             return pos_src, neg_dst, pos_timestamp
#         else:
#             # Fallback if no unseen nodes available
#             return self._generate_single_negative_random(pos_src, pos_dst, pos_timestamp)

#     def _generate_single_negative(self, pos_src, pos_dst, pos_timestamp):
#         """Generate one negative sample for a given positive edge"""
#         if self.negative_sampling_strategy == "random":
#             # Sample random nodes that aren't the positive edge
#             attempts = 0
#             max_attempts = 100
#             while attempts < max_attempts:
#                 src = torch.randint(1, self.num_nodes + 1, (1,)).item()
#                 dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
#                 if src != dst and (src, dst) not in self.positive_edge_set:
#                     return src, dst, pos_timestamp
#             # Fallback to simple random if can't find valid pair
#             src = torch.randint(1, self.num_nodes + 1, (1,)).item()
#             dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
#             while src == dst:
#                 dst = torch.randint(1, self.num_nodes+1, (1,)).item()
#             return src, dst, pos_timestamp
                    
#         elif self.negative_sampling_strategy == "historical":
#             # Sample from historical edges before pos_timestamp
#             valid_mask = self.timestamps < pos_timestamp
#             if valid_mask.sum() > 0:
#                 valid_indices = torch.where(valid_mask)[0]
#                 rand_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
#                 return self.edges[rand_idx, 0].item(), self.edges[rand_idx, 1].item(), pos_timestamp
#             else:
#                 # No historical edges to fallback to random sampling
#                 return self._generate_single_negative_random(pos_src, pos_dst, pos_timestamp)  # fallback
                
#         else:  # inductive
#             # For inductive, use future timestamp approximation
#             future_timestamp = pos_timestamp + (self.timestamps.max().item() - pos_timestamp) * 0.1
#             src = torch.randint(1, self.num_nodes + 1, (1,)).item()
#             dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
#             while src == dst:
#                 dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
#             return src, dst, future_timestamp

#     def _generate_single_negative_random(self, pos_src, pos_dst, pos_timestamp):
#         """Generate valid random negative samples."""
#         max_attempts = 100
#         for _ in range(max_attempts):
#             # Sample from actual node range (1 to num_nodes inclusive)
#             src = torch.randint(1, self.num_nodes + 1, (1,)).item()
#             dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
            
#             # Ensure valid edge (not self-loop and not positive edge)
#             if src != dst and (src, dst) not in self.positive_edge_set:
#                 return src, dst, pos_timestamp
        
#         # Final fallback: use different destination
#         src = pos_src
#         dst = (pos_dst % self.num_nodes) + 1
#         if src == dst:
#             dst = (dst % self.num_nodes) + 1
#         return src, dst, pos_timestamp
    
#     def _generate_random_negatives(self, num_negative: int):
#         """Sample from all possible edges not in positive set"""
#         negatives = []
#         for _ in range(num_negative):
#             while True:
#                 src = torch.randint(1, self.num_nodes + 1, (1,)).item()
#                 dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
#                 if src != dst and not self._is_positive_edge(src, dst):
#                     break
#             timestamp = self.timestamps[torch.randint(0, len(self.timestamps), (1,))].item()
#             negatives.append((src, dst, timestamp))
#         return negatives
    
#     def _is_positive_edge(self, src, dst):
#         """Check if edge exists in positive set (use hash set for efficiency)"""
#         # Pre-compute positive edge set in __init__
#         return (src, dst) in self.positive_edge_set or (dst, src) in self.positive_edge_set

    
    
            
#     def _generate_inductive_negatives(self, num_negative:int)->List[Tuple[int, int, float]]:
#         """Sample from future test edges (for inductive setting)."""
#         negatives = []
#         # For true inductive, you'd need access to test edges
#         # Simple approximation: sample from later timestamps
#         for _ in range(num_negative):
#             # Sample timestamp from upper half of time range
#             min_time = self.timestamps.median().item()
#             max_time = self.timestamps.max().item()
#             timestamp = min_time + torch.rand(1, device=self.device).item() * (max_time - min_time)
            
#             # Sample random nodes
#             src_node = torch.randint(1, self.num_nodes + 1, (1,), device=self.device).item()
#             dst_node = torch.randint(1, self.num_nodes + 1, (1,), device=self.device).item()
#             while dst_node == src_node:
#                 dst_node = torch.randint(1, self.num_nodes + 1, (1,), device=self.device).item()
                
#             negatives.append((src_node, dst_node, timestamp))
#         return negatives

#     def _sample_neighbors(
#         self,
#         node: int,
#         timestamp: float,
#         max_neighbors: int
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Sample temporal neighbors for a node.
        
#         Args:
#             node: Node ID
#             timestamp: Current timestamp
#             max_neighbors: Maximum number of neighbors to sample
            
#         Returns:
#             Tuple of (neighbor_indices, neighbor_timestamps)
#         """
#         # Get neighbors that appeared before the current timestamp
#         valid_neighbors = [
#             (neighbor, ts) for neighbor, ts in self.adj_list[node]
#             if ts < timestamp
#         ]
        
#         if len(valid_neighbors) == 0:
#             # No valid neighbors, return padding
#             neighbors = torch.zeros(max_neighbors, dtype=torch.long, device=self.device)
#             neighbor_times = torch.zeros(max_neighbors, dtype=torch.float32, device=self.device)
#         else:
#             # Take most recent neighbors
#             valid_neighbors = valid_neighbors[-max_neighbors:]
#             neighbor_ids, neighbor_timestamps = zip(*valid_neighbors)
            
#             neighbors = torch.tensor(neighbor_ids, dtype=torch.long, device=self.device)
#             neighbor_times = torch.tensor(neighbor_timestamps, dtype=torch.float32, device=self.device)
            
#             # Pad if necessary
#             if len(neighbors) < max_neighbors:
#                 padding_size = max_neighbors - len(neighbors)
#                 neighbors = torch.cat([
#                     neighbors,
#                     torch.zeros(padding_size, dtype=torch.long, device=self.device)
#                 ])
#                 neighbor_times = torch.cat([
#                     neighbor_times,
#                     torch.zeros(padding_size, dtype=torch.float32, device=self.device)
#                 ])
                
#         return neighbors, neighbor_times
        
#     def __len__(self) -> int:
#         return len(self.samples)
        
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         """Get a single sample."""
#         sample = self.samples[idx]
        
#         # Sample neighbors for source and destination nodes
#         src_neighbors, src_neighbor_times = self._sample_neighbors(
#             sample['src_node'],
#             sample['timestamp'],
#             self.max_neighbors
#         )
        
#         dst_neighbors, dst_neighbor_times = self._sample_neighbors(
#             sample['dst_node'],
#             sample['timestamp'],
#             self.max_neighbors
#         )
        
#         return {
#             'src_node': torch.tensor(sample['src_node'], dtype=torch.long, device=self.device),
#             'dst_node': torch.tensor(sample['dst_node'], dtype=torch.long, device=self.device),
#             'timestamp': torch.tensor(sample['timestamp'], dtype=torch.float32, device=self.device),
#             'label': torch.tensor(sample['label'], dtype=torch.float32, device=self.device),
#             'src_neighbors': src_neighbors,
#             'dst_neighbors': dst_neighbors,
#             'src_neighbor_times': src_neighbor_times,
#             'dst_neighbor_times': dst_neighbor_times,
#             'edge_feature': sample['edge_feature']
#         }
        
#     def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         """Collate function for batching."""
        
#         # Stack all tensors
#         batch_dict = {
#             'src_nodes': torch.stack([item['src_node'] for item in batch]),
#             'dst_nodes': torch.stack([item['dst_node'] for item in batch]),
#             'timestamps': torch.stack([item['timestamp'] for item in batch]),
#             'labels': torch.stack([item['label'] for item in batch]),
#             'src_neighbors': torch.stack([item['src_neighbors'] for item in batch]),
#             'dst_neighbors': torch.stack([item['dst_neighbors'] for item in batch]),
#             'src_neighbor_times': torch.stack([item['src_neighbor_times'] for item in batch]),
#             'dst_neighbor_times': torch.stack([item['dst_neighbor_times'] for item in batch]),
#             'batch_size': len(batch)
#         }
        
#         # Handle optional edge features
#         edge_features = [item['edge_feature'] for item in batch]
#         if any(ef is not None for ef in edge_features):
#             # Get feature dimension
#             feat_dim = next(ef.shape[0] for ef in edge_features if ef is not None)
#             # Stack edge features if they exist
#             batch_dict['edge_features'] = torch.stack([
#                 ef if ef is not None else torch.zeros(feat_dim, device=self.device)
#                 for ef in edge_features
#             ])
            
#         return batch_dict
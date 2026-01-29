"""Temporal dataset class for dynamic graph learning."""

from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset


class TemporalDataset(Dataset):
    """Dataset for temporal graph data.
    
    This dataset handles temporal edges with timestamps and provides
    functionality for negative sampling and neighborhood sampling.
    """
    
    def __init__(
        self,
        edges: torch.Tensor,
        timestamps: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        num_nodes: int = None,
        max_neighbors: int = 10,
        split: str = 'train',
        negative_sampling_ratio: float = 1.0,
        negative_sampling_strategy: str = "random",
        unseen_nodes = None,
        device: str = "cpu" # device fixed
    ):
        """Initialize temporal dataset.
        
        Args:
            edges: Edge indices [num_edges, 2]
            timestamps: Edge timestamps [num_edges]
            edge_features: Optional edge features [num_edges, feature_dim]
            num_nodes: Number of nodes (inferred if not provided)
            max_neighbors: Maximum number of neighbors to sample
            split: Dataset split ('train', 'val', 'test')
            negative_sampling_ratio: Ratio of negative to positive samples
        """
        super().__init__()
        
        self.edges = edges.to(device)
        print("Edges shape:", self.edges.shape)
        # assert self.edges.shape[1] == 2, f"Expected 2 columns for edges, got {self.edges.shape[1]}"
        self.timestamps = timestamps.to(device)
        print("Timestamps shape:", self.timestamps.shape)
        self.edge_features = edge_features.to(device) if edge_features is not None else None        
        self.max_neighbors = max_neighbors
        self.split = split
        self.negative_sampling_ratio = negative_sampling_ratio
        self.negative_sampling_strategy = negative_sampling_strategy
        self.device = device

        self.unseen_nodes = unseen_nodes if unseen_nodes is not None else torch.tensor([]) 

        
        
        # Infer number of nodes if not provided
        if num_nodes is None:
            self.num_nodes = int(edges.max().item())
        else:
            self.num_nodes = num_nodes
            
        self.num_edges = len(edges)
        
        # Sort edges by timestamp before creating samples
        sorted_indices = torch.argsort(self.timestamps)
        self.edges = self.edges[sorted_indices]
        self.timestamps = self.timestamps[sorted_indices]
        if self.edge_features is not None:
            self.edge_features = self.edge_features[sorted_indices]
        
        # Get actual number of nodes (excluding padding)
        actual_max_node = int(self.edges.max().item())
        self.num_nodes = actual_max_node  # This should be 9228 for Wikipedia


        # Build adjacency list for neighborhood sampling
        self._build_adjacency_list()
        
        # Create positive edge set for negative sampling
        self.positive_edge_set = set()
        for src, dst in self.edges.tolist():
            self.positive_edge_set.add((src, dst))
            self.positive_edge_set.add((dst, src))  # Undirected

        # Prepare positive and negative samples
        self._prepare_samples()
        
    def _build_adjacency_list(self):
        """Build adjacency list for efficient neighborhood sampling."""
        self.adj_list = {i: [] for i in range(self.num_nodes+1)}
        
        for idx, (src, dst) in enumerate(self.edges):
            timestamp = self.timestamps[idx].item()
            self.adj_list[int(src.item())].append((dst.item(), timestamp))
            # For undirected graphs, add reverse edge
            self.adj_list[int(dst.item())].append((src.item(), timestamp))
            
        # Sort neighbors by timestamp for temporal ordering
        for node in self.adj_list:
            self.adj_list[node].sort(key=lambda x: x[1])
            
    
    def _prepare_samples(self):
        """Prepare positive and negative samples."""
        self.samples = []
        
        # Add positive samples
        for idx in range(self.num_edges):
            self.samples.append({
                'idx': idx,
                'label': 1.0,
                'src_node': self.edges[idx, 0].item(),
                'dst_node': self.edges[idx, 1].item(),
                'timestamp': self.timestamps[idx].item(),
                'edge_feature': self.edge_features[idx] if self.edge_features is not None else None,
                'is_positive': True
            })
            
            # Add ONE negative sample per positive sample (for evaluation)
            if self.split != 'train':
                if self.negative_sampling_strategy == "historical":
                    neg_src, neg_dst, neg_ts = self._generate_single_historical_negative(
                        self.edges[idx, 0].item(), 
                        self.edges[idx, 1].item(), 
                        self.timestamps[idx].item()
                    )
                elif self.negative_sampling_strategy == "inductive":
                    neg_src, neg_dst, neg_ts = self._generate_single_inductive_negative(
                        self.edges[idx, 0].item(), 
                        self.edges[idx, 1].item(), 
                        self.timestamps[idx].item()
                    )
                else:
                    neg_src, neg_dst, neg_ts = self._generate_single_negative_random(
                        self.edges[idx, 0].item(), 
                        self.edges[idx, 1].item(), 
                        self.timestamps[idx].item()
                    )

                self.samples.append({
                    'idx': -1,
                    'label': 0.0,
                    'src_node': neg_src,
                    'dst_node': neg_dst,
                    'timestamp': neg_ts,
                    'edge_feature': None,
                    'is_positive': False
                })
        
        # For training: use negative_sampling_ratio
        # if self.split == 'train':
        #     num_negative = int(self.num_edges * self.negative_sampling_ratio)
            
        #     if self.negative_sampling_strategy == "historical":
        #         negative_samples = self._generate_historical_negatives(num_negative)
        #     elif self.negative_sampling_strategy == "inductive":
        #         negative_samples = self._generate_inductive_negatives(num_negative)
        #     else:  # random
        #         negative_samples = self._generate_random_negatives(num_negative)
            
        if self.split == "train":
            num_negative = int(self.num_edges*self.negative_sampling_ratio)
            negative_samples = self._generate_random_negatives(num_negative)

            # negative_samples = self._generate_random_negatives(num_negative)
            for src_node, dst_node, timestamp in negative_samples:
                self.samples.append({
                    'idx': -1,
                    'label': 0.0,
                    'src_node': src_node,
                    'dst_node': dst_node,
                    'timestamp': timestamp,
                    'edge_feature': None,
                    'is_positive': False
                })
        
        # Shuffle only for training
        if self.split == 'train':
            indices = torch.randperm(len(self.samples))
            self.samples = [self.samples[i] for i in indices]
            
    def _generate_single_historical_negative(self, pos_src, pos_dst, pos_timestamp):
        """Generate historical negative: (pos_src, w, pos_timestamp) where w is a past neighbor of pos_src ≠ pos_dst"""
        # Get all neighbors of pos_src BEFORE pos_timestamp
        mask = (self.edges[:, 0] == pos_src) & (self.timestamps < pos_timestamp)
        candidate_dsts = self.edges[mask, 1]
        
        # Filter out the positive destination
        valid_candidates = candidate_dsts[candidate_dsts != pos_dst]
        
        if len(valid_candidates) > 0:
            neg_dst = valid_candidates[torch.randint(0, len(valid_candidates), (1,))].item()
            return pos_src, neg_dst, pos_timestamp
        
        # Fallback to random if no historical neighbors
        return self._generate_single_negative_random(pos_src, pos_dst, pos_timestamp)

    def _generate_single_inductive_negative(self, pos_src, pos_dst, pos_timestamp):
        """Generate inductive negative: sample from future nodes"""
        """Generate inductive negative: (pos_src, unseen_node, pos_timestamp)"""
        if hasattr(self, 'unseen_nodes') and len(self.unseen_nodes) > 0:
            # Sample from truly unseen nodes
            unseen_idx = torch.randint(0, len(self.unseen_nodes), (1,)).item()
            neg_dst = self.unseen_nodes[unseen_idx].item()
            return pos_src, neg_dst, pos_timestamp
        else:
            # Fallback if no unseen nodes available
            return self._generate_single_negative_random(pos_src, pos_dst, pos_timestamp)

    def _generate_single_negative(self, pos_src, pos_dst, pos_timestamp):
        """Generate one negative sample for a given positive edge"""
        if self.negative_sampling_strategy == "random":
            # Sample random nodes that aren't the positive edge
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts:
                src = torch.randint(1, self.num_nodes + 1, (1,)).item()
                dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
                if src != dst and (src, dst) not in self.positive_edge_set:
                    return src, dst, pos_timestamp
            # Fallback to simple random if can't find valid pair
            src = torch.randint(1, self.num_nodes + 1, (1,)).item()
            dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
            while src == dst:
                dst = torch.randint(1, self.num_nodes+1, (1,)).item()
            return src, dst, pos_timestamp
                    
        elif self.negative_sampling_strategy == "historical":
            # Sample from historical edges before pos_timestamp
            valid_mask = self.timestamps < pos_timestamp
            if valid_mask.sum() > 0:
                valid_indices = torch.where(valid_mask)[0]
                rand_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
                return self.edges[rand_idx, 0].item(), self.edges[rand_idx, 1].item(), pos_timestamp
            else:
                # No historical edges to fallback to random sampling
                return self._generate_single_negative_random(pos_src, pos_dst, pos_timestamp)  # fallback
                
        else:  # inductive
            # For inductive, use future timestamp approximation
            future_timestamp = pos_timestamp + (self.timestamps.max().item() - pos_timestamp) * 0.1
            src = torch.randint(1, self.num_nodes + 1, (1,)).item()
            dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
            while src == dst:
                dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
            return src, dst, future_timestamp

    def _generate_single_negative_random(self, pos_src, pos_dst, pos_timestamp):
        """Helper method for random negative sampling without recursion"""
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            src = torch.randint(1, self.num_nodes + 1, (1,)).item()
            dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
            if src != dst and (src, dst) not in self.positive_edge_set:
                return src, dst, pos_timestamp
        attempts += 1
        
        # Final fallback
        src = torch.randint(1, self.num_nodes + 1, (1,)).item()
        dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
        while src == dst:
            dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
        return src, dst, pos_timestamp
    
    def _generate_random_negatives(self, num_negative: int):
        """Sample from all possible edges not in positive set"""
        negatives = []
        for _ in range(num_negative):
            while True:
                src = torch.randint(1, self.num_nodes + 1, (1,)).item()
                dst = torch.randint(1, self.num_nodes + 1, (1,)).item()
                if src != dst and not self._is_positive_edge(src, dst):
                    break
            timestamp = self.timestamps[torch.randint(0, len(self.timestamps), (1,))].item()
            negatives.append((src, dst, timestamp))
        return negatives
    
    def _is_positive_edge(self, src, dst):
        """Check if edge exists in positive set (use hash set for efficiency)"""
        # Pre-compute positive edge set in __init__
        return (src, dst) in self.positive_edge_set or (dst, src) in self.positive_edge_set

    
    
            
    def _generate_inductive_negatives(self, num_negative:int)->List[Tuple[int, int, float]]:
        """Sample from future test edges (for inductive setting)."""
        negatives = []
        # For true inductive, you'd need access to test edges
        # Simple approximation: sample from later timestamps
        for _ in range(num_negative):
            # Sample timestamp from upper half of time range
            min_time = self.timestamps.median().item()
            max_time = self.timestamps.max().item()
            timestamp = min_time + torch.rand(1, device=self.device).item() * (max_time - min_time)
            
            # Sample random nodes
            src_node = torch.randint(1, self.num_nodes + 1, (1,), device=self.device).item()
            dst_node = torch.randint(1, self.num_nodes + 1, (1,), device=self.device).item()
            while dst_node == src_node:
                dst_node = torch.randint(1, self.num_nodes + 1, (1,), device=self.device).item()
                
            negatives.append((src_node, dst_node, timestamp))
        return negatives

    def _sample_neighbors(
        self,
        node: int,
        timestamp: float,
        max_neighbors: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample temporal neighbors for a node.
        
        Args:
            node: Node ID
            timestamp: Current timestamp
            max_neighbors: Maximum number of neighbors to sample
            
        Returns:
            Tuple of (neighbor_indices, neighbor_timestamps)
        """
        # Get neighbors that appeared before the current timestamp
        valid_neighbors = [
            (neighbor, ts) for neighbor, ts in self.adj_list[node]
            if ts < timestamp
        ]
        
        if len(valid_neighbors) == 0:
            # No valid neighbors, return padding
            neighbors = torch.zeros(max_neighbors, dtype=torch.long, device=self.device)
            neighbor_times = torch.zeros(max_neighbors, dtype=torch.float32, device=self.device)
        else:
            # Take most recent neighbors
            valid_neighbors = valid_neighbors[-max_neighbors:]
            neighbor_ids, neighbor_timestamps = zip(*valid_neighbors)
            
            neighbors = torch.tensor(neighbor_ids, dtype=torch.long, device=self.device)
            neighbor_times = torch.tensor(neighbor_timestamps, dtype=torch.float32, device=self.device)
            
            # Pad if necessary
            if len(neighbors) < max_neighbors:
                padding_size = max_neighbors - len(neighbors)
                neighbors = torch.cat([
                    neighbors,
                    torch.zeros(padding_size, dtype=torch.long, device=self.device)
                ])
                neighbor_times = torch.cat([
                    neighbor_times,
                    torch.zeros(padding_size, dtype=torch.float32, device=self.device)
                ])
                
        return neighbors, neighbor_times
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Sample neighbors for source and destination nodes
        src_neighbors, src_neighbor_times = self._sample_neighbors(
            sample['src_node'],
            sample['timestamp'],
            self.max_neighbors
        )
        
        dst_neighbors, dst_neighbor_times = self._sample_neighbors(
            sample['dst_node'],
            sample['timestamp'],
            self.max_neighbors
        )
        
        return {
            'src_node': torch.tensor(sample['src_node'], dtype=torch.long, device=self.device),
            'dst_node': torch.tensor(sample['dst_node'], dtype=torch.long, device=self.device),
            'timestamp': torch.tensor(sample['timestamp'], dtype=torch.float32, device=self.device),
            'label': torch.tensor(sample['label'], dtype=torch.float32, device=self.device),
            'src_neighbors': src_neighbors,
            'dst_neighbors': dst_neighbors,
            'src_neighbor_times': src_neighbor_times,
            'dst_neighbor_times': dst_neighbor_times,
            'edge_feature': sample['edge_feature']
        }
        
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        
        # Stack all tensors
        batch_dict = {
            'src_nodes': torch.stack([item['src_node'] for item in batch]),
            'dst_nodes': torch.stack([item['dst_node'] for item in batch]),
            'timestamps': torch.stack([item['timestamp'] for item in batch]),
            'labels': torch.stack([item['label'] for item in batch]),
            'src_neighbors': torch.stack([item['src_neighbors'] for item in batch]),
            'dst_neighbors': torch.stack([item['dst_neighbors'] for item in batch]),
            'src_neighbor_times': torch.stack([item['src_neighbor_times'] for item in batch]),
            'dst_neighbor_times': torch.stack([item['dst_neighbor_times'] for item in batch]),
            'batch_size': len(batch)
        }
        
        # Handle optional edge features
        edge_features = [item['edge_feature'] for item in batch]
        if any(ef is not None for ef in edge_features):
            # Get feature dimension
            feat_dim = next(ef.shape[0] for ef in edge_features if ef is not None)
            # Stack edge features if they exist
            batch_dict['edge_features'] = torch.stack([
                ef if ef is not None else torch.zeros(feat_dim, device=self.device)
                for ef in edge_features
            ])
            
        return batch_dict
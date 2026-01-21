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
        self.device = device # store device

        
        # Infer number of nodes if not provided
        if num_nodes is None:
            self.num_nodes = int(edges.max().item() + 1)
        else:
            self.num_nodes = num_nodes
            
        self.num_edges = len(edges)
        
        # Build adjacency list for neighborhood sampling
        self._build_adjacency_list()
        
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
        
        # Positive samples
        self.positive_samples = list(range(self.num_edges))
        
        # Generate negative samples
        num_negative = int(self.num_edges * self.negative_sampling_ratio)
        self.negative_samples = self._generate_negative_samples(num_negative)
        
        # Combine positive and negative samples
        self.samples = []
        
        # Add positive samples
        for idx in self.positive_samples:
            self.samples.append({
                'idx': idx,
                'label': 1.0,
                'src_node': self.edges[idx, 0].item(),
                'dst_node': self.edges[idx, 1].item(),
                'timestamp': self.timestamps[idx].item(),
                'edge_feature': self.edge_features[idx] if self.edge_features is not None else None
            })
            
        # Add negative samples
        for src_node, dst_node, timestamp in self.negative_samples:
            self.samples.append({
                'idx': -1,  # Negative samples don't have original indices
                'label': 0.0,
                'src_node': src_node,
                'dst_node': dst_node,
                'timestamp': timestamp,
                'edge_feature': None
            })
            
        # Shuffle samples for training
        if self.split == 'train':
            indices = torch.randperm(len(self.samples))
            self.samples = [self.samples[i] for i in indices]
            
    def _generate_negative_samples(self, num_negative: int) -> List[Tuple[int, int, float]]:
        """Generate negative samples by random sampling.
        
        Args:
            num_negative: Number of negative samples to generate
            
        Returns:
            List of (src_node, dst_node, timestamp) tuples
        """
        negative_samples = []
        
        for _ in range(num_negative):
            # Randomly select a positive edge as reference
            idx = torch.randint(0, self.num_edges, (1,)).item()
            timestamp = self.timestamps[idx].item()
            
            # Random source node
            src_node = torch.randint(1, self.num_nodes+1, (1,), device=self.device).item()
            
            # Random destination node (ensure it's not the same as source)
            dst_node = src_node
            while dst_node == src_node:
                dst_node = torch.randint(1, self.num_nodes+1, (1,), device=self.device).item()
                
            negative_samples.append((src_node, dst_node, timestamp))
            
        return negative_samples
        
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
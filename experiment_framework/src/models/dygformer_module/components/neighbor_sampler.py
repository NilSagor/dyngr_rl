import numpy as np
import torch
from typing import List, Tuple, Optional, Union


class NeighborSampler:
    """
    Temporal neighbor sampler supporting multiple strategies:
    - 'uniform': Random sampling with equal probability
    - 'recent': Take most recent neighbors  
    - 'time_interval_aware': Bias toward recent interactions (CAWN-style)
    """

    def __init__(self, adj_list: List[List[Tuple[int, int, float]]], 
                 sample_neighbor_strategy: str = 'uniform', 
                 time_scaling_factor: float = 0.0, 
                 seed: Optional[int] = None):
        """
        :param adj_list: List where adj_list[i] contains tuples (neighbor_id, edge_id, timestamp)
        :param sample_neighbor_strategy: 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: Controls recency bias (larger = more recent preference)
        :param seed: Random seed for reproducibility
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed
        self.time_scaling_factor = time_scaling_factor

        # Pre-sorted neighbor storage per node
        self.nodes_neighbor_ids: List[np.ndarray] = []
        self.nodes_edge_ids: List[np.ndarray] = []
        self.nodes_neighbor_times: List[np.ndarray] = []
        
        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities: List[np.ndarray] = []

        # Build adjacency structures (sorted by timestamp)
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # Stable sort by timestamp
            sorted_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            
            self.nodes_neighbor_ids.append(
                np.array([n[0] for n in sorted_neighbors], dtype=np.int64))
            self.nodes_edge_ids.append(
                np.array([n[1] for n in sorted_neighbors], dtype=np.int64))
            self.nodes_neighbor_times.append(
                np.array([n[2] for n in sorted_neighbors], dtype=np.float64))

            # Pre-compute probabilities for time_interval_aware strategy
            if self.sample_neighbor_strategy == 'time_interval_aware':
                times = np.array([n[2] for n in sorted_neighbors], dtype=np.float64)
                probs = self._compute_sampled_probabilities(times)
                self.nodes_neighbor_sampled_probabilities.append(probs)

        # Initialize reproducible random state
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def _compute_sampled_probabilities(self, neighbor_times: np.ndarray) -> np.ndarray:
        """Compute time-aware sampling probabilities: more recent = higher probability."""
        if len(neighbor_times) == 0:
            return np.array([], dtype=np.float64)
        
        # Time deltas relative to most recent (all <= 0)
        time_deltas = neighbor_times - np.max(neighbor_times)
        
        # Exponential weighting
        weights = np.exp(self.time_scaling_factor * time_deltas)
        
        # Proper normalization
        total_weight = np.sum(weights)
        if total_weight > 1e-10:
            probabilities = weights / total_weight
        else:
            # Fallback to uniform if weights are invalid
            probabilities = np.ones(len(neighbor_times), dtype=np.float64) / len(neighbor_times)
        
        return probabilities

    def _find_neighbors_before(self, node_id: int, interact_time: float, 
                              return_probs: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Binary search to find all neighbors interacting before interact_time."""
        idx = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time, side='left')
        
        neighbor_ids = self.nodes_neighbor_ids[node_id][:idx]
        edge_ids = self.nodes_edge_ids[node_id][:idx]
        timestamps = self.nodes_neighbor_times[node_id][:idx]
        
        if return_probs and self.sample_neighbor_strategy == 'time_interval_aware':
            probs = self.nodes_neighbor_sampled_probabilities[node_id][:idx]
            return neighbor_ids, edge_ids, timestamps, probs
        
        return neighbor_ids, edge_ids, timestamps, None

    def get_historical_neighbors(self, node_ids: np.ndarray, 
                                  node_interact_times: np.ndarray, 
                                  num_neighbors: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample historical neighbors for batch of nodes."""
        assert num_neighbors > 0, 'num_neighbors must be positive'
        
        batch_size = len(node_ids)
        
        # Output arrays: zeros serve as padding for missing neighbors
        sampled_neighbor_ids = np.zeros((batch_size, num_neighbors), dtype=np.int64)
        sampled_edge_ids = np.zeros((batch_size, num_neighbors), dtype=np.int64)
        sampled_times = np.zeros((batch_size, num_neighbors), dtype=np.float32)

        for i, (node_id, interact_time) in enumerate(zip(node_ids, node_interact_times)):
            neighbor_ids, edge_ids, timestamps, probs = self._find_neighbors_before(
                node_id, interact_time, 
                return_probs=(self.sample_neighbor_strategy == 'time_interval_aware')
            )
            
            num_available = len(neighbor_ids)
            
            if num_available == 0:
                continue  # Keep zeros (padding)
            
            if self.sample_neighbor_strategy == 'recent':
                # Take most recent neighbors
                start_idx = max(0, num_available - num_neighbors)
                n_sampled = min(num_neighbors, num_available)
                sampled_neighbor_ids[i, :n_sampled] = neighbor_ids[start_idx:]
                sampled_edge_ids[i, :n_sampled] = edge_ids[start_idx:]
                sampled_times[i, :n_sampled] = timestamps[start_idx:]
                
            elif self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                if num_available <= num_neighbors:
                    # Use all available, pad remainder with zeros
                    sampled_neighbor_ids[i, :num_available] = neighbor_ids
                    sampled_edge_ids[i, :num_available] = edge_ids
                    sampled_times[i, :num_available] = timestamps
                else:
                    # Proper probability handling
                    if self.sample_neighbor_strategy == 'time_interval_aware' and probs is not None:
                        sampling_probs = probs
                    else:
                        sampling_probs = None  # Uniform sampling
                    
                    # Use consistent random state
                    rng = self.random_state if self.seed is not None else np.random
                    
                    sampled_idx = rng.choice(
                        num_available, 
                        size=num_neighbors, 
                        replace=False,
                        p=sampling_probs
                    )
                    
                    sampled_neighbor_ids[i, :] = neighbor_ids[sampled_idx]
                    sampled_edge_ids[i, :] = edge_ids[sampled_idx]
                    sampled_times[i, :] = timestamps[sampled_idx]
                    
                    # Optional: Sort by timestamp for model consistency
                    sort_order = np.argsort(sampled_times[i, :])
                    sampled_neighbor_ids[i, :] = sampled_neighbor_ids[i, :][sort_order]
                    sampled_edge_ids[i, :] = sampled_edge_ids[i, :][sort_order]
                    sampled_times[i, :] = sampled_times[i, :][sort_order]
            else:
                raise ValueError(f'Unknown strategy: {self.sample_neighbor_strategy}')

        return sampled_neighbor_ids, sampled_edge_ids, sampled_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, 
                                node_interact_times: np.ndarray, 
                                num_neighbors: int = 20) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Sample multi-hop temporal neighbors for walk-based models."""
        assert num_hops > 0, 'num_hops must be positive'
        
        # First hop
        neighbor_ids, edge_ids, timestamps = self.get_historical_neighbors(
            node_ids, node_interact_times, num_neighbors
        )
        
        ids_list = [neighbor_ids]
        edges_list = [edge_ids]
        times_list = [timestamps]
        
        # Subsequent hops
        for hop in range(1, num_hops):
            prev_ids = ids_list[-1].flatten()
            prev_times = times_list[-1].flatten()
            
            hop_ids, hop_edges, hop_times = self.get_historical_neighbors(
                prev_ids, prev_times, num_neighbors
            )
            
            new_shape = (len(node_ids), -1)
            hop_ids = hop_ids.reshape(new_shape)
            hop_edges = hop_edges.reshape(new_shape)
            hop_times = hop_times.reshape(new_shape)
            
            ids_list.append(hop_ids)
            edges_list.append(hop_edges)
            times_list.append(hop_times)
        
        return ids_list, edges_list, times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, 
                                     node_interact_times: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Get ALL first-hop neighbors (no sampling) - for variable-length sequence models."""
        neighbor_ids_list = []
        edge_ids_list = []
        timestamps_list = []
        
        for node_id, interact_time in zip(node_ids, node_interact_times):
            n_ids, e_ids, t_times, _ = self._find_neighbors_before(
                node_id, interact_time, return_probs=False)
            neighbor_ids_list.append(n_ids)
            edge_ids_list.append(e_ids)
            timestamps_list.append(t_times)
        
        return neighbor_ids_list, edge_ids_list, timestamps_list

    def reset_random_state(self):
        """Reset RNG for reproducible sampling."""
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)


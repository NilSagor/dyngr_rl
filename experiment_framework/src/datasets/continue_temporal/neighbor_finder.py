from typing import Dict, List, Optional, Tuple, Union
import torch
import bisect
import numpy as np

class NeighborFinder:
    """
    FIXED: Built ONLY from training edges to prevent temporal leakage.
    Industry standard: TGN (ICML 2020), DyGLib temporal splits.
    
    Thread-safe for read-only queries after initialization.
    """
    
    def __init__(self, 
                 train_edges: torch.Tensor, 
                 train_timestamps: torch.Tensor,
                 max_neighbors: int = 20,
                 undirected: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            train_edges: [E_train, 2] LongTensor of 0-indexed training edges ONLY
            train_timestamps: [E_train] FloatTensor of training timestamps ONLY
            max_neighbors: maximum neighbors to store per node
            undirected: if True, add reverse edges (default: True)
            undirected: if True, add reverse edges
            seed: random seed for reproducibility
        """
        # Input validation
        if not isinstance(train_edges, torch.Tensor):
            raise TypeError(
                f"train_edges must be torch.Tensor, got {type(train_edges)}"
            )
        if not isinstance(train_timestamps, torch.Tensor):
            raise TypeError(
                f"train_timestamps must be torch.Tensor, got {type(train_timestamps)}"
            )
        
        # Ensure correct shapes and types
        train_edges = train_edges.long().contiguous()
        train_timestamps = train_timestamps.float().contiguous()

        # Handle both [E, 2] and [2, E] formats - standardize to [2, E]
        if train_edges.shape[0] == 2:
            # Already [2, E]
            self.edge_index = train_edges
            edges_for_iteration = train_edges.t()  # [E, 2] for iteration
        elif train_edges.shape[1] == 2:
            # [E, 2] format
            self.edge_index = train_edges.t().contiguous()  # [2, E]
            edges_for_iteration = train_edges
        else:
            raise ValueError(f"train_edges must be [E, 2] or [2, E], got {train_edges.shape}")
        
        self.edge_time = train_timestamps
        self.max_neighbors = max_neighbors
        self.undirected = undirected
                
        
        
        # Single unified adjacency list: node -> [(neighbor, timestamp, edge_id), ...]
        self.edge_id_adj_list: Dict[int, List[Tuple[int, float, int]]] = {}
        self._timestamps_cache: Dict[int, List[float]] = {}

        # Build from training edges
        edges_list = edges_for_iteration.tolist()
        timestamps_list = train_timestamps.tolist()
        
        for edge_idx, ((src, dst), ts) in enumerate(zip(edges_list, timestamps_list)):
            self._add_edge_with_id(src, dst, ts, edge_idx)
            if undirected:
                self._add_edge_with_id(dst, src, ts, edge_idx)
        
        # Sort by timestamp (oldest first) for binary search
        for node in self.edge_id_adj_list:
            self.edge_id_adj_list[node].sort(key=lambda x: x[1])
            self._timestamps_cache[node] = [t for _, t, _ in self.edge_id_adj_list[node]]
        
        # Statistics
        self.num_nodes = len(self.edge_id_adj_list)
        self.num_edges = len(edges_list)
        self.max_timestamp = max(timestamps_list) if timestamps_list else 0.0
        self.min_timestamp = min(timestamps_list) if timestamps_list else 0.0
    
        
    
    def _add_edge_with_id(self, src: int, dst: int, ts: float, edge_id: int):
        """Add directed edge with original edge ID."""
        if src not in self.edge_id_adj_list:
            self.edge_id_adj_list[src] = []
        self.edge_id_adj_list[src].append((dst, ts, edge_id))   
    
    def find_neighbors(self, 
                      nodes: Union[List[int], torch.Tensor], 
                      timestamps: Union[List[float], torch.Tensor],
                      n_neighbors: Optional[int] = None) -> Tuple[List[List[int]], List[List[float]]]:
        """Backward-compatible interface without edge IDs."""
        nbrs, times, _ = self.find_neighbors_with_edge_ids(nodes, timestamps, n_neighbors)
        return nbrs, times    
    
   
    
    def find_neighbors_with_edge_ids(self, 
                                     nodes: Union[List[int], torch.Tensor], 
                                     timestamps: Union[List[float], torch.Tensor],
                                     n_neighbors: Optional[int] = None) -> Tuple[List[List[int]], List[List[float]], List[List[int]]]:
        """
        Find temporal neighbors with their original edge IDs.
        """
        if n_neighbors is None:
            n_neighbors = self.max_neighbors
        
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.tolist()
            
        if len(nodes) != len(timestamps):
            raise ValueError(f"Length mismatch: {len(nodes)} nodes vs {len(timestamps)} timestamps")
        
        all_neighbors: List[List[int]] = []
        all_edge_times: List[List[float]] = []
        all_edge_ids: List[List[int]] = []
        
        for node, ts in zip(nodes, timestamps):
            nbrs, times, eids = self._find_single_node_neighbors(node, ts, n_neighbors)
            all_neighbors.append(nbrs)
            all_edge_times.append(times)
            all_edge_ids.append(eids)
        
        return all_neighbors, all_edge_times, all_edge_ids   
    
    
    def _find_single_node_neighbors(self, node: int, ts: float, n_neighbors: int) -> Tuple[List[int], List[float], List[int]]:
        """Find neighbors with edge IDs for a single node."""
        if node not in self.edge_id_adj_list:
            return [], [], []
        
        node_times = self._timestamps_cache[node]
        
        # Binary search: find rightmost timestamp < ts
        cutoff_idx = bisect.bisect_left(node_times, ts)
        
        if cutoff_idx == 0:
            return [], [], []
        
        # Take most recent n_neighbors from valid range
        start_idx = max(0, cutoff_idx - n_neighbors)
        sampled = self.edge_id_adj_list[node][start_idx:cutoff_idx]
        
        if not sampled:
            return [], [], []
        
        neighbors, times, edge_ids = zip(*sampled)
        return list(neighbors), list(times), list(edge_ids)
    
    
    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=10):
        """TGN-compatible interface. Returns (neighbors, edge_idxs, edge_times)."""
        neighbors_list, times_list, edge_ids_list = self.find_neighbors_with_edge_ids(
            source_nodes, timestamps, n_neighbors=n_neighbors
        )
        
        batch_size = len(source_nodes)
        neighbors = np.zeros((batch_size, n_neighbors), dtype=np.int64)
        edge_idxs = np.zeros((batch_size, n_neighbors), dtype=np.int64)
        edge_times = np.zeros((batch_size, n_neighbors), dtype=np.float32)
        
        for i, (nbrs, tms, eids) in enumerate(zip(neighbors_list, times_list, edge_ids_list)):
            if len(nbrs) > 0:
                actual_n = min(len(nbrs), n_neighbors)
                neighbors[i, :actual_n] = nbrs[:actual_n]
                edge_idxs[i, :actual_n] = eids[:actual_n]
                edge_times[i, :actual_n] = tms[:actual_n]
        
        return neighbors, edge_idxs, edge_times
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Return construction statistics."""
        return {
            'num_nodes_with_edges': self.num_nodes,
            'num_train_edges': self.num_edges,
            'max_neighbors_config': self.max_neighbors,
            'undirected': self.undirected,
            'time_span': (self.min_timestamp, self.max_timestamp)
        }
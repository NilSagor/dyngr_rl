from typing import Dict, List, Optional, Tuple, Union
import torch
import bisect

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
                 undirected: bool = True):
        """
        Args:
            train_edges: [E_train, 2] LongTensor of 0-indexed training edges ONLY
            train_timestamps: [E_train] FloatTensor of training timestamps ONLY
            max_neighbors: maximum neighbors to store per node
            undirected: if True, add reverse edges (default: True)
        """
        if not isinstance(train_edges, torch.Tensor):
            raise TypeError(f"train_edges must be torch.Tensor, got {type(train_edges)}")
        if not isinstance(train_timestamps, torch.Tensor):
            raise TypeError(f"train_timestamps must be torch.Tensor, got {type(train_timestamps)}")
        if train_edges.shape[0] != train_timestamps.shape[0]:
            raise ValueError(f"Shape mismatch: edges {train_edges.shape[0]} != timestamps {train_timestamps.shape[0]}")
        if train_edges.dtype != torch.long:
            raise TypeError(f"train_edges must be LongTensor, got {train_edges.dtype}")
            
        self.max_neighbors = max_neighbors
        self.undirected = undirected
        self.adj_list: Dict[int, List[Tuple[int, float]]] = {}
        
        # Build from training edges only
        edges_list = train_edges.tolist()
        timestamps_list = train_timestamps.tolist()
        
        for (src, dst), ts in zip(edges_list, timestamps_list):
            self._add_edge(src, dst, ts)
            if undirected:
                self._add_edge(dst, src, ts)
        
        # Sort by timestamp (oldest first) for binary search
        self._timestamps_cache: Dict[int, List[float]] = {}
        for node in self.adj_list:
            self.adj_list[node].sort(key=lambda x: x[1])
            # Cache timestamps for O(log n) lookup
            self._timestamps_cache[node] = [t for _, t in self.adj_list[node]]
        
        # Statistics
        self.num_nodes = len(self.adj_list)
        self.num_edges = train_edges.shape[0]
        self.max_timestamp = max(timestamps_list) if timestamps_list else 0.0
        self.min_timestamp = min(timestamps_list) if timestamps_list else 0.0
    
    def _add_edge(self, src: int, dst: int, ts: float):
        """Internal: add single directed edge."""
        if src not in self.adj_list:
            self.adj_list[src] = []
        self.adj_list[src].append((dst, ts))
    
    def find_neighbors(self, 
                      nodes: Union[List[int], torch.Tensor], 
                      timestamps: Union[List[float], torch.Tensor],
                      n_neighbors: Optional[int] = None) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Find temporal neighbors that appeared BEFORE the given timestamp.
        
        Args:
            nodes: list/tensor of node IDs to query
            timestamps: list/tensor of cutoff timestamps (must align with nodes)
            n_neighbors: max neighbors per node (None = use default)
            
        Returns:
            neighbors: list of neighbor ID lists per node
            edge_times: list of edge timestamp lists per node
            
        Raises:
            ValueError: if nodes and timestamps length mismatch
            KeyError: if node ID invalid (optional, see strict_mode)
        """
        if n_neighbors is None:
            n_neighbors = self.max_neighbors
            
        # Normalize inputs
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.tolist()
            
        if len(nodes) != len(timestamps):
            raise ValueError(f"Length mismatch: {len(nodes)} nodes vs {len(timestamps)} timestamps")
        
        all_neighbors: List[List[int]] = []
        all_edge_times: List[List[float]] = []
        
        for node, ts in zip(nodes, timestamps):
            result = self._find_single_node_neighbors(node, ts, n_neighbors)
            all_neighbors.append(result[0])
            all_edge_times.append(result[1])
        
        return all_neighbors, all_edge_times
    
    def _find_single_node_neighbors(self, 
                                   node: int, 
                                   ts: float, 
                                   n_neighbors: int) -> Tuple[List[int], List[float]]:
        """Find neighbors for single node with binary search optimization."""
        # Unknown node or no history
        if node not in self.adj_list:
            return [], []
        
        node_neighbors = self.adj_list[node]
        node_times = self._timestamps_cache[node]
        
        # Binary search: find rightmost timestamp < ts
        # bisect_left returns first index where timestamp >= ts
        cutoff_idx = bisect.bisect_left(node_times, ts)
        
        if cutoff_idx == 0:
            return [], []
        
        # Take most recent n_neighbors from valid range [0:cutoff_idx]
        valid_neighbors = node_neighbors[:cutoff_idx]
        start_idx = max(0, cutoff_idx - n_neighbors)
        sampled = valid_neighbors[start_idx:cutoff_idx]
        
        # Reverse to get most recent first
        sampled.reverse()
        
        if not sampled:
            return [], []
        
        neighbors, times = zip(*sampled)
        return list(neighbors), list(times)
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Return construction statistics."""
        return {
            'num_nodes_with_edges': self.num_nodes,
            'num_train_edges': self.num_edges,
            'max_neighbors_config': self.max_neighbors,
            'undirected': self.undirected,
            'time_span': (self.min_timestamp, self.max_timestamp)
        }
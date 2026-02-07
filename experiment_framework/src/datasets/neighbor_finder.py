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
        # self.adj_list: Dict[int, List[Tuple[int, float]]] = {}
        
        # Single unified adjacency list with edge IDs: node -> [(neighbor, timestamp, edge_id), ...]
        self.edge_id_adj_list: Dict[int, List[Tuple[int, float, int]]] = {}


        # Build from training edges only
        edges_list = train_edges.tolist()
        timestamps_list = train_timestamps.tolist()
        
        for edge_idx, ((src, dst), ts) in enumerate(zip(edges_list, timestamps_list)):
            self._add_edge_with_id(src, dst, ts, edge_idx)
            if undirected:
                self._add_edge_with_id(dst, src, ts, edge_idx)
        
        # Sort by timestamp (oldest first) for binary search
        self._timestamps_cache: Dict[int, List[float]] = {}
        for node in self.edge_id_adj_list:
            self.edge_id_adj_list[node].sort(key=lambda x: x[1])
            # Cache timestamps for O(log n) lookup
            self._timestamps_cache[node] = [t for _, t, _ in self.edge_id_adj_list[node]]
        
         # Store original edge indices for feature lookup
        # self.edge_id_adj_list: Dict[int, List[Tuple[int, float, int]]] = {}
        # Statistics
        self.num_nodes = len(self.edge_id_adj_list)
        self.num_edges = train_edges.shape[0]
        self.max_timestamp = max(timestamps_list) if timestamps_list else 0.0
        self.min_timestamp = min(timestamps_list) if timestamps_list else 0.0


        # # Build from training edges with edge IDs
        # for edge_idx, ((src, dst), ts) in enumerate(zip(edges_list, timestamps_list)):
        #     self._add_edge_with_id(src, dst, ts, edge_idx)
        #     if undirected:
        #         self._add_edge_with_id(dst, src, ts, edge_idx)


        # Statistics
        # self.num_nodes = len(self.adj_list)
        # self.num_edges = train_edges.shape[0]
        # self.max_timestamp = max(timestamps_list) if timestamps_list else 0.0
        # self.min_timestamp = min(timestamps_list) if timestamps_list else 0.0
    
    def _add_edge_with_id(self, src: int, dst: int, ts: float, edge_id: int):
        """Add directed edge with original edge ID."""
        if src not in self.edge_id_adj_list:
            self.edge_id_adj_list[src] = []
        self.edge_id_adj_list[src].append((dst, ts, edge_id))
    
    
    # def _add_edge(self, src: int, dst: int, ts: float):
    #     """Internal: add single directed edge."""
    #     if src not in self.adj_list:
    #         self.adj_list[src] = []
    #     self.adj_list[src].append((dst, ts))
    
    # def _add_edge_with_id(self, src: int, dst: int, ts: float, edge_id: int):
    #     """Add directed edge with original edge ID."""
    #     if src not in self.edge_id_adj_list:
    #         self.edge_id_adj_list[src] = []
    #     self.edge_id_adj_list[src].append((dst, ts, edge_id))
        
    #     # Also update regular adj_list for backward compatibility
    #     if src not in self.adj_list:
    #         self.adj_list[src] = []
    #     self.adj_list[src].append((dst, ts))
    
    def find_neighbors(self, 
                      nodes: Union[List[int], torch.Tensor], 
                      timestamps: Union[List[float], torch.Tensor],
                      n_neighbors: Optional[int] = None) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Find temporal neighbors that appeared BEFORE the given timestamp.
        Backward-compatible interface without edge IDs.
        
        Args:
            nodes: list/tensor of node IDs to query
            timestamps: list/tensor of cutoff timestamps (must align with nodes)
            n_neighbors: max neighbors per node (None = use default)
            
        Returns:
            neighbors: list of neighbor ID lists per node
            edge_times: list of edge timestamp lists per node
        """
        nbrs, times, _ = self.find_neighbors_with_edge_ids(nodes, timestamps, n_neighbors)
        return nbrs, times
    
    
    # def find_neighbors(self, 
    #                   nodes: Union[List[int], torch.Tensor], 
    #                   timestamps: Union[List[float], torch.Tensor],
    #                   n_neighbors: Optional[int] = None) -> Tuple[List[List[int]], List[List[float]]]:
    #     """
    #     Find temporal neighbors that appeared BEFORE the given timestamp.
        
    #     Args:
    #         nodes: list/tensor of node IDs to query
    #         timestamps: list/tensor of cutoff timestamps (must align with nodes)
    #         n_neighbors: max neighbors per node (None = use default)
            
    #     Returns:
    #         neighbors: list of neighbor ID lists per node
    #         edge_times: list of edge timestamp lists per node
            
    #     Raises:
    #         ValueError: if nodes and timestamps length mismatch
    #         KeyError: if node ID invalid (optional, see strict_mode)
    #     """
    #     if n_neighbors is None:
    #         n_neighbors = self.max_neighbors
            
    #     # Normalize inputs
    #     if isinstance(nodes, torch.Tensor):
    #         nodes = nodes.tolist()
    #     if isinstance(timestamps, torch.Tensor):
    #         timestamps = timestamps.tolist()
            
    #     if len(nodes) != len(timestamps):
    #         raise ValueError(f"Length mismatch: {len(nodes)} nodes vs {len(timestamps)} timestamps")
        
    #     all_neighbors: List[List[int]] = []
    #     all_edge_times: List[List[float]] = []
        
    #     for node, ts in zip(nodes, timestamps):
    #         result = self._find_single_node_neighbors(node, ts, n_neighbors)
    #         all_neighbors.append(result[0])
    #         all_edge_times.append(result[1])
        
    #     return all_neighbors, all_edge_times
    
    def find_neighbors_with_edge_ids(self, 
                                     nodes: Union[List[int], torch.Tensor], 
                                     timestamps: Union[List[float], torch.Tensor],
                                     n_neighbors: Optional[int] = None) -> Tuple[List[List[int]], List[List[float]], List[List[int]]]:
        """
        Find temporal neighbors with their original edge IDs.
        
        Args:
            nodes: list/tensor of node IDs to query
            timestamps: list/tensor of cutoff timestamps (must align with nodes)
            n_neighbors: max neighbors per node (None = use default)
            
        Returns:
            neighbors: list of neighbor ID lists per node
            edge_times: list of edge timestamp lists per node
            edge_ids: list of original edge ID lists per node
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
    
    # def _find_single_node_neighbors_with_ids(self, node: int, ts: float, n_neighbors: int):
    #     """Find neighbors with edge IDs for a single node."""
    #     if node not in self.edge_id_adj_list:
    #         return [], [], []
        
    #     node_neighbors = self.edge_id_adj_list[node]
        
    #     # Sort by timestamp (should already be sorted from init)
    #     # Get timestamps for binary search
    #     node_times = [t for _, t, _ in node_neighbors]
        
    #     # Binary search: find rightmost timestamp < ts
    #     cutoff_idx = bisect.bisect_left(node_times, ts)
        
    #     if cutoff_idx == 0:
    #         return [], [], []
        
    #     # Take most recent n_neighbors from valid range
    #     valid_neighbors = node_neighbors[:cutoff_idx]
    #     start_idx = max(0, cutoff_idx - n_neighbors)
    #     sampled = valid_neighbors[start_idx:cutoff_idx]
        
    #     if not sampled:
    #         return [], [], []
        
    #     # Unpack neighbors, timestamps, and edge IDs
    #     neighbors, times, edge_ids = zip(*sampled)
    #     return list(neighbors), list(times), list(edge_ids)
    
    
    # def _find_single_node_neighbors(self, 
    #                                node: int, 
    #                                ts: float, 
    #                                n_neighbors: int) -> Tuple[List[int], List[float]]:
    #     """Find neighbors for single node with binary search optimization.
    #         Find neighbors without edge IDs (for backward compatibility)."""
    #     nbrs, times, _ = self._find_single_node_neighbors_with_ids(node, ts, n_neighbors)
    #     return nbrs, times
    
    def _find_single_node_neighbors(self, node: int, ts: float, n_neighbors: int)-> Tuple[List[int], List[float], List[int]]:
        """Find neighbors with edge IDs for a single node."""
        if node not in self.edge_id_adj_list:
            return [], [], []
        
        node_neighbors = self.edge_id_adj_list[node]
        
        # Sort by timestamp (should already be sorted from init)
        # Get timestamps for binary search
        node_times = self._timestamps_cache[node]
        
        # Binary search: find rightmost timestamp < ts
        cutoff_idx = bisect.bisect_left(node_times, ts)
        
        if cutoff_idx == 0:
            return [], [], []
        
        # Take most recent n_neighbors from valid range
        # valid_neighbors = node_neighbors[:cutoff_idx]
        start_idx = max(0, cutoff_idx - n_neighbors)
        sampled = node_neighbors[start_idx:cutoff_idx]
        
        if not sampled:
            return [], [], []
        
        # Unpack neighbors, timestamps, and edge IDs
        neighbors, times, edge_ids = zip(*sampled)
        return list(neighbors), list(times), list(edge_ids)
    
    
    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=10):
        """
        TGN-compatible interface. Returns (neighbors, edge_idxs, edge_times).
        edge_idxs are dummy indices (0,1,2...) for each neighbor position.
        """
        # Get neighbors, times, and actual edge IDs
        neighbors_list, times_list, edge_ids_list = self.find_neighbors_with_edge_ids(
            source_nodes, timestamps, n_neighbors=n_neighbors
        )
        
        # TGN expects numpy arrays with shape [batch_size, n_neighbors]
        # Pad to fixed size and create edge indices
        batch_size = len(source_nodes)
        
        neighbors = np.zeros((batch_size, n_neighbors), dtype=np.int64)
        edge_idxs = np.zeros((batch_size, n_neighbors), dtype=np.int64)
        edge_times = np.zeros((batch_size, n_neighbors), dtype=np.float32)
        
        for i, (nbrs, tms) in enumerate(zip(neighbors_list, times_list)):
            if len(nbrs) > 0:
                actual_n = min(len(nbrs), n_neighbors)
                neighbors[i, :actual_n] = nbrs[:actual_n]
                edge_idxs[i, :actual_n] = np.arange(actual_n)  # Dummy indices
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
# utils/neighbor_utils.py
import numpy as np
from typing import List, Tuple

def build_adj_list(src_node_ids: np.ndarray, 
                   dst_node_ids: np.ndarray, 
                   edge_ids: np.ndarray, 
                   timestamps: np.ndarray,
                   max_node_id: int = None) -> List[List[Tuple[int, int, float]]]:
    """
    Build adjacency list from edge lists for NeighborSampler.
    
    :param src_node_ids: array of source node IDs
    :param dst_node_ids: array of destination node IDs  
    :param edge_ids: array of edge IDs
    :param timestamps: array of interaction timestamps
    :param max_node_id: optional, max node ID (auto-computed if None)
    :return: adj_list where adj_list[i] = [(neighbor_id, edge_id, timestamp), ...]
    """
    if max_node_id is None:
        max_node_id = max(int(src_node_ids.max()), int(dst_node_ids.max()))
    
    # Initialize empty adjacency list (index 0 unused if nodes are 1-indexed)
    adj_list: List[List[Tuple[int, int, float]]] = [[] for _ in range(max_node_id + 1)]
    
    # Build undirected graph: add edges in both directions
    for src, dst, eid, ts in zip(src_node_ids, dst_node_ids, edge_ids, timestamps):
        src, dst, eid = int(src), int(dst), int(eid)
        ts = float(ts)
        adj_list[src].append((dst, eid, ts))
        adj_list[dst].append((src, eid, ts))
    
    return adj_list
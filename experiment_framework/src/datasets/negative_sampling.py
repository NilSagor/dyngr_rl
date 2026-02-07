from typing import Set, Tuple, Optional, TYPE_CHECKING
import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from .neighbor_finder import NeighborFinder


class NegativeSampler:
    """Industry-standard negative samplers (TGN ICML 2020 protocol)."""
    
    def __init__(self, 
                 edges: torch.Tensor,
                 timestamps: torch.Tensor,
                 num_nodes: int,
                 neighbor_finder: Optional['NeighborFinder'] = None,
                 seed: int = 42):
        """
        Args:
            edges: [E, 2] positive edge tensor (0-indexed)
            timestamps: [E] edge timestamps (unused for random/inductive)
            num_nodes: total nodes in graph
            neighbor_finder: required for historical sampling
            seed: random seed for reproducibility
        """
        if not isinstance(edges, torch.Tensor):
            raise TypeError(f"edges must be torch.Tensor, got {type(edges)}")
        if edges.dtype != torch.long:
            raise TypeError(f"edges must be LongTensor, got {edges.dtype}")
        if edges.dim() != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must be [E, 2], got {edges.shape}")
            
        self.num_nodes = int(num_nodes)
        self.neighbor_finder = neighbor_finder
        
        # Isolated RNG to prevent global state pollution
        self.rng = np.random.default_rng(seed)
        
        # Build positive edge set (undirected)
        self.positive_edges: Set[Tuple[int, int]] = set()
        edges_np = edges.cpu().numpy()
        for src, dst in edges_np:
            self.positive_edges.add((int(src), int(dst)))
            self.positive_edges.add((int(dst), int(src)))
        
        # Cache for faster lookup
        self._positive_neighbors: dict[int, Set[int]] = {}
        for src, dst in edges_np:
            src, dst = int(src), int(dst)
            self._positive_neighbors.setdefault(src, set()).add(dst)
            self._positive_neighbors.setdefault(dst, set()).add(src)
    
    def _is_valid_negative(self, src: int, dst: int) -> bool:
        """Check if (src, dst) is a valid negative edge."""
        return src != dst and (src, dst) not in self.positive_edges
    
    def random(self, src_nodes: np.ndarray) -> np.ndarray:
        """
        Random negative sampling (baseline).
        
        Args:
            src_nodes: [N] array of source node IDs
            
        Returns:
            [N] array of destination node IDs (negative samples)
        """
        src_nodes = np.asarray(src_nodes)
        if src_nodes.size == 0:
            return np.array([], dtype=np.int64)
            
        neg_dst = np.empty(len(src_nodes), dtype=np.int64)
        
        for i, src in enumerate(src_nodes):
            src = int(src)
            if not (0 <= src < self.num_nodes):
                raise ValueError(f"Invalid node ID: {src} (num_nodes={self.num_nodes})")
            
            # Get valid candidates (exclude self and positive neighbors)
            blocked = self._positive_neighbors.get(src, set()) | {src}
            candidates = np.array([
                n for n in range(self.num_nodes) 
                if n not in blocked
            ])
            
            if len(candidates) == 0:
                # DENSE GRAPH FALLBACK: Allow positive edges as negatives
                # This is necessary for datasets like SocialEvo
                logger.warning(f"Dense graph: node {src} connected to all others. "
                            f"Sampling from positive edges.")
                candidates = [n for n in range(self.num_nodes) if n != src]
                
                if len(candidates) == 0:
                    raise ValueError(f"Isolated node {src}, cannot sample negative")
            
            neg_dst[i] = self.rng.choice(candidates)
        
        return neg_dst
    
    def historical(self, 
                  src_nodes: np.ndarray, 
                  timestamps: np.ndarray) -> np.ndarray:
        """
        Historical negative sampling (TGN standard).
        
        Samples (u, w) where w was a neighbor of u BEFORE timestamp t.
        
        Args:
            src_nodes: [N] source node IDs
            timestamps: [N] cutoff timestamps
            
        Returns:
            [N] destination node IDs (negative samples)
        """        
        if self.neighbor_finder is None:
            raise ValueError("neighbor_finder required for historical sampling")
        
        src_nodes = np.asarray(src_nodes)
        timestamps = np.asarray(timestamps)

        
        neg_dsts = np.empty(len(src_nodes), dtype=np.int64)
    
        # Batch query all neighbors at once for efficiency
        all_neighbors_list, _, _ = self.neighbor_finder.find_neighbors_with_edge_ids(
            src_nodes.tolist(), timestamps.tolist(), n_neighbors=self.neighbor_finder.max_neighbors
        )
        
        for i, (src, neighbors) in enumerate(zip(src_nodes, all_neighbors_list)):
            src = int(src)
            
            # Filter: must be historical, not self, and not positive
            valid_candidates = [
                n for n in neighbors
                if n != src and (src, n) not in self.positive_edges
            ]
            
            if valid_candidates:
                # Choose random historical neighbor
                neg_dsts[i] = self.rng.choice(valid_candidates)
            else:
                # Fallback to random (per TGN protocol)
                neg_dsts[i] = self.random(np.array([src]))[0]

            # neighbors, _, _ = self.historical_with_edge_ids(int(src), float(ts))
            
            # if neighbors:
            #     # Choose random historical neighbor
            #     idx = self.rng.integers(0, len(neighbors))
            #     neg_dsts[i] = neighbors[idx]
            # else:
            #     # Fallback to random (per TGN protocol)
            #     neg_dsts[i] = self.random(np.array([src]))[0]
        
        return neg_dsts
    
    
    
    def historical_with_edge_ids(self, 
                              src_nodes: np.ndarray, 
                              timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Historical negative sampling with edge IDs for feature retrieval.
        
        Args:
            src_nodes: [N] source node IDs
            timestamps: [N] cutoff timestamps
            
        Returns:
            Tuple of (neg_dsts, edge_ids) where:
                - neg_dsts: [N] destination node IDs
                - edge_ids: [N] original edge indices (-1 if fallback to random)
        """
        if self.neighbor_finder is None:
            raise ValueError("neighbor_finder required for historical sampling")
        
        src_nodes = np.asarray(src_nodes)
        timestamps = np.asarray(timestamps)
        
        if len(src_nodes) != len(timestamps):
            raise ValueError(f"Length mismatch: {len(src_nodes)} vs {len(timestamps)}")
        
        neg_dsts = np.empty(len(src_nodes), dtype=np.int64)
        edge_ids = np.full(len(src_nodes), -1, dtype=np.int64)
        
        # Batch query all neighbors with edge IDs
        all_neighbors_list, _, all_edge_ids_list = self.neighbor_finder.find_neighbors_with_edge_ids(
            src_nodes.tolist(), timestamps.tolist(), n_neighbors=self.neighbor_finder.max_neighbors
        )
        
        for i, (src, neighbors, edge_ids_list) in enumerate(zip(src_nodes, all_neighbors_list, all_edge_ids_list)):
            src = int(src)
            
            # Filter: must be historical, not self, and not positive
            valid_candidates = []
            valid_edge_ids = []
            
            for nbr, eid in zip(neighbors, edge_ids_list):
                if nbr != src and (src, nbr) not in self.positive_edges:
                    valid_candidates.append(nbr)
                    valid_edge_ids.append(eid)
            
            if valid_candidates:
                # Choose random historical neighbor
                choice_idx = self.rng.integers(len(valid_candidates))
                neg_dsts[i] = valid_candidates[choice_idx]
                edge_ids[i] = valid_edge_ids[choice_idx]
            else:
                # Fallback to random (per TGN protocol)
                neg_dsts[i] = self.random(np.array([src]))[0]
                edge_ids[i] = -1  # Marker for random sample
        
        return neg_dsts, edge_ids
    
    
    
    def inductive(self, 
                 src_nodes: np.ndarray, 
                 unseen_nodes: np.ndarray) -> np.ndarray:
        """
        Inductive negative sampling.
        
        Samples (u, w) where w is from the unseen node set,
        excluding edges that are actually positive.
        
        Args:
            src_nodes: [N] source node IDs
            unseen_nodes: [M] unseen node IDs (M <= num_nodes)
            
        Returns:
            [N] destination node IDs from unseen set
        """
        src_nodes = np.asarray(src_nodes)
        unseen_nodes = np.asarray(unseen_nodes)
        
        if len(unseen_nodes) == 0:
            raise ValueError("unseen_nodes required for inductive sampling")
        
        neg_dst = np.empty(len(src_nodes), dtype=np.int64)
        unseen_set = set(unseen_nodes)
        
        for i, src in enumerate(src_nodes):
            src = int(src)
            
            # Filter unseen nodes that aren't already connected to src
            candidates = [
                n for n in unseen_nodes
                if n != src and (src, int(n)) not in self.positive_edges
            ]
            
            if candidates:
                neg_dst[i] = self.rng.choice(candidates)
            else:
                # Relaxed: any unseen node (even if positive, for evaluation only)
                # Or raise error depending on strictness requirements
                neg_dst[i] = self.rng.choice(unseen_nodes)
        
        return neg_dst
    
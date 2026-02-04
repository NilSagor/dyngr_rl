from typing import Set, Tuple, Optional, TYPE_CHECKING
import numpy as np
import torch

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
                raise ValueError(f"No valid negative samples for node {src}")
            
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
        
        if len(src_nodes) != len(timestamps):
            raise ValueError(f"Length mismatch: {len(src_nodes)} vs {len(timestamps)}")
        
        neg_dst = np.empty(len(src_nodes), dtype=np.int64)
        
        for i, (src, ts) in enumerate(zip(src_nodes, timestamps)):
            src = int(src)
            
            # Query historical neighbors before timestamp ts
            # FIXED: Use correct method name and unpack 2 values
            neighbors_list, _ = self.neighbor_finder.find_neighbors(
                [src], [float(ts)], n_neighbors=20
            )
            neighbors = neighbors_list[0]  # Unwrap batch dimension
            
            # Filter: must be historical, not self, and not positive
            candidates = [
                n for n in neighbors
                if n != src and (src, n) not in self.positive_edges
            ]
            
            if candidates:
                neg_dst[i] = self.rng.choice(candidates)
            else:
                # Fallback to random (per TGN protocol)
                neg_dst[i] = self.random(np.array([src]))[0]
        
        return neg_dst
    
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
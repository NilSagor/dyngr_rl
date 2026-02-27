import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from collections import defaultdict
import random
from loguru import logger

from .time_encoder import TimeEncoder

class MultiScaleWalkSampler(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            walk_length_short: int = 3,
            walk_length_long: int = 10,
            walk_length_tawr: int = 8,
            num_walks_short: int = 10,
            num_walks_long: int = 5,
            num_walks_tawr: int = 5,
            temperature: float = 0.1,
            memory_dim: int = 128,
            time_dim: int = 64,
    ):
        super(MultiScaleWalkSampler, self).__init__()
        
        # Walk configuration
        self.num_nodes = num_nodes
        self.walk_length_short = walk_length_short
        self.walk_length_long = walk_length_long
        self.walk_length_tawr = walk_length_tawr
        self.num_walks_short = num_walks_short
        self.num_walks_long = num_walks_long
        self.num_walks_tawr = num_walks_tawr
        self.temperature = temperature

        # Learnable restart probability parameters (for TAWR walks)
        self.restart_projection = nn.Linear(memory_dim + time_dim, 1)
        nn.init.xavier_uniform_(self.restart_projection.weight)
        nn.init.zeros_(self.restart_projection.bias)
        
        # Neighbor cache for efficient sampling
        self.neighbor_cache: Dict[int, List[Tuple[int, float]]] = {}  # Will be populated by update_neighbors()
        
        # Time encoding function (fixed, non-learnable)
        self.time_encoder = TimeEncoder(time_dim)

        # Flag to track if dense tables are built
        self._dense_tables_built = False

    
    def update_neighbors(self, edge_index: torch.Tensor, edge_time: torch.Tensor):
        """
        Update the neighbor cache with current graph structure.
        
        Args:
            edge_index: [2, num_edges] tensor of edges
            edge_time: [num_edges] tensor of timestamps
        """
        self.neighbor_cache = {}
        self._dense_tables_built = False
        
        # Convert to numpy for faster processing (or keep on GPU if possible)
        edge_index_np = edge_index.cpu().numpy() if edge_index.is_cuda else edge_index.numpy()
        edge_time_np = edge_time.cpu().numpy() if edge_time.is_cuda else edge_time.numpy()
        
        # Build neighbor lists for each node
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            t = float(edge_time_np[i])
            
            # Add both directions for undirected graphs (modify if directed)
            self._add_to_cache(src, dst, t)
            self._add_to_cache(dst, src, t)
        
        # Sort neighbors by time for each node (most recent first)
        for node in self.neighbor_cache:
            neighbors = self.neighbor_cache[node]
            # Sort by timestamp descending (most recent first)
            neighbors.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Updated neighbor cache with {len(self.neighbor_cache)} nodes")
    
    def _add_to_cache(self, src: int, dst: int, t: float):
        """Helper to add a neighbor to cache."""
        if src not in self.neighbor_cache:
            self.neighbor_cache[src] = []
        self.neighbor_cache[src].append((dst, t))
    
    
    
    def get_temporal_neighbors(self, node: int, current_time: float) -> List[Tuple[int, float]]:
        """
        Get all neighbors of node with timestamps < current_time.
        
        Args:
            node: Node ID
            current_time: Current walk time
            
        Returns:
            List of (neighbor_id, timestamp) tuples with timestamp < current_time
        """
        if node not in self.neighbor_cache:
            return []
        
        all_neighbors = self.neighbor_cache[node]
        valid_neighbors = [(n, t) for n, t in all_neighbors if t < current_time]
        
        return valid_neighbors
    
    def sample_neighbor_with_temporal_bias(
        self, 
        node: int, 
        current_time: float
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Sample a neighbor using temporal bias (Eq. 7 from paper).
        
        P(u, t' | v, t) = exp((t' - t_max)/τ) / sum(exp((t_i - t_max)/τ))
        
        Args:
            node: Current node
            current_time: Current walk time
            
        Returns:
            (neighbor_id, timestamp) or (None, None) if no valid neighbors
        """
        valid_neighbors = self.get_temporal_neighbors(node, current_time)
        
        if not valid_neighbors:
            return None, None
        
        if len(valid_neighbors) == 1:
            return valid_neighbors[0]
        
        # Extract timestamps
        timestamps = np.array([t for _, t in valid_neighbors], dtype=np.float32)
        t_max = np.max(timestamps)
        
        # Compute exponential weights (Eq. 7)
        weights = np.exp((timestamps - t_max) / self.temperature)
        probs = weights / (np.sum(weights) + 1e-10)
        
        # Sample based on probabilities
        idx = np.random.choice(len(valid_neighbors), p=probs)
        
        return valid_neighbors[idx]
    
    
    def compute_restart_probability(
        self,
        node: int,
        current_time: float,
        memory_state: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute learnable restart probability ρ_u(τ) for TAWR walks.
        
        ρ_u(τ) = σ(w_ρ^T · [m_u(τ) || Φ(τ)] + b_ρ)
        
        Args:
            node: Node ID
            current_time: Current walk time
            memory_state: Memory state m_u(τ) from SAM (if available)
            
        Returns:
            Restart probability between 0 and 1
        """
        if memory_state is None:
            # Default to 0.1 if no memory (e.g., during initialization)
            return 0.1
        
        device = memory_state.device
        # Clamp node index to valid range
        node_idx = min(max(node, 0), memory_state.size(0) - 1)
        
        # Get time encoding        
        time_tensor = torch.tensor([current_time], device=device, dtype=torch.float32)
        time_enc = self.time_encoder(time_tensor)  # [1, time_dim]
        

        # Combine memory and time encoding
        memory_tensor = memory_state[node_idx:node_idx+1]
        # memory_tensor = memory_state[node].unsqueeze(0)  # [1, memory_dim]
        combined = torch.cat([memory_tensor, time_enc], dim=-1)  # [1, memory_dim + time_dim]
        
        # Compute probability
        logits = self.restart_projection(combined)  # [1, 1]
        prob = torch.sigmoid(logits).item()
        
        return prob
    
    def build_dense_neighbor_table(self):
        """
        Convert the neighbor cache into dense tensors for fast GPU sampling.
        Returns:
            neighbor_ids: [num_nodes, max_degree] (padded with 0)
            neighbor_times: [num_nodes, max_degree] (padded with 0)
            neighbor_counts: [num_nodes] actual degree per node
        """
        if not self.neighbor_cache:
            logger.error("Cannot build dense table: neighbor_cache is empty!")
            raise RuntimeError("Cannot build dense neighbor table: neighbor_cache is empty")
        
        if self._dense_tables_built:
            return  # Skip if already built       
        
        
        max_degree = max(len(neighbors) for neighbors in self.neighbor_cache.values())
        
        # max_degree = max(len(neighbors) for neighbors in self.neighbor_cache.values()) if self.neighbor_cache else 0
        num_nodes = self.num_nodes
        
        device = next(self.parameters(), torch.tensor(0)).device       
        

        neighbor_ids = torch.zeros(num_nodes, max_degree, dtype=torch.long, device=device)
        neighbor_times = torch.zeros(num_nodes, max_degree, dtype=torch.float32, device=device)        
        neighbor_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        # if (neighbor_ids < 0).any() or (neighbor_ids >= self.num_nodes).any():
        #     logger.error("neighbor_ids contains invalid node indices")
        #     raise RuntimeError("Invalid neighbor_ids")
        # if torch.isnan(neighbor_times).any() or torch.isinf(neighbor_times).any():
        #     logger.error("neighbor_times contains NaN/Inf")
        #     raise RuntimeError("Invalid neighbor_times")
                
        
        for node, neighbors in self.neighbor_cache.items():
            if 0 <= node < num_nodes:  # Bounds check
                deg = len(neighbors)
                neighbor_counts[node] = deg
                for i, (nid, ts) in enumerate(neighbors):
                    if 0 <= nid < num_nodes:  # Bounds check for neighbor
                        neighbor_ids[node, i] = nid
                        neighbor_times[node, i] = float(ts)
        
        
        
        # Use register_buffer only once, or update existing buffers
        if not hasattr(self, 'dense_neighbor_ids'):
            self.register_buffer('dense_neighbor_ids', neighbor_ids, persistent=False)
            self.register_buffer('dense_neighbor_times', neighbor_times, persistent=False)
            self.register_buffer('dense_neighbor_counts', neighbor_counts, persistent=False)
        else:
            # Update existing buffers
            self.dense_neighbor_ids = neighbor_ids
            self.dense_neighbor_times = neighbor_times
            self.dense_neighbor_counts = neighbor_counts
        
        self._dense_tables_built = True
        logger.info(f"Built dense neighbor table: max_degree={max_degree}")
    
    def compute_restart_probabilities_batched(
        self,
        node_ids: torch.Tensor,        # [batch_size, num_walks]
        times: torch.Tensor,           # [batch_size, num_walks]
        memory_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute restart probabilities $$\rho_{u}(\tau)$$  for a batch of walks.
        Returns probabilities of shape [batch_size, num_walks].
        Compute restart probabilities for a batch of walks.
        """
        if memory_states is None:
            # Default probability when no memory is available
            return torch.full_like(node_ids, 0.1, dtype=torch.float, device=node_ids.device)

        batch_size, num_walks = node_ids.shape
        device = node_ids.device

        # Gather memory states for each node
        node_ids = node_ids.clamp(0, memory_states.size(0)-1)
        node_memory = memory_states[node_ids]  # [B, W, memory_dim]

        
        # Encode times (TimeEncoder handles 2D input)
        time_enc = self.time_encoder(times)    # [B, W, time_dim]

        # Concatenate memory and time encoding
        combined = torch.cat([node_memory, time_enc], dim=-1)  # [B, W, memory_dim+time_dim]

                
        # Flatten for linear projection
        combined_flat = combined.view(-1, combined.size(-1))   # [B*W, input_dim]
        logits_flat = self.restart_projection(combined_flat)   # [B*W, 1]
        logits = logits_flat.view(batch_size, num_walks)       # [B, W]

        


        return torch.sigmoid(logits)
    
    
    
    def _sample_walks_vectorized(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        num_walks: int,
        walk_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        Unified vectorized walk sampling with temporal bias.
        Handles short and long walks (without restart logic).
        """
        batch_size = source_nodes.size(0)
        device = source_nodes.device
        
        if not hasattr(self, 'dense_neighbor_ids'):
            self.build_dense_neighbor_table()
        
        max_deg = self.dense_neighbor_ids.size(1)
        eps = 1e-10

        # Expand to [B, W]
        curr_nodes = source_nodes.unsqueeze(1).expand(-1, num_walks)
        curr_times = current_times.unsqueeze(1).expand(-1, num_walks)

        # Initialize tensors
        walk_nodes = torch.zeros(batch_size, num_walks, walk_length, dtype=torch.long, device=device)
        walk_times = torch.zeros(batch_size, num_walks, walk_length, dtype=torch.float32, device=device)
        walk_masks = torch.zeros_like(walk_nodes, dtype=torch.float32)

        walk_nodes[:, :, 0] = curr_nodes
        walk_times[:, :, 0] = curr_times
        walk_masks[:, :, 0] = 1.0

        for step in range(1, walk_length):
            # Gather neighbors: [B, W, max_deg]
            neighbor_ids = self.dense_neighbor_ids[curr_nodes]
            neighbor_times = self.dense_neighbor_times[curr_nodes]

            # Temporal mask
            time_mask = neighbor_times < curr_times.unsqueeze(-1)
            valid_counts = time_mask.sum(dim=-1)
            has_valid = valid_counts > 0

            # Numerically stable temporal bias weights
            t_max = neighbor_times.max(dim=-1, keepdim=True)[0]
            t_max = torch.where(t_max > 0, t_max, torch.ones_like(t_max))
            weights = torch.exp((neighbor_times - t_max) / self.temperature)
            weights = weights.masked_fill(~time_mask, 0.0)

            # Normalize to probabilities with numerical stability
            prob_sums = weights.sum(dim=-1, keepdim=True)
            probs = torch.where(
                prob_sums > eps,
                weights / (prob_sums + eps),
                torch.ones_like(weights) / max_deg  # Uniform fallback
            )

            # Sample with multinomial (handle zero-prob rows)
            probs_flat = probs.view(-1, max_deg)
            sampled_idx = torch.multinomial(probs_flat, 1).view(batch_size, num_walks)

            # Gather results
            next_node = torch.gather(neighbor_ids, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)
            next_time = torch.gather(neighbor_times, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            # Handle nodes with no valid neighbors: stay at current node
            next_node = torch.where(has_valid, next_node, curr_nodes)
            next_time = torch.where(has_valid, next_time, curr_times)

            # Update walk tensors
            walk_nodes[:, :, step] = next_node
            walk_times[:, :, step] = next_time
            walk_masks[:, :, step] = 1.0

            curr_nodes = next_node
            curr_times = next_time

        return {
            'nodes': walk_nodes,
            'times': walk_times,
            'masks': walk_masks
        }
    
    def sample_short_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Sample short walks using vectorized implementation."""
        return self._sample_walks_vectorized(
            source_nodes, current_times,
            num_walks=self.num_walks_short,
            walk_length=self.walk_length_short
        )
    
    def sample_long_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample long walks (memory_states not used for basic walks)."""
        return self._sample_walks_vectorized(
            source_nodes, current_times,
            num_walks=self.num_walks_long,
            walk_length=self.walk_length_long
        )
    
    def sample_tawr_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Vectorized TAWR walk sampling with restart probabilities.
        """
        batch_size = len(source_nodes)
        device = source_nodes.device
        
        if not hasattr(self, 'dense_neighbor_ids'):
            self.build_dense_neighbor_table()
        
        max_deg = self.dense_neighbor_ids.size(1)
        eps = 1e-10
        # Expand to [B, W] for each walk
        # curr_nodes = source_nodes.unsqueeze(1).expand(-1, self.num_walks_tawr)   # [B, W]
        # curr_times = current_times.unsqueeze(1).expand(-1, self.num_walks_tawr)  # [B, W]

        curr_nodes = source_nodes.unsqueeze(1).expand(-1, self.num_walks_tawr)
        curr_times = current_times.unsqueeze(1).expand(-1, self.num_walks_tawr)


        # Original source nodes for restarts (same for all walks)
        # original_source = source_nodes.unsqueeze(1).expand(-1, self.num_walks_tawr)
        original_source = source_nodes.unsqueeze(1).expand(-1, self.num_walks_tawr)
        original_times = current_times.unsqueeze(1).expand(-1, self.num_walks_tawr)

        # Initialize walk tensors
        walk_nodes = torch.zeros(batch_size, self.num_walks_tawr, self.walk_length_tawr,
                                dtype=torch.long, device=device)
        walk_times = torch.zeros(batch_size, self.num_walks_tawr, self.walk_length_tawr,
                                dtype=torch.float32, device=device)
        walk_restart = torch.zeros_like(walk_nodes, dtype=torch.float32)
        walk_masks = torch.zeros_like(walk_nodes, dtype=torch.float32)

        # First step
        walk_nodes[:, :, 0] = curr_nodes
        walk_times[:, :, 0] = curr_times
        walk_masks[:, :, 0] = 1.0

        for step in range(1, self.walk_length_tawr):
            # Compute restart probabilities for all current nodes
            # restart_probs = self.compute_restart_probabilities_batched(
            #     curr_nodes, curr_times, memory_states
            # )  # [B, W]

            # # Random restart decisions
            # rand = torch.rand(batch_size, self.num_walks_tawr, device=device)
            # restart_mask = rand < restart_probs  # [B, W]

            # # ---- For walks that restart ----
            # next_node_restart = original_source
            # next_time_restart = current_times.unsqueeze(1).expand(-1, self.num_walks_tawr)

            # # ---- For walks that continue ----
            # # Get neighbor info for current nodes
            # neighbor_ids = self.dense_neighbor_ids[curr_nodes]       # [B, W, max_deg]
            # neighbor_times = self.dense_neighbor_times[curr_nodes]   # [B, W, max_deg]


            # # Temporal mask
            # time_mask = neighbor_times < curr_times.unsqueeze(-1)    # [B, W, max_deg]

            restart_probs = self.compute_restart_probabilities_batched(
                curr_nodes, curr_times, memory_states
            )
            
            rand = torch.rand(batch_size, self.num_walks_tawr, device=device)
            restart_mask = rand < restart_probs
            
            next_node_restart = original_source
            next_time_restart = curr_times
            
            neighbor_ids = self.dense_neighbor_ids[curr_nodes]
            neighbor_times = self.dense_neighbor_times[curr_nodes]
            time_mask = neighbor_times < curr_times.unsqueeze(-1)
            
            # Temporal bias weights
            t_max = neighbor_times.max(dim=-1, keepdim=True)[0]
            t_max = torch.where(t_max > 0, t_max, torch.ones_like(t_max))
            weights = torch.exp((neighbor_times - t_max) / self.temperature)
            weights = weights.masked_fill(~time_mask, 0.0)

            prob_sums = weights.sum(dim=-1, keepdim=True)
            probs = torch.where(
                prob_sums > eps,
                weights / (prob_sums + eps),
                torch.ones_like(weights) / max_deg
            )


            probs_flat = probs.view(-1, max_deg)
            sampled_idx = torch.multinomial(probs_flat, 1).view(batch_size, self.num_walks_tawr)
            
            if sampled_idx.max() >= max_deg or sampled_idx.min() < 0:
                logger.error(f"sampled_idx out of bounds: min={sampled_idx.min()}, max={sampled_idx.max()}, max_deg={max_deg}")
                raise RuntimeError("Sampled index out of bounds")


            next_node_continue = torch.gather(neighbor_ids, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)
            next_time_continue = torch.gather(neighbor_times, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            # Combine based on restart_mask
            next_node = torch.where(restart_mask, next_node_restart, next_node_continue)
            next_time = torch.where(restart_mask, next_time_restart, next_time_continue)
            
            # Record step
            walk_nodes[:, :, step] = next_node
            walk_times[:, :, step] = next_time
            walk_restart[:, :, step] = restart_mask.float()
            walk_masks[:, :, step] = 1.0

            # Update current nodes/times for next iteration
            curr_nodes = next_node
            curr_times = next_time

        return {
            'nodes': walk_nodes,
            'times': walk_times,
            'restart_flags': walk_restart,
            'masks': walk_masks
        }
    
    
    
    
    def anonymize_walks(
        self,
        walk_data: Dict[str, torch.Tensor],
        walk_type: str
    ) -> Dict[str, torch.Tensor]:
        """
        Anonymize walks by replacing node IDs with positional identifiers.
        
        Implements Algorithm 2 from the paper for anonymization.
        
        Args:
            walk_data: Dictionary with walk tensors
            walk_type: 'short', 'long', or 'tawr'
            
        Returns:
            Dictionary with anonymized walk tensors
        """
        nodes = walk_data['nodes']  # [batch, num_walks, walk_len]
        batch_size, num_walks, walk_len = nodes.shape
        
        device = nodes.device
        # Create anonymized tensor (same shape, but with anonymized IDs)
        anonymized = torch.zeros_like(nodes)
        
        for b in range(batch_size):
            # Get unique nodes for this batch item (excluding padding=0)
            batch_nodes = nodes[b]
            unique_nodes = torch.unique(batch_nodes[batch_nodes > 0])
            
            if len(unique_nodes) == 0:
                continue
            
            # Create mapping: original_id -> anonymized_id (1-based)
            mapping = {nid.item(): idx + 1 for idx, nid in enumerate(unique_nodes)}
            
            # Vectorized replacement using torch operations
            flat_nodes = batch_nodes.view(-1)
            flat_anon = torch.zeros_like(flat_nodes)
            
            for orig_id, anon_id in mapping.items():
                mask = flat_nodes == orig_id
                flat_anon[mask] = anon_id
            
            anonymized[b] = flat_anon.view(num_walks, walk_len)
        
        assert anonymized[b].min() >= 0 and anonymized[b].max() <= len(unique_nodes)
        
        walk_data['nodes_anon'] = anonymized
        return walk_data
    
    def clear_cache(self):
        self.neighbor_cache.clear()
        self._dense_tables_built = False
        if hasattr(self, 'dense_neighbor_ids'):
            delattr(self, 'dense_neighbor_ids')
    
    
    def forward(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_time: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate multi-scale walks for source and target nodes.
        
        Args:
            source_nodes: [batch_size] source node IDs
            target_nodes: [batch_size] target node IDs
            current_times: [batch_size] interaction times
            memory_states: [num_nodes, memory_dim] memory states from SAM
            edge_index: Optional edge index to update neighbor cache
            edge_time: Optional edge times to update neighbor cache
            
        Returns:
            Nested dictionary with walks for source and target:
                {
                    'source': {
                        'short': {...},
                        'long': {...},
                        'tawr': {...}
                    },
                    'target': {
                        'short': {...},
                        'long': {...},
                        'tawr': {...}
                    }
                }
        """                  
        
        # # Update neighbor cache if new edges provided
        # if edge_index is not None and edge_time is not None:
        #     self.update_neighbors(edge_index, edge_time)
        #     # Always rebuild dense table when edges are updated
        #     if self.neighbor_cache:  # Only if cache is not empty
        #         self.build_dense_neighbor_table()
        
        # # Check if we have neighbor data
        # if not hasattr(self, 'dense_neighbor_ids'):
        #     if not self.neighbor_cache:
        #         raise RuntimeError(
        #             "No neighbor data available. Please provide edge_index and edge_time "
        #             "with valid edges to initialize the neighbor cache."
        #         )
        #     else:
        #         # Cache exists but dense table not built yet
        #         self.build_dense_neighbor_table()

        # Update neighbor cache if new edges provided
        if edge_index is not None and edge_time is not None:
            self.update_neighbors(edge_index, edge_time)
        
        # Build dense tables if needed
        if self.neighbor_cache and not self._dense_tables_built:
            self.build_dense_neighbor_table()
        
        if not hasattr(self, 'dense_neighbor_ids'):
            raise RuntimeError(
                "No neighbor data available. Please provide edge_index and edge_time "
                "with valid edges to initialize the neighbor cache."
            )
        
        
        # Combine source and target nodes for efficient sampling
        all_nodes = torch.cat([source_nodes, target_nodes])
        all_times = torch.cat([current_times, current_times])
        
        # Sample all walk types
        short_walks = self.sample_short_walks(all_nodes, all_times)
        long_walks = self.sample_long_walks(all_nodes, all_times, memory_states)
        tawr_walks = self.sample_tawr_walks(all_nodes, all_times, memory_states)
        
        # Anonymize walks
        short_walks = self.anonymize_walks(short_walks, 'short')
        long_walks = self.anonymize_walks(long_walks, 'long')
        tawr_walks = self.anonymize_walks(tawr_walks, 'tawr')
        
        # Split back into source and target
        batch_size = len(source_nodes)
        
        result = {
            'source': {
                'short': {k: v[:batch_size] for k, v in short_walks.items()},
                'long': {k: v[:batch_size] for k, v in long_walks.items()},
                'tawr': {k: v[:batch_size] for k, v in tawr_walks.items()}
            },
            'target': {
                'short': {k: v[batch_size:] for k, v in short_walks.items()},
                'long': {k: v[batch_size:] for k, v in long_walks.items()},
                'tawr': {k: v[batch_size:] for k, v in tawr_walks.items()}
            }
        }

        # logger.debug("Walk sampler result keys:", result.keys())  # or use logger
        return result
    




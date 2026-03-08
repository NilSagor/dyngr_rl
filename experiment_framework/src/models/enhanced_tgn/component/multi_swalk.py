import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from loguru import logger

from .time_encoder import TimeEncoder

class MultiScaleWalkSampler(nn.Module):
    """
    Multi-scale temporal walk sampler with:
    - Temporal bias sampling 
    - TAWR walks with learnable restart probabilities
    - Vectorized GPU sampling for efficiency
    - Walk anonymization for privacy
    """    
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
        
        # Ensure temperature is never zero (prevent division by zero)
        self.temperature = max(float(temperature), 1e-6)
        self.register_buffer('safe_temperature', torch.tensor(self.temperature))

        # Learnable restart probability parameters (for TAWR walks)
        self.restart_projection = nn.Linear(memory_dim + time_dim, 1)
        nn.init.xavier_uniform_(self.restart_projection.weight)
        nn.init.constant_(self.restart_projection.bias, -2.197)
        
        # Track if edges have been initialize
        self._edges_initialized = False
        
        # Neighbor cache for efficient sampling
        self.neighbor_cache: Dict[int, List[Tuple[int, float]]] = {}  # Will be populated by update_neighbors()
        
        # Time encoding function (fixed, non-learnable)
        self.time_encoder = TimeEncoder(time_dim)

        # Flag to track if dense tables are built
        self._dense_tables_built = False
        self._cached_edge_index = None
        self._cached_edge_time = None

    def _get_device(self) -> torch.device:
        """Safely get device from module buffers or parameters."""
        for buf in self.buffers():
            return buf.device
        for param in self.parameters():
            return param.device
        return torch.device('cpu')
    
    
    def update_neighbors(
            self, 
            edge_index: torch.Tensor, 
            edge_time: torch.Tensor,
            force: bool = False
    ):
        """
        Update the neighbor cache with current graph structure.
        
        Args:
            edge_index: [2, num_edges] tensor of edges
            edge_time: [num_edges] tensor of timestamps
        """
        
        # Compute hash for change detection (more robust than .equal())
        edge_index_hash = hash(tuple(edge_index.cpu().numpy().flatten()))
        edge_time_hash = hash(tuple(edge_time.cpu().numpy()))
        
        # Skip if edges haven't changed
        if not force and self._edges_initialized:
            if (edge_index_hash == self._cached_edge_index_hash and 
                edge_time_hash == self._cached_edge_time_hash):
                return

        # Store hashes for next comparison
        self._cached_edge_index_hash = edge_index_hash
        self._cached_edge_time_hash = edge_time_hash
        self._edges_initialized = True
        self.neighbor_cache = {}
        self._dense_tables_built = False

        # Clear dense buffers if they exist
        for attr in ['dense_neighbor_ids', 'dense_neighbor_times', 'dense_neighbor_counts']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Convert to numpy for faster processing
        edge_index_np = edge_index.cpu().numpy()
        edge_time_np = edge_time.cpu().numpy()

        # Build neighbor lists for each node (undirected graph)
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            t = float(edge_time_np[i])
            
            # Add both directions
            self._add_to_cache(src, dst, t)
            self._add_to_cache(dst, src, t)

         # Sort neighbors by time descending (most recent first) for each node
        for node in self.neighbor_cache:
            self.neighbor_cache[node].sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Updated neighbor cache with {len(self.neighbor_cache)} nodes")
           
 
    def _add_to_cache(self, src: int, dst: int, t: float):
        """Helper to add a neighbor to cache."""
        if src not in self.neighbor_cache:
            self.neighbor_cache[src] = []
        self.neighbor_cache[src].append((dst, t))   
    
    def get_temporal_neighbors(self, node: int, current_time: float) -> List[Tuple[int, float]]:
        """Get all neighbors of node with timestamps < current_time."""
        if node not in self.neighbor_cache:
            return []
        
        return [(n, t) for n, t in self.neighbor_cache[node] if t < current_time]
    
    def sample_neighbor_with_temporal_bias(
        self, 
        node: int, 
        current_time: float
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Sample a neighbor using temporal bias (Eq. 7).
        P(u, t' | v, t) = exp((t' - t_max)/τ) / sum(exp((t_i - t_max)/τ))
        """
        valid_neighbors = self.get_temporal_neighbors(node, current_time)
        
        if not valid_neighbors:
            return None, None
        
        if len(valid_neighbors) == 1:
            return valid_neighbors[0]
        
        timestamps = np.array([t for _, t in valid_neighbors], dtype=np.float32)
        t_max = np.max(timestamps)
        
        # Compute exponential weights with numerical stability
        safe_temp = max(self.temperature, 1e-6)
        weights = np.exp((timestamps - t_max) / safe_temp)
        # Prevent overflow/underflow
        weights = np.clip(weights, 1e-10, 1e10)  
        probs = weights / (np.sum(weights) + 1e-10)
        
        idx = np.random.choice(len(valid_neighbors), p=probs)
        return valid_neighbors[idx]
    
    
    def compute_restart_probability(
        self,
        node: int,
        current_time: float,
        memory_state: Optional[torch.Tensor] = None
    ) -> float:
        """Compute learnable restart probability ρ_u(τ) for TAWR walks."""
        if memory_state is None:
            return 0.1
        
        device = memory_state.device
        # FIX: Use torch.clamp for safe indexing
        node_idx = torch.clamp(torch.tensor(node, device=device), 0, memory_state.size(0) - 1)
        
        # FIX: Ensure time_encoder is on same device without reassignment
        if next(self.time_encoder.parameters(), None) is not None:
            target_device = next(self.time_encoder.parameters()).device
            if device != target_device:
                self.time_encoder = self.time_encoder.to(device)
        
        # Get time encoding
        time_tensor = torch.tensor([current_time], device=device, dtype=torch.float32)
        time_enc = self.time_encoder(time_tensor)  # [1, time_dim]
        
        # Combine memory and time encoding
        memory_tensor = memory_state[node_idx:node_idx+1]  # [1, memory_dim]
        combined = torch.cat([memory_tensor, time_enc], dim=-1)  # [1, memory_dim + time_dim]
        
        # Compute probability with NaN protection
        logits = self.restart_projection(combined)
        prob = torch.sigmoid(logits).item()
        
        return np.clip(prob, 0.0, 1.0)
    
    def build_dense_neighbor_table(self):
        """Convert neighbor cache into dense tensors for fast GPU sampling."""
        if not self.neighbor_cache:
            logger.error("Cannot build dense table: neighbor_cache is empty!")
            raise RuntimeError("Cannot build dense neighbor table: neighbor_cache is empty")
        
        if self._dense_tables_built:
            return
        
        device = self._get_device()  # FIX: Safe device detection
        max_degree = max((len(neighbors) for neighbors in self.neighbor_cache.values()), default=1)
        max_degree = max(max_degree, 1)
        num_nodes = self.num_nodes
        
        # Pre-allocate dense tensors
        neighbor_ids = torch.zeros(num_nodes, max_degree, dtype=torch.long, device=device)
        neighbor_times = torch.zeros(num_nodes, max_degree, dtype=torch.float32, device=device)
        neighbor_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        # FIX: Populate dense tensors FIRST, then check bounds
        for node, neighbors in self.neighbor_cache.items():
            if 0 <= node < num_nodes and neighbors:
                deg = min(len(neighbors), max_degree)
                neighbor_counts[node] = deg
                for i, (nid, ts) in enumerate(neighbors[:deg]):
                    if 0 <= nid < num_nodes:
                        neighbor_ids[node, i] = nid
                        neighbor_times[node, i] = float(ts)
        
        # FIX: Proper buffer registration/update using copy_()
        if not hasattr(self, 'dense_neighbor_ids'):
            self.register_buffer('dense_neighbor_ids', neighbor_ids, persistent=False)
            self.register_buffer('dense_neighbor_times', neighbor_times, persistent=False)
            self.register_buffer('dense_neighbor_counts', neighbor_counts, persistent=False)
        else:
            self.dense_neighbor_ids.copy_(neighbor_ids)
            self.dense_neighbor_times.copy_(neighbor_times)
            self.dense_neighbor_counts.copy_(neighbor_counts)
        
        self._dense_tables_built = True
        logger.info(f"Built dense neighbor table: max_degree={max_degree}")
    
    def compute_restart_probabilities_batched(
        self,
        node_ids: torch.Tensor,
        times: torch.Tensor,
        memory_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute restart probabilities for a batch of walks."""
        if memory_states is None:
            return torch.full_like(node_ids, 0.1, dtype=torch.float, device=node_ids.device)

        batch_size, num_walks = node_ids.shape
        device = node_ids.device

        # Clamp node IDs to valid range
        node_ids_clamped = torch.clamp(node_ids, 0, memory_states.size(0) - 1)
        node_memory = memory_states[node_ids_clamped]  # [B, W, memory_dim]

        # Encode times
        time_enc = self.time_encoder(times)  # [B, W, time_dim]

        # Concatenate and project
        combined = torch.cat([node_memory, time_enc], dim=-1)  # [B, W, memory_dim+time_dim]
        combined_flat = combined.view(-1, combined.size(-1))
        logits_flat = self.restart_projection(combined_flat)
        logits = logits_flat.view(batch_size, num_walks)

        probs = torch.sigmoid(logits)
        return torch.clamp(probs, 0.0, 1.0)

    def _sample_walks_vectorized(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        num_walks: int,
        walk_length: int
    ) -> Dict[str, torch.Tensor]:
        """Unified vectorized walk sampling with temporal bias and NaN protection."""
        batch_size = source_nodes.size(0)
        device = source_nodes.device
        
        if not hasattr(self, 'dense_neighbor_ids'):
            self.build_dense_neighbor_table()
        
        safe_temp = max(self.temperature, 1e-6)  # FIX: Always use safe_temp
        max_deg = self.dense_neighbor_ids.size(1)
        eps = 1e-8

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

            # Base validity mask
            base_mask = (neighbor_ids > 0) & (neighbor_times > 0)
            temporal_mask = neighbor_times < curr_times.unsqueeze(-1)
            valid_neighbor_mask = base_mask & temporal_mask
            valid_counts = valid_neighbor_mask.sum(dim=-1)
            has_valid = valid_counts > 0

            # Compute temporal weights with numerical stability
            t_max = neighbor_times.masked_fill(~base_mask, -float('inf')).max(dim=-1, keepdim=True)[0]
            t_max = torch.where(t_max > -float('inf'), t_max, torch.zeros_like(t_max))
            
            # FIX: Use safe_temp and clamp to prevent overflow
            time_diff = (neighbor_times - t_max) / safe_temp
            time_diff = torch.clamp(time_diff, min=-50, max=50)
            temp_weights = torch.exp(time_diff)
            
            # Apply masks
            temp_weights = temp_weights.masked_fill(~valid_neighbor_mask, 0.0)
            fallback_weights = base_mask.float()
            
            # Choose weights based on validity
            weights = torch.where(has_valid.unsqueeze(-1), temp_weights, fallback_weights)
            sampling_mask = torch.where(has_valid.unsqueeze(-1), valid_neighbor_mask, base_mask)

            # Normalize to probabilities
            weights = weights.masked_fill(~sampling_mask, 0.0)
            prob_sums = weights.sum(dim=-1, keepdim=True)
            
            # Handle edge cases
            no_neighbors = ~base_mask.any(dim=-1)
            isolated_with_fallback = (~has_valid) & base_mask.any(dim=-1)
            
            probs = torch.zeros_like(weights)
            
            # Case 1: Valid temporal neighbors
            has_temporal = has_valid & ~no_neighbors
            if has_temporal.any():
                safe_sums = torch.where(prob_sums > eps, prob_sums, torch.ones_like(prob_sums))
                probs[has_temporal] = weights[has_temporal] / safe_sums[has_temporal]
            
            # Case 2: Has neighbors but none temporally valid
            if isolated_with_fallback.any():
                fallback_sums = fallback_weights[isolated_with_fallback].sum(dim=-1, keepdim=True)
                safe_fallback_sums = torch.where(fallback_sums > eps, fallback_sums, torch.ones_like(fallback_sums))
                probs[isolated_with_fallback] = fallback_weights[isolated_with_fallback] / safe_fallback_sums
            
            # Case 3: Completely isolated - uniform
            if no_neighbors.any():
                probs[no_neighbors] = 1.0 / max_deg
            
            # Final safety: clamp, NaN check, renormalize
            probs = torch.clamp(probs, min=0.0, max=1.0)
            probs = torch.nan_to_num(probs, nan=1.0/max_deg, posinf=1.0/max_deg, neginf=0.0)
            row_sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sums > eps, probs / row_sums, torch.ones_like(probs) / max_deg)

            # Sample with multinomial
            probs_flat = probs.view(-1, max_deg)
            valid_probs = (probs_flat >= 0) & torch.isfinite(probs_flat)
            
            if not valid_probs.all():
                probs_flat = torch.where(valid_probs, probs_flat, torch.ones_like(probs_flat) / max_deg)
                row_sums = probs_flat.sum(dim=-1, keepdim=True)
                probs_flat = torch.where(row_sums > eps, probs_flat / row_sums, torch.ones_like(probs_flat) / max_deg)
            
            sampled_idx = torch.multinomial(probs_flat, 1).view(batch_size, num_walks)
            sampled_idx = torch.clamp(sampled_idx, 0, max_deg - 1)

            # Gather results
            next_node = torch.gather(neighbor_ids, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)
            next_time = torch.gather(neighbor_times, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            # Handle isolated nodes: stay at current node
            next_node = torch.where(no_neighbors, curr_nodes, next_node)
            next_time = torch.where(no_neighbors, curr_times, next_time)

            # Bounds check
            next_node = torch.clamp(next_node, 0, self.num_nodes - 1)

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
        """Sample long walks."""
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
        """Vectorized TAWR walk sampling with restart probabilities and NaN protection."""
        batch_size = len(source_nodes)
        device = source_nodes.device
        
        if not hasattr(self, 'dense_neighbor_ids'):
            self.build_dense_neighbor_table()
        
        safe_temp = max(self.temperature, 1e-6)
        max_deg = self.dense_neighbor_ids.size(1)
        eps = 1e-8

        # Expand to [B, W]
        curr_nodes = source_nodes.unsqueeze(1).expand(-1, self.num_walks_tawr)
        curr_times = current_times.unsqueeze(1).expand(-1, self.num_walks_tawr)
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
            # Compute restart probabilities with NaN protection
            restart_probs = self.compute_restart_probabilities_batched(
                curr_nodes, curr_times, memory_states
            )
            restart_probs = torch.nan_to_num(restart_probs, nan=0.1, posinf=1.0, neginf=0.0)
            restart_probs = torch.clamp(restart_probs, 0.0, 1.0)
            
            # Sample restart decisions
            rand = torch.rand(batch_size, self.num_walks_tawr, device=device)
            restart_mask = rand < restart_probs
            
            # Restart path
            next_node_restart = original_source
            next_time_restart = original_times
            
            # Continue path: sample next neighbor
            neighbor_ids = self.dense_neighbor_ids[curr_nodes]
            neighbor_times = self.dense_neighbor_times[curr_nodes]
            
            base_mask = (neighbor_ids > 0) & (neighbor_times > 0)
            valid_neighbor_mask = base_mask & (neighbor_times < curr_times.unsqueeze(-1))

            has_temporal = valid_neighbor_mask.sum(dim=-1) > 0
            has_any = base_mask.sum(dim=-1) > 0
            
            sampling_mask = torch.where(has_temporal.unsqueeze(-1), valid_neighbor_mask, base_mask)
            
            # Compute weights with numerical stability
            t_max = neighbor_times.masked_fill(~base_mask, -float('inf')).max(dim=-1, keepdim=True)[0]
            t_max = torch.where(t_max > -float('inf'), t_max, torch.zeros_like(t_max))
            
            time_diff = (neighbor_times - t_max) / safe_temp  # FIX: Use safe_temp
            time_diff = torch.clamp(time_diff, min=-50, max=50)
            temporal_weights = torch.exp(time_diff)
            temporal_weights = temporal_weights.masked_fill(~valid_neighbor_mask, 0.0)

            uniform_weights = base_mask.float()
            weights = torch.where(has_temporal.unsqueeze(-1), temporal_weights, uniform_weights)
            
            # Normalize to probabilities
            prob_sums = weights.sum(dim=-1, keepdim=True)
            probs = torch.where(
                prob_sums > eps,
                weights / (prob_sums + eps),
                torch.ones_like(weights) / max_deg
            )
            
            # Apply sampling mask and renormalize
            probs = probs.masked_fill(~sampling_mask, 0.0)
            prob_sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(
                prob_sums > eps,
                probs / (prob_sums + eps),
                torch.ones_like(probs) / max_deg
            )
            
            # Final safety checks
            probs = torch.clamp(probs, min=0.0, max=1.0)
            probs = torch.nan_to_num(probs, nan=1.0/max_deg, posinf=1.0/max_deg, neginf=0.0)
            row_sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sums > eps, probs / row_sums, torch.ones_like(probs) / max_deg)
            
            # Sample
            probs_flat = probs.view(-1, max_deg)
            valid_probs = (probs_flat >= 0) & torch.isfinite(probs_flat)
            
            if not valid_probs.all():
                probs_flat = torch.where(valid_probs, probs_flat, torch.ones_like(probs_flat) / max_deg)
                row_sums = probs_flat.sum(dim=-1, keepdim=True)
                probs_flat = torch.where(row_sums > eps, probs_flat / row_sums, torch.ones_like(probs_flat) / max_deg)
            
            sampled_idx = torch.multinomial(probs_flat, 1).view(batch_size, self.num_walks_tawr)
            sampled_idx = torch.clamp(sampled_idx, 0, max_deg - 1)

            next_node_continue = torch.gather(neighbor_ids, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)
            next_time_continue = torch.gather(neighbor_times, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            # Handle isolated nodes
            next_node_continue = torch.where(has_any, next_node_continue, curr_nodes)
            next_time_continue = torch.where(has_any, next_time_continue, curr_times)
            next_node_continue = torch.clamp(next_node_continue, 0, self.num_nodes - 1)
            
            # Combine restart and continue paths
            next_node = torch.where(restart_mask, next_node_restart, next_node_continue)
            next_time = torch.where(restart_mask, next_time_restart, next_time_continue)

            # Record step
            walk_nodes[:, :, step] = next_node
            walk_times[:, :, step] = next_time
            walk_restart[:, :, step] = restart_mask.float()
            walk_masks[:, :, step] = 1.0

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
        Vectorized anonymization of walks by replacing node IDs with positional identifiers.
        FIX: Replaced slow Python loops with tensor-based mapping.
        """
        nodes = walk_data['nodes']  # [batch, num_walks, walk_len]
        masks = walk_data['masks']
        batch_size, num_walks, walk_len = nodes.shape
        
        device = nodes.device
        nodes_anon = torch.zeros_like(nodes)
        
        # Vectorized anonymization per batch item
        for b in range(batch_size):
            valid_mask = masks[b].bool()
            batch_nodes = nodes[b]
            valid_nodes = batch_nodes[valid_mask]
            
            if valid_nodes.numel() == 0:
                continue
            
            # Get unique nodes (excluding padding=0)
            unique_nodes = torch.unique(valid_nodes[valid_nodes > 0])
            
            if unique_nodes.numel() == 0:
                continue

            # FIX: Create mapping tensor for fast vectorized lookup
            max_node_id = unique_nodes.max().item() + 1
            mapping_tensor = torch.zeros(max_node_id + 1, dtype=torch.long, device=device)
            mapping_tensor[unique_nodes] = torch.arange(1, len(unique_nodes) + 1, device=device)
            
            # Apply mapping vectorized
            flat_nodes = batch_nodes.view(-1)
            flat_anon = torch.zeros_like(flat_nodes)
            flat_mask = masks[b].view(-1).bool()
            
            valid_flat = flat_mask & (flat_nodes > 0) & (flat_nodes < mapping_tensor.size(0))
            flat_anon[valid_flat] = mapping_tensor[flat_nodes[valid_flat]]
            
            nodes_anon[b] = flat_anon.view(num_walks, walk_len)
        
        # Sanity checks
        if nodes_anon.numel() > 0:
            assert nodes_anon.min() >= 0, "Anonymized nodes contain negative values"
            max_anon = nodes_anon.max().item()
            assert max_anon <= 10000, f"Anonymized ID {max_anon} seems too large"
        
        walk_data['nodes_anon'] = nodes_anon
        return walk_data
    
    def clear_cache(self):
        """Clear all cached data."""
        self.neighbor_cache.clear()
        self._dense_tables_built = False
        self._cached_edge_index_hash = None
        self._cached_edge_time_hash = None
        
        for attr in ['dense_neighbor_ids', 'dense_neighbor_times', 'dense_neighbor_counts']:
            if hasattr(self, attr):
                delattr(self, attr)
    
    
    def forward(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_time: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate multi-scale walks for source and target nodes."""
        # Update neighbor cache if new edges provided
        if edge_index is not None and edge_time is not None:
            self.update_neighbors(edge_index, edge_time, force=False)
        
        # Build dense tables if needed
        if self.neighbor_cache and not self._dense_tables_built:
            self.build_dense_neighbor_table()
        
        # Fallback if no neighbor data available
        if not hasattr(self, 'dense_neighbor_ids'):
            batch_size = len(source_nodes)
            empty_walk = {
                'nodes': torch.zeros(batch_size, 1, 1, dtype=torch.long, device=source_nodes.device),
                'times': torch.zeros(batch_size, 1, 1, dtype=torch.float32, device=source_nodes.device),
                'masks': torch.zeros(batch_size, 1, 1, dtype=torch.float32, device=source_nodes.device)
            }
            return {
                'source': {'short': empty_walk, 'long': empty_walk, 'tawr': empty_walk},
                'target': {'short': empty_walk, 'long': empty_walk, 'tawr': empty_walk}
            }
        
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
        
        return result
    




import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from loguru import logger

from .time_encoder import TimeEncoder

class MultiScaleWalkSampler(nn.Module):
    """
    Enhanced Multi-scale temporal walk sampler with:
    - Temporal bias sampling 
    - TAWR walks with learnable restart probabilities
    - Vectorized GPU sampling for efficiency
    - Walk anonymization for privacy
    - [NEW] Dynamic temporal noise injection
    - [NEW] Temporal masking for augmentation
    - [NEW] Adaptive walk lengths based on time gaps
    - [NEW] Co-occurrence weighting for HCT alignment
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
            # New Configuration Parameters
            time_noise_std: float = 0.0,       # Std dev for temporal noise (0.0 = disabled)
            mask_prob: float = 0.0,            # Probability to mask a node (0.0 - 0.1 recommended)
            adaptive_length_factor: float = 0.0, # Factor to scale length based on time gap
            min_time_gap: float = 1.0,         # Reference time unit for adaptive scaling
    ):
        super(MultiScaleWalkSampler, self).__init__()
        
        # Walk configuration
        self.num_nodes = num_nodes
        self.base_walk_length_short = walk_length_short
        self.base_walk_length_long = walk_length_long
        self.base_walk_length_tawr = walk_length_tawr
        
        self.num_walks_short = num_walks_short
        self.num_walks_long = num_walks_long
        self.num_walks_tawr = num_walks_tawr
        
        # Ensure temperature is never zero
        self.temperature = max(float(temperature), 1e-6)
        self.register_buffer('safe_temperature', torch.tensor(self.temperature))

        # New Hyperparameters
        self.time_noise_std = time_noise_std
        self.mask_prob = max(0.0, min(1.0, mask_prob))
        self.adaptive_length_factor = adaptive_length_factor
        self.min_time_gap = min_time_gap

        # Learnable restart probability parameters
        self.restart_projection = nn.Linear(memory_dim + time_dim, 1)
        nn.init.xavier_uniform_(self.restart_projection.weight)
        nn.init.constant_(self.restart_projection.bias, -2.197)
        
        # Track initialization
        self._edges_initialized = False
        self.neighbor_cache: Dict[int, List[Tuple[int, float]]] = {}
        
        # Time encoding function
        self.time_encoder = TimeEncoder(time_dim)

        # Dense table flags
        self._dense_tables_built = False
        self._cached_edge_hash = None
        
        # Track global time range for adaptive lengths
        self.register_buffer('global_min_time', torch.tensor(0.0))
        self.register_buffer('global_max_time', torch.tensor(0.0))

    def _get_device(self) -> torch.device:
        """Safely get device from module buffers or parameters."""
        for buf in self.buffers():
            return buf.device
        for param in self.parameters():
            return param.device
        return torch.device('cpu')
    
    def _compute_edge_hash(self, edge_index: torch.Tensor, edge_time: torch.Tensor) -> int:
        """Fast hash using summary statistics."""
        shape_hash = hash((edge_index.shape, edge_time.shape))
        
        if edge_index.numel() > 0:
            ei_flat = edge_index.flatten()
            et_flat = edge_time.flatten()
            sample_hash = hash((
                ei_flat[:10].sum().item() if len(ei_flat) >= 10 else ei_flat.sum().item(),
                ei_flat[-10:].sum().item() if len(ei_flat) >= 10 else 0,
                et_flat[:10].sum().item() if len(et_flat) >= 10 else et_flat.sum().item(),
                et_flat[-10:].sum().item() if len(et_flat) >= 10 else 0,
            ))
        else:
            sample_hash = hash(0)
        
        return hash((shape_hash, sample_hash))
    
    def update_neighbors(
            self, 
            edge_index: torch.Tensor, 
            edge_time: torch.Tensor,
            force: bool = False
    ):
        """Update the neighbor cache and track global time range."""
        
        edge_hash = self._compute_edge_hash(edge_index, edge_time)
        if not force and self._edges_initialized and edge_hash == self._cached_edge_hash:
            return
        
        self._cached_edge_hash = edge_hash
        self._edges_initialized = True
        self.neighbor_cache = {}
        self._dense_tables_built = False

        # Update global time range for adaptive calculations
        if edge_time.numel() > 0:
            self.global_min_time.fill_(edge_time.min().item())
            self.global_max_time.fill_(edge_time.max().item())

        # Clear dense buffers
        for attr in ['dense_neighbor_ids', 'dense_neighbor_times', 'dense_neighbor_counts']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Convert to numpy
        edge_index_np = edge_index.cpu().numpy()
        edge_time_np = edge_time.cpu().numpy()

        # Build neighbor lists
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            t = float(edge_time_np[i])            
            self._add_to_cache(src, dst, t)
            self._add_to_cache(dst, src, t)

        # Sort neighbors by time descending
        for node in self.neighbor_cache:
            self.neighbor_cache[node].sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Updated neighbor cache: {len(self.neighbor_cache)} nodes, Time Range: [{self.global_min_time.item():.2f}, {self.global_max_time.item():.2f}]")
           
    def _add_to_cache(self, src: int, dst: int, t: float):
        if src not in self.neighbor_cache:
            self.neighbor_cache[src] = []
        self.neighbor_cache[src].append((dst, t))   
    
    def get_temporal_neighbors(self, node: int, current_time: float) -> List[Tuple[int, float]]:
        if node not in self.neighbor_cache:
            return []
        return [(n, t) for n, t in self.neighbor_cache[node] if t < current_time]
    
    def _ensure_module_device(self, module: nn.Module, target_device: torch.device):
        for buf in module.buffers():
            if buf.device != target_device:
                return module.to(target_device)
        for param in self.parameters():
            if param.device != target_device:
                return module.to(target_device)
        return module
    
    def compute_restart_probability(
        self,
        node: int,
        current_time: float,
        memory_state: Optional[torch.Tensor] = None
    ) -> float:
        if memory_state is None:
            return 0.1
        
        device = memory_state.device
        node_idx = torch.clamp(torch.tensor(node, device=device), 0, memory_state.size(0) - 1)
        self.time_encoder = self._ensure_module_device(self.time_encoder, device)

        time_tensor = torch.tensor([current_time], device=device, dtype=torch.float32)
        time_enc = self.time_encoder(time_tensor)
        
        memory_tensor = memory_state[node_idx:node_idx+1]
        combined = torch.cat([memory_tensor, time_enc], dim=-1)
        
        logits = self.restart_projection(combined)
        prob = torch.sigmoid(logits).item()
        
        return np.clip(prob, 0.0, 1.0)
    
    def build_dense_neighbor_table(self):
        if not self.neighbor_cache:
            logger.error("Cannot build dense table: neighbor_cache is empty!")
            raise RuntimeError("Cannot build dense neighbor table: neighbor_cache is empty")
        
        if self._dense_tables_built:
            return
        
        device = self._get_device()
        max_degree = max((len(neighbors) for neighbors in self.neighbor_cache.values()), default=1)
        max_degree = max(max_degree, 1)
        num_nodes = self.num_nodes
        
        neighbor_ids = torch.zeros(num_nodes, max_degree, dtype=torch.long, device=device)
        neighbor_times = torch.zeros(num_nodes, max_degree, dtype=torch.float32, device=device)
        neighbor_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        for node, neighbors in self.neighbor_cache.items():
            if 0 <= node < num_nodes and neighbors:
                deg = min(len(neighbors), max_degree)
                neighbor_counts[node] = deg
                for i, (nid, ts) in enumerate(neighbors[:deg]):
                    if 0 <= nid < num_nodes:
                        neighbor_ids[node, i] = nid
                        neighbor_times[node, i] = float(ts)
        
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
        if memory_states is None:
            return torch.full_like(node_ids, 0.1, dtype=torch.float, device=node_ids.device)

        batch_size, num_walks = node_ids.shape
        device = node_ids.device

        node_ids_clamped = torch.clamp(node_ids, 0, memory_states.size(0) - 1)
        node_memory = memory_states[node_ids_clamped]

        time_enc = self.time_encoder(times)
        combined = torch.cat([node_memory, time_enc], dim=-1)
        combined_flat = combined.view(-1, combined.size(-1))
        logits_flat = self.restart_projection(combined_flat)
        logits = logits_flat.view(batch_size, num_walks)

        probs = torch.sigmoid(logits)
        return torch.clamp(probs, 0.0, 1.0)

    def _calculate_adaptive_length(self, current_times: torch.Tensor, base_length: int) -> int:
        """
        Calculate adaptive walk length based on time gap.
        Logic: Larger gap (older event) -> Longer walk to capture more context.
        Smaller gap (recent event) -> Shorter walk for locality.
        """
        if self.adaptive_length_factor <= 0 or self.global_max_time.item() == 0:
            return base_length
        
        device = current_times.device
        # Normalize time gap relative to global range
        time_range = self.global_max_time - self.global_min_time
        if time_range <= 0:
            return base_length
            
        # How far back is this event from the "now" (global max)?
        # Gap = (Max_Time - Current_Time) / Range  -> [0, 1]
        gaps = (self.global_max_time - current_times) / (time_range + 1e-9)
        
        # Average gap for the batch to determine a single length for this call
        # (Vectorized sampling requires fixed length per call, so we use the mean or max)
        avg_gap = gaps.mean().item()
        
        # Scale: base_length * (1 + factor * gap)
        # If gap is 1 (oldest), length = base * (1 + factor)
        # If gap is 0 (newest), length = base
        scaled_length = int(base_length * (1.0 + self.adaptive_length_factor * avg_gap))
        
        # Clamp to reasonable bounds (at least 2, at most 2x base)
        return max(2, min(scaled_length, base_length * 2))

    def _apply_temporal_noise(self, times: torch.Tensor) -> torch.Tensor:
        """Inject Gaussian noise into timestamps."""
        if self.time_noise_std <= 0:
            return times
        
        noise = torch.randn_like(times) * self.time_noise_std
        # Ensure time doesn't go below global min or above global max significantly
        noisy_times = times + noise
        return torch.clamp(noisy_times, self.global_min_time, self.global_max_time + 1.0)

    def _apply_temporal_masking(self, masks: torch.Tensor) -> torch.Tensor:
        """Randomly mask out nodes in the walk."""
        if self.mask_prob <= 0:
            return masks
        
        # Create a random mask where True means "keep", False means "mask out"
        keep_prob = 1.0 - self.mask_prob
        random_mask = torch.rand_like(masks) < keep_prob
        
        # Apply to existing validity mask
        # Note: We don't mask the first step (index 0) to ensure we always have a start node
        augmented_masks = masks * random_mask
        
        # Ensure first step is never masked
        aug_slice = [slice(None)] * augmented_masks.dim()
        aug_slice[-1] = slice(0, 1) # Assuming last dim is sequence length
        augmented_masks[tuple(aug_slice)] = masks[tuple(aug_slice)]
        
        return augmented_masks

    def _sample_walks_vectorized(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        num_walks: int,
        walk_length: int,
        memory_states: Optional[torch.Tensor] = None,
        co_occurrence_scores: Optional[torch.Tensor] = None # [num_edges] or mapped to neighbors
    ) -> Dict[str, torch.Tensor]:
        """Unified vectorized walk sampling with all new enhancements."""
        batch_size = source_nodes.size(0)
        device = source_nodes.device
        
        # Sanity Check: Ensure inputs match
        if current_times.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: source_nodes ({batch_size}) vs current_times ({current_times.size(0)})")
        
        
        # 1. Apply Dynamic Temporal Noise
        effective_times = self._apply_temporal_noise(current_times)
        
        if not hasattr(self, 'dense_neighbor_ids'):
            self.build_dense_neighbor_table()
        
        safe_temp = max(self.temperature, 1e-6)
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


        cooc_weights = None
        if co_occurrence_scores is not None and isinstance(co_occurrence_scores, dict):
            # Determine which walk type we are sampling (based on num_walks arg matching config)
            type_key = 'short'
            if num_walks == self.num_walks_long: type_key = 'long'
            elif num_walks == self.num_walks_tawr: type_key = 'tawr'
            
            if type_key in co_occurrence_scores:
                full_cooc = co_occurrence_scores[type_key] # [B_orig*2, W, W]
                
                # Handle shape mismatch due to Hard Negative Mining
                if full_cooc.size(0) == batch_size:
                    cooc_full = full_cooc
                elif full_cooc.size(0) * 2 == batch_size or full_cooc.size(0) < batch_size:
                    # Mismatch: Cooc was calculated on original batch, now we have augmented.
                    logger.warning(f"Co-occurrence shape {full_cooc.shape} mismatch...")
                    cooc_full = None  # Silently disables co-occurrence weighting
                else:
                    cooc_full = full_cooc[:batch_size] # Truncate if larger
                
                if cooc_full is not None:
                    # Map [B, W, W] -> [B, W, Max_Deg]
                    # Heuristic: The neighbor at position k in the dense table belongs to some walk index.
                    # Since we don't store walk indices in the dense table, we approximate:
                    # Weight = Mean Co-occurrence of current walk i with ALL walks (global structural importance)
                    # OR simply use the diagonal (self-co-occurrence) if available.
                    
                    # Better Approach for Dense Table: 
                    # Use the row-sum of co-occurrence as a node-specific structural weight.
                    # Score[i] = Sum_j Cooccur[i, j]
                    # Then expand to [B, W, 1] -> broadcast to [B, W, Max_Deg]
                    
                    row_sums = cooc_full.sum(dim=-1) # [B, W]
                    cooc_weights = row_sums.unsqueeze(-1).expand(-1, -1, max_deg) # [B, W, Max_Deg]
                    cooc_weights = torch.clamp(cooc_weights, min=0.0, max=10.0)

        for step in range(1, walk_length):
            # Gather neighbors
            curr_nodes_clamped = torch.clamp(curr_nodes, 0, self.num_nodes - 1)
            
            neighbor_ids = self.dense_neighbor_ids[curr_nodes_clamped]
            neighbor_times = self.dense_neighbor_times[curr_nodes_clamped]
            neighbor_counts = self.dense_neighbor_counts[curr_nodes_clamped]

            pos_mask = torch.arange(max_deg, device=device).unsqueeze(0).unsqueeze(0)
            node_degree = neighbor_counts.unsqueeze(-1)
            base_mask = pos_mask < node_degree

            temporal_mask = neighbor_times < curr_times.unsqueeze(-1)
            valid_neighbor_mask = base_mask & temporal_mask
            
            valid_counts = valid_neighbor_mask.sum(dim=-1)
            has_valid = valid_counts > 0

            # Compute temporal weights
            t_max = neighbor_times.masked_fill(~base_mask, -float('inf')).max(dim=-1, keepdim=True)[0]
            t_max = torch.where(t_max > -float('inf'), t_max, torch.zeros_like(t_max))

            time_diff = (neighbor_times - t_max) / safe_temp
            time_diff = torch.clamp(time_diff, min=-50, max=50)
            temp_weights = torch.exp(time_diff)
            
            # 2. Apply Co-occurrence Weighting
            # --- APPLY CO-OCCURRENCE WEIGHTING ---
            if cooc_weights is not None:
                # Multiply temporal weights by structural co-occurrence importance
                # Shape: [B, W, Max_Deg] * [B, W, Max_Deg]
                temp_weights = temp_weights * (cooc_weights + 1e-6)

            temp_weights = temp_weights.masked_fill(~valid_neighbor_mask, 0.0)
            fallback_weights = base_mask.float()
            
            has_valid = valid_neighbor_mask.sum(dim=-1) > 0
            weights = torch.where(has_valid.unsqueeze(-1), temp_weights, fallback_weights)
            sampling_mask = torch.where(has_valid.unsqueeze(-1), valid_neighbor_mask, base_mask)
            
                        
            # Normalize
            weights = weights.masked_fill(~sampling_mask, 0.0)
            prob_sums = weights.sum(dim=-1, keepdim=True)
            
            no_neighbors = ~base_mask.any(dim=-1)
            isolated_with_fallback = (~has_valid) & base_mask.any(dim=-1)
            
            probs = torch.zeros_like(weights)
            
            has_temporal = has_valid & ~no_neighbors
            
            if has_temporal.any():
                safe_sums = torch.where(prob_sums > eps, prob_sums, torch.ones_like(prob_sums))
                probs[has_temporal] = weights[has_temporal] / safe_sums[has_temporal]
            
            if isolated_with_fallback.any():
                fallback_sums = fallback_weights[isolated_with_fallback].sum(dim=-1, keepdim=True)
                safe_fallback_sums = torch.where(fallback_sums > eps, fallback_sums, torch.ones_like(fallback_sums))
                probs[isolated_with_fallback] = fallback_weights[isolated_with_fallback] / safe_fallback_sums
            
            if no_neighbors.any():
                probs[no_neighbors] = 1.0 / max_deg
            
            probs = torch.clamp(probs, min=0.0, max=1.0)
            probs = torch.nan_to_num(probs, nan=1.0/max_deg, posinf=1.0/max_deg, neginf=0.0)
            row_sums_final = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sums_final > eps, probs / row_sums_final, torch.ones_like(probs) / max_deg)

            # Sample
            probs_flat = probs.view(-1, max_deg)
            valid_probs = (probs_flat >= 0) & torch.isfinite(probs_flat)
            
            if not valid_probs.all():
                probs_flat = torch.where(valid_probs, probs_flat, torch.ones_like(probs_flat) / max_deg)
                row_sums = probs_flat.sum(dim=-1, keepdim=True)
                probs_flat = torch.where(row_sums > eps, probs_flat / row_sums, torch.ones_like(probs_flat) / max_deg)
            
            sampled_idx = torch.multinomial(probs_flat, 1).view(batch_size, num_walks)
            sampled_idx = torch.clamp(sampled_idx, 0, max_deg - 1)

            next_node = torch.gather(neighbor_ids, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)
            next_time = torch.gather(neighbor_times, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            next_node = torch.where(no_neighbors, curr_nodes, next_node)
            next_time = torch.where(no_neighbors, curr_times, next_time)
            next_node = torch.clamp(next_node, 0, self.num_nodes - 1)

            # Update walk tensors
            walk_nodes[:, :, step] = next_node
            walk_times[:, :, step] = next_time
            walk_masks[:, :, step] = 1.0

            curr_nodes = next_node
            curr_times = next_time

        # 3. Apply Temporal Masking (Augmentation)
        # walk_masks = self._apply_temporal_masking(walk_masks)

        return {
            'nodes': walk_nodes,
            'times': walk_times,
            'masks': walk_masks
        }
    
    def sample_short_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        co_occurrence_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Adaptive Length Logic
        walk_len = self._calculate_adaptive_length(current_times, self.base_walk_length_short)
        
        return self._sample_walks_vectorized(
            source_nodes, current_times,
            num_walks=self.num_walks_short,
            walk_length=walk_len,
            co_occurrence_scores=co_occurrence_scores
        )
    
    def sample_long_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: Optional[torch.Tensor] = None,
        co_occurrence_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        walk_len = self._calculate_adaptive_length(current_times, self.base_walk_length_long)
        
        return self._sample_walks_vectorized(
            source_nodes, current_times,
            num_walks=self.num_walks_long,
            walk_length=walk_len,
            memory_states=memory_states,
            co_occurrence_scores=co_occurrence_scores
        )
    
    def sample_tawr_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: torch.Tensor,
        co_occurrence_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(source_nodes)
        device = source_nodes.device
        
        # Adaptive Length
        walk_len = self._calculate_adaptive_length(current_times, self.base_walk_length_tawr)
        
        if not hasattr(self, 'dense_neighbor_ids'):
            self.build_dense_neighbor_table()
        
        safe_temp = max(self.temperature, 1e-6)
        max_deg = self.dense_neighbor_ids.size(1)
        eps = 1e-8

        # Apply Noise
        effective_times = self._apply_temporal_noise(current_times)

        curr_nodes = source_nodes.unsqueeze(1).expand(-1, self.num_walks_tawr)
        curr_times = effective_times.unsqueeze(1).expand(-1, self.num_walks_tawr)
        original_source = source_nodes.unsqueeze(1).expand(-1, self.num_walks_tawr)
        original_times = effective_times.unsqueeze(1).expand(-1, self.num_walks_tawr)

        walk_nodes = torch.zeros(batch_size, self.num_walks_tawr, walk_len, dtype=torch.long, device=device)
        walk_times = torch.zeros(batch_size, self.num_walks_tawr, walk_len, dtype=torch.float32, device=device)
        walk_restart = torch.zeros_like(walk_nodes, dtype=torch.float32)
        walk_masks = torch.zeros_like(walk_nodes, dtype=torch.float32)

        walk_nodes[:, :, 0] = curr_nodes
        walk_times[:, :, 0] = curr_times
        walk_masks[:, :, 0] = 1.0

        for step in range(1, walk_len):
            restart_probs = self.compute_restart_probabilities_batched(
                curr_nodes, curr_times, memory_states
            )
            restart_probs = torch.nan_to_num(restart_probs, nan=0.1, posinf=1.0, neginf=0.0)
            restart_probs = torch.clamp(restart_probs, 0.0, 1.0)
            
            rand = torch.rand(batch_size, self.num_walks_tawr, device=device)
            restart_mask = rand < restart_probs
            
            next_node_restart = original_source
            next_time_restart = original_times
            
            neighbor_ids = self.dense_neighbor_ids[curr_nodes]
            neighbor_times = self.dense_neighbor_times[curr_nodes]
            neighbor_counts = self.dense_neighbor_counts[curr_nodes]
            
            pos_mask = torch.arange(max_deg, device=device).unsqueeze(0).unsqueeze(0)
            node_degree = neighbor_counts.unsqueeze(-1)
            base_mask = pos_mask < node_degree
            
            temporal_mask = neighbor_times < curr_times.unsqueeze(-1)
            valid_neighbor_mask = base_mask & temporal_mask

            has_temporal = valid_neighbor_mask.sum(dim=-1) > 0
            has_any = base_mask.sum(dim=-1) > 0
            
            sampling_mask = torch.where(has_temporal.unsqueeze(-1), valid_neighbor_mask, base_mask)
            
            t_max = neighbor_times.masked_fill(~base_mask, -float('inf')).max(dim=-1, keepdim=True)[0]
            t_max = torch.where(t_max > -float('inf'), t_max, torch.zeros_like(t_max))
            
            time_diff = (neighbor_times - t_max) / safe_temp
            time_diff = torch.clamp(time_diff, min=-50, max=50)
            temporal_weights = torch.exp(time_diff)
            temporal_weights = temporal_weights.masked_fill(~valid_neighbor_mask, 0.0)

            # Co-occurrence Weighting
            if co_occurrence_scores is not None and isinstance(co_occurrence_scores, torch.Tensor) and co_occurrence_scores.shape == neighbor_ids.shape:
                temporal_weights = temporal_weights * (co_occurrence_scores + 1e-6)

            uniform_weights = base_mask.float()
            weights = torch.where(has_temporal.unsqueeze(-1), temporal_weights, uniform_weights)
            
            prob_sums = weights.sum(dim=-1, keepdim=True)
            probs = torch.where(
                prob_sums > eps,
                weights / (prob_sums + eps),
                torch.ones_like(weights) / max_deg
            )
            
            probs = probs.masked_fill(~sampling_mask, 0.0)
            prob_sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(
                prob_sums > eps,
                probs / (prob_sums + eps),
                torch.ones_like(probs) / max_deg
            )
            
            probs = torch.clamp(probs, min=0.0, max=1.0)
            probs = torch.nan_to_num(probs, nan=1.0/max_deg, posinf=1.0/max_deg, neginf=0.0)
            row_sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sums > eps, probs / row_sums, torch.ones_like(probs) / max_deg)
            
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

            next_node_continue = torch.where(has_any, next_node_continue, curr_nodes)
            next_time_continue = torch.where(has_any, next_time_continue, curr_times)
            next_node_continue = torch.clamp(next_node_continue, 0, self.num_nodes - 1)
            
            next_node = torch.where(restart_mask, next_node_restart, next_node_continue)
            next_time = torch.where(restart_mask, next_time_restart, next_time_continue)

            walk_nodes[:, :, step] = next_node
            walk_times[:, :, step] = next_time
            walk_restart[:, :, step] = restart_mask.float()
            walk_masks[:, :, step] = 1.0

            curr_nodes = next_node
            curr_times = next_time

        # Apply Temporal Masking
        walk_masks = self._apply_temporal_masking(walk_masks)

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
        nodes = walk_data['nodes']
        masks = walk_data['masks']
        batch_size, num_walks, walk_len = nodes.shape
        
        device = nodes.device
        nodes_anon = torch.zeros_like(nodes)
        
        for b in range(batch_size):
            valid_mask = masks[b].bool()
            batch_nodes = nodes[b]
            valid_nodes = batch_nodes[valid_mask]
            
            if valid_nodes.numel() == 0:
                continue
            
            unique_nodes = torch.unique(valid_nodes[valid_nodes > 0])
            if unique_nodes.numel() == 0:
                continue
            
            max_node_id = unique_nodes.max().item()
            
            if max_node_id > 100000:
                node_to_anon = {nid.item(): idx+1 for idx, nid in enumerate(unique_nodes)}
                flat_nodes = batch_nodes.view(-1)
                flat_mask = masks[b].view(-1).bool()
                flat_anon = torch.zeros_like(flat_nodes)
                
                for i, nid in enumerate(flat_nodes):
                    if flat_mask[i] and nid.item() > 0 and nid.item() in node_to_anon:
                        flat_anon[i] = node_to_anon[nid.item()]
                
                nodes_anon[b] = flat_anon.view(num_walks, walk_len)
            else:
                mapping_tensor = torch.zeros(max_node_id + 2, dtype=torch.long, device=device)
                mapping_tensor[unique_nodes] = torch.arange(1, len(unique_nodes) + 1, device=device)
                
                flat_nodes = batch_nodes.view(-1)
                flat_anon = torch.zeros_like(flat_nodes)
                flat_mask = masks[b].view(-1).bool()
                
                valid_flat = flat_mask & (flat_nodes > 0) & (flat_nodes < mapping_tensor.size(0))
                flat_anon[valid_flat] = mapping_tensor[flat_nodes[valid_flat]]
                
                nodes_anon[b] = flat_anon.view(num_walks, walk_len)
        
        if nodes_anon.numel() > 0:
            assert nodes_anon.min() >= 0, "Anonymized nodes contain negative values"
            max_anon = nodes_anon.max().item()
            if max_anon > 10000:
                logger.warning(f"Anonymized ID {max_anon} seems large (walk_type={walk_type})")
        
        
        walk_data['nodes_anon'] = nodes_anon
        return walk_data
    
    def clear_cache(self):
        self.neighbor_cache.clear()
        self._dense_tables_built = False
        
        for attr in ['_cached_edge_index_hash', '_cached_edge_time_hash', '_cached_edge_hash']:
            if hasattr(self, attr):
                delattr(self, attr)
        
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
        edge_time: Optional[torch.Tensor] = None,
        co_occurrence_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate multi-scale walks.
        
        Args:
            co_occurrence_scores: Optional tensor of scores. 
                If using dense neighbor tables, this should ideally be pre-mapped to [num_nodes, max_degree]
                matching the structure of `dense_neighbor_ids`. 
                If passed as a flat edge list, internal mapping would be required (omitted here for speed, 
                assuming user pre-processes scores to match dense table shape if needed).
        """
        if edge_index is not None and edge_time is not None:
            self.update_neighbors(edge_index, edge_time, force=False)
        
        if self.neighbor_cache and not self._dense_tables_built:
            self.build_dense_neighbor_table()
        
        if not hasattr(self, 'dense_neighbor_ids'):
            batch_size = len(source_nodes)
            device = source_nodes.device

            def _empty_walk_dict(B: int, num_walks: int, walk_length: int, device: torch.device, include_restart: bool = False) -> Dict:
                result = {
                    'nodes': torch.zeros(B, num_walks, walk_length, dtype=torch.long, device=device),
                    'times': torch.zeros(B, num_walks, walk_length, dtype=torch.float32, device=device),
                    'masks': torch.zeros(B, num_walks, walk_length, dtype=torch.float32, device=device)
                }
                if include_restart:
                    result['restart_flags'] = torch.zeros(B, num_walks, walk_length, dtype=torch.float32, device=device)
                return result

            return {
                'source': {
                    'short': _empty_walk_dict(batch_size, self.num_walks_short, self.base_walk_length_short, device),
                    'long': _empty_walk_dict(batch_size, self.num_walks_long, self.base_walk_length_long, device),
                    'tawr': _empty_walk_dict(batch_size, self.num_walks_tawr, self.base_walk_length_tawr, device, include_restart=True)
                },
                'target': {
                    'short': _empty_walk_dict(batch_size, self.num_walks_short, self.base_walk_length_short, device),
                    'long': _empty_walk_dict(batch_size, self.num_walks_long, self.base_walk_length_long, device),
                    'tawr': _empty_walk_dict(batch_size, self.num_walks_tawr, self.base_walk_length_tawr, device, include_restart=True)
                }
            }
        
        all_nodes = torch.cat([source_nodes, target_nodes])
        all_times = torch.cat([current_times, current_times])
        
        # Note: For co_occurrence_scores, if it's specific to source/target, you might need to split/cat them too.
        # Here we assume if passed, it's either global or broadcastable. 
        # If specific per-batch item, ensure co_occurrence_scores is concatenated similarly.
        all_co_scores = None
        if co_occurrence_scores is not None:
            # Placeholder logic: if user passes a single tensor, we pass it down. 
            # In a real scenario, you might need to cat source_scores and target_scores.
            all_co_scores = co_occurrence_scores 

        short_walks = self.sample_short_walks(all_nodes, all_times, co_occurrence_scores=all_co_scores)
        long_walks = self.sample_long_walks(all_nodes, all_times, memory_states, co_occurrence_scores=all_co_scores)
        tawr_walks = self.sample_tawr_walks(all_nodes, all_times, memory_states, co_occurrence_scores=all_co_scores)
        
        short_walks = self.anonymize_walks(short_walks, 'short')
        long_walks = self.anonymize_walks(long_walks, 'long')
        tawr_walks = self.anonymize_walks(tawr_walks, 'tawr')
        
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
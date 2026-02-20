import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from collections import defaultdict
import random

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
        self.neighbor_cache = {}  # Will be populated by update_neighbors()
        
        # Time encoding function (fixed, non-learnable)
        self.time_encoder = TimeEncoder(time_dim)

    
    def update_neighbors(self, edge_index: torch.Tensor, edge_time: torch.Tensor):
        """
        Update the neighbor cache with current graph structure.
        
        Args:
            edge_index: [2, num_edges] tensor of edges
            edge_time: [num_edges] tensor of timestamps
        """
        self.neighbor_cache = {}
        
        # Convert to numpy for faster processing (or keep on GPU if possible)
        edge_index_np = edge_index.cpu().numpy() if edge_index.is_cuda else edge_index.numpy()
        edge_time_np = edge_time.cpu().numpy() if edge_time.is_cuda else edge_time.numpy()
        
        # Build neighbor lists for each node
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            t = edge_time_np[i]
            
            # Add both directions for undirected graphs (modify if directed)
            self._add_to_cache(src, dst, t)
            self._add_to_cache(dst, src, t)
        
        # Sort neighbors by time for each node (most recent first)
        for node in self.neighbor_cache:
            neighbors = self.neighbor_cache[node]
            # Sort by timestamp descending (most recent first)
            neighbors.sort(key=lambda x: x[1], reverse=True)
    
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
        timestamps = np.array([t for _, t in valid_neighbors])
        t_max = np.max(timestamps)
        
        # Compute exponential weights (Eq. 7)
        weights = np.exp((timestamps - t_max) / self.temperature)
        probs = weights / np.sum(weights)
        
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
        
        # Get time encoding
        time_tensor = torch.tensor([current_time], device=self.device).float()
        time_enc = self.time_encoder(time_tensor)  # [1, time_dim]
        
        # Combine memory and time encoding
        memory_tensor = memory_state[node].unsqueeze(0)  # [1, memory_dim]
        combined = torch.cat([memory_tensor, time_enc], dim=-1)  # [1, memory_dim + time_dim]
        
        # Compute probability
        logits = self.restart_projection(combined)  # [1, 1]
        prob = torch.sigmoid(logits).item()
        
        return prob
    
    
    def sample_short_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample short walks for multiple source nodes in parallel.
        
        Args:
            source_nodes: [batch_size] tensor of source node IDs
            current_times: [batch_size] tensor of current times
            memory_states: [num_nodes, memory_dim] memory states (optional)
            
        Returns:
            Dictionary with walk tensors:
                - 'nodes': [batch_size, num_walks_short, walk_length_short]
                - 'times': [batch_size, num_walks_short, walk_length_short]
                - 'masks': [batch_size, num_walks_short, walk_length_short] (1 if valid)
        """
        batch_size = len(source_nodes)
        
        # Convert to numpy for iterative sampling
        source_np = source_nodes.cpu().numpy()
        times_np = current_times.cpu().numpy()
        
        # Initialize walk tensors
        walk_nodes = np.zeros((batch_size, self.num_walks_short, self.walk_length_short), dtype=np.int64)
        walk_times = np.zeros((batch_size, self.num_walks_short, self.walk_length_short), dtype=np.float32)
        walk_masks = np.zeros((batch_size, self.num_walks_short, self.walk_length_short), dtype=np.float32)
        
        # Sample walks for each batch item
        for b in range(batch_size):
            source = source_np[b]
            current_time = times_np[b]
            
            for w in range(self.num_walks_short):
                # Initialize walk
                walk_nodes[b, w, 0] = source
                walk_times[b, w, 0] = current_time
                walk_masks[b, w, 0] = 1.0
                
                curr_node = source
                curr_time = current_time
                
                # Extend walk
                for step in range(1, self.walk_length_short):
                    # Sample next node with temporal bias
                    next_node, next_time = self.sample_neighbor_with_temporal_bias(
                        curr_node, curr_time
                    )
                    
                    if next_node is None:
                        # No more valid neighbors, pad with zeros
                        break
                    
                    walk_nodes[b, w, step] = next_node
                    walk_times[b, w, step] = next_time
                    walk_masks[b, w, step] = 1.0
                    
                    curr_node = next_node
                    curr_time = next_time
        
        # Convert to tensors
        return {
            'nodes': torch.tensor(walk_nodes, device=self.device),
            'times': torch.tensor(walk_times, device=self.device),
            'masks': torch.tensor(walk_masks, device=self.device)
        }   
    
    
    def sample_long_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample long walks (same as short walks but with longer length).
        """
        batch_size = len(source_nodes)
        source_np = source_nodes.cpu().numpy()
        times_np = current_times.cpu().numpy()
        
        walk_nodes = np.zeros((batch_size, self.num_walks_long, self.walk_length_long), dtype=np.int64)
        walk_times = np.zeros((batch_size, self.num_walks_long, self.walk_length_long), dtype=np.float32)
        walk_masks = np.zeros((batch_size, self.num_walks_long, self.walk_length_long), dtype=np.float32)
        
        for b in range(batch_size):
            source = source_np[b]
            current_time = times_np[b]
            
            for w in range(self.num_walks_long):
                walk_nodes[b, w, 0] = source
                walk_times[b, w, 0] = current_time
                walk_masks[b, w, 0] = 1.0
                
                curr_node = source
                curr_time = current_time
                
                for step in range(1, self.walk_length_long):
                    next_node, next_time = self.sample_neighbor_with_temporal_bias(
                        curr_node, curr_time
                    )
                    
                    if next_node is None:
                        break
                    
                    walk_nodes[b, w, step] = next_node
                    walk_times[b, w, step] = next_time
                    walk_masks[b, w, step] = 1.0
                    
                    curr_node = next_node
                    curr_time = next_time
        
        return {
            'nodes': torch.tensor(walk_nodes, device=self.device),
            'times': torch.tensor(walk_times, device=self.device),
            'masks': torch.tensor(walk_masks, device=self.device)
        }
    
    
    def sample_tawr_walks(
        self,
        source_nodes: torch.Tensor,
        current_times: torch.Tensor,
        memory_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Sample TAWR (Temporal Anonymous Walks with Restart) walks.
        
        These walks have a learnable restart probability ρ_u(τ) that
        can reset the walk to the source node.
        
        Args:
            source_nodes: [batch_size] tensor of source node IDs
            current_times: [batch_size] tensor of current times
            memory_states: [num_nodes, memory_dim] memory states from SAM
            
        Returns:
            Dictionary with walk tensors:
                - 'nodes': [batch_size, num_walks_tawr, walk_length_tawr]
                - 'times': [batch_size, num_walks_tawr, walk_length_tawr]
                - 'restart_flags': [batch_size, num_walks_tawr, walk_length_tawr] (1 if restart occurred)
                - 'masks': [batch_size, num_walks_tawr, walk_length_tawr]
        """
        batch_size = len(source_nodes)
        source_np = source_nodes.cpu().numpy()
        times_np = current_times.cpu().numpy()
        
        walk_nodes = np.zeros((batch_size, self.num_walks_tawr, self.walk_length_tawr), dtype=np.int64)
        walk_times = np.zeros((batch_size, self.num_walks_tawr, self.walk_length_tawr), dtype=np.float32)
        walk_restart = np.zeros((batch_size, self.num_walks_tawr, self.walk_length_tawr), dtype=np.float32)
        walk_masks = np.zeros((batch_size, self.num_walks_tawr, self.walk_length_tawr), dtype=np.float32)
        
        for b in range(batch_size):
            source = source_np[b]
            original_source = source  # Keep original source for restarts
            current_time = times_np[b]
            
            for w in range(self.num_walks_tawr):
                # Initialize walk
                walk_nodes[b, w, 0] = source
                walk_times[b, w, 0] = current_time
                walk_restart[b, w, 0] = 0.0
                walk_masks[b, w, 0] = 1.0
                
                curr_node = source
                curr_time = current_time
                
                for step in range(1, self.walk_length_tawr):
                    # Compute restart probability for current node
                    restart_prob = self.compute_restart_probability(
                        curr_node, curr_time, memory_states
                    )
                    
                    # Decide whether to restart
                    if random.random() < restart_prob:
                        # Restart to source node
                        walk_nodes[b, w, step] = original_source
                        walk_times[b, w, step] = current_time  # Reset to original time
                        walk_restart[b, w, step] = 1.0
                        walk_masks[b, w, step] = 1.0
                        
                        curr_node = original_source
                        curr_time = current_time
                    else:
                        # Continue normally
                        next_node, next_time = self.sample_neighbor_with_temporal_bias(
                            curr_node, curr_time
                        )
                        
                        if next_node is None:
                            break
                        
                        walk_nodes[b, w, step] = next_node
                        walk_times[b, w, step] = next_time
                        walk_restart[b, w, step] = 0.0
                        walk_masks[b, w, step] = 1.0
                        
                        curr_node = next_node
                        curr_time = next_time
        
        return {
            'nodes': torch.tensor(walk_nodes, device=self.device),
            'times': torch.tensor(walk_times, device=self.device),
            'restart_flags': torch.tensor(walk_restart, device=self.device),
            'masks': torch.tensor(walk_masks, device=self.device)
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
        
        # Create anonymized tensor (same shape, but with anonymized IDs)
        anonymized = torch.zeros_like(nodes)
        
        for b in range(batch_size):
            # Get all unique nodes appearing in walks for this batch item
            unique_nodes = torch.unique(nodes[b])
            
            # Create mapping from node ID to anonymized ID
            # Anonymized IDs are assigned based on first occurrence position
            node_to_anon = {}
            
            # First pass: assign anonymized IDs based on first occurrence
            for w in range(num_walks):
                for step in range(walk_len):
                    node = nodes[b, w, step].item()
                    if node == 0:  # Padding
                        continue
                    if node not in node_to_anon:
                        node_to_anon[node] = len(node_to_anon) + 1  # 1-based IDs (0 for padding)
            
            # Second pass: replace with anonymized IDs
            for w in range(num_walks):
                for step in range(walk_len):
                    node = nodes[b, w, step].item()
                    if node == 0:
                        anonymized[b, w, step] = 0
                    else:
                        anonymized[b, w, step] = node_to_anon[node]
        
        walk_data['nodes_anon'] = anonymized
        return walk_data
    
    
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
        # Update neighbor cache if new edges provided
        if edge_index is not None and edge_time is not None:
            self.update_neighbors(edge_index, edge_time)
        
        # Combine source and target nodes for efficient sampling
        all_nodes = torch.cat([source_nodes, target_nodes])
        all_times = torch.cat([current_times, current_times])
        
        # Sample all walk types
        short_walks = self.sample_short_walks(all_nodes, all_times, memory_states)
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
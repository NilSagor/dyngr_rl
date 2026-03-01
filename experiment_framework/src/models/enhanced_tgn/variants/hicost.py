import os
import torch 
import torch.nn as nn
import numpy as np

from typing import Dict, Optional, Tuple
from loguru import logger

from ..base_enhance_tgn import BaseEnhancedTGN
from ..component.time_encoder import TimeEncoder
from ..component.sam_module import StabilityAugmentedMemory
from ..component.multi_swalk import MultiScaleWalkSampler
from ..component.walk_encoder import WalkEncoder
from ..component.co_transformer import HierarchicalCooccurrenceTransformer
from ..component.stode_module import SpectralTemporalODE
from ..component.transformer_encoder import MergeLayer



class HiCoST(nn.Module):
    """
     Hierarchical Co‑occurrence Spectral Temporal
     hierarchical co‑occurrence + spectral‑temporal ODE
     SAM + Multi-Scale Walk Sampler + Hierarchical Co-occurrence Transformer (HCT) + ST ODE
    
    Architecture:
    1. SAM maintains stable node memory via prototypes
    2. Multi-scale walk sampler generates short/long/TAWR walks with anonymization
    3. HCT processes walks: intra-walk -> co-occurrence -> inter-walk -> fusion
    4. Combine HCT walk embeddings with base TGN embeddings
    5. Spectral-Temporal ODE for continuous-time node representation learning.
    6. Link prediction on combined representations
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_features: int = 0,
        hidden_dim: int = 172,
        time_encoding_dim: int = 32,
        memory_dim: int = 172,
        message_dim: int = 172,
        edge_features_dim: int = 172,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_heads: int = 2,
        n_neighbors: int = 20,
        use_memory: bool = True,
        embedding_module_type: str = "graph_attention",
        message_function_type: str = "mlp",
        aggregator_type: str = "last",
        memory_updater_type: str = "sam",
        # SAM params
        num_prototypes: int = 5,
        similarity_metric: str = "cosine",
        # Multi-scale walk params
        walk_length_short: int = 2, # 3
        walk_length_long: int = 5, # 10
        walk_length_tawr: int = 4, # 8
        num_walks_short: int = 5, # 10
        num_walks_long: int = 3,    # 5
        num_walks_tawr: int = 3, # 5
        walk_temperature: float = 0.1,
        use_walk_encoder: bool = True,  # Toggle to ablate walks        
        use_hct: bool = True,  # Toggle HCT vs base walk encoder
        # HCT params
        hct_d_model: int = 128,
        hct_nhead: int = 4,
        hct_num_intra_layers: int = 2,
        hct_num_inter_layers: int = 2,
        hct_dim_feedforward: int = 256,
        hct_cooccurrence_sigma: float = 2.0,
        hct_cooccurrence_gamma: float = 0.5,
        # ST ODE
        num_eigenvectors: int = 10,
        mu: float = 0.0,
        adaptive_mu: bool = False,
        use_gru_ode: bool = True,
        ode_method: str = 'rk4',
        ode_step_size: Optional[float] = 1000,        
        adjoint: bool = True,        
        aggregation: str = 'mean',  # 'mean', 'sum', 'max'
        time_precision: int = 6,  # Decimal places for time hashing
        use_checkpoint: bool = True,
        use_st_ode: bool = True,
        **kwargs
    ):
        # Initialize base TGN WITHOUT memory updater (SAM replaces it)
        super().__init__(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=hidden_dim,
            time_encoding_dim=time_encoding_dim,
            memory_dim=memory_dim,
            message_dim=message_dim,
            edge_features_dim=edge_features_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_heads=n_heads,
            n_neighbors=n_neighbors,
            use_memory=use_memory,
            embedding_module_type=embedding_module_type,
            message_function_type=message_function_type,
            aggregator_type=aggregator_type,            
            memory_updater_type="none",  # Disable GRU updater - SAM handles updates
            **kwargs
        )
        # torch.autograd.set_detect_anomaly(True)
        if os.environ.get('DEBUG_GRADIENTS'):
            torch.autograd.set_detect_anomaly(True)
        
        self.use_st_ode = use_st_ode
        self.directed = kwargs.get('directed', False)

        self.time_encoder = TimeEncoder(time_encoding_dim)
        
        # Initialize SAM with ALL required parameters
        self.sam_module = StabilityAugmentedMemory(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            node_feat_dim=node_features,
            edge_feat_dim=edge_features_dim,
            time_dim=time_encoding_dim,
            num_prototypes=num_prototypes,
            similarity_metric=similarity_metric,
            dropout=dropout            
        )
        
        # Store SAM config for logging/debugging
        self.sam_config = {
            'num_prototypes': num_prototypes,
            'similarity_metric': similarity_metric,
            'memory_dim': memory_dim
        }
        
        logger.info(f" SAM initialized: {num_prototypes} prototypes, {similarity_metric} similarity")

        # Initialize Multi-Scale Walk Sampler
        self.walk_sampler = MultiScaleWalkSampler(
            num_nodes=num_nodes,
            walk_length_short=walk_length_short,
            walk_length_long=walk_length_long,
            walk_length_tawr=walk_length_tawr,
            num_walks_short=num_walks_short,
            num_walks_long=num_walks_long,
            num_walks_tawr=num_walks_tawr,
            temperature=walk_temperature,
            memory_dim=memory_dim,
            time_dim=time_encoding_dim
        )

        
        
        
        
        # Initialize HCT (replaces old WalkEncoder)
        self.use_hct = use_hct
        if use_hct:
            # HCT d_model can be different from memory_dim
            # We project SAM memory into HCT space
            self.hct = HierarchicalCooccurrenceTransformer(
                d_model=hct_d_model, # HCT operates in this space
                memory_dim=memory_dim,
                nhead=hct_nhead,
                num_intra_layers=hct_num_intra_layers,
                num_inter_layers=hct_num_inter_layers,
                dim_feedforward=hct_dim_feedforward,
                dropout=dropout,
                max_walk_length=max(walk_length_short, walk_length_long, walk_length_tawr),
                max_num_walks=num_walks_short + num_walks_long + num_walks_tawr,
                cooccurrence_sigma=hct_cooccurrence_sigma,
                cooccurrence_gamma=hct_cooccurrence_gamma,
                use_walk_type_embedding=True
            )
            
            # SAM memory [memory_dim] -> HCT [d_model]            
            # Project SAM memory to HCT dimension if different
            if memory_dim != hct_d_model:
                self.memory_proj = nn.Linear(memory_dim, hct_d_model)
            else:
                self.memory_proj = nn.Identity()
                
            # Project HCT output back to hidden_dim for fusion
            # HCT output [d_model] -> TGN hidden [hidden_dim]
            # Project HCT output [d_model] -> TGN hidden [hidden_dim]
            if hct_d_model != hidden_dim:
                self.hct_to_tgn = nn.Sequential(
                    nn.Linear(hct_d_model, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            else:
                self.hct_to_tgn = nn.Identity()
        
        else:
            # Fallback to simple walk encoder (TGNv4 style)            
            self.walk_encoder = WalkEncoder(
                walk_length_short=walk_length_short,
                walk_length_long=walk_length_long,
                walk_length_tawr=walk_length_tawr,
                memory_dim=memory_dim,
                output_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout
            )

        if self.use_st_ode:
            # ST-ODE evolves full node memory [num_nodes, memory_dim]
            self.st_ode = SpectralTemporalODE(
                    hidden_dim = memory_dim,
                    num_nodes = num_nodes,
                    num_eigenvectors = num_eigenvectors,  # Small for speed
                    mu = mu,
                    adaptive_mu = adaptive_mu,
                    use_gru_ode = use_gru_ode,
                    ode_method = ode_method,  # Fast for testing
                    ode_step_size=ode_step_size,
                    num_layers=num_layers,
                    adjoint=adjoint,
                    dropout=dropout,  # Disable for deterministic testing
                    aggregation = aggregation,
                    time_precision = time_precision,
                    use_checkpoint = use_checkpoint
            )

            # Project walk observations to memory space if needed
            if hct_d_model != memory_dim:
                self.walk_to_memory = nn.Linear(hct_d_model, memory_dim) 
            else:
                self.walk_to_memory = nn.Identity()

            # Time of last memory update
            self.register_buffer('last_update_time', torch.tensor(0.0))      
        
        # Fusion layer: combine HCT walk embeddings with base TGN embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.edge_index = None
        self.edge_time = None

        self._time_proj = None

        self._walk_cache = {}
        self._cache_walks = False  # Toggle

        # self.link_predictor = nn.Sequential(
        #     nn.Linear(hidden_dim*2, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, 1)
        # )
        self.link_predictor = MergeLayer(
            input_dim1=hidden_dim,
            input_dim2=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        )
        
        
        logger.info(f"TGNv6 initialized: SAM({num_prototypes}p) + WalkSampler + "
                   f"HCT({hct_d_model}d, {hct_nhead}h)" if use_hct else "SimpleWalkEncoder")
    
    def _prepare_stode_observations(self, batch):
        """
        Convert batch interactions to ST-ODE observation format.
        ST-ODE expects walk_encodings as [num_nodes, hidden_dim] and 
        walk_masks as [num_nodes] boolean tensor.
        """
        device = self.device
        num_nodes = self.num_nodes
        
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']

        # Fill with actual interaction data
        # Node embeddings from SAM memory
        src_emb = self.sam_module.raw_memory[src_nodes]
        dst_emb = self.sam_module.raw_memory[dst_nodes]
        
        # Time encoding (project to memory dim if needed)
        time_emb = self.time_encoder(timestamps.float())
        if time_emb.shape[-1] != self.memory_dim:
            if self._time_proj is None:
                self._time_proj = nn.Linear(self.time_encoding_dim, self.memory_dim).to(device)
            time_emb = self._time_proj(time_emb)
        
        # # Build adjacency for the entire batch (or use a sliding window)
        # Sparse adjacency for the whole batch (same for all times in this batch)
        adj_t = self._build_temporal_adjacency(src_nodes, dst_nodes, num_nodes)
        
        # Group edges by unique timestamp
        unique_times, inverse = torch.unique(timestamps, sorted=True, return_inverse=True)
        T = len(unique_times)

        # Aggregate node observations per time (sum over edges at that time)
        node_obs_per_time = torch.zeros(T, num_nodes, self.memory_dim, device=device)
        for t_idx in range(T):
            mask = (inverse == t_idx)
            src_t = src_nodes[mask]
            dst_t = dst_nodes[mask]
            src_emb_t = src_emb[mask]
            dst_emb_t = dst_emb[mask]
            time_emb_t = time_emb[mask]

            # Add source, destination, and time contributions
            node_obs_per_time[t_idx].index_add_(0, src_t, src_emb_t)
            node_obs_per_time[t_idx].index_add_(0, dst_t, dst_emb_t)
            node_obs_per_time[t_idx].index_add_(0, src_t, time_emb_t)
            node_obs_per_time[t_idx].index_add_(0, dst_t, time_emb_t)

        # Boolean mask indicating which nodes have any observation at each time
        masks = (node_obs_per_time.abs().sum(dim=-1) > 0)  # [T, N]

        # Reshape to [N, T, 1, H] and [N, T, 1]
        walk_encodings = node_obs_per_time.permute(1, 0, 2).unsqueeze(2)   # [N, T, 1, H]
        walk_times = unique_times.view(1, T, 1).expand(num_nodes, -1, -1)  # [N, T, 1]
        walk_masks = masks.T.unsqueeze(-1)                                 # [N, T, 1] (boolean)

        return {
            'encodings': walk_encodings,
            'times': walk_times,
            'masks': walk_masks,
            'adjs': [adj_t] * T   # same adjacency for each time slice
        }       
        
        
        # # Get current batch time (single time point for ODE evolution)
        # current_time = batch['timestamps'].max()
        
        # # Create observation encodings from ALL interactions in the batch
        # # Treat the entire batch as observations at the current time
        # src_nodes = batch['src_nodes']
        # dst_nodes = batch['dst_nodes']

        # # Build adjacency for the batch
        # adj_t = self._build_temporal_adjacency(src_nodes, dst_nodes, num_nodes)
        
        # # Create observation encodings: [num_nodes, num_walks, walk_len, memory_dim]
        # # For batch interactions, we create a single "walk" per interacting node
        # obs_encoding = self._create_observation_encoding(
        #     src_nodes, dst_nodes, batch.get('edge_features'), batch['timestamps']
        # )
        
        # # Reshape to [N, W, L, H] format expected by ST-ODE
        # # Current shape is [N, 1, 3, memory_dim], need to ensure it's 4D
        # if obs_encoding.dim() == 5:
        #     # If somehow we got [T, N, W, L, H], take the last time step or mean
        #     obs_encoding = obs_encoding[-1]  # Take last time step: [N, W, L, H]
        


        # # Get unique timestamps in batch
        # unique_times, inverse_indices = torch.unique(
        #     batch['timestamps'], sorted=True, return_inverse=True
        # )
        
        # # For each unique time, create "walk" observations
        # # These represent the structural perturbations at that time
        
        # observations_list = []
        # adj_matrices = []
        
        # for t in unique_times:
        #     # Find interactions at this time
        #     mask = (batch['timestamps'] == t)
        #     src_nodes = batch['src_nodes'][mask]
        #     dst_nodes = batch['dst_nodes'][mask]
            
        #     # Build adjacency at time t
        #     # This is the graph structure for the spectral regularization
        #     adj_t = self._build_temporal_adjacency(src_nodes, dst_nodes, num_nodes)
        #     adj_matrices.append(adj_t)
            
        #     # Create observation encodings from edge features + node features
        #     # Shape: [num_nodes, num_walks, walk_len, memory_dim]
        #     # For interactions, we treat each edge as a "walk" of length 2 (src->dst)
        #     obs_encoding = self._create_observation_encoding(
        #         src_nodes, dst_nodes, batch.get('edge_features'), t
        #     )
        #     observations_list.append(obs_encoding)
        
        # # Stack: [T, num_nodes, 1, 3, memory_dim]
        # encodings = torch.stack(observations_list)  # [T, N, 1, 3, H]
        
        
        # # Get unique timestamps and their indices
        # unique_times, inverse_indices = torch.unique(timestamps, sorted=True, return_inverse=True)
        
        # # For ST-ODE, we need to provide observations at specific times
        # # The ODE will evolve from last_update_time to current_time
        # current_time = timestamps.max()


        # # Build adjacency for the entire batch (or use a sliding window)
        # adj_t = self._build_temporal_adjacency(src_nodes, dst_nodes, num_nodes)
        
        # # Create sparse observation tensor: [num_nodes, num_walks=1, walk_len=3, memory_dim]
        # # This represents the "perturbation" from new edges
        
        # # Fill with actual interaction data
        # src_emb = self.sam_module.raw_memory[src_nodes]
        # dst_emb = self.sam_module.raw_memory[dst_nodes]
        
        # # Time encoding for each interaction
        # time_emb = self.time_encoder(timestamps.float())
        # if time_emb.shape[-1] != self.memory_dim:
        #     if self._time_proj is None:
        #         self._time_proj = nn.Linear(self.time_encoding_dim, self.memory_dim).to(device)
        #     time_emb = self._time_proj(time_emb)

        # # Aggregate observations per node by summing contributions
        # # Initialize [num_nodes, memory_dim]
        # node_obs = torch.zeros(num_nodes, self.memory_dim, device=device)
        
        # # Add src embeddings
        # node_obs.index_add_(0, src_nodes, src_emb)
        # # Add dst embeddings  
        # node_obs.index_add_(0, dst_nodes, dst_emb)
        # # Add time embeddings to both
        # node_obs.index_add_(0, src_nodes, time_emb)
        # node_obs.index_add_(0, dst_nodes, time_emb)               

        # # Create mask for nodes that appear in this batch
        # unique_nodes = torch.unique(torch.cat([src_nodes, dst_nodes]))
        # mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        # mask[unique_nodes] = True


        # # Reshape to [N, 1, 1, H] for walk_encodings
        # encodings_4d = node_obs.unsqueeze(1).unsqueeze(2)          # [N, 1, 1, H]
        
        # # Current time (scalar) for all nodes/walks
        # current_time = batch['timestamps'].max().to(node_obs.device)
        # times_4d = current_time.view(1, 1, 1).expand(num_nodes, 1, 1)   # [N, 1, 1]
        
        # # Boolean mask for nodes that have any observation
        # masks_4d = mask.unsqueeze(1).unsqueeze(2)                  # [N, 1, 1]
        
        # # Note: This requires careful reshaping to match [N, W, L, H] format
        
        # return {
        #     'encodings': encodings_4d,
        #     'times': times_4d,
        #     'masks': masks_4d,
        #     'adjs': [adj_t]
        # }
        
    
    
    def _build_temporal_adjacency(self, src_nodes, dst_nodes, num_nodes):
        """Build SPARSE adjacency matrix for specific time snapshot."""
        device = src_nodes.device
        
        edges = torch.stack([src_nodes, dst_nodes], dim=0)  # [2, E]
        
        if not self.directed:
            edges = torch.cat([edges, edges.flip(0)], dim=1) # undirected
        
        
        # Add self-loops
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        edges = torch.cat([edges, self_loops], dim=1)

        values = torch.ones(edges.shape[1], device=device)
        # Create sparse tensor
        adj = torch.sparse_coo_tensor(
            edges, 
            values,
            size=(num_nodes, num_nodes),
            device=device
        )
        
        
        # adj = torch.zeros(num_nodes, num_nodes, device=src_nodes.device)
        
        # # Add edges
        # for s, d in zip(src_nodes, dst_nodes):
        #     adj[s, d] = 1.0
        #     if not self.directed:  # Assuming undirected
        #         adj[d, s] = 1.0
        
        # # Add self-loops for numerical stability
        # adj = adj + torch.eye(num_nodes, device=adj.device)
        
        return adj.coalesce()   # ensure coalesced for efficiency
    
    def _create_observation_encoding(self, src_nodes, dst_nodes, edge_feats, time):
        """
        Create walk-like encoding from interaction.
        time: scalar or 1D tensor of timestamps
        Returns: [B, 1, 3, memory_dim] for batch interactions
        """
        # Create [num_nodes, 1, 3, memory_dim] with zeros for non-interacting nodes
        device = src_nodes.device

        # Ensure time is at least 1D for TimeEncoder
        if time.dim() == 0:  # scalar
            time = time.unsqueeze(0)  # [1]

        # Initialize sparse observations for ALL nodes: [N, 1, 3, memory_dim]
        obs = torch.zeros(self.num_nodes, 1, 3, self.memory_dim, device=device)
        
        # Get node embeddings from SAM memory
        src_emb = self.sam_module.raw_memory[src_nodes]  # [B, memory_dim]
        dst_emb = self.sam_module.raw_memory[dst_nodes]  # [B, memory_dim]
        
        # Encode time: [1] or [B] -> [B, time_encoding_dim]
        time_emb = self.time_encoder(time)  # [B, time_encoding_dim]
        
               
        
        # Project time embedding to memory_dim if needed
        if time_emb.shape[-1] != self.memory_dim:
            if self._time_proj is None:
                self._time_proj = nn.Linear(self.time_encoding_dim, self.memory_dim).to(device)
            time_emb = self._time_proj(time_emb)  # [B, memory_dim]
        
        
       
        # Fill observations ONLY for interacting nodes (sparse update)
        # Position 0: src embedding, Position 1: dst embedding, Position 2: time context
        obs[src_nodes, 0, 0] = src_emb  # src at position 0
        obs[dst_nodes, 0, 1] = dst_emb  # dst at position 1
        obs[src_nodes, 0, 2] = time_emb  # time context at src
        obs[dst_nodes, 0, 2] = time_emb  # time context at dst
                
        # Add walk dimension: [B, 1, 3, memory_dim]
        return obs
    
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with proper SAM update sequencing.
        """
        # Step 1: Compute embeddings using CURRENT memory state
        # (Memory from previous batches, not yet updated with this batch)
        source_emb, dest_emb = self.compute_temporal_embeddings(
            batch['src_nodes'],
            batch['dst_nodes'],
            batch['timestamps']
        )
        
        # Step 2: Store interactions for SAM update (AFTER forward, BEFORE optimizer)
        # This ensures gradients flow through embedding computation first
        if self.training and self.use_memory:
            self._store_sam_interactions(batch)
        
        # Step 3: Link prediction
        # scores = self.link_predictor(
        #     torch.cat([source_emb, dest_emb], dim=-1)
        # ).squeeze(-1)
        scores = self.link_predictor(
            source_emb, dest_emb
        ).squeeze(-1)
        
        return scores 
        
    
    
    def _simple_walk_embed(self, walk_data: Dict) -> torch.Tensor:
        """Fallback: simple mean of walk node features."""
        nodes = walk_data['nodes']  # [B, num_walks, L]
        masks = walk_data['masks']  # [B, num_walks, L]
        
        flat_nodes = nodes.reshape(-1)
        flat_feats = self.sam_module.raw_memory[flat_nodes]
        flat_feats = flat_feats.view(nodes.size(0), nodes.size(1), nodes.size(2), -1)
        
        # Mean over valid steps and walks
        masks_expanded = masks.unsqueeze(-1).float()
        sum_feats = (flat_feats * masks_expanded).sum(dim=[1, 2])  # [B, memory_dim]
        count = masks.sum(dim=[1, 2]).unsqueeze(-1) + 1e-8
        return sum_feats / count
    
    
    
    def compute_temporal_embeddings(
        self,
        source_nodes: torch.Tensor,
        destination_nodes: torch.Tensor,
        edge_times: torch.Tensor,
        n_neighbors: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified embedding computation with HCT.
        """
        device = self.device
        
        # Convert inputs
        src_tensor = source_nodes.to(device) if torch.is_tensor(source_nodes) \
            else torch.from_numpy(source_nodes).long().to(device)
        dst_tensor = destination_nodes.to(device) if torch.is_tensor(destination_nodes) \
            else torch.from_numpy(destination_nodes).long().to(device)
        ts_tensor = edge_times.to(device) if torch.is_tensor(edge_times) \
            else torch.from_numpy(edge_times).float().to(device)
        
        # cache_key = (src_tensor.nbytes(), ts_tensor[0].item())  # Simple key
        
        # 1. Generate walks (indices only)
        if self._cache_walks:
            cache_key = self._get_cache_key(src_tensor, ts_tensor)
            if cache_key in self._walk_cache:
                walk_data = self._walk_cache[cache_key]
            else:
                walk_data = self.walk_sampler(
                source_nodes=src_tensor,
                target_nodes=dst_tensor,
                current_times=ts_tensor,
                memory_states=self.sam_module.raw_memory,  # For TAWR restart prob
                edge_index=self.edge_index,
                edge_time=self.edge_time
            )
                if self.training:  # Only cache during training
                    self._walk_cache[cache_key] = walk_data
        else:
            walk_data = self.walk_sampler(
            source_nodes=src_tensor,
            target_nodes=dst_tensor,
            current_times=ts_tensor,
            memory_states=self.sam_module.raw_memory,  # For TAWR restart prob
            edge_index=None,
            edge_time=None
        )
        
        
        # 1. Generate walks (indices only)
        
        
        
        # 2. HCT encodes walks by looking up SAM memory
        if self.use_hct:
            # Pass walks_dict with 'nodes' (for lookup) and 'nodes_anon' (for co-occurrence)
            hct_src_output = self.hct(
                walks_dict=walk_data['source'],
                node_memory=self.sam_module.raw_memory,
                # memory_proj=self.memory_proj,
                return_all=True  # Return dict with all intermediate outputs
            )
            
            hct_dst_output = self.hct(
                walks_dict=walk_data['target'],
                node_memory=self.sam_module.raw_memory,
                # memory_proj=self.memory_proj,
                return_all=True
            )
            
            # Extract fused embeddings
            hct_src_emb = hct_src_output['fused']  # [B, d_model]
            hct_dst_emb = hct_dst_output['fused']
            
            # Store for later logging (only keep source for brevity, or store both)
            self._last_hct_info = hct_src_output  # Contains 'fused', 'short', 'long', 'tawr'
            
            # Project to TGN hidden dimension
            walk_src_emb = self.hct_to_tgn(hct_src_emb)  # [B, hidden_dim]
            walk_dst_emb = self.hct_to_tgn(hct_dst_emb)
        else:
            # Fallback: simple mean of walk node memories (no HCT)
            walk_src_emb = self._simple_walk_embed(walk_data['source'])
            walk_dst_emb = self._simple_walk_embed(walk_data['target'])
        
        # if self.use_hct:
        #     print(f"HCT output: mean={hct_src_emb.mean():.4f}, std={hct_src_emb.std():.4f}")
        #     print(f"HCT output range: [{hct_src_emb.min():.4f}, {hct_src_emb.max():.4f}]")
        
        # if self.training and torch.rand(1).item() < 0.01:  # log ~1% of batches
        #     logger.info(f"Short walks (first batch): {walk_data['source']['short']['nodes'][0, :, :]}")
        #     logger.info(f"Anonymized short: {walk_data['source']['short']['nodes_anon'][0, :, :]}")
        #     logger.info(f"Masks: {walk_data['source']['short']['masks'][0, :, :]}")

        
        # 3. Base TGN embeddings from SAM + graph attention
        if self.embedding_module is not None:
            base_src_emb = self.embedding_module.compute_embedding(
                memory=self.sam_module.raw_memory,
                source_nodes=source_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            base_dst_emb = self.embedding_module.compute_embedding(
                memory=self.sam_module.raw_memory,
                source_nodes=destination_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
        else:
            base_src_emb = self.sam_module.raw_memory[src_tensor]
            base_dst_emb = self.sam_module.raw_memory[dst_tensor]
        
        # 4. Fusion
        combined_src = torch.cat([base_src_emb, walk_src_emb], dim=-1)
        combined_dst = torch.cat([base_dst_emb, walk_dst_emb], dim=-1)
        
        final_src = self.fusion_layer(combined_src)
        final_dst = self.fusion_layer(combined_dst)
        
        return final_src, final_dst

    def _store_sam_interactions(self, batch: Dict[str, torch.Tensor]):
        """Store interactions in buffer for deferred SAM update."""
        device = self.device
        
        src_nodes = batch['src_nodes'].to(device)
        dst_nodes = batch['dst_nodes'].to(device)
        timestamps = batch['timestamps'].to(device)
        
        # Get edge features
        if 'edge_features' in batch and batch['edge_features'] is not None:
            edge_feats = batch['edge_features'].to(device)
        else:
            edge_feats = torch.zeros(len(src_nodes), self.edge_features_dim, device=device)
        
        # Store in buffer for on_train_batch_end
        self._sam_batch_buffer = {
            'src_nodes': src_nodes,
            'dst_nodes': dst_nodes,
            'timestamps': timestamps,
            'edge_features': edge_feats
        }
        
    def set_neighbor_finder(self, neighbor_finder):
        # First call the base method to initialize embedding module, etc.
        super().set_neighbor_finder(neighbor_finder)

        # Extract edge_index and edge_time from the neighbor finder
        if hasattr(neighbor_finder, 'edge_index') and hasattr(neighbor_finder, 'edge_time'):
            edge_index = neighbor_finder.edge_index
            edge_time = neighbor_finder.edge_time

            # Ensure they are torch tensors on CPU (update_neighbors handles device later)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index)
            if not isinstance(edge_time, torch.Tensor):
                edge_time = torch.tensor(edge_time)

            # Initialize the walk sampler's neighbor cache
            self.walk_sampler.update_neighbors(edge_index, edge_time)
            logger.info(f"Walk sampler initialized with {edge_index.size(1)} edges")
        else:
            logger.warning("Neighbor finder missing edge_index/edge_time; walks will not work")
    
    def set_graph(self, edge_index: torch.Tensor, edge_time: torch.Tensor):
        """
        Provide the full training graph to the walk sampler.
        Must be called before any forward pass.
        """
        self.edge_index = edge_index
        self.edge_time = edge_time
        
        self.walk_sampler.update_neighbors(edge_index, edge_time)
        self.walk_sampler.build_dense_neighbor_table()  # ADD THIS LINE
        
        logger.info(f"Walk sampler initialized with {edge_index.size(1)} edges, "
                    f"max_degree={self.walk_sampler.dense_neighbor_ids.size(1) if hasattr(self.walk_sampler, 'dense_neighbor_ids') else 'N/A'}")

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Override to use SAM memory instead of GRU memory."""
        return self.sam_module.get_memory(node_ids)
       
    def _get_cache_key(self, src_tensor: torch.Tensor, ts_tensor: torch.Tensor) -> tuple:
        """Safe cache key generation."""
        ts_hash = hash((ts_tensor.min().item(), ts_tensor.max().item(), len(ts_tensor)))
        return (src_tensor.data_ptr(), src_tensor.shape, ts_hash)
    
    def on_load_checkpoint(self, checkpoint):
        """Called when loading a checkpoint."""
        # Rebuild neighbor cache from the dataset
        if hasattr(self, 'neighbor_finder') and self.neighbor_finder is not None:
            self._initialize_walk_sampler_from_neighbor_finder()
    
    def _initialize_walk_sampler_from_neighbor_finder(self):
        """Extract edge_index and edge_time from neighbor_finder and initialize walk sampler."""
        nf = self.neighbor_finder
        
        # Try to get edges from neighbor_finder
        if hasattr(nf, '_edges') and hasattr(nf, '_timestamps'):
            # Your neighbor finder stores edges internally
            edge_index = torch.tensor(nf._edges, dtype=torch.long).t()  # [2, num_edges]
            edge_time = torch.tensor(nf._timestamps, dtype=torch.float)  # [num_edges]
            
            self.walk_sampler.update_neighbors(edge_index, edge_time)
            self.walk_sampler.build_dense_neighbor_table()
            logger.info(f"Walk sampler reinitialized with {edge_index.shape[1]} edges")
        elif hasattr(nf, 'edge_index') and hasattr(nf, 'edge_time'):
            # Direct attributes
            self.walk_sampler.update_neighbors(nf.edge_index, nf.edge_time)
            self.walk_sampler.build_dense_neighbor_table()
        else:
            logger.error("Cannot find edge data in neighbor_finder for walk sampler")
    
    def on_fit_start(self):
        """Initialize SAM memory at start of training."""
        super().on_fit_start()
        if self.use_memory:
            self.sam_module.reset_memory()
            logger.info(" SAM memory initialized for training")
    
       
    def on_train_batch_start(self, batch, batch_idx):
        times = batch['timestamps']
        logger.info(f"Batch {batch_idx}: min={times.min():.0f}, max={times.max():.0f}")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        ST-ODE-based memory update: continuous evolution + observation update.
        """
        # 1. SAM discrete update
        if self.training and self.use_memory and hasattr(self, '_sam_batch_buffer'):
            buffer = self._sam_batch_buffer
            
            mem_before_norm = torch.norm(self.sam_module.raw_memory).item()
            mem_before_std = self.sam_module.raw_memory.std().item()
            
            self.sam_module.update_memory_batch(
                source_nodes=buffer['src_nodes'],
                target_nodes=buffer['dst_nodes'],
                edge_features=buffer['edge_features'],
                current_time=buffer['timestamps'],
                node_features=self.node_raw_features
            )
            
            mem_after_norm = torch.norm(self.sam_module.raw_memory).item()
            mem_after_std = self.sam_module.raw_memory.std().item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: SAM | "
                        f"norm: {mem_before_norm:.2f} -> {mem_after_norm:.2f}, "
                        f"std: {mem_before_std:.4f} -> {mem_after_std:.4f}")
            
            delattr(self, '_sam_batch_buffer')

        # 2. ST-ODE continuous evolution
        if self.use_st_ode:
            obs_data = self._prepare_stode_observations(batch)
            current_time = batch['timestamps'].max()
            
            # Safety check: skip if times are before last_update_time
            if obs_data['times'].min() <= self.last_update_time:
                logger.debug(f"Batch {batch_idx}: Skipping ST-ODE (times before last_update_time)")
                self.last_update_time = current_time  # Still advance time
            else:
                # Run ST-ODE
                evolved_memory = self.st_ode(
                    node_states=self.sam_module.raw_memory,
                    walk_encodings=obs_data['encodings'],
                    walk_times=obs_data['times'],
                    walk_masks=obs_data['masks'],
                    adj_matrices=obs_data['adjs'],
                    t_init=self.last_update_time,
                    return_all=False
                )
                
                # Extract final state
                if hasattr(evolved_memory, 'final_state'):
                    evolved_state = evolved_memory.final_state
                else:
                    evolved_state = evolved_memory
                
                # Update memory
                self.sam_module.raw_memory.data.copy_(evolved_state)
                self.last_update_time = current_time
    
    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     """ST-ODE-based memory update: continuous evolution + observation update."""
    
    #     # if self.use_st_ode and self.training and self.use_memory:
    #     if self.use_st_ode and self.training and self.use_memory:
    #         current_time = batch['timestamps'].max()
    #         # First batch: only record the time, do not run ODE
    #         if self.last_update_time == 0:
    #             self.last_update_time = current_time
    #             return
    #         # Prepare observations from batch interactions
    #         # Prepare observations and run ODE for subsequent batches
    #         obs_data = self._prepare_stode_observations(batch)
            
    #         # Evolve memory from last_update_time to current batch time
    #         # current_time = batch['timestamps'].max()
            
    #         # Ensure times are properly shaped
    #         # if current_time.dim() == 0:
    #         #     current_time = current_time.unsqueeze(0)
            
    #         # SAFETY: Skip if observation times are before last_update_time
    #         if obs_data['times'].min() <= self.last_update_time:
    #             logger.debug(f"Batch {batch_idx}: Skipping ST-ODE (times before last_update_time)")
    #             # Still update last_update_time to current batch max
    #             self.last_update_time = current_time
    #         else:
    #             evolved_memory = self.st_ode(
    #                 node_states=self.sam_module.raw_memory,
    #                 walk_encodings=obs_data['encodings'],
    #                 walk_times=obs_data['times'],
    #                 walk_masks=obs_data['masks'],
    #                 adj_matrices=obs_data['adjs'],
    #                 t_init=self.last_update_time,
    #                 return_all=False
    #             )
        
    #     # Update memory
    #     if hasattr(evolved_memory, 'final_state'):
    #         evolved_state = evolved_memory.final_state
    #     else:
    #         evolved_state = evolved_memory
            
    #     self.sam_module.raw_memory.data.copy_(evolved_state)
    #     # self.last_update_time = current_time.item() if current_time.numel() == 1 else current_time[-1].item()
    #     self.last_update_time = current_time
        
        # """
        # ST-ODE-based memory update: continuous evolution + observation update.
        # """
        
        # if self.training and self.use_memory and hasattr(self, '_sam_batch_buffer'):
        #     buffer = self._sam_batch_buffer
            
        #     # Log L2 norm (more meaningful for symmetric distributions)
        #     # mem_before_norm = torch.norm(self.sam_module.raw_memory).item()
        #     # mem_before_std = self.sam_module.raw_memory.std().item()
            
        #     # FIX: Replace ... with actual arguments from buffer
        #     # self.sam_module.update_memory_batch(
        #     #     source_nodes=buffer['src_nodes'],
        #     #     target_nodes=buffer['dst_nodes'],
        #     #     edge_features=buffer['edge_features'],
        #     #     current_time=buffer['timestamps'],
        #     #     node_features=self.node_raw_features
        #     # )
            
        #     # mem_after_norm = torch.norm(self.sam_module.raw_memory).item()
        #     # mem_after_std = self.sam_module.raw_memory.std().item()
            
        #     # if batch_idx % 100 == 0:
        #     #     logger.info(f"Batch {batch_idx}: SAM | "
        #     #             f"norm: {mem_before_norm:.2f} -> {mem_after_norm:.2f}, "
        #     #             f"std: {mem_before_std:.4f} -> {mem_after_std:.4f}")
            
            
        #     # CRITICAL: Delete buffer to prevent memory leak
        #     # Clear buffer
        #     # delattr(self, '_sam_batch_buffer')

        # if self.use_st_ode:
        #     # Prepare observations from batch interactions
        #     # Each interaction is an "observation" that perturbs the smooth evolution
            
        #     obs_data = self._prepare_stode_observations(batch)
            
        #     # Evolve memory from last_update_time to current batch time
        #     # This is the core ST-ODE integration
        #     current_time = batch['timestamps'].max()
            
        #     evolved_memory = self.st_ode(
        #         node_states=self.sam_module.raw_memory,  # [num_nodes, memory_dim]
        #         walk_encodings=obs_data['encodings'],    # Observations as walk-like data
        #         walk_times=obs_data['times'],
        #         walk_masks=obs_data['masks'],
        #         adj_matrices=obs_data['adjs'],           # Graph snapshots over time
        #         t_init=self.last_update_time,
        #         return_all=False
        #     )
        #     if hasattr(evolved_memory, 'final_state'):
        #         evolved_state = evolved_memory.final_state
        #     else:
        #         evolved_state = evolved_memory  # Assume direct tensor return
            
        #     # Update memory with evolved state
        #     self.sam_module.raw_memory.data.copy_(evolved_state)
        #     self.last_update_time = current_time
    def on_train_epoch_start(self):
        """Reset ST-ODE temporal state at epoch start."""
        super().on_train_epoch_start()
        if self.use_st_ode:
            self.last_update_time = torch.tensor(0.0, device=self.device)
            logger.info(" ST-ODE last_update_time reset for new epoch")
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self._walk_cache.clear()
        self._last_hct_info = None
        torch.cuda.empty_cache()
    
    def on_validation_epoch_start(self):
        """Clone SAM memory for validation (preserve training state)."""
        super().on_validation_epoch_start()
        if self.use_memory:
            # Clone SAM memory state
            self._sam_validation_memory = self.sam_module.raw_memory.clone().detach()
            self._sam_validation_last_update = self.sam_module.last_update.clone().detach()
            logger.info(" Cloned SAM memory for validation")
    
    def on_validation_epoch_end(self):
        """Restore SAM memory after validation."""
        super().on_validation_epoch_end()
        if self.use_memory:
            self.sam_module.raw_memory.data.copy_(self._sam_validation_memory)
            self.sam_module.last_update.data.copy_(self._sam_validation_last_update)
            logger.info(" Restored SAM memory after validation")
        
        # Safe logging of HCT info
        if hasattr(self, '_last_hct_info') and self._last_hct_info is not None:
            try:
                for walk_type in ['short', 'long', 'tawr']:
                    if walk_type in self._last_hct_info:
                        cooc = self._last_hct_info[walk_type]['cooccurrence']
                        logger.info(f"HCT {walk_type}: cooc_mean={cooc.mean():.3f}, max={cooc.max():.3f}")
            except (KeyError, TypeError, AttributeError) as e:
                logger.debug(f"Could not log HCT co-occurrence: {e}")
            
            # Clear after logging to prevent stale data
            self._last_hct_info = None

    def on_test_epoch_start(self):
        """Reset SAM memory for cold-start test evaluation."""
        super().on_test_epoch_start()
        if self.use_memory:
            self.sam_module.reset_memory()
            logger.info("SAM memory reset for TEST (cold-start evaluation)")


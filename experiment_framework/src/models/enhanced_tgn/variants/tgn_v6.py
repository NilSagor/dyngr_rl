import os
import torch 
import torch.nn as nn
import numpy as np

from typing import Dict, Optional, Tuple
from loguru import logger

import torch.nn.functional as F


from ..base_enhance_tgn import BaseEnhancedTGN
from ..component.time_encoder import TimeEncoder
from ..component.sam_module import StabilityAugmentedMemory
from ..component.multi_swalk import MultiScaleWalkSampler
from ..component.walk_encoder import WalkEncoder
from ..component.co_transformer import HierarchicalCooccurrenceTransformer
from ..component.stode_module import SpectralTemporalODE
from ..component.transformer_encoder import MergeLayer



class TGNv6(BaseEnhancedTGN):
    """
    TGN + SAM + Multi-Scale Walk Sampler + Hierarchical Co-occurrence Transformer (HCT) + ST ODE
    
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
        use_sam: bool = True,
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
        self.edge_feat_scale = nn.Parameter(torch.ones(1))

        if self.use_st_ode:
            # Initialize time projection upfront (not lazily)
            if time_encoding_dim != memory_dim:
                self._time_proj = nn.Linear(time_encoding_dim, memory_dim)
                nn.init.xavier_uniform_(self._time_proj.weight)
                nn.init.zeros_(self._time_proj.bias)
            else:
                self._time_proj = nn.Identity()
        else:
            self._time_proj = None

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

        
        logger.debug(f"TGNv6 __init__: Walk Sampler Initialization = {self.walk_sampler}")
        logger.debug(f"TGNv6 __init__: use_hct = {use_hct}")
        
        
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
        
        # else:
        #     # Fallback to simple walk encoder (TGNv4 style)            
        #     self.walk_encoder = WalkEncoder(
        #         walk_length_short=walk_length_short,
        #         walk_length_long=walk_length_long,
        #         walk_length_tawr=walk_length_tawr,
        #         memory_dim=memory_dim,
        #         output_dim=hidden_dim,
        #         num_heads=n_heads,
        #         dropout=dropout
        #     )

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
        
        if edge_features_dim == 0:
            logger.warning("Edge features disabled (dim=0). SAM will receive zero edge input.")
            logger.warning("Consider setting edge_features_dim > 0 or using dummy features.")
        else:
            logger.info(f"Edge features enabled: input_dim={edge_features_dim}, projected_dim={memory_dim}")

        
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

        # Use pre-initialized _time_proj (not lazy)
        if self._time_proj is not None and not isinstance(self._time_proj, nn.Identity):
            time_emb = self._time_proj(time_emb.to(self._time_proj.weight.device))
        
        # if not isinstance(self._time_proj, nn.Identity):
        #     time_emb = self._time_proj(time_emb)

        # if time_emb.shape[-1] != self.memory_dim:
        #     if self._time_proj is None:
        #         self._time_proj = nn.Linear(self.time_encoding_dim, self.memory_dim).to(device)
        #     time_emb = self._time_proj(time_emb)
        
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
            if mask.sum() == 0:
                continue
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
        
   
    def _build_temporal_adjacency(self, src_nodes, dst_nodes, num_nodes):
        """Build SPARSE adjacency matrix for specific time snapshot."""
        device = src_nodes.device
        
        edges = torch.stack([src_nodes, dst_nodes], dim=0)  # [2, E]
        
        if not self.directed:
            edges = torch.cat([edges, edges.flip(0)], dim=1) # undirected
        
        
        # Add self-loops
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        edges = torch.cat([edges, self_loops], dim=1)

        # values = torch.ones(edges.shape[1], device=device)
        # Create sparse tensor
        # adj = torch.sparse_coo_tensor(
        #     edges, 
        #     values,
        #     size=(num_nodes, num_nodes),
        #     device=device
        # )
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edges[0], edges[1]] = 1.0
        
        return adj  # ensure coalesced for efficiency
    
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

        # Debug NaN
        if torch.isnan(scores).any():
            logger.error(
                f"NaN in scores! source_emb: {torch.isnan(source_emb).any()}, dest_emb: {torch.isnan(dest_emb).any()}")
            
        return scores 
        
    
    
    def _simple_walk_embed(self, walk_data: Dict) -> torch.Tensor:
        """
        Fallback: simple mean of walk node features across all walk types.
        walk_data: dict with keys 'short', 'long', 'tawr', each containing
                'nodes' and 'masks'.
        Returns: [batch_size, hidden_dim] tensor.
        """
        

        all_nodes = []
        all_masks = []
        max_len = 0

        # First, find the maximum walk length among all types
        for wt in ['short', 'long', 'tawr']:
            if wt in walk_data:
                nodes = walk_data[wt]['nodes']  # [B, num_walks, L]
                max_len = max(max_len, nodes.size(2))

        # Pad each type to max_len and collect
        for wt in ['short', 'long', 'tawr']:
            if wt in walk_data:
                nodes = walk_data[wt]['nodes']
                masks = walk_data[wt]['masks']
                L = nodes.size(2)
                if L < max_len:
                    pad_size = max_len - L
                    nodes = F.pad(nodes, (0, pad_size), mode='constant', value=0)
                    masks = F.pad(masks, (0, pad_size), mode='constant', value=0)
                all_nodes.append(nodes)
                all_masks.append(masks)

        if not all_nodes:
            # No walks – return zeros (should not happen)
            return torch.zeros(self.hidden_dim, device=self.device)

        # Concatenate along walk dimension
        nodes = torch.cat(all_nodes, dim=1)  # [B, total_walks, max_len]
        masks = torch.cat(all_masks, dim=1)  # [B, total_walks, max_len]

        # Flatten for memory lookup
        flat_nodes = nodes.reshape(-1)
        flat_feats = self.sam_module.raw_memory[flat_nodes]
        flat_feats = flat_feats.view(nodes.size(0), nodes.size(1), nodes.size(2), -1)

        # Weighted mean over valid steps
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
        
        # Check and sanitize inputs
        if torch.isnan(edge_times).any():
            logger.error("NaN in edge_times!")
            edge_times = torch.nan_to_num(edge_times, nan=0.0)
        
        # Convert inputs
        src_tensor = source_nodes.to(device) if torch.is_tensor(source_nodes) \
            else torch.from_numpy(source_nodes).long().to(device)
        dst_tensor = destination_nodes.to(device) if torch.is_tensor(destination_nodes) \
            else torch.from_numpy(destination_nodes).long().to(device)
        ts_tensor = edge_times.to(device) if torch.is_tensor(edge_times) \
            else torch.from_numpy(edge_times).float().to(device)
        
        # 1. Generate walks
        walk_data = self.walk_sampler(
            source_nodes=src_tensor,
            target_nodes=dst_tensor,
            current_times=ts_tensor,
            memory_states=self.sam_module.raw_memory,
            edge_index=self.edge_index,
            edge_time=self.edge_time
        )

        # Validate walk data for NaN/Inf
        for side in ['source', 'target']:
            for walk_type in ['short', 'long', 'tawr']:
                if walk_type in walk_data[side]:
                    nodes = walk_data[side][walk_type]['nodes']
                    if not torch.isfinite(nodes).all():
                        logger.error(f"{side}/{walk_type} nodes contain NaN/Inf! Replacing with source/target nodes")
                        # Fallback to source/target nodes
                        if side == 'source':
                            dummy = src_tensor.unsqueeze(1).unsqueeze(2).expand(-1, nodes.shape[1], nodes.shape[2])
                        else:
                            dummy = dst_tensor.unsqueeze(1).unsqueeze(2).expand(-1, nodes.shape[1], nodes.shape[2])
                        walk_data[side][walk_type]['nodes'] = torch.clamp(dummy, 0, self.num_nodes - 1)
                        if 'masks' in walk_data[side][walk_type]:
                            walk_data[side][walk_type]['masks'] = torch.ones_like(walk_data[side][walk_type]['masks'])
        
        # Before HCT, ensure memory is clean
        if not torch.isfinite(self.sam_module.raw_memory).all():
            logger.warning("SAM memory has NaN, resetting affected nodes")
            self.sam_module.raw_memory.data = torch.nan_to_num(
                self.sam_module.raw_memory.data,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0
            )
        
        # 2. VALIDATE WALK DATA BEFORE HCT
        if self.use_hct:
            for side in ['source', 'target']:
                for walk_type in ['short', 'long', 'tawr']:
                    if walk_type in walk_data[side]:
                        nodes = walk_data[side][walk_type]['nodes']
                        masks = walk_data[side][walk_type].get('masks', None)
                        
                        # CRITICAL: Check for all-zero indices (walk sampler failure)
                        if nodes.max().item() == 0 and nodes.numel() > 0:
                            logger.error(f"Walk sampler returned all-zero indices for {side}/{walk_type}!")
                            logger.error(f"Check walk_sampler neighbor table initialization")
                            # Fallback: use direct node memory instead of walks
                            batch_size = nodes.shape[0]
                            num_walks = nodes.shape[1]
                            walk_len = nodes.shape[2]
                            if side == 'source':
                                dummy_nodes = src_tensor.unsqueeze(1).unsqueeze(2).expand(-1, num_walks, walk_len)
                            else:
                                dummy_nodes = dst_tensor.unsqueeze(1).unsqueeze(2).expand(-1, num_walks, walk_len)
                            walk_data[side][walk_type]['nodes'] = dummy_nodes
                            if masks is not None:
                                walk_data[side][walk_type]['masks'] = torch.ones_like(masks)
                        
                        # Validate node indices are within bounds
                        if nodes.max().item() >= self.num_nodes:
                            logger.warning(f"Clamping {side} {walk_type} walk nodes")
                            walk_data[side][walk_type]['nodes'] = torch.clamp(
                                nodes, 0, self.num_nodes - 1
                            )
                        
                        # Validate masks
                        if masks is not None and masks.sum() == 0:
                            logger.warning(f"All {side} {walk_type} masks are False!")
                        
                        # Validate NaN/Inf in nodes
                        if not torch.isfinite(nodes).all():
                            logger.error(f"{side} {walk_type} nodes contain NaN/Inf!")
                            walk_data[side][walk_type]['nodes'] = torch.nan_to_num(
                                nodes, nan=0, posinf=self.num_nodes-1, neginf=0
                            ).long()
        
        if not torch.isfinite(self.sam_module.raw_memory).all():
            logger.error("SAM memory NaN before HCT - RESETTING")
            self.sam_module.reset_memory()
        
        
        # 3. HCT encodes walks
        if self.use_hct:
            # TRACE: Check memory before HCT
            if not torch.isfinite(self.sam_module.raw_memory).all():
                logger.error(f"SAM memory NaN before HCT! Resetting...")
                self.sam_module.reset_memory()
            
            # TRACE: Check walk indices
            if 'source' in walk_data and 'nodes' in walk_data['source']:
                src_walk_nodes = walk_data['source']['nodes']
                if src_walk_nodes.max().item() >= self.num_nodes:
                    logger.error(f"Invalid walk node index! max={src_walk_nodes.max().item()}, num_nodes={self.num_nodes}")
                    walk_data['source']['nodes'] = torch.clamp(src_walk_nodes, 0, self.num_nodes - 1)
            
            if 'target' in walk_data and 'nodes' in walk_data['target']:
                dst_walk_nodes = walk_data['target']['nodes']
                if dst_walk_nodes.max().item() >= self.num_nodes:
                    logger.error(f"Invalid walk node index! max={dst_walk_nodes.max().item()}, num_nodes={self.num_nodes}")
                    walk_data['target']['nodes'] = torch.clamp(dst_walk_nodes, 0, self.num_nodes - 1)
            
            hct_src_output = self.hct(
                walks_dict=walk_data['source'],
                node_memory=self.sam_module.raw_memory,
                return_all=True
            )
            hct_dst_output = self.hct(
                walks_dict=walk_data['target'],
                node_memory=self.sam_module.raw_memory,
                return_all=True
            )
            
            hct_src_emb = hct_src_output['fused']
            hct_dst_emb = hct_dst_output['fused']
            
            # TRACE: Check HCT outputs (with type checking)
            for key in ['short', 'long', 'tawr', 'fused']:
                if key in hct_src_output:
                    val = hct_src_output[key]
                    if isinstance(val, torch.Tensor):
                        if not torch.isfinite(val).all():
                            logger.error(f"HCT src {key} NaN! shape={val.shape}, "
                                    f"min={val.min().item():.4f}, max={val.max().item():.4f}")
                    elif isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, torch.Tensor):
                                if not torch.isfinite(sub_val).all():
                                    logger.error(f"HCT src {key}.{sub_key} NaN! shape={sub_val.shape}")
            
            # SANITIZE HCT outputs BEFORE projection
            if not torch.isfinite(hct_src_emb).all():
                logger.warning(f"HCT src NaN detected! Clamping...")
                hct_src_emb = torch.nan_to_num(hct_src_emb, nan=0.0, posinf=10.0, neginf=-10.0)
            if not torch.isfinite(hct_dst_emb).all():
                logger.warning(f"HCT dst NaN detected! Clamping...")
                hct_dst_emb = torch.nan_to_num(hct_dst_emb, nan=0.0, posinf=10.0, neginf=-10.0)
            
            walk_src_emb = self.hct_to_tgn(hct_src_emb)
            walk_dst_emb = self.hct_to_tgn(hct_dst_emb)
        else:
            walk_src_emb = self._simple_walk_embed(walk_data['source'])
            walk_dst_emb = self._simple_walk_embed(walk_data['target'])
        
        # AFTER HCT call
        if 'fused' in hct_src_output:
            if not torch.isfinite(hct_src_output['fused']).all():
                logger.error("HCT output NaN - DISABLING HCT for this batch")
                hct_src_emb = torch.zeros_like(hct_src_output['fused'])
        
        # SANITIZE after projection
        if not torch.isfinite(walk_src_emb).all():
            logger.warning(f"walk_src_emb NaN after projection! Clamping...")
            walk_src_emb = torch.nan_to_num(walk_src_emb, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(walk_dst_emb).all():
            logger.warning(f"walk_dst_emb NaN after projection! Clamping...")
            walk_dst_emb = torch.nan_to_num(walk_dst_emb, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 4. Base TGN embeddings
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
        
        # SANITIZE base embeddings
        if not torch.isfinite(base_src_emb).all():
            logger.warning(f"base_src_emb NaN! Clamping...")
            base_src_emb = torch.nan_to_num(base_src_emb, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(base_dst_emb).all():
            logger.warning(f"base_dst_emb NaN! Clamping...")
            base_dst_emb = torch.nan_to_num(base_dst_emb, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 5. Stabilized memory
        stabilized_src = self.sam_module.get_stabilized_memory(
            node_ids=source_nodes,
            current_time=edge_times,
        )
        stabilized_dst = self.sam_module.get_stabilized_memory(
            node_ids=destination_nodes,
            current_time=edge_times,
        )
        
        # SANITIZE stabilized memory
        if not torch.isfinite(stabilized_src).all():
            logger.warning(f"stabilized_src NaN! Clamping...")
            stabilized_src = torch.zeros_like(stabilized_src)
        if not torch.isfinite(stabilized_dst).all():
            logger.warning(f"stabilized_dst NaN! Clamping...")
            stabilized_dst = torch.zeros_like(stabilized_dst)
        
        base_src_emb = base_src_emb + 0.01 * stabilized_src.detach()
        base_dst_emb = base_dst_emb + 0.01 * stabilized_dst.detach()
        
        # 6. Fusion
        combined_src = torch.cat([base_src_emb, walk_src_emb], dim=-1)
        combined_dst = torch.cat([base_dst_emb, walk_dst_emb], dim=-1)
        
        final_src = self.fusion_layer(combined_src)
        final_dst = self.fusion_layer(combined_dst)
        
        # FINAL SANITIZE
        if not torch.isfinite(final_src).all():
            logger.error(f"final_src NaN after fusion! Clamping...")
            final_src = torch.nan_to_num(final_src, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(final_dst).all():
            logger.error(f"final_dst NaN after fusion! Clamping...")
            final_dst = torch.nan_to_num(final_dst, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return final_src, final_dst   
    
      
   

    def _store_sam_interactions(self, batch: Dict[str, torch.Tensor]):
        """Store interactions in buffer for deferred SAM update."""
        device = self.device
        
        src_nodes = batch['src_nodes'].to(device)
        dst_nodes = batch['dst_nodes'].to(device)
        timestamps = batch['timestamps'].to(device)
        
        # edge_feats = edge_feats / 100.0

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
    
    def training_step(self, batch, batch_idx):
        src = batch['src_nodes']
        dst = batch['dst_nodes'] 
        ts = batch['timestamps']
        labels = batch['labels']
        edge_feats = batch.get('edge_features')

        # 1. FORWARD PASS: Compute embeddings using CURRENT memory state
        # This establishes the computational graph for link prediction loss
        
        # 1. FORWARD PASS: Compute embeddings using CURRENT memory state
        source_emb, dest_emb = self.compute_temporal_embeddings(src, dst, ts)
        
        # 2. LINK PREDICTION
        scores = self.link_predictor(source_emb, dest_emb).squeeze(-1)
    
        
        # Debug NaN in scores
        if torch.isnan(scores).any():
            logger.error(f"NaN in scores! source_emb: {torch.isnan(source_emb).any()}, "
                        f"dest_emb: {torch.isnan(dest_emb).any()}")
        
        if torch.isnan(scores).any():
            logger.warning(f"Batch {batch_idx} contains NaN scores, skipping")
            return None   # Lightning will skip this batch
        
        
        # 3. COMPUTE LOSS
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        
       


        # 4. DEFERRED SAM UPDATE: Store interactions for update AFTER backward
        # We don't update here to avoid creating gradients that pollute the optimizer
        if self.training and self.use_memory:
            self._store_sam_interactions(batch)
        
        # 5. LOGGING
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        
        if torch.isnan(loss):
            logger.error(f"NaN loss! scores range: [{scores.min():.4f}, {scores.max():.4f}]")            
        
            logger.error(f"NaN loss! scores: {scores.min():.4f}/{scores.max():.4f}, labels: {batch['labels'].unique()}")
        
            
        return {'loss': loss}
        
           
    
    def on_fit_start(self):
        """Initialize SAM memory at start of training."""
        super().on_fit_start()
        if self.use_memory:
            self.sam_module.reset_memory()
            logger.info(" SAM memory initialized for training")
    
       
    # def on_train_batch_start(self, batch, batch_idx):
    #     times = batch['timestamps']
    #     logger.info(f"Batch {batch_idx}: min={times.min():.0f}, max={times.max():.0f}")
    
    def on_train_batch_start(self, batch, batch_idx):
        times = batch['timestamps']
        current_max = times.max().item()
        
        # if batch_idx == 0 and self.use_hct:
        if batch_idx == 0:
            # Generate test walks
            walk_data = self.walk_sampler(
                source_nodes=batch['src_nodes'][:2],
                target_nodes=batch['dst_nodes'][:2],
                current_times=batch['timestamps'][:2],
                memory_states=self.sam_module.raw_memory,
                edge_index=self.edge_index,
                edge_time=self.edge_time
            )
            
            # Inspect structure
            logger.info(f"Walk data keys: {walk_data.keys()}")
            if 'source' in walk_data:
                logger.info(f"Source keys: {walk_data['source'].keys()}")
                for walk_type in ['short', 'long', 'tawr']:
                    if walk_type in walk_data['source']:
                        nodes = walk_data['source'][walk_type]['nodes']
                        logger.info(f"{walk_type}: shape={nodes.shape}, "
                                f"min={nodes.min().item()}, max={nodes.max().item()}")
        
        
        
        if batch_idx == 0:
            edge_feats = batch.get('edge_features')
            if edge_feats is not None:
                logger.info(f"Batch 0 edge_features: shape={edge_feats.shape}, "
                        f"finite={torch.isfinite(edge_feats).all().item()}, "
                        f"range=[{edge_feats.min().item():.4f}, {edge_feats.max().item():.4f}]")
            else:
                logger.info("Batch 0: No edge_features in batch")
        
        if batch_idx == 0:
            logger.info(f"Epoch start - Batch 0 time range: [{times.min():.0f}, {times.max():.0f}]")
        
        if hasattr(self, '_prev_max_time'):
            if current_max < self._prev_max_time:
                logger.error(f"TEMPORAL VIOLATION: Batch {batch_idx} max {current_max:.0f} < prev {self._prev_max_time:.0f}")
                # Print first few timestamps for debugging
                logger.error(f"First 10 timestamps: {times[:10].tolist()}")
        
        self._prev_max_time = current_max
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        CRITICAL: SAM update happens HERE after backward pass, with NO_GRAD protection.
        This prevents gradient accumulation and NaN explosions.
        """
        if batch_idx == 0:
            logger.info(
                f"Batch edge_features: shape={batch['edge_features'].shape}, "
                f"min={batch['edge_features'].min().item():.4f}, "
                f"max={batch['edge_features'].max().item():.4f}, "
                f"norm={batch['edge_features'].norm().item():.4f}"
            )
        
        # 1. SAM DISCRETE UPDATE (with gradient protection)
        if self.training and self.use_memory and hasattr(self, '_sam_batch_buffer'):
            buffer = self._sam_batch_buffer
            
            # Check buffer validity before any update
            if not torch.isfinite(buffer['edge_features']).all():
                logger.error(f"Batch {batch_idx}: Edge features already NaN in buffer, skipping SAM update")
                delattr(self, '_sam_batch_buffer')
                return
            
            
            # Wrap entire update in no_grad to prevent gradient pollution
            with torch.no_grad():
                
                # Get node features (detach to be safe)
                node_feats = self.node_embedding.weight.detach()

                # SAM update - no gradients should flow from this
                self.sam_module.update_memory_batch(
                    source_nodes=buffer['src_nodes'],
                    target_nodes=buffer['dst_nodes'],
                    edge_features=buffer['edge_features'],  # Already detached in buffer
                    current_time=buffer['timestamps'],
                    node_features=node_feats
                )

                # clamp(NaN, -10, 10) returns NaN, so nan_to_num must come first
                self.sam_module.raw_memory.data = torch.nan_to_num(
                    self.sam_module.raw_memory.data,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0
                ).clamp_(-10, 10)

                # Also sanitize prototypes
                if not torch.isfinite(self.sam_module.all_prototypes).all():
                    logger.error(f"Batch {batch_idx}: Prototypes NaN after update, resetting")
                    self.sam_module.all_prototypes.data.normal_(0, 0.01)
                else:
                    self.sam_module.all_prototypes.data = torch.nan_to_num(
                        self.sam_module.all_prototypes.data,
                        nan=0.0,
                        posinf=10.0,
                        neginf=-10.0
                    ).clamp_(-10, 10)

                if not torch.isfinite(self.sam_module.raw_memory).all():
                    logger.error(f"Batch {batch_idx}: SAM memory NaN AFTER update! RESETTING")
                    self.sam_module.reset_memory()
                
                if not torch.isfinite(self.sam_module.all_prototypes).all():
                    logger.error(f"Batch {batch_idx}: Prototypes NaN! RESETTING")
                    self.sam_module.all_prototypes.data.normal_(0, 0.01)
                
                # EMERGENCY: Hard clamp memory to prevent explosion                
                self.sam_module.raw_memory.data.clamp_(-10, 10)
                self.sam_module.all_prototypes.data.clamp_(-10, 10)
                
                # Verify update didn't explode
                       
                
                # Safety check: if memory exploded, reset and log
                # After update, verify memory health
                if not torch.isfinite(self.sam_module.all_prototypes).all():
                    logger.error("Prototypes NaN after optimizer step – resetting")
                    self.sam_module.all_prototypes.data.normal_(0, 0.01)
                else:
                    self.sam_module.all_prototypes.data.clamp_(-10.0, 10.0)
            
            # Clean up buffer
            delattr(self, '_sam_batch_buffer')

        
        
        
        # 2. ST-ODE CONTINUOUS EVOLUTION (only if time has advanced)
        if self.use_st_ode:
            current_time = batch['timestamps'].max()
            time_delta = current_time - self.last_update_time
            
            # Skip if no time has passed
            if time_delta < 1e-6:
                self.last_update_time = current_time
                return
            
            # Check memory health before ST-ODE
            if not torch.isfinite(self.sam_module.raw_memory).all():
                logger.error(f"Batch {batch_idx}: Memory NaN before ST-ODE, skipping")
                self.sam_module.reset_memory()
                self.last_update_time = current_time
                return
            
            # Prepare and filter observations
            try:
                obs_data = self._prepare_stode_observations(batch)
                
                # Filter to valid times only
                times_flat = obs_data['times'].reshape(-1)
                valid_mask = times_flat > (self.last_update_time + 1e-6)
                
                if not valid_mask.any():
                    self.last_update_time = current_time
                    return
                
                # Get valid time indices
                time_dim = obs_data['times'].shape[1]
                original_times = times_flat[:time_dim]
                valid_time_mask = original_times > (self.last_update_time + 1e-6)
                valid_indices = torch.where(valid_time_mask)[0]
                
                if len(valid_indices) == 0:
                    self.last_update_time = current_time
                    return
                
                # Filter tensors
                filtered_encodings = obs_data['encodings'][:, valid_indices]
                filtered_times = obs_data['times'][:, valid_indices]
                filtered_masks = obs_data['masks'][:, valid_indices]
                filtered_adjs = [obs_data['adjs'][i] for i in valid_indices.tolist()]
                
                # Verify ST-ODE module exists
                if self.st_ode is None:
                    logger.error(f"Batch {batch_idx}: ST-ODE module is None!")
                    self.last_update_time = current_time
                    return
                
                # Run ST-ODE (wrapped in no_grad since this is state evolution)
                with torch.no_grad():
                    evolved_memory = self.st_ode(
                        node_states=self.sam_module.raw_memory,
                        walk_encodings=filtered_encodings,
                        walk_times=filtered_times,
                        walk_masks=filtered_masks,
                        adj_matrices=filtered_adjs,
                        t_init=self.last_update_time,
                        return_all=False
                    )
                    
                    # Extract final state
                    evolved_state = evolved_memory.final_state if hasattr(evolved_memory, 'final_state') else evolved_memory
                    
                    # Validate output
                    if not torch.isfinite(evolved_state).all():
                        logger.error(f"Batch {batch_idx}: ST-ODE produced NaN, skipping update")
                        # Don't update memory, keep previous state
                    elif evolved_state.abs().max() < 1e-8:
                        logger.warning(f"Batch {batch_idx}: ST-ODE produced near-zero state")
                    else:
                        self.sam_module.raw_memory.data.copy_(evolved_state)
                        self.last_update_time = current_time
                    
                    
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx}: ST-ODE failed ({e})")
                self.last_update_time = current_time
        # Check if loss is NaN and skip update if so
        
    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.sam_module.parameters(), max_norm=1.0)    
        # Extra clipping for sensitive components
        if hasattr(self, 'hct'):
            torch.nn.utils.clip_grad_norm_(self.hct.parameters(), max_norm=0.5)
        
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


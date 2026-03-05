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
from ..component.hct_module import HierarchicalCooccurrenceTransformer
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
    5. Spectral-Temporal ODE for continuous-time node representation learning
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
        walk_length_short: int = 2,
        walk_length_long: int = 5,
        walk_length_tawr: int = 4,
        num_walks_short: int = 5,
        num_walks_long: int = 3,
        num_walks_tawr: int = 3,
        walk_temperature: float = 0.1,
        use_walk_encoder: bool = True,
        debug_simple_walk: bool = False,
        # HCT params
        use_hct: bool = True,
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
        aggregation: str = 'mean',
        time_precision: int = 6,
        use_checkpoint: bool = True,
        use_st_ode: bool = True,
        **kwargs
    ):
        # Verify base class compatibility before initialization
        if hasattr(BaseEnhancedTGN, '_has_memory_buffer'):
            logger.warning("Base class has memory buffer - may conflict with SAM")
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
            memory_updater_type="none",
            **kwargs
        )
        # torch.autograd.set_detect_anomaly(True)
        if os.environ.get('DEBUG_GRADIENTS'):
            torch.autograd.set_detect_anomaly(True)
        
        self.use_st_ode = use_st_ode
        self.use_hct = use_hct
        self.directed = kwargs.get('directed', False)
        
        self.use_sam = use_sam
        self.debug_simple_walk = debug_simple_walk

        # Initialize time projection only if ST-ODE is enabled
        if self.use_st_ode:
            if time_encoding_dim != memory_dim:
                self._time_proj = nn.Linear(time_encoding_dim, memory_dim)
                nn.init.xavier_uniform_(self._time_proj.weight)
                nn.init.zeros_(self._time_proj.bias)
            else:
                self._time_proj = nn.Identity()
        else:
            self._time_proj = None
        
        self.time_encoder = TimeEncoder(time_encoding_dim)
        
        self.sam_config = {
            'num_prototypes': num_prototypes,
            'similarity_metric': similarity_metric,
            'memory_dim': memory_dim
        }
                
        # SAM initialization (only if use_sam=True)
        if self.use_sam:
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
            logger.info(f"SAM initialized: {num_prototypes} prototypes, {similarity_metric} similarity")
        else:
            self.sam_module = None
            logger.info(f"SAM disabled – using base TGN memory")
                
        
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
                
        # HCT initialization
        if self.use_hct and not self.debug_simple_walk:
            self.hct = HierarchicalCooccurrenceTransformer(
                d_model=hct_d_model,
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
            if hct_d_model != hidden_dim:
                self.hct_to_tgn = nn.Sequential(
                    nn.Linear(hct_d_model, hidden_dim),
                    nn.LayerNorm(hidden_dim, eps=1e-4),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            else:
                self.hct_to_tgn = nn.Identity()
            logger.info("HCT enabled")
        else:
            self.hct = None
            self.hct_to_tgn = None
            if self.debug_simple_walk:
                logger.info("Using simple walk mean pooling (bypass HCT)")
      
        # Initialize ST-ODE only if enabled AND SAM is used (ST-ODE relies on SAM memory)
        if self.use_st_ode:
            if not self.use_sam:
                logger.warning("ST-ODE disabled because SAM is off (ST-ODE requires SAM memory).")
                self.st_ode = None
                self.walk_to_memory = None
            else:
                self.st_ode = SpectralTemporalODE(
                    hidden_dim=memory_dim,
                    num_nodes=num_nodes,
                    num_eigenvectors=num_eigenvectors,
                    mu=mu,
                    adaptive_mu=adaptive_mu,
                    use_gru_ode=use_gru_ode,
                    ode_method=ode_method,
                    ode_step_size=ode_step_size,
                    num_layers=num_layers,
                    adjoint=adjoint,
                    dropout=dropout,
                    aggregation=aggregation,
                    time_precision=time_precision,
                    use_checkpoint=use_checkpoint
                )
                if self.use_hct and hct_d_model != memory_dim:
                    self.walk_to_memory = nn.Linear(hct_d_model, memory_dim)
                else:
                    self.walk_to_memory = nn.Identity()
        else:
            self.st_ode = None
            self.walk_to_memory = None
        
        # Time of last memory update (only used if ST-ODE enabled)
        self.register_buffer('last_update_time', torch.tensor(0.0))
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_index = None
        self.edge_time = None
        self._walk_cache = {}
        self._cache_walks = False
        
        self._sam_batch_buffer = None
        
        # Simple walk projection (if needed)
        if self.hidden_dim != memory_dim:
            self.simple_walk_proj = nn.Linear(memory_dim, self.hidden_dim)
        else:
            self.simple_walk_proj = nn.Identity()

        self.link_predictor = MergeLayer(
            input_dim1=hidden_dim,
            input_dim2=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        )
        
        if edge_features_dim == 0:
            logger.warning("Edge features disabled (dim=0). SAM will receive zero edge input.")
        else:
            logger.info(f"Edge features enabled: input_dim={edge_features_dim}, projected_dim={memory_dim}")
        
        logger.info(f"TGNv6 initialized: SAM({num_prototypes}p) + WalkSampler + "
                   f"{'HCT(' + str(hct_d_model) + 'd, ' + str(hct_nhead) + 'h)' if use_hct else 'SimpleWalkEncoder'}")

    def _prepare_stode_observations(self, batch):
        """Convert batch interactions to ST-ODE observation format."""
        
        # Only run if ST-ODE is enabled AND SAM is used
        if not self.use_st_ode or not self.use_sam:
            return None
        
        device = self.device
        num_nodes = self.num_nodes
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']
        
        src_emb = self.sam_module.raw_memory[src_nodes]
        dst_emb = self.sam_module.raw_memory[dst_nodes]
        time_emb = self.time_encoder(timestamps.float())
        
        if self._time_proj is not None and not isinstance(self._time_proj, nn.Identity):
            time_emb = self._time_proj(time_emb)
        
        adj_t = self._build_temporal_adjacency(src_nodes, dst_nodes, num_nodes)
        unique_times, inverse = torch.unique(timestamps, sorted=True, return_inverse=True)
        T = len(unique_times)
        
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
            
            node_obs_per_time[t_idx].index_add_(0, src_t, src_emb_t)
            node_obs_per_time[t_idx].index_add_(0, dst_t, dst_emb_t)
            node_obs_per_time[t_idx].index_add_(0, src_t, time_emb_t)
            node_obs_per_time[t_idx].index_add_(0, dst_t, time_emb_t)
        
        masks = (node_obs_per_time.abs().sum(dim=-1) > 0)
        walk_encodings = node_obs_per_time.permute(1, 0, 2).unsqueeze(2)
        walk_times = unique_times.view(1, T, 1).expand(num_nodes, -1, -1)
        walk_masks = masks.T.unsqueeze(-1)
        
        return {
            'encodings': walk_encodings,
            'times': walk_times,
            'masks': walk_masks,
            'adjs': [adj_t] * T
        }       
        
   
    def _build_temporal_adjacency(self, src_nodes, dst_nodes, num_nodes):
        """Build sparse adjacency matrix for specific time snapshot."""
        device = src_nodes.device
        edges = torch.stack([src_nodes, dst_nodes], dim=0)
        
        if not self.directed:
            edges = torch.cat([edges, edges.flip(0)], dim=1)
        
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        edges = torch.cat([edges, self_loops], dim=1)
        
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edges[0], edges[1]] = 1.0
        
        return adj
    
    def _create_observation_encoding(self, src_nodes, dst_nodes, edge_feats, time):
        """Create walk-like encoding from interaction."""
        device = src_nodes.device
        
        if time.dim() == 0:
            time = time.unsqueeze(0)
        
        obs = torch.zeros(self.num_nodes, 1, 3, self.memory_dim, device=device)
        src_emb = self.sam_module.raw_memory[src_nodes]
        dst_emb = self.sam_module.raw_memory[dst_nodes]
        time_emb = self.time_encoder(time)
        
        # Only project if ST-ODE enabled and projection exists
        if self.use_st_ode and self._time_proj is not None and not isinstance(self._time_proj, nn.Identity):
            if time_emb.shape[-1] != self.memory_dim:
                time_emb = self._time_proj(time_emb)
        
        obs[src_nodes, 0, 0] = src_emb
        obs[dst_nodes, 0, 1] = dst_emb
        obs[src_nodes, 0, 2] = time_emb
        obs[dst_nodes, 0, 2] = time_emb
        
        return obs
    
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with proper SAM update sequencing."""
        source_emb, dest_emb = self.compute_temporal_embeddings(
            batch['src_nodes'],
            batch['dst_nodes'],
            batch['timestamps']
        )
        
        if self.training and self.use_memory:
            self._store_sam_interactions(batch)
        
        scores = self.link_predictor(source_emb, dest_emb).squeeze(-1)
        
        if torch.isnan(scores).any():
            logger.error(f"NaN in scores! source_emb: {torch.isnan(source_emb).any()}, "
                        f"dest_emb: {torch.isnan(dest_emb).any()}")
        
        return scores
        
    
    
    def _simple_walk_embed(self, walk_data_side: Dict, node_memory) -> torch.Tensor:
        """
        Fallback walk embedding: mean of all node features from all walks.
        Expects walk_data_side to contain 'short', 'long', 'tawr' each with 'nodes' and 'masks'.
        """
        all_embs = []
        
        for wt in ['short', 'long', 'tawr']:
            if wt in walk_data_side:
                nodes = walk_data_side[wt]['nodes']          # [B, num_walks, walk_len]
                masks = walk_data_side[wt]['masks']          # same shape
                flat_nodes = nodes.reshape(-1)
                flat_feats = node_memory[flat_nodes]         # [B*num_walks*walk_len, D]
                feats = flat_feats.view(*nodes.shape, -1)    # [B, num_walks, walk_len, D]
                masked_sum = (feats * masks.unsqueeze(-1)).sum(dim=[1,2])  # [B, D]
                count = masks.sum(dim=[1,2]).unsqueeze(-1)   # [B, 1]
                count = torch.where(count > 1e-6, count, torch.ones_like(count))
                all_embs.append(masked_sum / count)
        
        if not all_embs:
            return torch.zeros(node_memory.size(0), self.hidden_dim, device=node_memory.device)
        
        # Average across walk types
        pooled = torch.stack(all_embs, dim=0).mean(dim=0)   # [B, D_mem]

        # Project to hidden_dim if needed        
        pooled = self.simple_walk_proj(pooled)
        
        return pooled
    
    def compute_temporal_embeddings(
        self,
        source_nodes: torch.Tensor,
        destination_nodes: torch.Tensor,
        edge_times: torch.Tensor,
        n_neighbors: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified embedding computation with HCT."""
        device = self.device
        
        # ----- 1. Get current node memory (SAM or base TGN) -----
        if self.use_sam and self.sam_module is not None:
            node_memory = self.sam_module.raw_memory  # already a tensor
        else:
            # Base TGN memory might be a Memory object; extract the raw tensor.
            if hasattr(self.memory, 'memory'):
                node_memory = self.memory.memory
            else:
                node_memory = self.memory

        # Basic input validation
        assert torch.isfinite(node_memory).all(), "node_memory contains NaN/Inf"        
        assert (source_nodes >= 0).all() and (source_nodes < self.num_nodes).all(), "source_nodes out of bounds"
        assert (destination_nodes >= 0).all() and (destination_nodes < self.num_nodes).all(), "dst_nodes out of bounds"
        assert torch.isfinite(edge_times).all(), "edge_times contains NaN/Inf"

        # Reset SAM memory if needed (only when SAM is used)
        if self.use_sam and not torch.isfinite(self.sam_module.raw_memory).all():
            logger.error("SAM memory NaN before compute_temporal_embeddings! RESETTING")
            self.sam_module.reset_memory()
            node_memory = self.sam_module.raw_memory  # refresh

        # Fix NaN in edge_times
        if torch.isnan(edge_times).any():
            logger.error("NaN in edge_times! Replacing with 0.")
            edge_times = torch.nan_to_num(edge_times, nan=0.0)

        # Ensure tensors are on correct device and type
        src_tensor = source_nodes.to(device).long()
        dst_tensor = destination_nodes.to(device).long()
        ts_tensor = edge_times.to(device).float()

        # Memory snapshot for walk sampler (detached)
        with torch.no_grad():
            walk_memory = node_memory.detach().clone()
        
        # ----- 2. Generate walks -----
        walk_data = self.walk_sampler(
            source_nodes=src_tensor,
            target_nodes=dst_tensor,
            current_times=ts_tensor,
            memory_states=walk_memory,
            edge_index=self.edge_index,
            edge_time=self.edge_time
        )

        
        # ----- 3. Validate walk data (once, before any branching) -----
        for side in ['source', 'target']:
            for wt in ['short', 'long', 'tawr']:
                if wt in walk_data[side]:
                    nodes = walk_data[side][wt]['nodes']
                    if not torch.isfinite(nodes).all():
                        logger.error(f"{side}/{wt} nodes contain NaN/Inf! Replacing with zeros.")
                        walk_data[side][wt]['nodes'] = torch.zeros_like(nodes)
                    # Clamp to valid node indices
                    nodes.clamp_(0, self.num_nodes - 1)
        
        
        # ----- 4. Obtain walk embeddings (HCT or simple mean) -----
        if self.use_hct and self.hct is not None and not self.debug_simple_walk:
            # Use HCT
            with torch.enable_grad():
                hct_src_output = self.hct(
                    walks_dict=walk_data['source'],
                    node_memory=node_memory,
                    return_all=True
                )
                hct_dst_output = self.hct(
                    walks_dict=walk_data['target'],
                    node_memory=node_memory,
                    return_all=True
                )

            # Take fused output
            hct_src_fused = hct_src_output['fused']
            hct_dst_fused = hct_dst_output['fused']

            # FIX: Actually clamp the tensors, not just a local variable
            if not torch.isfinite(hct_src_fused).all():
                logger.warning("HCT src output contains NaN/Inf! Clamping.")
                hct_src_fused = torch.nan_to_num(hct_src_fused, nan=0.0, posinf=10.0, neginf=-10.0)
            if not torch.isfinite(hct_dst_fused).all():
                logger.warning("HCT dst output contains NaN/Inf! Clamping.")
                hct_dst_fused = torch.nan_to_num(hct_dst_fused, nan=0.0, posinf=10.0, neginf=-10.0)

            # Project to hidden_dim (if needed)
            walk_src_emb = self.hct_to_tgn(hct_src_fused)
            walk_dst_emb = self.hct_to_tgn(hct_dst_fused)
        else:
            # Fallback: simple mean pooling (ensure output dimension = hidden_dim)
            walk_src_emb = self._simple_walk_embed(walk_data['source'], node_memory)
            walk_dst_emb = self._simple_walk_embed(walk_data['target'], node_memory)

        
        # ----- 5. Base TGN embeddings (using node_memory) -----
        if self.embedding_module is not None:
            base_src_emb = self.embedding_module.compute_embedding(
                memory=node_memory,
                source_nodes=source_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            base_dst_emb = self.embedding_module.compute_embedding(
                memory=node_memory,
                source_nodes=destination_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
        else:
            base_src_emb = node_memory[source_nodes]
            base_dst_emb = node_memory[destination_nodes]
        
        # ----- 6. Fusion -----
        combined_src = torch.cat([base_src_emb, walk_src_emb], dim=-1)
        combined_dst = torch.cat([base_dst_emb, walk_dst_emb], dim=-1)
        
        final_src = self.fusion_layer(combined_src)
        final_dst = self.fusion_layer(combined_dst)
        
        # Final safety clamp
        if not torch.isfinite(final_src).all():
            final_src = torch.nan_to_num(final_src, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(final_dst).all():
            final_dst = torch.nan_to_num(final_dst, nan=0.0, posinf=10.0, neginf=-10.0)
       
        return final_src, final_dst    
    
    
    def _store_sam_interactions(self, batch: Dict[str, torch.Tensor]):
        """Store interactions in buffer for deferred SAM update."""
        if not self.use_sam:  # FIX: Skip if SAM not used
            return
        device = self.device
        src_nodes = batch['src_nodes'].to(device)
        dst_nodes = batch['dst_nodes'].to(device)
        timestamps = batch['timestamps'].to(device)
        
        if 'edge_features' in batch and batch['edge_features'] is not None:
            edge_feats = batch['edge_features'].to(device)
        else:
            edge_feats = torch.zeros(len(src_nodes), self.edge_features_dim, device=device)
        
        self._sam_batch_buffer = {
            'src_nodes': src_nodes,
            'dst_nodes': dst_nodes,
            'timestamps': timestamps,
            'edge_features': edge_feats
        }
        
    def set_neighbor_finder(self, neighbor_finder):
        super().set_neighbor_finder(neighbor_finder)
        
        if hasattr(neighbor_finder, 'edge_index') and hasattr(neighbor_finder, 'edge_time'):
            edge_index = neighbor_finder.edge_index
            edge_time = neighbor_finder.edge_time
            
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index)
            if not isinstance(edge_time, torch.Tensor):
                edge_time = torch.tensor(edge_time)
            
            self.walk_sampler.update_neighbors(edge_index, edge_time)
            logger.info(f"Walk sampler initialized with {edge_index.size(1)} edges")
        else:
            logger.warning("Neighbor finder missing edge_index/edge_time")

    def set_graph(self, edge_index: torch.Tensor, edge_time: torch.Tensor):
        """Provide the full training graph to the walk sampler."""
        self.edge_index = edge_index
        self.edge_time = edge_time
        self.walk_sampler.update_neighbors(edge_index, edge_time)
        self.walk_sampler.build_dense_neighbor_table()
        logger.info(f"Walk sampler initialized with {edge_index.size(1)} edges")

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Return memory tensor for given nodes."""
        if self.use_sam and self.sam_module is not None:
            return self.sam_module.get_memory(node_ids)
        else:
            # Ensure we index the raw tensor, not the Memory object
            if hasattr(self.memory, 'memory'):
                base_memory = self.memory.memory
            else:
                base_memory = self.memory
            return base_memory[node_ids]
       
    def _get_cache_key(self, src_tensor: torch.Tensor, ts_tensor: torch.Tensor) -> tuple:
        """Safer cache key generation without float hashing."""
        ts_hash = hash((int(ts_tensor.min().item() * 1000),
                       int(ts_tensor.max().item() * 1000),
                       len(ts_tensor)))
        return (src_tensor.data_ptr(), src_tensor.shape, ts_hash)
    
    def on_load_checkpoint(self, checkpoint):
        """Called when loading a checkpoint."""
        if hasattr(self, 'neighbor_finder') and self.neighbor_finder is not None:
            self._initialize_walk_sampler_from_neighbor_finder()

    
    def _initialize_walk_sampler_from_neighbor_finder(self):
        """Extract edge_index and edge_time from neighbor_finder."""
        nf = self.neighbor_finder
        
        if hasattr(nf, '_edges') and hasattr(nf, '_timestamps'):
            edge_index = torch.tensor(nf._edges, dtype=torch.long).t()
            edge_time = torch.tensor(nf._timestamps, dtype=torch.float)
            self.walk_sampler.update_neighbors(edge_index, edge_time)
            self.walk_sampler.build_dense_neighbor_table()
            logger.info(f"Walk sampler reinitialized with {edge_index.shape[1]} edges")
        elif hasattr(nf, 'edge_index') and hasattr(nf, 'edge_time'):
            self.walk_sampler.update_neighbors(nf.edge_index, nf.edge_time)
            self.walk_sampler.build_dense_neighbor_table()
        else:
            logger.error("Cannot find edge data in neighbor_finder")
    
    def training_step(self, batch, batch_idx):
        src = batch['src_nodes']
        dst = batch['dst_nodes']
        ts = batch['timestamps']
        labels = batch['labels']
        
        # Check memory only if SAM is used
        if self.use_sam and not torch.isfinite(self.sam_module.raw_memory).all():
            logger.error(f"Batch {batch_idx}: Memory NaN at training_step start! RESETTING")
            self.sam_module.reset_memory()
        
        source_emb, dest_emb = self.compute_temporal_embeddings(src, dst, ts)
        scores = self.link_predictor(source_emb, dest_emb).squeeze(-1)
        
        if torch.isnan(scores).any():
            logger.error(f"Batch {batch_idx}: NaN scores! src_emb NaN: {torch.isnan(source_emb).any()}, dst_emb NaN: {torch.isnan(dest_emb).any()}")
            return None
        
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        
        if self.training and self.use_memory:
            self._store_sam_interactions(batch)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        
        if torch.isnan(loss):
            logger.error(f"Batch {batch_idx}: NaN loss!")
            return None
        
        return loss    
    def on_after_optimizer_step(self, optimizer):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"Parameter {name} is NaN after optimizer step")

    def on_fit_start(self):
        """Initialize SAM memory at start of training."""
        super().on_fit_start()
        if self.use_sam and self.use_memory:
            self.sam_module.reset_memory()
            logger.info("SAM memory initialized for training")

    def on_train_batch_start(self, batch, batch_idx):
        times = batch['timestamps']
        current_max = times.max().item()
        
        # FIX: Only check SAM memory if SAM is used
        if self.use_sam and batch_idx > 0:
            mem_finite = torch.isfinite(self.sam_module.raw_memory).all().item()
            proto_finite = torch.isfinite(self.sam_module.all_prototypes).all().item()
            
            if not mem_finite:
                logger.error(f"Batch {batch_idx} START: SAM memory already contains NaN!")
                nan_count = (~torch.isfinite(self.sam_module.raw_memory)).sum().item()
                logger.error(f"NaN count in memory: {nan_count}/{self.sam_module.raw_memory.numel()}")
                self.sam_module.reset_memory()
            
            if not proto_finite:
                logger.error(f"Batch {batch_idx} START: Prototypes contain NaN!")
                with torch.no_grad():
                    self.sam_module.all_prototypes.data.normal_(0, 0.01)
        
        if batch_idx == 0 and self.use_sam:  # FIX: Only log if SAM exists
            walk_data = self.walk_sampler(
                source_nodes=batch['src_nodes'][:2],
                target_nodes=batch['dst_nodes'][:2],
                current_times=batch['timestamps'][:2],
                memory_states=self.sam_module.raw_memory,
                edge_index=self.edge_index,
                edge_time=self.edge_time
            )
            
            logger.info(f"Walk data keys: {walk_data.keys()}")
            if 'source' in walk_data:
                logger.info(f"Source keys: {walk_data['source'].keys()}")
                for walk_type in ['short', 'long', 'tawr']:
                    if walk_type in walk_data['source']:
                        nodes = walk_data['source'][walk_type]['nodes']
                        logger.info(f"{walk_type}: shape={nodes.shape}, "
                                   f"min={nodes.min().item()}, max={nodes.max().item()}")
            
            edge_feats = batch.get('edge_features')
            if edge_feats is not None:
                logger.info(f"Batch 0 edge_features: shape={edge_feats.shape}, "
                           f"finite={torch.isfinite(edge_feats).all().item()}")
            
            logger.info(f"Epoch start - Batch 0 time range: [{times.min():.0f}, {times.max():.0f}]")
        
        if hasattr(self, '_prev_max_time'):
            if current_max < self._prev_max_time - 1e-6:
                logger.error(f"TEMPORAL VIOLATION: Batch {batch_idx} max {current_max:.0f} < prev {self._prev_max_time:.0f}")
        
        self._prev_max_time = current_max
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """SAM update happens HERE after backward pass."""
        # Guard all SAM operations
        if self.use_sam:
            if not torch.isfinite(self.sam_module.raw_memory).all():
                logger.error(f"Batch {batch_idx}: Memory already NaN at batch end start! RESETTING")
                self.sam_module.reset_memory()
        
        # 1. SAM DISCRETE UPDATE
        if self.training and self.use_sam and self._sam_batch_buffer is not None:
            buffer = self._sam_batch_buffer
            
            # Check buffer validity
            if not torch.isfinite(buffer['edge_features']).all():
                logger.error(f"Batch {batch_idx}: Edge features NaN, skipping SAM update")
                self._sam_batch_buffer = None
                return
            
            with torch.no_grad():
                node_feats = self.node_embedding.weight.detach()
                
                try:
                    # SAM update
                    self.sam_module.update_memory_batch(
                        source_nodes=buffer['src_nodes'],
                        target_nodes=buffer['dst_nodes'],
                        edge_features=buffer['edge_features'],
                        current_time=buffer['timestamps'],
                        node_features=node_feats
                    )
                except Exception as e:
                    logger.error(f"Batch {batch_idx}: SAM update failed with error: {e}")
                    self.sam_module.reset_memory()

                # Verify memory AFTER SAM update
                if not torch.isfinite(self.sam_module.raw_memory).all():
                    logger.error(f"Batch {batch_idx}: SAM memory NaN AFTER update! RESETTING")
                    self.sam_module.reset_memory()
                
                if not torch.isfinite(self.sam_module.all_prototypes).all():
                    logger.error(f"Batch {batch_idx}: Prototypes NaN! RESETTING")
                    self.sam_module.all_prototypes.data.normal_(0, 0.01)
                
                # Final sanitization
                self.sam_module.raw_memory.data = torch.nan_to_num(
                    self.sam_module.raw_memory.data,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0
                ).clamp_(-10, 10)
                
                self.sam_module.all_prototypes.data = torch.nan_to_num(
                    self.sam_module.all_prototypes.data,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0
                ).clamp_(-10, 10)
            
            self._sam_batch_buffer = None
        
        # 2. ST-ODE CONTINUOUS EVOLUTION (only if enabled AND SAM used)
        if self.use_st_ode and self.use_sam:
            current_time = batch['timestamps'].max()
            time_delta = current_time - self.last_update_time
            
            if time_delta < 1e-6:
                self.last_update_time = current_time
                return
            
            if not torch.isfinite(self.sam_module.raw_memory).all():
                logger.error(f"Batch {batch_idx}: Memory NaN before ST-ODE, skipping")
                self.sam_module.reset_memory()
                self.last_update_time = current_time
                return
            
            try:
                obs_data = self._prepare_stode_observations(batch)
                if obs_data is None:
                    self.last_update_time = current_time
                    return
                
                times_flat = obs_data['times'].reshape(-1)
                valid_mask = times_flat > (self.last_update_time + 1e-6)
                
                if not valid_mask.any():
                    self.last_update_time = current_time
                    return
                
                time_dim = obs_data['times'].shape[1]
                original_times = times_flat[:time_dim]
                valid_time_mask = original_times > (self.last_update_time + 1e-6)
                valid_indices = torch.where(valid_time_mask)[0]
                
                if len(valid_indices) == 0:
                    self.last_update_time = current_time
                    return
                
                filtered_encodings = obs_data['encodings'][:, valid_indices]
                filtered_times = obs_data['times'][:, valid_indices]
                filtered_masks = obs_data['masks'][:, valid_indices]
                filtered_adjs = [obs_data['adjs'][i] for i in valid_indices.tolist()]
                
                if self.st_ode is None:
                    logger.error(f"Batch {batch_idx}: ST-ODE module is None!")
                    self.last_update_time = current_time
                    return
                
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
                    
                    evolved_state = evolved_memory.final_state if hasattr(
                        evolved_memory, 'final_state') else evolved_memory
                    
                    if not torch.isfinite(evolved_state).all():
                        logger.error(f"Batch {batch_idx}: ST-ODE produced NaN, skipping")
                    elif evolved_state.abs().max() < 1e-8:
                        logger.warning(f"Batch {batch_idx}: ST-ODE produced near-zero state")
                    else:
                        self.sam_module.raw_memory.data.copy_(evolved_state)
                        self.last_update_time = current_time
            
            except Exception as e:
                logger.error(f"Batch {batch_idx}: ST-ODE failed ({e})")
                self.last_update_time = current_time
        
    def on_after_backward(self):
        # Clip ALL parameters, not just SAM and HCT
        modules_to_clip = [
            ('sam_module', 1.0),
            ('hct', 0.5),
            ('hct_to_tgn', 0.5),
            ('fusion_layer', 0.5),
            ('embedding_module', 0.5),
            ('link_predictor', 0.5),
            ('message_fn', 0.5),
            ('node_embedding', 0.5),
            ('time_proj', 0.5),           # if exists
            ('walk_to_memory', 0.5),       # if exists
        ]
        for name, max_norm in modules_to_clip:
            module = getattr(self, name, None)
            if module is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=max_norm)
        
        

        if hasattr(self, 'sam_module') and self.sam_module is not None:
            torch.nn.utils.clip_grad_norm_(
                self.sam_module.parameters(), max_norm=1.0
            )
            # Extra clipping for sensitive prototypes
            torch.nn.utils.clip_grad_norm_(
                self.sam_module.all_prototypes, 
                max_norm=0.5  # Stricter for prototypes
            )
            
        
        if self.use_hct and hasattr(self, 'hct') and self.hct is not None:
            torch.nn.utils.clip_grad_norm_(self.hct.parameters(), max_norm=0.5)
        
                
        if hasattr(self, 'hct_to_tgn') and self.hct_to_tgn is not None:
            torch.nn.utils.clip_grad_norm_(self.hct_to_tgn.parameters(), max_norm=0.5)
        
        if hasattr(self, 'fusion_layer'):
            torch.nn.utils.clip_grad_norm_(self.fusion_layer.parameters(), max_norm=0.5)
        # if hasattr(self, 'embedding_module') and self.embedding_module is not None:
        #     torch.nn.utils.clip_grad_norm_(self.embedding_module.parameters(), max_norm=0.5)

        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"NaN gradient detected in {name}")
                # Optionally zero the gradient to prevent propagation
                param.grad.zero_()
    
    def on_train_epoch_start(self):
        """Reset ST-ODE temporal state at epoch start."""
        super().on_train_epoch_start()
        if self.use_st_ode:
            self.last_update_time = torch.tensor(0.0, device=self.device)
            logger.info("ST-ODE last_update_time reset for new epoch")

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self._walk_cache.clear()
        self._last_hct_info = None
        torch.cuda.empty_cache()
    
    def on_validation_epoch_start(self):
        """Clone SAM memory for validation."""
        super().on_validation_epoch_start()
        if self.use_sam and self.use_memory:
            self._sam_validation_memory = self.sam_module.raw_memory.clone().detach()
            self._sam_validation_last_update = self.sam_module.last_update.clone().detach()
            logger.info("Cloned SAM memory for validation")
    
    def on_validation_epoch_end(self):
        """Restore SAM memory after validation."""
        super().on_validation_epoch_end()
        if self.use_sam and self.use_memory:
            self.sam_module.raw_memory.data.copy_(self._sam_validation_memory)
            self.sam_module.last_update.data.copy_(self._sam_validation_last_update)
            logger.info("Restored SAM memory after validation")
        
        if hasattr(self, '_last_hct_info') and self._last_hct_info is not None:
            try:
                for walk_type in ['short', 'long', 'tawr']:
                    if walk_type in self._last_hct_info:
                        cooc = self._last_hct_info[walk_type]['cooccurrence']
                        logger.info(f"HCT {walk_type}: cooc_mean={cooc.mean():.3f}")
            except (KeyError, TypeError, AttributeError) as e:
                logger.debug(f"Could not log HCT co-occurrence: {e}")
            
            self._last_hct_info = None
    
    
    def on_test_epoch_start(self):
        """Reset SAM memory for cold-start test evaluation."""
        super().on_test_epoch_start()
        if self.use_sam and self.use_memory:
            self.sam_module.reset_memory()
            logger.info("SAM memory reset for TEST")


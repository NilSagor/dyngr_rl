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
        use_hct: bool = True,
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
        aggregation: str = 'mean',
        time_precision: int = 6,
        use_checkpoint: bool = True,
        use_st_ode: bool = True,
        **kwargs
    ):
        # FIX 14: Verify base class compatibility before initialization
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
      
        # Initialize time projection safely (no double initialization)
        if self.use_st_ode and time_encoding_dim != memory_dim:
            self._time_proj = nn.Linear(time_encoding_dim, memory_dim)
            nn.init.xavier_uniform_(self._time_proj.weight)
            nn.init.zeros_(self._time_proj.bias)
        elif self.use_st_ode:
            self._time_proj = nn.Identity()
        else:
            self._time_proj = None
        
        self.time_encoder = TimeEncoder(time_encoding_dim)
        
        # Initialize SAM
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
        
        self.sam_config = {
            'num_prototypes': num_prototypes,
            'similarity_metric': similarity_metric,
            'memory_dim': memory_dim
        }
        logger.info(f"SAM initialized: {num_prototypes} prototypes, {similarity_metric} similarity")
        
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
        
        # Initialize HCT
        if self.use_hct:
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
            
            if memory_dim != hct_d_model:
                self.memory_proj = nn.Linear(memory_dim, hct_d_model)
            else:
                self.memory_proj = nn.Identity()
            
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
            self.hct = None
            self.memory_proj = None
            self.hct_to_tgn = None
        
        # Initialize ST-ODE
        if self.use_st_ode:
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
        
        # Time of last memory update
        self.register_buffer('last_update_time', torch.tensor(0.0))
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_index = None
        self.edge_time = None
        self._walk_cache = {}
        self._cache_walks = False
        
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
        device = self.device
        num_nodes = self.num_nodes
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']
        
        src_emb = self.sam_module.raw_memory[src_nodes]
        dst_emb = self.sam_module.raw_memory[dst_nodes]
        
        time_emb = self.time_encoder(timestamps.float())
        
        # FIX 3: Safe time projection with Identity check
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
        
        # FIX 12: Remove lazy initialization (only use __init__ projection)
        if self._time_proj is not None and not isinstance(self._time_proj, nn.Identity):
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
        
    
    
    def _simple_walk_embed(self, walk_data: Dict) -> torch.Tensor:
        """Fallback: simple mean of walk node features."""
        all_nodes = []
        all_masks = []
        max_len = 0
        
        for wt in ['short', 'long', 'tawr']:
            if wt in walk_data:
                nodes = walk_data[wt]['nodes']
                max_len = max(max_len, nodes.size(2))
        
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
            return torch.zeros(self.hidden_dim, device=self.device)
        
        nodes = torch.cat(all_nodes, dim=1)
        masks = torch.cat(all_masks, dim=1)
        
        flat_nodes = nodes.reshape(-1)
        flat_feats = self.sam_module.raw_memory[flat_nodes]
        flat_feats = flat_feats.view(nodes.size(0), nodes.size(1), nodes.size(2), -1)
        
        masks_expanded = masks.unsqueeze(-1).float()
        sum_feats = (flat_feats * masks_expanded).sum(dim=[1, 2])
        
        # Better handling for near-zero count
        count = masks.sum(dim=[1, 2]).unsqueeze(-1)
        count_safe = torch.where(count > 1e-6, count, torch.ones_like(count))
        
        return sum_feats / count_safe
    
    def compute_temporal_embeddings(
        self,
        source_nodes: torch.Tensor,
        destination_nodes: torch.Tensor,
        edge_times: torch.Tensor,
        n_neighbors: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified embedding computation with HCT."""
        device = self.device
        
        if torch.isnan(edge_times).any():
            logger.error("NaN in edge_times!")
            edge_times = torch.nan_to_num(edge_times, nan=0.0)
        
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
        
        # Validate walk data
        for side in ['source', 'target']:
            for walk_type in ['short', 'long', 'tawr']:
                if walk_type in walk_data[side]:
                    nodes = walk_data[side][walk_type]['nodes']
                    if not torch.isfinite(nodes).all():
                        logger.error(f"{side}/{walk_type} nodes contain NaN/Inf!")
                        if side == 'source':
                            dummy = src_tensor.unsqueeze(1).unsqueeze(2).expand(-1, nodes.shape[1], nodes.shape[2])
                        else:
                            dummy = dst_tensor.unsqueeze(1).unsqueeze(2).expand(-1, nodes.shape[1], nodes.shape[2])
                        walk_data[side][walk_type]['nodes'] = torch.clamp(dummy, 0, self.num_nodes - 1)
                        if 'masks' in walk_data[side][walk_type]:
                            walk_data[side][walk_type]['masks'] = torch.ones_like(
                                walk_data[side][walk_type]['masks'])
        
        # Check memory before HCT
        if not torch.isfinite(self.sam_module.raw_memory).all():
            logger.warning("SAM memory has NaN, resetting affected nodes")
            self.sam_module.raw_memory.data = torch.nan_to_num(
                self.sam_module.raw_memory.data, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 2. VALIDATE WALK DATA BEFORE HCT (FIX 2: Move validation before memory access)
        if self.use_hct:
            for side in ['source', 'target']:
                for walk_type in ['short', 'long', 'tawr']:
                    if walk_type in walk_data[side]:
                        nodes = walk_data[side][walk_type]['nodes']
                        masks = walk_data[side][walk_type].get('masks', None)
                        
                        # Check for all-zero indices
                        if nodes.max().item() == 0 and nodes.numel() > 0:
                            logger.error(f"Walk sampler returned all-zero indices for {side}/{walk_type}!")
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
                        
                        # Validate bounds BEFORE memory lookup
                        if nodes.max().item() >= self.num_nodes or nodes.min().item() < 0:
                            logger.warning(f"Clamping {side} {walk_type} walk nodes")
                            walk_data[side][walk_type]['nodes'] = torch.clamp(
                                nodes, 0, self.num_nodes - 1)
                        
                        if masks is not None and masks.sum() == 0:
                            logger.warning(f"All {side} {walk_type} masks are False!")
                        
                        if not torch.isfinite(nodes).all():
                            logger.error(f"{side} {walk_type} nodes contain NaN/Inf!")
                            walk_data[side][walk_type]['nodes'] = torch.nan_to_num(
                                nodes, nan=0, posinf=self.num_nodes-1, neginf=0).long()
        
        if not torch.isfinite(self.sam_module.raw_memory).all():
            logger.error("SAM memory NaN before HCT - RESETTING")
            self.sam_module.reset_memory()
        
        # 3. HCT encodes walks
        if self.use_hct:
            if not torch.isfinite(self.sam_module.raw_memory).all():
                logger.error(f"SAM memory NaN before HCT! Resetting...")
                self.sam_module.reset_memory()
            
            # Validate walk indices before HCT memory access
            if 'source' in walk_data and 'nodes' in walk_data['source']:
                src_walk_nodes = walk_data['source']['nodes']
                if src_walk_nodes.max().item() >= self.num_nodes or src_walk_nodes.min().item() < 0:
                    logger.error(f"Invalid walk node index! max={src_walk_nodes.max().item()}")
                    walk_data['source']['nodes'] = torch.clamp(src_walk_nodes, 0, self.num_nodes - 1)
            
            if 'target' in walk_data and 'nodes' in walk_data['target']:
                dst_walk_nodes = walk_data['target']['nodes']
                if dst_walk_nodes.max().item() >= self.num_nodes or dst_walk_nodes.min().item() < 0:
                    logger.error(f"Invalid walk node index! max={dst_walk_nodes.max().item()}")
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
            
            # Check HCT outputs
            for key in ['short', 'long', 'tawr', 'fused']:
                if key in hct_src_output:
                    val = hct_src_output[key]
                    if isinstance(val, torch.Tensor) and not torch.isfinite(val).all():
                        logger.error(f"HCT src {key} NaN! shape={val.shape}")
                    elif isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, torch.Tensor) and not torch.isfinite(sub_val).all():
                                logger.error(f"HCT src {key}.{sub_key} NaN!")
            
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
        
        if not torch.isfinite(walk_src_emb).all():
            walk_src_emb = torch.nan_to_num(walk_src_emb, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(walk_dst_emb).all():
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
        
        if not torch.isfinite(base_src_emb).all():
            base_src_emb = torch.nan_to_num(base_src_emb, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(base_dst_emb).all():
            base_dst_emb = torch.nan_to_num(base_dst_emb, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Remove stabilized memory call to prevent double ST-ODE execution
        
        # 6. Fusion
        combined_src = torch.cat([base_src_emb, walk_src_emb], dim=-1)
        combined_dst = torch.cat([base_dst_emb, walk_dst_emb], dim=-1)
        
        final_src = self.fusion_layer(combined_src)
        final_dst = self.fusion_layer(combined_dst)
        
        if not torch.isfinite(final_src).all():
            final_src = torch.nan_to_num(final_src, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(final_dst).all():
            final_dst = torch.nan_to_num(final_dst, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return final_src, final_dst   
    
      
   

    def _store_sam_interactions(self, batch: Dict[str, torch.Tensor]):
        """Store interactions in buffer for deferred SAM update."""
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
        """Override to use SAM memory instead of GRU memory."""
        return self.sam_module.get_memory(node_ids)
       
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
        
        source_emb, dest_emb = self.compute_temporal_embeddings(src, dst, ts)
        
        scores = self.link_predictor(source_emb, dest_emb).squeeze(-1)
        
        if torch.isnan(scores).any():
            logger.error(f"NaN in scores!")
            logger.warning(f"Batch {batch_idx} contains NaN scores, skipping")
            return None
        
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        
        if self.training and self.use_memory:
            self._store_sam_interactions(batch)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        
        if torch.isnan(loss):
            logger.error(f"NaN loss!")
        
        # Return scalar loss for Lightning compatibility
        return loss   
    

    def on_fit_start(self):
        """Initialize SAM memory at start of training."""
        super().on_fit_start()
        if self.use_memory:
            self.sam_module.reset_memory()
            logger.info("SAM memory initialized for training")

    def on_train_batch_start(self, batch, batch_idx):
        times = batch['timestamps']
        current_max = times.max().item()
        
        if batch_idx == 0:
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
            # FIX 9: Add epsilon tolerance for temporal violation
            if current_max < self._prev_max_time - 1e-6:
                logger.error(f"TEMPORAL VIOLATION: Batch {batch_idx} max {current_max:.0f} < prev {self._prev_max_time:.0f}")
        
        self._prev_max_time = current_max
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """SAM update happens HERE after backward pass."""
        # 1. SAM DISCRETE UPDATE
        if self.training and self.use_memory and hasattr(self, '_sam_batch_buffer'):
            buffer = self._sam_batch_buffer
            
            if not torch.isfinite(buffer['edge_features']).all():
                logger.error(f"Batch {batch_idx}: Edge features NaN, skipping SAM update")
                delattr(self, '_sam_batch_buffer')
                return
            
            with torch.no_grad():
                node_feats = self.node_embedding.weight.detach()
                
                self.sam_module.update_memory_batch(
                    source_nodes=buffer['src_nodes'],
                    target_nodes=buffer['dst_nodes'],
                    edge_features=buffer['edge_features'],
                    current_time=buffer['timestamps'],
                    node_features=node_feats
                )
                
                # FIX 5: Remove manual raw_memory.data copy (SAM handles it internally)
                # Only sanitize prototypes
                if not torch.isfinite(self.sam_module.raw_memory).all():
                    logger.error(f"Batch {batch_idx}: SAM memory NaN AFTER update! RESETTING")
                    self.sam_module.reset_memory()
                
                if not torch.isfinite(self.sam_module.all_prototypes).all():
                    logger.error(f"Batch {batch_idx}: Prototypes NaN! RESETTING")
                    self.sam_module.all_prototypes.data.normal_(0, 0.01)
                
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
            
            delattr(self, '_sam_batch_buffer')
        
        # 2. ST-ODE CONTINUOUS EVOLUTION
        if self.use_st_ode:
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
                    
                    # FIX 8: Handle eigen decomposition failures in ST-ODE
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
        # Safe gradient clipping with proper hasattr checks
        if hasattr(self, 'sam_module') and self.sam_module is not None:
            torch.nn.utils.clip_grad_norm_(self.sam_module.parameters(), max_norm=1.0)
        
        # Check if hct exists before clipping
        if self.use_hct and hasattr(self, 'hct') and self.hct is not None:
            torch.nn.utils.clip_grad_norm_(self.hct.parameters(), max_norm=0.5)
        
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
        if self.use_memory:
            self._sam_validation_memory = self.sam_module.raw_memory.clone().detach()
            self._sam_validation_last_update = self.sam_module.last_update.clone().detach()
            logger.info("Cloned SAM memory for validation")
    
    def on_validation_epoch_end(self):
        """Restore SAM memory after validation."""
        super().on_validation_epoch_end()
        if self.use_memory:
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
        if self.use_memory:
            self.sam_module.reset_memory()
            logger.info("SAM memory reset for TEST")


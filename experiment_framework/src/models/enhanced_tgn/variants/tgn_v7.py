import os
import torch 
import torch.nn as nn
import numpy as np

from typing import Dict, Optional, Tuple
from loguru import logger

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from src.datasets.sam_dataloading.neighbor_finder import NeighborFinder

from ..base_enhance_tgn import BaseEnhancedTGN
from ..component.time_encoder import TimeEncoder
from ..component.sam_module import StabilityAugmentedMemory
from ..component.multi_swalk import MultiScaleWalkSampler
from ..component.walk_encoder import WalkEncoder
from ..component.hct_module import HierarchicalCooccurrenceTransformer
from ..component.stode_module import SpectralTemporalODE
from ..component.transformer_encoder import MergeLayer
from ..component.mrp_module import MutualRefineAndPooling


# === CONFIG: Disable debug features for training ===
DEBUG_VALIDATION = False  # Set True only for debugging
DEBUG_LOGGING = False    # Disable verbose logging


# === Helper function ==========

def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


class TGNv7(BaseEnhancedTGN):
    """
    TGN + SAM + Multi-Scale Walk Sampler + Hierarchical Co-occurrence Transformer (HCT) + ST ODE
    + Mutual Refined and Pooling
    Architecture:
    1. SAM maintains stable node memory via prototypes
    2. Multi-scale walk sampler generates short/long/TAWR walks with anonymization
    3. HCT processes walks: intra-walk -> co-occurrence -> inter-walk -> fusion
    4. Combine HCT walk embeddings with base TGN embeddings
    5. Spectral-Temporal ODE for continuous-time node representation learning
    6. Mutual Refinement and Pooling combined representations
    7. Link prediction on combined representations
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
        ode_method: str = 'dopri5',
        ode_step_size: Optional[float] = 1000,
        rtol: float = 1e-6,  # Add this
        atol: float = 1e-8,
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
        
        self.use_st_ode = parse_bool(use_st_ode)
        self.use_hct = parse_bool(use_hct)
        self.directed = kwargs.get('directed', False)
        
        self.use_sam = parse_bool(use_sam)
        self.debug_simple_walk = parse_bool(debug_simple_walk)

        self.st_ode_update_interval = 10  # batches
        self._st_ode_batch_counter = 0

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
        
         # ===== Time Encoder =====
        self.time_encoder = TimeEncoder(time_encoding_dim)
        
        self.sam_config = {
            'num_prototypes': num_prototypes,
            'similarity_metric': similarity_metric,
            'memory_dim': memory_dim
        }
                
        # ===== SAM Memory =====
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
            sam_info = f"SAM({num_prototypes}p) + " if self.use_sam else ""
            logger.info(f"TGNv7 initialized: {sam_info}WalkSampler + "
                        f"{'HCT(' + str(hct_d_model) + 'd, ' + str(hct_nhead) + 'h)' if use_hct else 'SimpleWalkEncoder'}")
            
        else:
            self.sam_module = None
            logger.info(f"SAM disabled – using base TGN memory")
                
        
         # ===== Multi-Scale Walk Sampler =====
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
                
        # ===== Walk Processor (HCT or Simple) =====
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
                    # aggregation=aggregation,
                    # time_precision=time_precision,
                    
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
        
        # Mutual Refinement
        self.mutual_refine = MutualRefineAndPooling(
            d_model=hidden_dim,
            nhead=n_heads,
            # num_walk_types=,
            # max_walks_per_type=
        )
        
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
        # Cache for walk data (avoid recomputation)
        self._walk_cache = None
        self._cache_batch_key = None
        self._prev_max_time = -float('inf')
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
            output_dim=1,
            dropout=dropout,
            use_temperature=True  # Enable temperature scaling
        )
        
        if edge_features_dim == 0:
            logger.warning("Edge features disabled (dim=0). SAM will receive zero edge input.")
        else:
            logger.info(f"Edge features enabled: input_dim={edge_features_dim}, projected_dim={memory_dim}")
        
        logger.info(f"TGNv6 initialized: SAM({num_prototypes}p) + WalkSampler + "
                   f"{'HCT(' + str(hct_d_model) + 'd, ' + str(hct_nhead) + 'h)' if use_hct else 'SimpleWalkEncoder'}")

    def _get_batch_key(self, batch):
        """Generate cache key from batch data."""
        src_nodes = batch['src_nodes']
        timestamps = batch['timestamps'] if 'timestamps' in batch else batch.get('ts', torch.tensor([]))
        
        # Sample first 5 elements for key (or all if smaller)
        src_sample = src_nodes[:min(5, len(src_nodes))]
        ts_sample = timestamps[:min(5, len(timestamps))] if len(timestamps) > 0 else torch.tensor([0.0])
        
        # Convert to numpy for hashing (cpu().numpy().tobytes())
        src_bytes = src_sample.cpu().numpy().tobytes()
        ts_bytes = ts_sample.cpu().numpy().tobytes()
        
        return (src_bytes, ts_bytes, len(src_nodes))
    
    def _get_device(self) -> torch.device:
        """Get device from model parameters."""
        return next(self.parameters()).device
    
    def _prepare_stode_observations(self, batch):
        """Convert batch interactions to ST-ODE observation format."""
        
        """Fully vectorized ST-ODE observation preparation."""
        if not self.use_st_ode or not self.use_sam:
            return None
        
        device = self.device
        num_nodes = self.num_nodes
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']
        
        # Get embeddings
        src_emb = self.sam_module.raw_memory[src_nodes]
        dst_emb = self.sam_module.raw_memory[dst_nodes]
        time_emb = self.time_encoder(timestamps.float())
        
        if self._time_proj is not None and not isinstance(self._time_proj, nn.Identity):
            time_emb = self._time_proj(time_emb)
        
        # Build sparse adjacency once
        adj_t = self._build_temporal_adjacency(src_nodes, dst_nodes, num_nodes)
        
        # === VECTORIZED TIME AGGREGATION ===
        # Sort by time for efficient grouping
        sorted_ts, sort_idx = torch.sort(timestamps)
        sorted_src = src_nodes[sort_idx]
        sorted_dst = dst_nodes[sort_idx]
        sorted_src_emb = src_emb[sort_idx] + time_emb[sort_idx]
        sorted_dst_emb = dst_emb[sort_idx] + time_emb[sort_idx]
        
        # Find time boundaries using diff
        time_diff = torch.diff(sorted_ts, prepend=sorted_ts[:1])
        time_boundaries = torch.where(time_diff > 0)[0]
        T = len(time_boundaries) + 1
        
        # Use scatter_add for vectorized aggregation
        node_obs = torch.zeros(T, num_nodes, self.memory_dim, device=device)
        counts = torch.zeros(T, num_nodes, device=device)
        
        # Create segment IDs for each edge
        segment_ids = torch.searchsorted(time_boundaries, torch.arange(len(sorted_ts), device=device))
        
        # Flatten for scatter_add: [T*N, D] indexing
        flat_idx_src = segment_ids * num_nodes + sorted_src
        flat_idx_dst = segment_ids * num_nodes + sorted_dst
        
        # Scatter add (much faster than loop)
        node_obs_flat = node_obs.view(-1, self.memory_dim)
        counts_flat = counts.view(-1)
        
        node_obs_flat.scatter_add_(0, flat_idx_src.unsqueeze(-1).expand(-1, self.memory_dim), sorted_src_emb)
        node_obs_flat.scatter_add_(0, flat_idx_dst.unsqueeze(-1).expand(-1, self.memory_dim), sorted_dst_emb)
        counts_flat.scatter_add_(0, flat_idx_src, torch.ones_like(sorted_src, dtype=torch.float))
        counts_flat.scatter_add_(0, flat_idx_dst, torch.ones_like(sorted_dst, dtype=torch.float))
        
        # Reshape and normalize
        node_obs = node_obs / counts.unsqueeze(-1).clamp(min=1)
        
        # Reshape for ST-ODE: [N, T, 1, D]
        walk_encodings = node_obs.permute(1, 0, 2).unsqueeze(2)
        
        # Use actual unique times
        unique_times = torch.cat([sorted_ts[:1], sorted_ts[time_boundaries]])
        walk_times = unique_times.view(1, T, 1).expand(num_nodes, -1, -1)
        walk_masks = (counts > 0).T.unsqueeze(-1).float()
        
        return {
            'encodings': walk_encodings,
            'times': walk_times,
            'masks': walk_masks,
            'adjs': adj_t
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
    
    
    def forward(self, batch: Dict[str, torch.Tensor], return_probs: bool = False) -> torch.Tensor:
        """Forward pass with proper SAM update sequencing."""
        source_emb, dest_emb = self.compute_temporal_embeddings(
            batch['src_nodes'],
            batch['dst_nodes'],
            batch['timestamps'],
            batch = batch,
        )
        
        if self.training and self.use_memory:
            self._store_sam_interactions(batch)
        
        # Get raw logits
        logits = self.link_predictor(source_emb, dest_emb).squeeze(-1) # [B]
        
        if return_probs:
            # Apply temperature if enabled
            if hasattr(self.link_predictor, 'temperature') and self.link_predictor.use_temperature:
                temp = self.link_predictor.temperature.clamp(min=0.1, max=10.0)
                scaled_logits = logits / temp
            else:
                scaled_logits = logits
            probs = torch.sigmoid(scaled_logits)
            return logits, probs
        
        if torch.isnan(logits).any():
            logger.error(f"NaN in scores! source_emb: {torch.isnan(source_emb).any()}, "
                        f"dest_emb: {torch.isnan(dest_emb).any()}")
        
        return logits
        
    
    
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
        n_neighbors: int = 20,
        batch: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified embedding computation with HCT."""
        device = self.device
        
        # ----- 1. Get current node memory (SAM or base TGN) -----
        if self.use_sam and self.sam_module is not None:
            # FIX: Detach memory to prevent gradient leakage
            node_memory = self.sam_module.raw_memory.detach()
        else:
            if hasattr(self.memory, 'memory'):
                node_memory = self.memory.memory
            else:
                node_memory = self.memory
        
        # FIX: Always validate memory state (not just in debug mode)
        if self.use_sam and not torch.isfinite(self.sam_module.raw_memory).all():
            logger.error("SAM memory NaN before compute_temporal_embeddings! RESETTING")
            self.sam_module.reset_memory()
            node_memory = self.sam_module.raw_memory.detach()
        
        # Fix NaN in edge_times
        if torch.isnan(edge_times).any():
            logger.error("NaN in edge_times! Replacing with 0.")
            edge_times = torch.nan_to_num(edge_times, nan=0.0)
        
        src_tensor = source_nodes.to(device).long()
        dst_tensor = destination_nodes.to(device).long()
        ts_tensor = edge_times.to(device).float()
        
        # ----- 2. Generate walks -----
        cache_key = self._get_batch_key(batch) if batch is not None else None
        if cache_key == self._cache_batch_key and self._walk_cache is not None:
            walk_data = self._walk_cache
        else:
            with torch.no_grad():
                walk_memory = node_memory.detach()
                walk_data = self.walk_sampler(
                    source_nodes=src_tensor,
                    target_nodes=dst_tensor,
                    current_times=ts_tensor,
                    memory_states=walk_memory,
                    edge_index=None,
                    edge_time=None
                )
            self._walk_cache = walk_data
            self._cache_batch_key = cache_key
        
        # ----- 3. Validate walk data -----
        for side in ['source', 'target']:
            for wt in ['short', 'long', 'tawr']:
                if wt in walk_data[side]:
                    nodes = walk_data[side][wt]['nodes']
                    if not torch.isfinite(nodes).all():
                        logger.error(f"{side}/{wt} nodes contain NaN/Inf! Replacing with zeros.")
                        walk_data[side][wt]['nodes'] = torch.zeros_like(nodes)
                    nodes.clamp_(0, self.num_nodes - 1)
        
        # ----- 4. Obtain walk embeddings (HCT or simple mean) -----
        src_per_type = None
        dst_per_type = None
        src_masks = None
        dst_masks = None
        
        if self.use_hct and self.hct is not None and not self.debug_simple_walk:
            combined_walks = {
                'short': {
                    'nodes': torch.cat([walk_data['source']['short']['nodes'],
                                    walk_data['target']['short']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['short']['nodes_anon'],
                                            walk_data['target']['short']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['short']['masks'],
                                    walk_data['target']['short']['masks']], dim=0),
                },
                'long': {
                    'nodes': torch.cat([walk_data['source']['long']['nodes'],
                                    walk_data['target']['long']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['long']['nodes_anon'],
                                            walk_data['target']['long']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['long']['masks'],
                                    walk_data['target']['long']['masks']], dim=0),
                },
                'tawr': {
                    'nodes': torch.cat([walk_data['source']['tawr']['nodes'],
                                    walk_data['target']['tawr']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['tawr']['nodes_anon'],
                                            walk_data['target']['tawr']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['tawr']['masks'],
                                    walk_data['target']['tawr']['masks']], dim=0),
                }
            }
            
            with torch.enable_grad():
                hct_output = self.hct(
                    walks_dict=combined_walks,
                    node_memory=node_memory,
                    return_all=False
                )
            
            batch_size = src_tensor.size(0)
            hct_src = hct_output[:batch_size]
            hct_dst = hct_output[batch_size:]
            
            # Ensure projection to hidden_dim
            if self.hct_to_tgn is not None and not isinstance(self.hct_to_tgn, nn.Identity):
                walk_src_emb = self.hct_to_tgn(hct_src)
                walk_dst_emb = self.hct_to_tgn(hct_dst)
            else:
                walk_src_emb = hct_src
                walk_dst_emb = hct_dst
            
            # Prepare per-type embeddings for hierarchical pooling in MRP
            src_per_type, src_masks = self._prepare_per_type_embeddings(
                walk_data['source'], node_memory, device
            )
            dst_per_type, dst_masks = self._prepare_per_type_embeddings(
                walk_data['target'], node_memory, device
            )
            
            # FIX: Project per-type embeddings to hidden_dim if needed
            if self.hidden_dim != self.memory_dim and self.hidden_dim != hct_src.size(-1):
                if not hasattr(self, '_per_type_proj'):
                    self._per_type_proj = nn.Linear(hct_src.size(-1), self.hidden_dim).to(device)
                if src_per_type is not None:
                    src_per_type = self._per_type_proj(src_per_type)
                if dst_per_type is not None:
                    dst_per_type = self._per_type_proj(dst_per_type)
        else:
            # Fallback: simple mean pooling
            walk_src_emb = self._simple_walk_embed(walk_data['source'], node_memory)
            walk_dst_emb = self._simple_walk_embed(walk_data['target'], node_memory)
        
        # ----- 5. Base TGN embeddings -----
        if self.embedding_module is not None:
            base_src_emb = self.embedding_module.compute_embedding(
                memory=node_memory,
                source_nodes=src_tensor,
                timestamps=ts_tensor,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            base_dst_emb = self.embedding_module.compute_embedding(
                memory=node_memory,
                source_nodes=dst_tensor,
                timestamps=ts_tensor,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
        else:
            base_src_emb = node_memory[src_tensor]
            base_dst_emb = node_memory[dst_tensor]
        
        # ----- 6. MUTUAL REFINEMENT & POOLING -----
        combined_walk = torch.cat([walk_src_emb, walk_dst_emb], dim=0)  # [2B, D]
        combined_base = torch.cat([base_src_emb, base_dst_emb], dim=0)  # [2B, D]
        
        # Validate dimensions match
        assert combined_walk.size(-1) == combined_base.size(-1), \
            f"Walk embedding dim {combined_walk.size(-1)} != base embedding dim {combined_base.size(-1)}"
        
        if src_per_type is not None and dst_per_type is not None:
            combined_per_type = torch.cat([src_per_type, dst_per_type], dim=0)
            combined_masks = torch.cat([src_masks, dst_masks], dim=0)
            # Ensure boolean masks
            combined_masks = combined_masks.bool()
        else:
            combined_per_type = None
            combined_masks = None
        
        refined_combined_walk, refined_combined_base = self.mutual_refine(
            src_walk=combined_walk,
            dst_walk=combined_base,
            src_per_type=combined_per_type,
            dst_per_type=None,
            src_masks=combined_masks,
            dst_masks=None,
        )
        
        # Split back to source and destination
        final_src = refined_combined_walk[:batch_size]
        final_dst = refined_combined_walk[batch_size:]
        
        # Final NaN protection
        if not torch.isfinite(final_src).all():
            logger.warning("NaN in final_src, sanitizing")
            final_src = torch.nan_to_num(final_src, nan=0.0, posinf=10.0, neginf=-10.0)
        if not torch.isfinite(final_dst).all():
            logger.warning("NaN in final_dst, sanitizing")
            final_dst = torch.nan_to_num(final_dst, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return final_src, final_dst    
    
    
    def _prepare_per_type_embeddings(
        self, 
        walk_data_side: Dict, 
        node_memory: torch.Tensor,
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepare per-type walk embeddings for MutualRefineAndPooling.
        
        Returns:
            per_type_embeds: [B, num_types, max_walks, D] or None
            masks: [B, num_types, max_walks] or None
        """
        if not walk_data_side:
            return None, None
        
        type_names = ['short', 'long', 'tawr']
        type_embeds = []
        type_masks = []
        
        max_walks = self.walk_sampler.num_walks_short  # Use as reference
        
        for wt in type_names:
            if wt in walk_data_side:
                nodes = walk_data_side[wt]['nodes']      # [B, num_walks, walk_len]
                masks = walk_data_side[wt]['masks']      # [B, num_walks, walk_len]
                
                # Get node features and average over walk length
                B, num_walks, walk_len = nodes.shape
                flat_nodes = nodes.reshape(-1)
                flat_feats = node_memory[flat_nodes]     # [B*num_walks*walk_len, D]
                feats = flat_feats.view(B, num_walks, walk_len, -1)  # [B, num_walks, walk_len, D]
                
                # Average over walk length (only valid positions)
                walk_masks = masks.any(dim=-1)  # [B, num_walks] - which walks are valid
                walk_masks = walk_masks.bool()
                walk_embeds = (feats * masks.unsqueeze(-1)).sum(dim=2) / masks.sum(dim=2, keepdim=True).clamp(min=1)
                # [B, num_walks, D]
                
                # Pad or truncate to max_walks
                if num_walks < max_walks:
                    pad_size = max_walks - num_walks
                    walk_embeds = F.pad(walk_embeds, (0, 0, 0, pad_size))
                    walk_masks = F.pad(walk_masks, (0, pad_size), value=False)
                else:
                    walk_embeds = walk_embeds[:, :max_walks]
                    walk_masks = walk_masks[:, :max_walks]
                
                type_embeds.append(walk_embeds)
                type_masks.append(walk_masks)
            else:
                # Create empty placeholder
                B = node_memory.size(0)
                D = node_memory.size(-1)
                type_embeds.append(torch.zeros(B, max_walks, D, device=device))
                type_masks.append(torch.zeros(B, max_walks, dtype=torch.bool, device=device))
        
        # Stack to [B, num_types, max_walks, D]
        per_type_embeds = torch.stack(type_embeds, dim=1)
        masks = torch.stack(type_masks, dim=1)
        
        return per_type_embeds, masks


    def _store_sam_interactions(self, batch: Dict[str, torch.Tensor]):
        """Store interactions in buffer for deferred SAM update."""
        if not self.use_sam:  # Skip if SAM not used
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
        self.walk_sampler._freeze_neighbors = True
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
            self._initialize_walk_sampler_from_neighbor_finder(self.neighbor_finder)
        else:
            logger.warning("neighbor_finder not available when loading checkpoint - walk sampler may not be initialized")
        
    def _initialize_walk_sampler_from_neighbor_finder(self, neighbor_finder):
        """
        Initialize walk sampler from NeighborFinder by reconstructing edge_index/edge_time.
        
        NeighborFinder stores edges in edge_id_adj_list: {node: [(neighbor, ts, edge_id), ...]}
        We reconstruct the original [2, E] edge_index and [E] edge_time tensors.
        """
        edge_index = None
        edge_time = None
        
        # === Strategy 1: Standard tensor attributes (for compatibility) ===
        if hasattr(neighbor_finder, 'edge_index') and hasattr(neighbor_finder, 'edge_time'):
            edge_index = neighbor_finder.edge_index
            edge_time = neighbor_finder.edge_time
            logger.info(f"Found edge_index {edge_index.shape} and edge_time {edge_time.shape} in neighbor_finder")
        
        # === Strategy 2: Legacy numpy attributes ===
        elif hasattr(neighbor_finder, '_edges') and hasattr(neighbor_finder, '_timestamps'):
            edges_np = neighbor_finder._edges
            timestamps_np = neighbor_finder._timestamps
            edge_index = torch.from_numpy(edges_np).T  # [E, 2] -> [2, E]
            edge_time = torch.from_numpy(timestamps_np)
            logger.info(f"Converted _edges/_timestamps to tensors: {edge_index.shape}, {edge_time.shape}")
        
        # === Strategy 3: Reconstruct from edge_id_adj_list (ACTUAL NeighborFinder format) ===
        elif hasattr(neighbor_finder, 'edge_id_adj_list'):
            logger.info("Reconstructing edge_index from edge_id_adj_list")
            
            # Collect unique edges by edge_id to avoid duplicates from undirected graph
            # edge_id_adj_list: {src: [(dst, timestamp, edge_id), ...]}
            edge_map: Dict[int, Tuple[int, int, float]] = {}
            
            for src_node, neighbors in neighbor_finder.edge_id_adj_list.items():
                for dst_node, timestamp, edge_id in neighbors:
                    if edge_id not in edge_map:
                        # Store first occurrence of this edge_id (avoids undirected duplicates)
                        edge_map[edge_id] = (src_node, dst_node, timestamp)
            
            if not edge_map:
                logger.error("edge_id_adj_list is empty - cannot initialize walk sampler")
                raise ValueError("NeighborFinder has no edges to initialize walk sampler")
            
            num_edges = len(edge_map)
            logger.info(f"Reconstructed {num_edges} unique edges from edge_id_adj_list")
            
            # Build edge_index [2, E] and edge_time [E]
            edge_index = torch.zeros((2, num_edges), dtype=torch.long)
            edge_time = torch.zeros(num_edges, dtype=torch.float32)
            
            for edge_id, (src, dst, ts) in edge_map.items():
                edge_index[0, edge_id] = src
                edge_index[1, edge_id] = dst
                edge_time[edge_id] = ts
            
            logger.info(f"Built edge_index shape {edge_index.shape}, edge_time shape {edge_time.shape}")
        
        # === Strategy 4: Fallback error with available attributes ===
        else:
            available = [attr for attr in dir(neighbor_finder) if not attr.startswith('__') and not callable(getattr(neighbor_finder, attr))]
            logger.error(f"Cannot find edge data in neighbor_finder. Available attrs: {available}")
            raise ValueError(
                "neighbor_finder missing required edge data. Expected one of: "
                "edge_index/edge_time, _edges/_timestamps, or edge_id_adj_list"
            )
        
        # === Verify and normalize shapes ===
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            logger.warning(f"Transposing edge_index from {edge_index.shape} to [2, E]")
            edge_index = edge_index.T
        
        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got {edge_index.shape}")
        
        if edge_time.shape[0] != edge_index.shape[1]:
            raise ValueError(f"Shape mismatch: edge_index[1]={edge_index.shape[1]} != edge_time[0]={edge_time.shape[0]}")
        
        # === Initialize walk sampler ===
        # Note: MultiScaleWalkSampler uses update_neighbors(), NOT initialize_from_edges()
        try:
            self.walk_sampler.update_neighbors(edge_index, edge_time)
            self.walk_sampler.build_dense_neighbor_table()
            logger.info(f"✓ Walk sampler initialized with {edge_index.shape[1]} edges")
        except Exception as e:
            logger.error(f"Failed to initialize walk sampler: {e}")
            raise
    
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
    
    # def validation_step(self, batch, batch_idx):
    #     """Store predictions for threshold optimization."""
    #     logits = self(batch)
    #     probs = torch.sigmoid(logits)
    #     labels = batch['labels'].float()
        
    #     loss = F.binary_cross_entropy_with_logits(logits, labels)
        
    #     # Store for threshold optimization
    #     if not hasattr(self, '_val_predictions'):
    #         self._val_predictions = []
        
    #     self._val_predictions.append({
    #         'probs': probs.detach().cpu(),
    #         'labels': labels.detach().cpu()
    #     })
        
    #     # Standard metrics
    #     self.log('val_loss', loss, prog_bar=True)
        
    #     # Use current optimal threshold if available
    #     thresh = getattr(self, '_optimal_threshold', 0.5)
    #     preds = (probs > thresh).float()
    #     acc = (preds == labels).float().mean()
    #     self.log('val_acc', acc, prog_bar=True)
        
    #     return loss
        
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

    def on_after_backward(self):        
        # Collect all parameters into single list
        # Single global clip (faster)
        all_params = [p for p in self.parameters() if p.grad is not None]
        if all_params:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        
        # Or: Use gradient scaling instead of clipping for stability
        # self.trainer.strategy.precision_plugin.scaler.unscale_(optimizer)
        
        if DEBUG_VALIDATION:
            for name, param in self.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    logger.error(f"NaN gradient in {name}")
                    param.grad.zero_()        
        #     param_groups = [
        #     (self.sam_module.parameters() if self.use_sam and self.sam_module else [], 1.0),
        #     ([self.sam_module.all_prototypes] if self.use_sam and self.sam_module and hasattr(self.sam_module, 'all_prototypes') else [], 0.5),
        #     (self.hct.parameters() if self.use_hct and self.hct else [], 0.5),
        #     (self.hct_to_tgn.parameters() if self.hct_to_tgn is not None and not isinstance(self.hct_to_tgn, nn.Identity) else [], 0.5),
        #     (self.fusion_layer.parameters(), 0.5),
        #     (self.embedding_module.parameters() if self.embedding_module else [], 0.5),
        #     (self.link_predictor.parameters() if self.link_predictor else [], 0.5),
        #     (self.message_fn.parameters() if self.message_fn else [], 0.5),
        #     (self._time_proj.parameters() if self._time_proj is not None and not isinstance(self._time_proj, nn.Identity) else [], 0.5),
        #     (self.walk_to_memory.parameters() if self.walk_to_memory is not None and not isinstance(self.walk_to_memory, nn.Identity) else [], 0.5),
        # ]

        #     for params, max_norm in param_groups:
        #         # Filter out empty parameter lists and ensure we have actual parameters
        #         params_list = list(params)
        #         if len(params_list) > 0:
        #             # Check if any parameter has a gradient
        #             has_grad = any(p.grad is not None for p in params_list)
        #             if has_grad:
        #                 try:
        #                     torch.nn.utils.clip_grad_norm_(params_list, max_norm=max_norm)
        #                 except Exception as e:
        #                     logger.warning(f"Gradient clipping failed for a parameter group: {e}")

        #     if DEBUG_VALIDATION:
        #         for name, param in self.named_parameters():
        #             if param.grad is not None and torch.isnan(param.grad).any():
        #                 logger.error(f"NaN gradient detected in {name}")
        #                 # Optionally zero the gradient to prevent propagation
        #                 param.grad.zero_()

    def on_train_batch_start(self, batch, batch_idx):
        batch_times = batch.get('timestamps', batch.get('ts', torch.tensor([0])))
        batch_min = float(batch_times.min())
        batch_max = float(batch_times.max())
        
        # Check SAM memory if SAM is used
        if self.use_sam and batch_idx > 0:
            mem_finite = torch.isfinite(self.sam_module.raw_memory).all().item()
            
            if not mem_finite:
                logger.error(f"Batch {batch_idx} START: SAM memory already contains NaN!")
                nan_count = (~torch.isfinite(self.sam_module.raw_memory)).sum().item()
                logger.error(f"NaN count in memory: {nan_count}/{self.sam_module.raw_memory.numel()}")
                self.sam_module.reset_memory()
        
        # Log epoch start info (only for batch 0)
        if batch_idx == 0:
            logger.info(f"Epoch start - Batch 0 time range: [{batch_min:.0f}, {batch_max:.0f}]")
        
        # Check temporal continuity
        if hasattr(self, 'prev_batch_time') and self.prev_batch_time is not None:
            if batch_max < self.prev_batch_time:
                logger.warning(f"Temporal reset detected: {batch_max} < {self.prev_batch_time}")
                # Reset ST-ODE state for new epoch
                if hasattr(self, 'st_ode') and self.st_ode is not None:
                    self.st_ode.reset_temporal_state()
        
        self.prev_batch_time = batch_max
        # Initialize _prev_max_time if it doesn't exist
        # if not hasattr(self, '_prev_max_time'):
        #     self._prev_max_time = -float('inf')
        
        # # Check for temporal violation (only after initialization)
        # if current_max < self._prev_max_time - 1e-6:
        #     logger.error(f"TEMPORAL VIOLATION: Batch {batch_idx} max {current_max:.0f} < prev {self._prev_max_time:.0f}")
        
        # # Handle epoch reset or continuous tracking
        # if batch_idx == 0 and self.current_epoch > 0:
        #     # Expected reset at epoch start - don't treat as violation
        #     self._prev_max_time = current_max  # Reset tracking
        # else:
        #     self._prev_max_time = max(self._prev_max_time, current_max)
        
        # Clear walk cache for new batch
        self._walk_cache = None
        self._cache_batch_key = None
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Optimized batch end with throttled ST-ODE."""
        if not self.use_sam:
            return
        
        # Periodic memory check (every 50 batches)
        if batch_idx % 50 == 0 and torch.isnan(self.sam_module.raw_memory).any():
            logger.error(f"Batch {batch_idx}: Memory NaN, resetting")
            self.sam_module.reset_memory()
        
        # 1. SAM DISCRETE UPDATE (always)
        if self.training and self._sam_batch_buffer is not None:
            buffer = self._sam_batch_buffer
            
            with torch.no_grad():
                try:
                    self.sam_module.update_memory_batch(
                        source_nodes=buffer['src_nodes'],
                        target_nodes=buffer['dst_nodes'],
                        edge_features=buffer['edge_features'],
                        current_time=buffer['timestamps'],
                        node_features=self.node_embedding.weight.detach()
                    )
                except Exception as e:
                    logger.error(f"SAM update failed: {e}")
                    self.sam_module.reset_memory()
            
            self._sam_batch_buffer = None
        
        # 2. ST-ODE CONTINUOUS EVOLUTION (throttled)
        if self.use_st_ode and self.st_ode is not None:
            self._st_ode_batch_counter += 1
            current_time = batch['timestamps'].max()
            time_delta = current_time - self.last_update_time
            
            # Run every 10 batches OR when time gap > 1000
            if self._st_ode_batch_counter >= 10 or time_delta > 1000:
                self._st_ode_batch_counter = 0
                
                if time_delta > 1e-6 and torch.isfinite(self.sam_module.raw_memory).all():
                    try:
                        obs_data = self._prepare_stode_observations(batch)
                        if obs_data is not None:
                            with torch.no_grad():
                                evolved = self.st_ode(
                                    node_states=self.sam_module.raw_memory,
                                    walk_encodings=obs_data['encodings'],
                                    walk_times=obs_data['times'],
                                    walk_masks=obs_data['masks'],
                                    adj_matrix=obs_data['adjs'],
                                )
                                
                                if torch.isfinite(evolved).all():
                                    self.sam_module.raw_memory.data.copy_(evolved)
                                    self.last_update_time = current_time
                    except Exception as e:
                        logger.warning(f"ST-ODE skipped: {e}")
                        self.last_update_time = current_time
        
    
    
    def on_train_epoch_start(self):
        """Reset ST-ODE temporal state at epoch start."""
        super().on_train_epoch_start()
        # if self.use_st_ode:
        #     self.last_update_time = torch.tensor(0.0, device=self.device)
        #     logger.info("ST-ODE last_update_time reset for new epoch")
        # Reset ST-ODE temporal state for new epoch
        self.prev_batch_time = 0
        if hasattr(self, 'st_ode') and self.st_ode is not None:
            self.st_ode.reset_temporal_state()
        logger.info(f"Epoch {self.current_epoch} start - Temporal state reset")

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self._walk_cache = None
        self._cache_batch_key = None
        self._sam_batch_buffer = None
        self._last_hct_info = None
        torch.cuda.empty_cache()
        logger.info("Epoch end: cleared all caches")
    
    def on_validation_epoch_start(self):
        """Clone SAM memory for validation."""
        super().on_validation_epoch_start()
        if self.use_sam and self.use_memory:
            self._sam_validation_memory = self.sam_module.raw_memory.clone().detach()
            self._sam_validation_last_update = self.sam_module.last_update.clone().detach()
            logger.info("Cloned SAM memory for validation")
    
    # def on_validation_epoch_end(self):
    #     """Restore SAM memory after validation."""
    #     super().on_validation_epoch_end()
    #     if self.use_sam and self.use_memory:
    #         self.sam_module.raw_memory.data.copy_(self._sam_validation_memory)
    #         self.sam_module.last_update.data.copy_(self._sam_validation_last_update)
    #         logger.info("Restored SAM memory after validation")
        
    #     if hasattr(self, '_last_hct_info') and self._last_hct_info is not None:
    #         try:
    #             for walk_type in ['short', 'long', 'tawr']:
    #                 if walk_type in self._last_hct_info:
    #                     cooc = self._last_hct_info[walk_type]['cooccurrence']
    #                     logger.info(f"HCT {walk_type}: cooc_mean={cooc.mean():.3f}")
    #         except (KeyError, TypeError, AttributeError) as e:
    #             logger.debug(f"Could not log HCT co-occurrence: {e}")
            
    #         self._last_hct_info = None
    
    def on_validation_epoch_end(self):
        """
        Find optimal temperature for calibration using validation set.
        """
        super().on_validation_epoch_end()
        
        # Restore SAM memory after validation (existing logic)
        if self.use_sam and self.use_memory:
            self.sam_module.raw_memory.data.copy_(self._sam_validation_memory)
            self.sam_module.last_update.data.copy_(self._sam_validation_last_update)
        
        # Temperature calibration (if using temperature scaling)
        if hasattr(self, '_val_logits_labels') and len(self._val_logits_labels) > 0:
            all_logits = torch.cat([x['logits'] for x in self._val_logits_labels])
            all_labels = torch.cat([x['labels'] for x in self._val_logits_labels])
            
            # Find temperature that minimizes BCE loss
            best_temp = 1.0
            best_loss = float('inf')
            
            for temp in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                scaled_logits = all_logits / temp
                loss = F.binary_cross_entropy_with_logits(scaled_logits, all_labels)
                if loss < best_loss:
                    best_loss = loss
                    best_temp = temp
            
            # Update model temperature
            if hasattr(self.link_predictor, 'temperature'):
                self.link_predictor.temperature.data.fill_(best_temp)
                logger.info(f"Validation: Optimal temperature = {best_temp:.2f} (loss={best_loss:.4f})")
            
            self._val_logits_labels = []

    def validation_step(self, batch, batch_idx):
        """Enhanced validation step with storage for calibration."""
        logits = self(batch)
        labels = batch['labels'].float()
        
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        
        # Standard metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ap', self._compute_ap(probs, labels), prog_bar=True)
        self.log('val_auc', self._compute_auc(probs, labels), prog_bar=True)
        
        # Store for temperature calibration
        if not hasattr(self, '_val_logits_labels'):
            self._val_logits_labels = []
        
        self._val_logits_labels.append({
            'logits': logits.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        # Check accuracy with current temperature
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Clean test step - only standard metrics for CSV export.
        """
        logits, probs = self(batch, return_probs=True)
        labels = batch['labels'].float()
        
        # Compute probabilities
        # probs = torch.sigmoid(logits)
        
        
        # Standard metrics
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        threshold = getattr(self, '_optimal_threshold', 0.5)
        preds = (probs > threshold).float()
        # preds = (probs > 0.5).float()
        accuracy = (preds == labels).float().mean()
        
        # Log standard metrics (aggregated by Lightning at epoch level)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Store predictions for AP/AUC calculation at epoch end
        if not hasattr(self, '_test_predictions'):
            self._test_predictions = []
        
        self._test_predictions.append({
            'probs': probs.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        # Silent diagnostics - only log once at start of testing
        if batch_idx == 0 and not hasattr(self, '_test_diagnostics_logged'):
            self._log_test_diagnostics(probs, logits, labels)
            self._test_diagnostics_logged = True
        
        return loss

    def on_test_epoch_start(self):
        """
        FIXED: Warm-start test evaluation to preserve learned temporal patterns.
        
        Cold-start (reset memory) causes catastrophic performance drop because
        the model loses all learned node representations.
        """
        super().on_test_epoch_start()
        
        if self.use_sam and self.use_memory:
            # CRITICAL FIX: Use warm-start instead of cold-start
            # Keep training memory state - do NOT reset
            logger.info("TEST: Warm-start evaluation (training memory preserved)")
            
            # # Verify memory is in good state
            # mem_finite = torch.isfinite(self.sam_module.raw_memory).all().item()
            # mem_mean = self.sam_module.raw_memory.mean().item()
            # mem_std = self.sam_module.raw_memory.std().item()
            
            # logger.info(f"TEST: Memory stats finite={mem_finite}, mean={mem_mean:.3f}, std={mem_std:.3f}")
            
            # if not mem_finite:
            #     logger.error("TEST: Memory contains NaN! Attempting recovery...")
            #     # Only reset if corrupted, otherwise keep training state
            #     # self.sam_module.reset_memory()
            #     # Try to warm-start with node features if available
            #     if hasattr(self, 'node_embedding'):
            #         self.sam_module.initialize_from_features(self.node_embedding.weight)

    def _log_test_diagnostics(self, probs, logits, labels):
        """Internal diagnostics - not metrics, just logs."""
        # Find best threshold for reference
        acc_05 = ((probs > 0.5).float() == labels).float().mean().item()
        
        # Try adaptive
        adaptive_thresh = probs.mean().item()
        acc_adaptive = ((probs > adaptive_thresh).float() == labels).float().mean().item()
        
        # Try balanced
        pos_ratio = labels.mean().item()
        balanced_thresh = probs.quantile(1 - pos_ratio).item() if pos_ratio > 0 else 0.5
        acc_balanced = ((probs > balanced_thresh).float() == labels).float().mean().item()
        
        best_acc = max(acc_05, acc_adaptive, acc_balanced)
        best_method = 'standard' if best_acc == acc_05 else ('adaptive' if best_acc == acc_adaptive else 'balanced')
        
        logger.info(f"Test calibration check - Best: {best_method}={best_acc:.3f} "
                    f"(std={acc_05:.3f}, adapt={acc_adaptive:.3f}, bal={acc_balanced:.3f})")
    
    def on_test_epoch_end(self):
        """
        Compute and log AP/AUC - the final two standard metrics.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        if hasattr(self, '_test_predictions') and len(self._test_predictions) > 0:
            # Aggregate all predictions
            all_probs = torch.cat([p['probs'] for p in self._test_predictions])
            all_labels = torch.cat([p['labels'] for p in self._test_predictions])
            
            # Convert to numpy for sklearn
            probs_np = all_probs.numpy()
            labels_np = all_labels.numpy()
            
            # Compute final metrics
            ap = average_precision_score(labels_np, probs_np)
            auc = roc_auc_score(labels_np, probs_np)

            adaptive_thresh = all_probs.mean().item()  # or use validation-optimal
            preds_adaptive = (all_probs > adaptive_thresh).float()
            acc_adaptive = (preds_adaptive == all_labels).float().mean()
            
            # Log final metrics (completes the 4 standard metrics)
            self.log('test_ap', ap, prog_bar=True)
            self.log('test_auc', auc, prog_bar=True)
            
            # Clean summary
            logger.info(f"Test: AP={ap:.4f}, AUC={auc:.4f}, " 
                        f"Acc@0.5={(all_probs>0.5).float().eq(all_labels).float().mean():.3f}, "
                        f"Acc@adaptive={acc_adaptive:.3f}")
            
            # Cleanup
            self._test_predictions = []
            if hasattr(self, '_test_diagnostics_logged'):
                delattr(self, '_test_diagnostics_logged')


    def _compute_ap(self, probs, labels):
        """Compute Average Precision."""        
        return average_precision_score(labels.cpu().numpy(), probs.cpu().numpy())

    def _compute_auc(self, probs, labels):
        """Compute AUC."""        
        return roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy())


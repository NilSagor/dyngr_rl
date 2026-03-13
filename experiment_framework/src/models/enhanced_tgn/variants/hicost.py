import os
import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import lightning as L

from ..component.time_encoder import TimeEncoder
from ..component.sam_module import StabilityAugmentedMemory
from ..component.multi_swalk import MultiScaleWalkSampler
# from ..component.walk_encoder import WalkEncoder
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


class HiCoST(L.LightningModule):
    """
     HiCoST: Hierarchical Co-occurrence Spatio-Temporal Network
     hierarchical co‑occurrence + spectral‑temporal ODE
     SAM + Multi-Scale Walk Sampler + Hierarchical Co-occurrence Transformer (HCT) + ST ODE + MutulRefinementPooling
    
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
        edge_features_dim: int = 172,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_heads: int = 2,
        # === REMOVED: All TGN-specific params (n_neighbors, embedding_module_type, etc.) ===
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
        rtol: float = 1e-6,
        atol: float = 1e-8,
        adjoint: bool = True,
        use_st_ode: bool = True,
        directed: bool = False,
        **kwargs
    ):        
        super().__init__()
        # torch.autograd.set_detect_anomaly(True)
        if os.environ.get('DEBUG_GRADIENTS'):
            torch.autograd.set_detect_anomaly(True)
        
        
        # self.directed = kwargs.get('directed', False)

        # Core dimensions
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.time_encoding_dim = time_encoding_dim
        self.edge_features_dim = edge_features_dim
        self.dropout = dropout
        self.directed = directed
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # Feature flags (cleaned up)
        self.use_sam = parse_bool(use_sam)
        self.use_hct = parse_bool(use_hct)
        self.use_st_ode = parse_bool(use_st_ode)
        self.debug_simple_walk = parse_bool(debug_simple_walk)
        
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # ST-ODE tracking
        self.st_ode_update_interval = 10
        self._st_ode_batch_counter = 0
        self.register_buffer('last_update_time', torch.tensor(0.0))
        
        self.save_hyperparameters()
        
        # === Time Encoder ===
        self.time_encoder = TimeEncoder(time_encoding_dim)
        
        
        
        # === SAM Memory (primary memory - no TGN fallback) ===
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
            logger.info(f"HiCoST initialized: SAM({num_prototypes}p) + WalkSampler + "
                        f"{'HCT(' + str(hct_d_model) + 'd, ' + str(hct_nhead) + 'h)' if use_hct else 'SimpleWalkEncoder'}")
        else:
            raise ValueError("SAM must be enabled (no TGN fallback available)")
        

        # === Multi-Scale Walk Sampler ===
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
        
        # === Walk Processor (HCT or Simple Mean Pooling) ===
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
            # Project HCT output to hidden_dim if needed
            if hct_d_model != hidden_dim:
                self.hct_to_hidden = nn.Sequential(
                    nn.Linear(hct_d_model, hidden_dim),
                    nn.LayerNorm(hidden_dim, eps=1e-4),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            else:
                self.hct_to_hidden = nn.Identity()
            logger.info("HCT enabled as primary walk encoder")
        else:
            self.hct = None
            self.hct_to_hidden = None
            # Simple walk projection (if needed)
            if memory_dim != hidden_dim:
                self.simple_walk_proj = nn.Linear(memory_dim, hidden_dim)
            else:
                self.simple_walk_proj = nn.Identity()
            logger.info("Using simple walk mean pooling (HCT disabled)")
        
        # === ST-ODE (only if SAM is enabled) ===
        if self.use_st_ode:
            # Time projection for ST-ODE (match memory dim)
            if time_encoding_dim != memory_dim:
                self._time_proj = nn.Linear(time_encoding_dim, memory_dim)
                nn.init.xavier_uniform_(self._time_proj.weight)
                nn.init.zeros_(self._time_proj.bias)
            else:
                self._time_proj = nn.Identity()
            
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
            )
            # Project walk embeddings to memory dim for ST-ODE if needed
            if self.use_hct and hct_d_model != memory_dim:
                self.walk_to_memory = nn.Linear(hct_d_model, memory_dim)
            else:
                self.walk_to_memory = nn.Identity()
        else:
            self.st_ode = None
            self._time_proj = None
            self.walk_to_memory = None

        
        # === Mutual Refinement and Pooling ===
        self.mutual_refine = MutualRefineAndPooling(
            d_model=hidden_dim,
            nhead=n_heads,
        )
        
        
        # self.edge_index = None
        # self.edge_time = None

        # self._time_proj = None

        # self._walk_cache = {}
        # self._cache_walks = False  # Toggle        
        
        # === Link Predictor (with temperature scaling) ===
        self.link_predictor = MergeLayer(
            input_dim1=hidden_dim,
            input_dim2=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
            use_temperature=True
        )
        
        # === Cache/Buffer Initialization ===
        self.edge_index = None
        self.edge_time = None
        self._walk_cache = None
        self._cache_batch_key = None
        self._sam_batch_buffer = None
        self.prev_batch_time = 0
        
        
        # Log initialization
        if edge_features_dim == 0:
            logger.warning("Edge features disabled (dim=0). SAM will receive zero edge input.")
        else:
            logger.info(f"Edge features enabled: input_dim={edge_features_dim}, projected_dim={memory_dim}")
    
        logger.info(f"HiCoST initialized: SAM({num_prototypes}p) + WalkSampler + "
                   f"{'HCT(' + str(hct_d_model) + 'd, ' + str(hct_nhead) + 'h)' if use_hct else 'SimpleWalkEncoder'}")
    
    def _get_batch_key(self, batch):
        """Generate cache key from batch data."""
        src_nodes = batch['src_nodes']
        timestamps = batch['timestamps'] if 'timestamps' in batch else batch.get('ts', torch.tensor([]))
        
        # Sample first 5 elements for key (or all if smaller)
        src_sample = src_nodes[:min(5, len(src_nodes))]
        ts_sample = timestamps[:min(5, len(timestamps))] if len(timestamps) > 0 else torch.tensor([0.0])
        
        # Convert to numpy for hashing
        src_bytes = src_sample.cpu().numpy().tobytes()
        ts_bytes = ts_sample.cpu().numpy().tobytes()
        
        return (src_bytes, ts_bytes, len(src_nodes))  
    
    def set_raw_features(self, node_features: Optional[torch.Tensor], edge_features: Optional[torch.Tensor]):
        """
        Store raw node and edge features (required for compatibility with training pipeline).
        
        Args:
            node_features: Tensor of shape [num_nodes, node_feat_dim] (node features)
            edge_features: Tensor of shape [num_edges, edge_feat_dim] (edge features)
        """
        # Store raw features as instance variables (matching TGN variant naming)
        self.node_raw_features = node_features
        self.edge_raw_features = edge_features
        
        # Move features to the model's device
        if node_features is not None:
            self.node_raw_features = self.node_raw_features.to(self.device)
        if edge_features is not None:
            self.edge_raw_features = self.edge_raw_features.to(self.device)
        
        # Optional: Update SAM module with node features if needed
        if self.use_sam and hasattr(self.sam_module, 'set_node_features'):
            self.sam_module.set_node_features(self.node_raw_features)
        
        logger.debug(f"Set raw features: node_feat_dim={self.node_raw_features.shape[-1] if self.node_raw_features is not None else 0}, "
                    f"edge_feat_dim={self.edge_raw_features.shape[-1] if self.edge_raw_features is not None else 0}")

    
    
    
    def _prepare_stode_observations(self, batch):
        """Convert batch interactions to ST-ODE observation format (vectorized)."""
        if not self.use_st_ode or not self.use_sam:
            return None
        
        device = self.device
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']
        
        # Get embeddings from SAM memory
        src_emb = self.sam_module.raw_memory[src_nodes]
        dst_emb = self.sam_module.raw_memory[dst_nodes]
        time_emb = self.time_encoder(timestamps.float())
        
        # Project time embedding if needed
        if self._time_proj is not None and not isinstance(self._time_proj, nn.Identity):
            time_emb = self._time_proj(time_emb)
        
        # Build sparse adjacency matrix
        adj_t = self._build_temporal_adjacency(src_nodes, dst_nodes, self.num_nodes)
        
        # Vectorized time aggregation
        sorted_ts, sort_idx = torch.sort(timestamps)
        sorted_src = src_nodes[sort_idx]
        sorted_dst = dst_nodes[sort_idx]
        sorted_src_emb = src_emb[sort_idx] + time_emb[sort_idx]
        sorted_dst_emb = dst_emb[sort_idx] + time_emb[sort_idx]
        
        # Find time boundaries
        time_diff = torch.diff(sorted_ts, prepend=sorted_ts[:1])
        time_boundaries = torch.where(time_diff > 0)[0]
        T = len(time_boundaries) + 1
        
        # Scatter add for efficient aggregation
        node_obs = torch.zeros(T, self.num_nodes, self.memory_dim, device=device)
        counts = torch.zeros(T, self.num_nodes, device=device)
        
        segment_ids = torch.searchsorted(time_boundaries, torch.arange(len(sorted_ts), device=device))
        flat_idx_src = segment_ids * self.num_nodes + sorted_src
        flat_idx_dst = segment_ids * self.num_nodes + sorted_dst
        
        node_obs_flat = node_obs.view(-1, self.memory_dim)
        counts_flat = counts.view(-1)
        
        node_obs_flat.scatter_add_(0, flat_idx_src.unsqueeze(-1).expand(-1, self.memory_dim), sorted_src_emb)
        node_obs_flat.scatter_add_(0, flat_idx_dst.unsqueeze(-1).expand(-1, self.memory_dim), sorted_dst_emb)
        counts_flat.scatter_add_(0, flat_idx_src, torch.ones_like(sorted_src, dtype=torch.float))
        counts_flat.scatter_add_(0, flat_idx_dst, torch.ones_like(sorted_dst, dtype=torch.float))
        
        # Normalize
        node_obs = node_obs / counts.unsqueeze(-1).clamp(min=1)
        
        # Reshape for ST-ODE
        walk_encodings = node_obs.permute(1, 0, 2).unsqueeze(2)
        unique_times = torch.cat([sorted_ts[:1], sorted_ts[time_boundaries]])
        walk_times = unique_times.view(1, T, 1).expand(self.num_nodes, -1, -1)
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
    
    def _simple_walk_embed(self, walk_data_side: Dict, node_memory) -> torch.Tensor:
        """Fallback walk embedding: mean of all node features from all walks."""
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
            return torch.zeros(len(node_memory), self.hidden_dim, device=node_memory.device)
        
        # Average across walk types and project to hidden dim
        pooled = torch.stack(all_embs, dim=0).mean(dim=0)
        pooled = self.simple_walk_proj(pooled)
        
        return pooled
    
    def _prepare_per_type_embeddings(
        self, 
        walk_data_side: Dict, 
        node_memory: torch.Tensor,
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare per-type walk embeddings for MutualRefineAndPooling."""
        if not walk_data_side:
            return None, None
        
        type_names = ['short', 'long', 'tawr']
        type_embeds = []
        type_masks = []
        
        max_walks = self.walk_sampler.num_walks_short  # Reference max walks
        
        for wt in type_names:
            if wt in walk_data_side:
                nodes = walk_data_side[wt]['nodes']      # [B, num_walks, walk_len]
                masks = walk_data_side[wt]['masks']      # [B, num_walks, walk_len]
                
                # Average over walk length
                B, num_walks, walk_len = nodes.shape
                flat_nodes = nodes.reshape(-1)
                flat_feats = node_memory[flat_nodes]     # [B*num_walks*walk_len, D]
                feats = flat_feats.view(B, num_walks, walk_len, -1)
                
                # Masked average
                walk_masks = masks.any(dim=-1)  # [B, num_walks]
                walk_embeds = (feats * masks.unsqueeze(-1)).sum(dim=2) / masks.sum(dim=2, keepdim=True).clamp(min=1)
                
                # Pad/truncate to max walks
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
                # Empty placeholder
                B = len(node_memory)
                D = node_memory.size(-1)
                type_embeds.append(torch.zeros(B, max_walks, D, device=device))
                type_masks.append(torch.zeros(B, max_walks, dtype=torch.bool, device=device))
        
        # Stack to [B, num_types, max_walks, D]
        per_type_embeds = torch.stack(type_embeds, dim=1)
        masks = torch.stack(type_masks, dim=1)
        
        return per_type_embeds, masks
    
    def compute_temporal_embeddings(
        self,
        source_nodes: torch.Tensor,
        destination_nodes: torch.Tensor,
        edge_times: torch.Tensor,
        batch: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Clean embedding computation (no Base TGN embeddings):
        1. Get SAM memory
        2. Generate multi-scale walks
        3. Encode walks with HCT (or simple pooling)
        4. Apply mutual refinement
        5. Return final source/destination embeddings
        """
        device = self.device
        
        # 1. Get SAM memory (detach to prevent gradient leakage during walk sampling)
        node_memory = self.sam_module.raw_memory.detach()
        
        # Validate memory (critical for stability)
        if not torch.isfinite(node_memory).all():
            logger.error("SAM memory contains NaN/Inf! Resetting...")
            self.sam_module.reset_memory()
            node_memory = self.sam_module.raw_memory.detach()
        
        # Fix NaN in timestamps
        edge_times = torch.nan_to_num(edge_times, nan=0.0)
        
        # Convert to proper device/dtype
        src_tensor = source_nodes.to(device).long()
        dst_tensor = destination_nodes.to(device).long()
        ts_tensor = edge_times.to(device).float()
        batch_size = src_tensor.size(0)
        
        # 2. Generate walks (cached if possible)
        cache_key = self._get_batch_key(batch) if batch is not None else None
        if cache_key == self._cache_batch_key and self._walk_cache is not None:
            walk_data = self._walk_cache
        else:
            with torch.no_grad():
                walk_data = self.walk_sampler(
                    source_nodes=src_tensor,
                    target_nodes=dst_tensor,
                    current_times=ts_tensor,
                    memory_states=node_memory,
                    edge_index=None,
                    edge_time=None
                )
            self._walk_cache = walk_data
            self._cache_batch_key = cache_key
        
        # 3. Validate walk data
        for side in ['source', 'target']:
            for wt in ['short', 'long', 'tawr']:
                if wt in walk_data[side]:
                    nodes = walk_data[side][wt]['nodes']
                    if not torch.isfinite(nodes).all():
                        logger.error(f"NaN in {side}/{wt} walk nodes! Sanitizing...")
                        walk_data[side][wt]['nodes'] = torch.nan_to_num(nodes, nan=0.0)
                    nodes.clamp_(0, self.num_nodes - 1)
        
        # 4. Encode walks (HCT or simple pooling)
        src_per_type = None
        dst_per_type = None
        src_masks = None
        dst_masks = None
        
        if self.use_hct and not self.debug_simple_walk:
            # Combine source/target walks for batch processing
            combined_walks = {
                'short': {
                    'nodes': torch.cat([walk_data['source']['short']['nodes'], walk_data['target']['short']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['short']['nodes_anon'], walk_data['target']['short']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['short']['masks'], walk_data['target']['short']['masks']], dim=0),
                },
                'long': {
                    'nodes': torch.cat([walk_data['source']['long']['nodes'], walk_data['target']['long']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['long']['nodes_anon'], walk_data['target']['long']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['long']['masks'], walk_data['target']['long']['masks']], dim=0),
                },
                'tawr': {
                    'nodes': torch.cat([walk_data['source']['tawr']['nodes'], walk_data['target']['tawr']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['tawr']['nodes_anon'], walk_data['target']['tawr']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['tawr']['masks'], walk_data['target']['tawr']['masks']], dim=0),
                }
            }
            
            # Run HCT
            hct_output = self.hct(
                walks_dict=combined_walks,
                node_memory=node_memory,
                return_all=False
            )
            
            # Split back to source/target and project to hidden dim
            hct_src = hct_output[:batch_size]
            hct_dst = hct_output[batch_size:]
            walk_src_emb = self.hct_to_hidden(hct_src)
            walk_dst_emb = self.hct_to_hidden(hct_dst)
            
            # Prepare per-type embeddings for mutual refinement
            src_per_type, src_masks = self._prepare_per_type_embeddings(walk_data['source'], node_memory, device)
            dst_per_type, dst_masks = self._prepare_per_type_embeddings(walk_data['target'], node_memory, device)
            
            # Project per-type embeddings to hidden dim if needed
            if src_per_type is not None and src_per_type.size(-1) != self.hidden_dim:
                if not hasattr(self, '_per_type_proj'):
                    self._per_type_proj = nn.Linear(src_per_type.size(-1), self.hidden_dim).to(device)
                src_per_type = self._per_type_proj(src_per_type)
                dst_per_type = self._per_type_proj(dst_per_type)
        else:
            # Simple mean pooling fallback
            walk_src_emb = self._simple_walk_embed(walk_data['source'], node_memory)
            walk_dst_emb = self._simple_walk_embed(walk_data['target'], node_memory)
        
        # 5. Mutual Refinement (walk embeddings only - no TGN base embeddings)
        combined_walk = torch.cat([walk_src_emb, walk_dst_emb], dim=0)
        
        # Prepare per-type embeddings for refinement
        if src_per_type is not None and dst_per_type is not None:
            combined_per_type = torch.cat([src_per_type, dst_per_type], dim=0)
            combined_masks = torch.cat([src_masks, dst_masks], dim=0).bool()
        else:
            combined_per_type = None
            combined_masks = None
        
        # Run mutual refinement (walk embeddings self-refinement)
        refined_walk, _ = self.mutual_refine(
            src_walk=combined_walk,
            dst_walk=combined_walk,  # Self-refinement (no TGN base to refine with)
            src_per_type=combined_per_type,
            dst_per_type=None,
            src_masks=combined_masks,
            dst_masks=None,
        )
        
        # Split back to source/destination
        final_src = refined_walk[:batch_size]
        final_dst = refined_walk[batch_size:]
        
        # Final NaN protection
        final_src = torch.nan_to_num(final_src, nan=0.0, posinf=10.0, neginf=-10.0)
        final_dst = torch.nan_to_num(final_dst, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return final_src, final_dst
    
    def forward(self, batch: Dict[str, torch.Tensor], return_probs: bool = False) -> torch.Tensor:
        """Clean forward pass (no TGN base logic)."""
        # Compute embeddings
        source_emb, dest_emb = self.compute_temporal_embeddings(
            batch['src_nodes'],
            batch['dst_nodes'],
            batch['timestamps'],
            batch=batch,
        )
        
        # Store interactions for SAM update (training only)
        if self.training and self.use_sam:
            self._store_sam_interactions(batch)
        
        # Link prediction logits
        logits = self.link_predictor(source_emb, dest_emb).squeeze(-1)
        
        # Return probabilities if requested (with temperature scaling)
        if return_probs:
            if hasattr(self.link_predictor, 'temperature') and self.link_predictor.use_temperature:
                temp = self.link_predictor.temperature.clamp(min=0.1, max=10.0)
                scaled_logits = logits / temp
            else:
                scaled_logits = logits
            probs = torch.sigmoid(scaled_logits)
            return logits, probs
        
        # NaN check
        if torch.isnan(logits).any():
            logger.error(f"NaN in logits! source_emb NaN: {torch.isnan(source_emb).any()}, dest_emb NaN: {torch.isnan(dest_emb).any()}")
        
        return logits
    
    def _store_sam_interactions(self, batch: Dict[str, torch.Tensor]):
        """Store interactions in buffer for deferred SAM update."""
        device = self.device
        src_nodes = batch['src_nodes'].to(device)
        dst_nodes = batch['dst_nodes'].to(device)
        timestamps = batch['timestamps'].to(device)
        
        # Handle edge features (zero if not available)
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
        """Initialize walk sampler from NeighborFinder."""
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
        """Provide full graph to walk sampler."""
        self.edge_index = edge_index
        self.edge_time = edge_time
        self.walk_sampler.update_neighbors(edge_index, edge_time)
        self.walk_sampler.build_dense_neighbor_table()
        self.walk_sampler._freeze_neighbors = True
        logger.info(f"Walk sampler initialized with {edge_index.size(1)} edges")
    
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Return SAM memory for given nodes."""
        return self.sam_module.get_memory(node_ids)
    
    # === Training/Validation/Test Steps (Cleaned - No TGN Logic) ===
    def training_step(self, batch, batch_idx):
        """Training step with SAM/ST-ODE updates (no TGN memory)."""
        # Validate SAM memory
        if not torch.isfinite(self.sam_module.raw_memory).all():
            logger.error(f"Batch {batch_idx}: SAM memory NaN! Resetting...")
            self.sam_module.reset_memory()
        
        # Forward pass
        source_emb, dest_emb = self.compute_temporal_embeddings(
            batch['src_nodes'], batch['dst_nodes'], batch['timestamps'], batch=batch
        )
        scores = self.link_predictor(source_emb, dest_emb).squeeze(-1)
        
        # NaN check
        if torch.isnan(scores).any():
            logger.error(f"Batch {batch_idx}: NaN scores detected!")
            return None
        
        # Loss calculation
        loss = F.binary_cross_entropy_with_logits(scores, batch['labels'].float())
        
        # Store interactions for SAM update
        if self.training:
            self._store_sam_interactions(batch)
        
        # Log loss
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with temperature calibration support."""
        logits = self(batch)
        labels = batch['labels'].float()
        
        # Loss calculation
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        
        # Log standard metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ap', average_precision_score(labels.cpu().numpy(), probs.cpu().numpy()), prog_bar=True)
        self.log('val_auc', roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy()), prog_bar=True)
        
        # Store for temperature calibration
        if not hasattr(self, '_val_logits_labels'):
            self._val_logits_labels = []
        self._val_logits_labels.append({
            'logits': logits.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        # Accuracy with 0.5 threshold
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step with standard metrics (AP/AUC)."""
        logits, probs = self(batch, return_probs=True)
        labels = batch['labels'].float()
        
        # Loss calculation
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Log basic metrics
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Store predictions for epoch-level AP/AUC calculation
        if not hasattr(self, '_test_predictions'):
            self._test_predictions = []
        self._test_predictions.append({
            'probs': probs.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        # Diagnostics (first batch only)
        if batch_idx == 0 and not hasattr(self, '_test_diagnostics_logged'):
            self._log_test_diagnostics(probs, logits, labels)
            self._test_diagnostics_logged = True
        
        return loss
    
    def configure_optimizers(self):
        """
        Required by Lightning: Define optimizer and learning rate scheduler (if needed).
        Matches the learning rate/weight decay from your __init__ params.
        """
        # Create optimizer (AdamW is recommended over Adam for stability)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Optional: Add learning rate scheduler (uncomment if needed)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     verbose=True
        # )
        
        # Return optimizer (or optimizer + scheduler)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val_loss'  # Monitor validation loss for scheduling
            # }
        }


    # === Callback Methods (Cleaned - No TGN Logic) ===
    def on_after_optimizer_step(self, optimizer):
        """Gradient clipping and NaN check after optimizer step."""
        # Global gradient clipping (stable)
        all_params = [p for p in self.parameters() if p.grad is not None]
        if all_params:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        
        # NaN parameter check
        if DEBUG_VALIDATION:
            for name, param in self.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    logger.error(f"NaN gradient in {name}! Zeroing...")
                    param.grad.zero_()
    
    def on_train_batch_start(self, batch, batch_idx):
        """Batch start setup (temporal checks + cache clearing)."""
        # Temporal continuity check
        batch_times = batch.get('timestamps', torch.tensor([0]))
        batch_max = float(batch_times.max())
        
        if batch_max < self.prev_batch_time:
            logger.warning(f"Temporal reset detected: {batch_max} < {self.prev_batch_time}")
            if self.st_ode is not None:
                self.st_ode.reset_temporal_state()
        
        self.prev_batch_time = batch_max
        
        # Clear walk cache
        self._walk_cache = None
        self._cache_batch_key = None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Batch end processing (SAM/ST-ODE updates)."""
        if not self.use_sam:
            return
        
        # 1. SAM memory update
        if self.training and self._sam_batch_buffer is not None:
            buffer = self._sam_batch_buffer
            with torch.no_grad():
                try:
                    self.sam_module.update_memory_batch(
                        source_nodes=buffer['src_nodes'],
                        target_nodes=buffer['dst_nodes'],
                        edge_features=buffer['edge_features'],
                        current_time=buffer['timestamps'],
                        node_features=self.node_raw_features if hasattr(self, 'node_raw_features') else None  # Adjust if node features exist
                    )
                except Exception as e:
                    logger.error(f"SAM update failed: {e}")
                    self.sam_module.reset_memory()
            self._sam_batch_buffer = None
        
        # 2. ST-ODE update (throttled for efficiency)
        if self.use_st_ode and self.st_ode is not None:
            self._st_ode_batch_counter += 1
            current_time = batch['timestamps'].max()
            time_delta = current_time - self.last_update_time
            
            if self._st_ode_batch_counter >= self.st_ode_update_interval or time_delta > 1000:
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
                        logger.warning(f"ST-ODE update skipped: {e}")
                        self.last_update_time = current_time
    
    def on_train_epoch_start(self):
        """Epoch start setup (temporal state reset)."""
        self.prev_batch_time = 0
        if self.st_ode is not None:
            self.st_ode.reset_temporal_state()
        logger.info(f"Epoch start - Temporal state reset")
    
    def on_train_epoch_end(self):
        """Epoch end cleanup (cache clearing)."""
        self._walk_cache = None
        self._cache_batch_key = None
        self._sam_batch_buffer = None
        torch.cuda.empty_cache()
        logger.info("Epoch end: Cleared all caches")
    
    def on_validation_epoch_start(self):
        """Validation start (SAM memory clone)."""
        if self.use_sam:
            self._sam_validation_memory = self.sam_module.raw_memory.clone().detach()
            self._sam_validation_last_update = self.sam_module.last_update.clone().detach()
            logger.info("Cloned SAM memory for validation")
    
    def on_validation_epoch_end(self):
        """Validation end (SAM memory restore + temperature calibration)."""
        # Restore SAM memory
        if self.use_sam:
            self.sam_module.raw_memory.data.copy_(self._sam_validation_memory)
            self.sam_module.last_update.data.copy_(self._sam_validation_last_update)
        
        # Temperature calibration
        if hasattr(self, '_val_logits_labels') and len(self._val_logits_labels) > 0:
            all_logits = torch.cat([x['logits'] for x in self._val_logits_labels])
            all_labels = torch.cat([x['labels'] for x in self._val_logits_labels])
            
            # Find optimal temperature
            best_temp = 1.0
            best_loss = float('inf')
            for temp in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                scaled_logits = all_logits / temp
                loss = F.binary_cross_entropy_with_logits(scaled_logits, all_labels)
                if loss < best_loss:
                    best_loss = loss
                    best_temp = temp
            
            # Update link predictor temperature
            if hasattr(self.link_predictor, 'temperature'):
                self.link_predictor.temperature.data.fill_(best_temp)
                logger.info(f"Optimal validation temperature: {best_temp:.2f} (loss={best_loss:.4f})")
            
            self._val_logits_labels = []
    
    def on_test_epoch_start(self):
        """Test start (warm-start SAM memory)."""
        logger.info("Test: Warm-start evaluation (preserving SAM memory from training)")
        
        # Validate memory state
        if not torch.isfinite(self.sam_module.raw_memory).all():
            logger.error("Test: SAM memory contains NaN!")
    
    def on_test_epoch_end(self):
        """Test end (compute final AP/AUC)."""
        if hasattr(self, '_test_predictions') and len(self._test_predictions) > 0:
            # Aggregate predictions
            all_probs = torch.cat([p['probs'] for p in self._test_predictions])
            all_labels = torch.cat([p['labels'] for p in self._test_predictions])
            
            # Compute final metrics
            ap = average_precision_score(all_labels.numpy(), all_probs.numpy())
            auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
            
            # Log final metrics
            self.log('test_ap', ap, prog_bar=True)
            self.log('test_auc', auc, prog_bar=True)
            
            # Adaptive accuracy
            adaptive_thresh = all_probs.mean().item()
            acc_adaptive = (all_probs > adaptive_thresh).float().eq(all_labels).float().mean()
            
            logger.info(f"Test Results: AP={ap:.4f}, AUC={auc:.4f}, Acc@0.5={(all_probs>0.5).float().eq(all_labels).float().mean():.3f}, Acc@adaptive={acc_adaptive:.3f}")
            
            # Cleanup
            self._test_predictions = []
            if hasattr(self, '_test_diagnostics_logged'):
                delattr(self, '_test_diagnostics_logged')
    
    # === Helper Methods ===
    def _log_test_diagnostics(self, probs, logits, labels):
        """Log test calibration diagnostics."""
        acc_05 = ((probs > 0.5).float() == labels).float().mean().item()
        adaptive_thresh = probs.mean().item()
        acc_adaptive = ((probs > adaptive_thresh).float() == labels).float().mean().item()
        pos_ratio = labels.mean().item()
        balanced_thresh = probs.quantile(1 - pos_ratio).item() if pos_ratio > 0 else 0.5
        acc_balanced = ((probs > balanced_thresh).float() == labels).float().mean().item()
        
        best_acc = max(acc_05, acc_adaptive, acc_balanced)
        best_method = 'standard' if best_acc == acc_05 else ('adaptive' if best_acc == acc_adaptive else 'balanced')
        
        logger.info(f"Test Calibration: Best={best_method} ({best_acc:.3f}), Standard={acc_05:.3f}, Adaptive={acc_adaptive:.3f}, Balanced={acc_balanced:.3f}")

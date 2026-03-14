import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score
import lightning as L

from torch_geometric.utils import get_laplacian

# --- Import Your Optimized Components ---
# Ensure these files are in your project structure
from ..component.time_encoder import TimeEncoder
from ..component.sam_modulev2 import RobustStabilityAugmentedMemory  # The enhanced version
from ..component.multi_swalkv2 import MultiScaleWalkSampler          # The enhanced version
from ..component.hct_modulev2 import StabilizedHCT                   # The enhanced version
from ..component.stode_modulev2 import NumericallyStabilizedSTODE    # The enhanced version
from ..component.mrp_modulev2 import GatedMutualRefinementPooling    # The enhanced version
from ..component.hardnegative_module import HardNegativeMiner # The enhanced version
from ..component.transformer_encoder import MergeLayer             # Standard link predictor head

def parse_bool(value):
    if isinstance(value, bool): return value
    if isinstance(value, str): return value.lower() in ('true', '1', 'yes')
    return bool(value)

class HiCoSTv2(L.LightningModule):
    """
    HiCoST: Hierarchical Co-occurrence Spatio-Temporal Network (Production Version)
    
    Architecture Flow:
    1. Input Graph -> MultiScaleWalkSampler (Vectorized, Adaptive, Noisy)
    2. Walks -> StabilizedHCT (Causal, Temporal Co-occurrence, DropPath)
    3. Memory -> RobustStabilityAugmentedMemory (Spectral Norm, Residual, Decay)
    4. Temporal -> NumericallyStabilizedSTODE (Velocity Gating, Low-Rank Spectral)
    5. Fusion -> GatedMutualRefinementPooling (Tri-Modal Gating, Vectorized Pooling)
    6. Loss -> HardNegativeMiner + Label Smoothing + Cosine LR
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int = 0,
        edge_feat_dim: int = 64,
        hidden_dim: int = 172,
        time_dim: int = 64,
        memory_dim: int = 172,
        # Learning Params
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: float = 0.1,
        min_lr_ratio: float = 0.01,
        # SAM Params (Robust)
        num_prototypes: int = 5,
        sam_residual_alpha: float = 0.8,
        sam_time_decay: float = 0.99,
        # Walk Sampler Params
        debug_simple_walk: bool = False,
        walk_length_short: int = 3,
        walk_length_long: int = 10,
        walk_length_tawr: int = 8,
        num_walks_short: int = 5,
        num_walks_long: int = 3,
        num_walks_tawr: int = 3,
        walk_temperature: float = 0.1,
        walk_time_noise_std: float = 0.0,
        walk_mask_prob: float = 0.05,
        walk_adaptive_factor: float = 0.5,
        # HCT Params (Stabilized)
        use_hct: bool = True,
        hct_d_model: int = 128,
        hct_nhead: int = 4,
        hct_num_layers: int = 2,
        hct_drop_path: float = 0.1,
        hct_sigma_time: float = 5.0,
        # ST-ODE Params (Stabilized)
        use_st_ode: bool = True,
        ode_method: str = "dopri5",
        ode_mu: float = 0.1,
        ode_num_eig: int = 10,
        ode_velocity_gate_thresh: float = 5.0,
        st_ode_update_interval: int = 10,
        # Hard Negative Mining
        neg_sample_ratio: int = 5,
        hard_neg_threshold: float = 0.7,
        label_smoothing: float = 0.1,
        dropout: float = 0.1,
        # Ablation Flags (NEW)
        ablation_sam_prototypes: bool = True,  # Enable SAM prototype mechanism
        ablation_hct_hierarchical: bool = True, # Enable HCT hierarchical co-occurrence
        ablation_mrp_gating: bool = True,       # Enable MRP gated refinement
        ablation_walk_multi_scale: bool = True, # Enable multi-scale walks (else single scale)
        **kwargs
    ):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.save_hyperparameters()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.edge_feat_dim = edge_feat_dim
        self.use_st_ode = parse_bool(use_st_ode)
        self.st_ode_update_interval = st_ode_update_interval
        self.use_hct = use_hct
        self.debug_simple_walk = debug_simple_walk
        
        # State Tracking
        self.register_buffer('last_ode_update_time', torch.tensor(0.0))
        self._ode_batch_counter = 0
        self._sam_epoch_backup = None

         # Store ablation flags (NEW)
        self.ablation_sam_prototypes = ablation_sam_prototypes
        self.ablation_hct_hierarchical = ablation_hct_hierarchical
        self.ablation_mrp_gating = ablation_mrp_gating
        self.ablation_walk_multi_scale = ablation_walk_multi_scale

        
        # === 1. Time Encoder ===
        self.time_encoder = TimeEncoder(time_dim)
        
        # === Robust Stability Augmented Memory (SAM) ===
        # === 2. SAM Module (Ablation: Prototypes) ===
        # If ablating prototypes, force num_prototypes=1 and effectively disable attention
        effective_prototypes = num_prototypes if ablation_sam_prototypes else 1
        self.sam_module = RobustStabilityAugmentedMemory(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            time_dim=time_dim,
            num_prototypes=effective_prototypes,
            residual_alpha=sam_residual_alpha,
            time_decay_factor=sam_time_decay,
            dropout=dropout
        )
        
        
        # === 3. Walk Sampler (Ablation: Multi-Scale) ===
        if not ablation_walk_multi_scale:
            # Force single scale (short) for ablation
            logger.info("Ablation: Disabling multi-scale walks (Single Scale Only)")
            w_len_short = walk_length_short
            w_num_short = num_walks_short + num_walks_long + num_walks_tawr # Combine all budget into short
            w_len_long = 0
            w_num_long = 0
            w_len_tawr = 0
            w_num_tawr = 0
        else:
            w_len_short = walk_length_short
            w_num_short = num_walks_short
            w_len_long = walk_length_long
            w_num_long = num_walks_long
            w_len_tawr = walk_length_tawr
            w_num_tawr = num_walks_tawr
        
        
        
        self.walk_sampler = MultiScaleWalkSampler(
            num_nodes=num_nodes,
            walk_length_short=w_len_short,
            walk_length_long=w_len_long if w_len_long > 0 else 1, # Avoid 0 length errors
            walk_length_tawr=w_len_tawr if w_len_tawr > 0 else 1,
            num_walks_short=w_num_short,
            num_walks_long=w_num_long,
            num_walks_tawr=w_num_tawr,
            temperature=walk_temperature,
            memory_dim=memory_dim,
            time_dim=time_dim,
            time_noise_std=walk_time_noise_std,
            mask_prob=walk_mask_prob,
            adaptive_length_factor=walk_adaptive_factor if ablation_walk_multi_scale else 0.0 # Disable adaptive if single scale
        )
        
        # === 4. HCT Module (Ablation: Hierarchical) ===
        if ablation_hct_hierarchical:
            self.hct = StabilizedHCT(
                d_model=hct_d_model,
                memory_dim=memory_dim,
                nhead=hct_nhead,
                num_intra_layers=hct_num_layers,
                num_inter_layers=hct_num_layers,
                dim_feedforward=hct_d_model * 4,
                dropout=dropout,
                drop_path_rate=hct_drop_path,
                max_walk_length=max(w_len_short, w_len_long, w_len_tawr),
                cooccurrence_sigma_time=hct_sigma_time
            )
            self.hct_proj = nn.Linear(hct_d_model, hidden_dim) if hct_d_model != hidden_dim else nn.Identity()
            logger.info("HCT: Full Hierarchical Co-occurrence Enabled")
        else:
            self.hct = None
            # Fallback projection for simple pooling
            self.hct_proj = nn.Identity()
            logger.info("Ablation: HCT Hierarchical Disabled (Using Simple Pooling)")
            
            
        # === 5. ST-ODE (Ablation: Existence) ===
        if self.use_st_ode:
            self.st_ode = NumericallyStabilizedSTODE(
                hidden_dim=memory_dim,
                num_nodes=num_nodes,
                num_eigenvectors=ode_num_eig,
                mu=ode_mu,
                adaptive_mu=True,
                use_gru_ode=True,
                ode_method=ode_method,
                dropout=dropout
            )
        else:
            self.st_ode = None
            logger.info("Ablation: ST-ODE Disabled")
            
        # === 6. Refiner (Ablation: Gating) ===
        max_walks = max(w_num_short, w_num_long, w_num_tawr)
        if ablation_mrp_gating:
            self.refiner = GatedMutualRefinementPooling(
                d_model=hidden_dim,
                nhead=4,
                dropout=dropout,
                num_walk_types=3 if ablation_walk_multi_scale else 1,
                max_walks_per_type=max_walks,
                modalities=3
            )
            logger.info("Refiner: Gated Mutual Refinement Enabled")
        else:
            # Simple Fusion Fallback: Concatenate modalities + Linear
            self.refiner = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            logger.info("Ablation: MRP Gating Disabled (Using Simple Concat Fusion)")
        
        
        # === 7. Link Predictor ===
        self.link_predictor = MergeLayer(
            input_dim1=hidden_dim,
            input_dim2=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
            use_temperature=True
        )
        
        # === 8. Hard Negative Miner (Ablation: Ratio) ===
        # If ratio=0, miner should just return batch with smoothing (or no smoothing if desired)
        self.hard_neg_miner = HardNegativeMiner(
            neg_sample_ratio=neg_sample_ratio,
            hard_neg_threshold=hard_neg_threshold,
            label_smoothing=label_smoothing if neg_sample_ratio > 0 else 0.0 # Disable smoothing if no mining? Optional.
        )
        
        # Buffers for graph data
        self.edge_index = None
        self.edge_time = None
        self.node_raw_features = None
        self.edge_raw_features = None
        self.validation_step_outputs = []   # For accumulating validation results
        
        logger.info(f"HiCoST Initialized: Robust-SAM + Stabilized-HCT + {'ST-ODE' if use_st_ode else 'No-ODE'} + Gated-Refinement")

    def set_graph(self, edge_index: torch.Tensor, edge_time: torch.Tensor):
        """Initialize graph structure for walk sampling."""
        self.edge_index = edge_index
        self.edge_time = edge_time
        self.walk_sampler.update_neighbors(edge_index, edge_time)
        logger.info(f"Graph initialized: {edge_index.shape[1]} edges")

    def set_raw_features(self, node_features: Optional[torch.Tensor], edge_features: Optional[torch.Tensor]):
        """Set static node/edge features."""
        if node_features is not None:
            self.node_raw_features = node_features.to(self.device)
        if edge_features is not None:
            # Store for batch usage if needed, usually passed in batch
            pass

    def set_neighbor_finder(self, neighbor_finder):
        """
        Set the neighbor finder and initialize the walk sampler with the graph structure.
        
        Args:
            neighbor_finder: An object that should contain `edge_index` (2, num_edges)
                            and `edge_time` (num_edges) tensors.
        """
        if hasattr(neighbor_finder, 'edge_index') and hasattr(neighbor_finder, 'edge_time'):
            # Move to the model's device
            device = next(self.parameters()).device
            edge_index = neighbor_finder.edge_index.to(device)
            edge_time = neighbor_finder.edge_time.to(device)
            self.sampler.update_neighbors(edge_index, edge_time, force=True)
            logger.info("Neighbor finder passed to walk sampler.")
        else:
            logger.warning(
                "Neighbor finder does not provide 'edge_index' and 'edge_time'. "
                "Walk sampler will not be initialized. Graph structure updates may fail."
            )
    
    
    def _get_hct_embeddings(self, walk_data: Dict, node_memory: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process walks through StabilizedHCT."""
        device = node_memory.device
        
        
        # Handle Ablation: If no HCT, perform simple mean pooling
        if self.hct is None:
            # Simple pooling logic
            all_nodes = torch.cat([walk_data['source']['short']['nodes'], walk_data['target']['short']['nodes']], dim=0)
            all_masks = torch.cat([walk_data['source']['short']['masks'], walk_data['target']['short']['masks']], dim=0)
            
            B_total = all_nodes.size(0)
            flat_n = all_nodes.view(-1)
            flat_f = node_memory[flat_n].view(B_total, all_nodes.size(1), all_nodes.size(2), -1)
            
            m_sum = all_masks.sum(dim=-1, keepdim=True).clamp(min=1)
            pooled = (flat_f * all_masks.unsqueeze(-1)).sum(dim=2) / m_sum # [B, W, D]
            # Pool over walks
            w_sum = pooled.sum(dim=1) / pooled.size(1)
            
            src_simple = w_sum[:batch_size]
            dst_simple = w_sum[batch_size:]
            return src_simple, dst_simple, walk_data        
        
        
        
        # Combine source/target walks for batched HCT processing
        combined_walks = {}
        types = ['short', 'long', 'tawr'] if self.ablation_walk_multi_scale else ['short']
        
        for wt in types:
            if wt not in walk_data['source']: continue
            combined_walks[wt] = {
                'nodes': torch.cat([walk_data['source'][wt]['nodes'], walk_data['target'][wt]['nodes']], dim=0),
                'nodes_anon': torch.cat([walk_data['source'][wt]['nodes_anon'], walk_data['target'][wt]['nodes_anon']], dim=0),
                'masks': torch.cat([walk_data['source'][wt]['masks'], walk_data['target'][wt]['masks']], dim=0),
                'times': torch.cat([walk_data['source'][wt]['times'], walk_data['target'][wt]['times']], dim=0)
            }
            if 'restart_flags' in walk_data['source'][wt]:
                combined_walks[wt]['restart_flags'] = torch.cat([
                    walk_data['source'][wt]['restart_flags'],
                    walk_data['target'][wt]['restart_flags']
                ], dim=0)
        
        # Run HCT
        # Returns [B*2, D] if return_all=False (fused) or dict if True
        # We need per-type for refiner, so we might need to adjust HCT forward or extract intermediates
        # For this integration, let's assume HCT returns fused [B*2, D] AND we extract per-type from walk_data separately if needed
        # Actually, GatedRefiner needs per-type raw walks. 
        # Strategy: HCT gives global structural context. Refiner takes HCT output + Raw Walks.
        
        hct_out = self.hct(
            walks_dict=combined_walks,
            node_memory=node_memory,
            return_all=False
        ) # [B*2, D_hct]
        
        hct_out = self.hct_proj(hct_out) # [B*2, D_hidden]
        
        src_hct = hct_out[:batch_size]
        dst_hct = hct_out[batch_size:]
        
        return src_hct, dst_hct, combined_walks

    def compute_temporal_embeddings(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        batch: Optional[Dict] = None,
        co_occurrence_matrix: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full Forward Pass:
        1. Sample Walks
        2. Encode Walks (HCT)
        3. Evolve Memory (ST-ODE) - Optional/Batched
        4. Fuse Modalities (Refiner)
        """
        device = src_nodes.device
        batch_size = src_nodes.size(0)
        
        # 1. Get Current Memory (SAM)
        # Detach for walk sampling to prevent gradient loops through history
        current_memory = self.sam_module.raw_memory.detach()
        
        src_tensor = src_nodes.to(device).long()
        dst_tensor = dst_nodes.to(device).long()
        ts_tensor = timestamps.to(device).float()

        if ts_tensor.size(0) != batch_size:
            raise RuntimeError(f"Internal Error: Timestamps ({ts_tensor.size(0)}) do not match nodes ({batch_size})")
        
        # 2. Sample Multi-Scale Walks
        walk_data = self.walk_sampler(
            source_nodes=src_tensor,
            target_nodes=dst_tensor,
            current_times=ts_tensor,
            memory_states=current_memory,
            edge_index=self.edge_index,
            edge_time=self.edge_time,
            co_occurrence_scores=co_occurrence_matrix # Pass the HCT co-occurrence here
        )
        
        # Ensure you pass the augmented data to HCT
        if self.use_hct and not self.debug_simple_walk:
            combined_walks = {
                'short': {
                    'nodes': torch.cat([walk_data['source']['short']['nodes'], walk_data['target']['short']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['short']['nodes_anon'], walk_data['target']['short']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['short']['masks'], walk_data['target']['short']['masks']], dim=0),
                    'times': torch.cat([walk_data['source']['short']['times'], walk_data['target']['short']['times']], dim=0),
                },
                'long': {
                    'nodes': torch.cat([walk_data['source']['long']['nodes'], walk_data['target']['long']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['long']['nodes_anon'], walk_data['target']['long']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['long']['masks'], walk_data['target']['long']['masks']], dim=0),
                    'times': torch.cat([walk_data['source']['long']['times'], walk_data['target']['long']['times']], dim=0),
                },
                'tawr': {
                    'nodes': torch.cat([walk_data['source']['tawr']['nodes'], walk_data['target']['tawr']['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source']['tawr']['nodes_anon'], walk_data['target']['tawr']['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source']['tawr']['masks'], walk_data['target']['tawr']['masks']], dim=0),
                    'times': torch.cat([walk_data['source']['tawr']['times'], walk_data['target']['tawr']['times']], dim=0),
                },
            }
        
        # 3. Encode Walks (HCT) -> Structural Embeddings
        src_hct, dst_hct, combined_walks = self._get_hct_embeddings(walk_data, current_memory, batch_size)
        
        # 4. Evolve Memory (ST-ODE) -> Temporal Embeddings
        # Only update periodically or if enough time has passed to save compute
        src_stode = current_memory[src_nodes] # Default to current memory if ODE skipped
        dst_stode = current_memory[dst_nodes]
        
        if self.use_st_ode and self.training and self.st_ode is not None:
            self._ode_batch_counter += 1
            t_max = timestamps.max()
            dt = t_max - self.last_ode_update_time
            
            # Update condition: Interval reached OR large time jump
            if self._ode_batch_counter >= self.st_ode_update_interval or dt > 100.0:
                self._ode_batch_counter = 0
                
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                if free_memory < 2 * 1024**3:  # Less than 2GB free
                    logger.warning(f"Skipping ST-ODE due to low memory ({free_memory/1024**3:.2f} GB free)")
                else:
                # Here we simulate obs using the sampled walk nodes and times
                    obs_encodings = self._prepare_ode_observations(walk_data, current_memory, batch_size)
                
                if obs_encodings is not None:
                    
                    observed_nodes = obs_encodings['node_map']
                    observed_memory = current_memory[observed_nodes]
                    try:
                        evolved_observed = self.st_ode(
                            node_states=observed_memory,  # Only observed nodes
                            walk_encodings=obs_encodings['encodings'],
                            walk_times=obs_encodings['times'],
                            walk_masks=obs_encodings['masks'],
                            adj_matrix=obs_encodings['adj']
                        )
                        
                        # Update global memory state (Residual update handled inside ST-ODE or here)
                        # RobustSAM already has residual, but ST-ODE evolves the whole graph
                        with torch.no_grad():
                            # Simple blending for stability
                            self.sam_module.raw_memory.data[observed_nodes] = \
                                0.9 * self.sam_module.raw_memory.data[observed_nodes] + \
                                0.1 * evolved_observed
                            
                        self.last_ode_update_time = t_max
                        
                        # Fetch updated memory for src/dst
                        src_stode = self.sam_module.raw_memory[src_nodes]
                        dst_stode = self.sam_module.raw_memory[dst_nodes]
                    except RuntimeError as e:
                        logger.warning(f"ST-ODE failed: {e}")
                        # Fall back to current memory
                        src_stode = current_memory[src_nodes]
                        dst_stode = current_memory[dst_nodes]
                    
                    # Clean up
                    del obs_encodings
                    torch.cuda.empty_cache()

        # 5. Get SAM Memory Embeddings (Direct Lookup) -> Memory Embeddings
        src_sam = current_memory[src_nodes]
        dst_sam = current_memory[dst_nodes]
        
        # 6. Gated Mutual Refinement (Fuse SAM, HCT, STODE)
        # Prepare per-type raw embeddings for the refiner
        src_per_type, src_masks = self._extract_per_type_embeds(walk_data['source'], current_memory)
        dst_per_type, dst_masks = self._extract_per_type_embeds(walk_data['target'], current_memory)
        
        if self.ablation_mrp_gating:
            final_src, final_dst = self.refiner(
                src_hct=src_hct, dst_hct=dst_hct,
                src_sam=src_sam, dst_sam=dst_sam,
                src_stode=src_stode, dst_stode=dst_stode,
                src_per_type=src_per_type, dst_per_type=dst_per_type,
                src_masks=src_masks, dst_masks=dst_masks
            )
        else:
            # Simple Concat Fusion
            stacked_src = torch.cat([src_sam, src_hct, src_stode], dim=-1)
            stacked_dst = torch.cat([dst_sam, dst_hct, dst_stode], dim=-1)
            final_src = self.refiner(stacked_src)
            final_dst = self.refiner(stacked_dst)

        
        return final_src, final_dst

    def _prepare_ode_observations(self, walk_data, memory, batch_size) -> Optional[Dict]:
        """Prepare sparse observations for ST-ODE from walk data - Memory Efficient Version."""
        device = memory.device
        
        # Use only the current batch nodes instead of all nodes
        key = 'short' if not self.ablation_walk_multi_scale else 'short'
        
        # Get nodes from walk data (only source and target from current batch)
        src_nodes = walk_data['source'][key]['nodes'].view(batch_size, -1)  # [B, W*L]
        tgt_nodes = walk_data['target'][key]['nodes'].view(batch_size, -1)  # [B, W*L]
        src_times = walk_data['source'][key]['times'].view(batch_size, -1)  # [B, W*L]
        tgt_times = walk_data['target'][key]['times'].view(batch_size, -1)  # [B, W*L]
        
        # Combine and get unique nodes for this batch only
        batch_nodes = torch.cat([src_nodes, tgt_nodes], dim=1)  # [B, 2*W*L]
        batch_times = torch.cat([src_times, tgt_times], dim=1)  # [B, 2*W*L]
        
        # Flatten and get unique
        flat_nodes = batch_nodes.view(-1)
        flat_times = batch_times.view(-1)
        
        # Filter valid nodes (non-zero)
        valid_mask = flat_nodes > 0
        valid_nodes = flat_nodes[valid_mask]
        valid_times = flat_times[valid_mask]
        
        if valid_nodes.numel() == 0:
            return None
        
        # Get unique nodes and times for this batch only
        unique_nodes, node_inv = torch.unique(valid_nodes, return_inverse=True)
        unique_times, time_inv = torch.unique(valid_times, return_inverse=True)
        
        num_unique_nodes = unique_nodes.size(0)
        num_unique_times = unique_times.size(0)
        
        # CRITICAL FIX: Aggressively limit dimensions to prevent OOM
        max_nodes = min(num_unique_nodes, 500)   # Cap at 500 nodes
        max_times = min(num_unique_times, 50)     # Cap at 50 time points
        
        # Sample nodes if too many
        if num_unique_nodes > max_nodes:
            # Keep nodes with most observations
            node_counts = torch.bincount(node_inv, minlength=num_unique_nodes)
            top_node_indices = torch.topk(node_counts, max_nodes).indices
            unique_nodes = unique_nodes[top_node_indices]
            # Remap indices
            node_map = torch.full((num_unique_nodes,), -1, dtype=torch.long, device=device)
            node_map[top_node_indices] = torch.arange(max_nodes, device=device)
            node_inv = node_map[node_inv]
            valid_mask = node_inv >= 0
            valid_nodes = valid_nodes[valid_mask]
            valid_times = valid_times[valid_mask]
            node_inv = node_inv[valid_mask]
            time_inv = time_inv[valid_mask]
            num_unique_nodes = max_nodes
        
        # Sample recent times if too many
        if num_unique_times > max_times:
            sorted_times, sorted_idx = torch.sort(unique_times, descending=True)
            selected_times = sorted_times[:max_times]
            time_map = torch.full((num_unique_times,), -1, dtype=torch.long, device=device)
            time_map[sorted_idx[:max_times]] = torch.arange(max_times, device=device)
            time_inv = time_map[time_inv]
            valid_mask = time_inv >= 0
            valid_nodes = valid_nodes[valid_mask]
            valid_times = valid_times[valid_mask]
            node_inv = node_inv[valid_mask]
            time_inv = time_inv[valid_mask]
            unique_times = selected_times
            num_unique_times = max_times
        
        # Create 2D indices for scatter operation
        # Flatten (node, time) pairs to 1D index: idx = node * num_times + time
        flat_idx = node_inv * num_unique_times + time_inv  # [num_valid]
        
        # Get features for valid nodes
        node_features = memory[unique_nodes]  # [num_unique_nodes, memory_dim]
        feats_expanded = node_features[node_inv]  # [num_valid, memory_dim]
        
        # CRITICAL FIX: Use scatter_add_ with proper 2D indexing
        # Create output tensor [num_unique_nodes * num_unique_times, memory_dim]
        total_slots = num_unique_nodes * num_unique_times
        obs_flat = torch.zeros(total_slots, self.memory_dim, device=device)
        counts_flat = torch.zeros(total_slots, device=device)
        
        # Scatter add features
        # feats_expanded: [num_valid, memory_dim]
        # flat_idx: [num_valid]
        # obs_flat: [total_slots, memory_dim]
        obs_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, self.memory_dim), feats_expanded)
        counts_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
        
        # Reshape to 3D: [num_unique_nodes, num_unique_times, memory_dim]
        obs = obs_flat.view(num_unique_nodes, num_unique_times, self.memory_dim)
        counts = counts_flat.view(num_unique_nodes, num_unique_times)
        
        # Normalize
        counts = counts.clamp(min=1.0).unsqueeze(-1)  # [num_nodes, num_times, 1]
        obs = obs / counts  # Broadcasting: [N, T, D] / [N, T, 1] -> [N, T, D]
        
        # Create sparse adjacency matrix for unique nodes only
        adj = torch.eye(num_unique_nodes, device=device)
        
        # Create time tensor [num_unique_nodes, num_unique_times, 1]
        times_matrix = unique_times.view(1, num_unique_times, 1).expand(num_unique_nodes, -1, -1)
        
        # Create mask [num_unique_nodes, num_unique_times, 1]
        masks = (counts.squeeze(-1) > 0).float().unsqueeze(-1)
        
        return {
            'encodings': obs.unsqueeze(2),  # [num_nodes, num_times, 1, memory_dim]
            'times': times_matrix,  # [num_nodes, num_times, 1]
            'masks': masks,  # [num_nodes, num_times, 1]
            'adj': adj,  # [num_nodes, num_nodes]
            'node_map': unique_nodes,  # Mapping back to global node IDs
        }

    def _extract_per_type_embeds(self, side_data, memory) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract [B, Types, MaxWalks, D] and masks for Refiner."""
        types = ['short', 'long', 'tawr'] if self.ablation_walk_multi_scale else ['short']
        embeds = []
        masks = []
        max_w = max(self.hparams.num_walks_short, self.hparams.num_walks_long, self.hparams.num_walks_tawr)
        
        B = side_data['short']['nodes'].size(0)
        D = memory.size(-1)
        device = memory.device
        
        for t in types:
            nodes = side_data[t]['nodes'] # [B, W, L]
            m = side_data[t]['masks']     # [B, W, L]
            W, L = nodes.shape[1], nodes.shape[2]
            
            # Lookup embeddings
            flat_n = nodes.view(-1)
            flat_f = memory[flat_n].view(B, W, L, D)
            
            # Pool over Length
            m_sum = m.sum(dim=-1, keepdim=True).clamp(min=1)
            pooled = (flat_f * m.unsqueeze(-1)).sum(dim=2) / m_sum # [B, W, D]
            
            # Pad/Trim to max_w
            if W < max_w:
                pad = max_w - W
                pooled = F.pad(pooled, (0,0,0,pad))
                m_valid = m.any(dim=-1)
                m_valid = F.pad(m_valid, (0, pad), value=False)
            else:
                pooled = pooled[:, :max_w]
                m_valid = m.any(dim=-1)[:, :max_w]
                
            embeds.append(pooled)
            masks.append(m_valid)
            
        return torch.stack(embeds, dim=1), torch.stack(masks, dim=1)

    def forward(self, batch: Dict, return_probs: bool = False, co_occurrence_matrix: Optional[Dict] = None) -> torch.Tensor:
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']  
        
        src, dst = self.compute_temporal_embeddings(
            src_nodes, 
            dst_nodes, 
            timestamps,   
            batch=batch,
            co_occurrence_matrix=co_occurrence_matrix
        )
        logits = self.link_predictor(src, dst).squeeze(-1)
        
        if return_probs:
            temp = self.link_predictor.temperature.clamp(0.1, 10.0)
            return logits, torch.sigmoid(logits / temp)
        
        return logits

    def training_step(self, batch, batch_idx):
        
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        # 1. Mine Hard Negatives & Smooth Labels        
        # Get HCT co-occurrence scores for hard negative mining (NEW)
        current_memory = self.sam_module.raw_memory.detach()

        walk_data = self.walk_sampler(
            source_nodes=batch['src_nodes'],
            target_nodes=batch['dst_nodes'],
            current_times=batch['timestamps'],
            memory_states=current_memory,
            edge_index=self.edge_index,
            edge_time=self.edge_time
        )

        # Get HCT co-occurrence matrix (from StabilizedHCT)
        hct_cooccur = self.hct.get_cooccurrence_matrix(walk_data, current_memory)
        
        
        # Pass HCT co-occurrence to HNM (NEW)
        aug_batch = self.hard_neg_miner(
            batch, 
            memory=self.sam_module.raw_memory,
            cooccurrence_matrix=hct_cooccur  
        )
        
        # 2. Forward Pass
        logits = self(aug_batch, co_occurrence_matrix=hct_cooccur)
        
        # 3. Loss (BCE works with smoothed labels)
        loss = F.binary_cross_entropy_with_logits(logits, aug_batch['labels'])
        
        # 4. Log
        self.log('train_loss', loss, prog_bar=True)

        # Clean up intermediate tensors
        del walk_data, hct_cooccur, aug_batch, current_memory

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update SAM Memory after batch."""
        if self.training:
            src = batch['src_nodes']
            dst = batch['dst_nodes']
            ts = batch['timestamps']
            edge_feat = batch.get('edge_features', torch.zeros(len(src), self.edge_feat_dim, device=self.device))
            
            with torch.no_grad():
                self.sam_module.update_memory_batch(
                    source_nodes=src.detach(),
                    target_nodes=dst.detach(),
                    edge_features=edge_feat.detach(),
                    current_time=ts.detach(),
                    node_features=self.node_raw_features.detach() if self.node_raw_features is not None else None
                )
            
                # Reset prototypes if NaN detected
                all_nodes = torch.unique(torch.cat([src, dst]))
                self.sam_module.reset_prototypes_if_needed(all_nodes)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Safely determine steps_per_epoch for the cosine scheduler
        if (self.trainer.train_dataloader is not None and 
                hasattr(self.trainer.train_dataloader, '__len__')):
            steps_per_epoch = len(self.trainer.train_dataloader) // self.trainer.accumulate_grad_batches
        elif hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches is not None:
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = total_steps // self.trainer.max_epochs
        else:
            # Fallback to the known value from dataset logs (862 batches/epoch)
            logger.warning("Could not determine steps_per_epoch. Using default of 862 steps (from dataset info).")
            steps_per_epoch = 862

        scheduler = self.hard_neg_miner.get_cosine_scheduler(
            optimizer,
            num_epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=self.hparams.warmup_epochs,
            min_lr_ratio=self.hparams.min_lr_ratio
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def validation_step(self, batch, batch_idx):
        logits, probs = self(batch, return_probs=True)
        labels = batch['labels'].float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        # Compute AP/AUC on epoch end to save time
        # Store for epoch-end aggregation
        self.validation_step_outputs.append({'probs': probs, 'labels': labels})
        return {'probs': probs, 'labels': labels}

    def on_validation_epoch_start(self):
        """Backup SAM memory to avoid contamination during validation (NEW)"""
        if hasattr(self, '_sam_val_backup'): delattr(self, '_sam_val_backup')
        if self.training:
            self._sam_val_backup = self.sam_module.raw_memory.clone().detach()
            self._stode_val_backup = self.st_ode.node_states.clone().detach() if self.use_st_ode else None


    def on_validation_epoch_end(self):
        """Restore SAM/ST-ODE memory after validation (NEW)"""
        if hasattr(self, '_sam_val_backup'):
            self.sam_module.raw_memory.data.copy_(self._sam_val_backup)
        
        if self.use_st_ode and hasattr(self, '_stode_val_backup'):
            self.st_ode.node_states.data.copy_(self._stode_val_backup)
        
        # Aggregate validation outputs
        outputs = self.validation_step_outputs
        if not outputs:
            self.validation_step_outputs.clear()
            return
        
        all_probs = torch.cat([x['probs'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])
        
        ap = average_precision_score(all_labels.cpu(), all_probs.cpu())
        auc = roc_auc_score(all_labels.cpu(), all_probs.cpu())
        
        self.log('val_ap', ap, prog_bar=True)
        self.log('val_auc', auc, prog_bar=True)

        # Clear the list for next validation epoch
        self.validation_step_outputs.clear()


    # Ablation Flags actively modify architecture behavior:
    # - ablation_walk_multi_scale=False -> Single scale walks + Simple Pooling
    # - ablation_sam_prototypes=False   -> SAM uses raw memory only (no prototype attention)
    # - ablation_hct_hierarchical=False -> HCT skips co-occurrence & inter-walk attention
    # - ablation_mrp_gating=False       -> Replaces Gated Refiner with Concat+Linear
    # - use_st_ode=False                -> Skips ODE evolution entirely
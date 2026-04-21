# === HiCoST v4 

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score
import lightning as L
import gc

from ..component.hicost_config import HiCoSTConfig
from ..component.time_encoder import TimeEncoder
from ..component.sam_modulev2 import RobustStabilityAugmentedMemory
from ..component.multi_swalkv2 import MultiScaleWalkSampler
from ..component.hct_modulev2 import StabilizedHCT
from ..component.stode_modulev2 import NumericallyStabilizedSTODE
from ..component.mrp_modulev2 import GatedMutualRefinementPooling
from ..component.hardnegative_modulev2 import HardNegativeMiner, _DummyHardNegativeMiner
from ..component.transformer_encoder import MergeLayer
from src.utils.hicost_utils import parse_bool, parse_float


class HiCoSTv4(L.LightningModule):
    def __init__(self, config: HiCoSTConfig):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)        
        self.save_hyperparameters()
        self.config = config 

        
        self.num_nodes = config.num_nodes
        self.hidden_dim = config.hidden_dim
        self.memory_dim = config.memory_dim
        self.time_dim = config.time_dim
        self.edge_feat_dim = config.edge_feat_dim
        self.dropout = config.dropout

        # Projection for dimension alignment
        if config.memory_dim != config.hidden_dim:
            self.mem_to_hidden = nn.Linear(config.memory_dim, config.hidden_dim)
            logger.info(f"Projection: memory_dim({config.memory_dim}) → hidden_dim({config.hidden_dim})")
        else:
            self.mem_to_hidden = nn.Identity()
        
        self.use_st_ode = parse_bool(config.use_st_ode)
        self.use_hct = parse_bool(config.use_hct)
        self.use_gated_refinement = parse_bool(config.use_gated_refinement)
        self.use_multi_scale_walks = parse_bool(config.use_multi_scale_walks)
        self.use_prototype_attention = parse_bool(config.use_prototype_attention)
        self.use_hard_negative_mining = parse_bool(config.use_hard_negative_mining)
        self.use_hct_hierarchical = parse_bool(config.use_hct_hierarchical)
        
        self.st_ode_update_interval = int(config.st_ode_update_interval)
        self.debug_simple_walk = parse_bool(config.debug_simple_walk)
        
        edge_feat_dim = config.edge_features_dim if config.edge_feat_dim == 64 else config.edge_feat_dim
        
        self.register_buffer('last_ode_update_time', torch.tensor(0.0))
        self._ode_batch_counter = 0
        
        
        self.time_encoder = TimeEncoder(config.time_dim)
        
        
        effective_prototypes = config.num_prototypes if self.use_prototype_attention else 1
        self.sam_module = RobustStabilityAugmentedMemory(
            num_nodes=config.num_nodes,
            memory_dim=config.memory_dim,
            node_feat_dim=config.node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            time_dim=config.time_dim,
            num_prototypes=effective_prototypes,
            residual_alpha=config.sam_residual_alpha,
            time_decay_factor=config.sam_time_decay,
            dropout=config.dropout,
        )

        # Add last_update buffer if not present (required for memory update)
        if not hasattr(self.sam_module, 'last_update'):
            self.sam_module.register_buffer('last_update', torch.zeros(config.num_nodes, device=self.device))
        logger.info(f"SAM: {effective_prototypes} prototype(s) ({'attention' if self.use_prototype_attention else 'single'})")

        # === Walk Sampler (simplified if multi-scale disabled) ===
        if self.use_multi_scale_walks:
            w_len_short = config.walk_length_short
            w_num_short = config.num_walks_short
            w_len_long = config.walk_length_long
            w_num_long = config.num_walks_long
            w_len_tawr = config.walk_length_tawr
            w_num_tawr = config.num_walks_tawr
        else:
            # Single scale only – sum all walks into short walks
            w_len_short = config.walk_length_short
            w_num_short = config.num_walks_short + config.num_walks_long + config.num_walks_tawr
            w_len_long = w_len_tawr = 0
            w_num_long = w_num_tawr = 0
        
        self.walk_sampler = MultiScaleWalkSampler(
            num_nodes=config.num_nodes,
            walk_length_short=w_len_short,
            walk_length_long=w_len_long if w_len_long > 0 else 1,
            walk_length_tawr=w_len_tawr if w_len_tawr > 0 else 1,
            num_walks_short=w_num_short,
            num_walks_long=w_num_long,
            num_walks_tawr=w_num_tawr,
            temperature=config.walk_temperature,
            memory_dim=config.memory_dim,
            time_dim=config.time_dim,
            time_noise_std=config.walk_time_noise_std,
            mask_prob=config.walk_mask_prob,
            adaptive_length_factor=config.walk_adaptive_factor if self.use_multi_scale_walks else 0.0
        )
        
        
        if self.use_hct_hierarchical:
            self.hct = StabilizedHCT(
                d_model=config.hct_d_model,
                memory_dim=config.memory_dim,
                nhead=config.hct_nhead,
                num_intra_layers=config.hct_num_intra_layers,
                num_inter_layers=config.hct_num_inter_layers,
                dim_feedforward=config.hct_d_model * 4,
                dropout=config.dropout,
                drop_path_rate=config.hct_drop_path,
                max_walk_length=max(w_len_short, w_len_long, w_len_tawr),
                cooccurrence_sigma_time=config.hct_sigma_time
            )
            self.hct_proj = nn.Linear(config.hct_d_model, config.hidden_dim) if config.hct_d_model != config.hidden_dim else nn.Identity()
            logger.info("HCT: Full hierarchical co-occurrence enabled")
        else:
            self.hct = None
            self.hct_proj = nn.Identity()
            logger.info("HCT: Simple pooling only (ablation mode)")
            
        
        if self.use_st_ode:
            self.st_ode = NumericallyStabilizedSTODE(
                hidden_dim=config.memory_dim,
                num_nodes=config.num_nodes,
                num_eigenvectors=config.ode_num_eig,
                mu=config.ode_mu,
                adaptive_mu=config.adaptive_mu,
                use_gru_ode=config.use_gru_ode,
                ode_method=config.ode_method,
                ode_step_size=config.ode_step_size,
                adjoint=config.adjoint,
                dropout=config.dropout
            )
            logger.info(f"ST-ODE: Enabled (method={config.ode_method})")
        else:
            self.st_ode = None
            logger.info("ST-ODE: Disabled (faster training)")
            
        
        if self.use_gated_refinement:
            self.refiner = GatedMutualRefinementPooling(
                d_model=config.hidden_dim,
                nhead=config.mrp_nhead,
                dropout=config.dropout,
                num_walk_types=3 if self.use_multi_scale_walks else 1,
                modalities=config.mrp_modalities,
                fusion_mode=config.mrp_fusion_mode,
                pool_attn_heads=config.mrp_pool_attn_heads,
            )
            logger.info(f"Refiner: Gated mutual refinement enabled")
        else:
            self.refiner = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2), 
                nn.LayerNorm(config.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim) 
            )
            logger.info("Refiner: Simple concat fusion (ablation mode)")
        
        
        self.link_predictor = MergeLayer(
            input_dim1=config.hidden_dim,
            input_dim2=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            dropout=config.dropout,
            use_temperature=True
        )
        
        
        if self.use_hard_negative_mining and config.neg_sample_ratio > 0:
            self.hard_neg_miner = HardNegativeMiner(
                neg_sample_ratio=config.neg_sample_ratio,
                hard_neg_threshold=config.hard_neg_threshold,
                label_smoothing=config.label_smoothing
            )
            logger.info(f"Hard Negative Mining: Enabled (ratio={config.neg_sample_ratio})")
        else:
            self.hard_neg_miner = _DummyHardNegativeMiner(config.label_smoothing)
            logger.info("Hard Negative Mining: Disabled")

        # Memory GRU (for TAWRMAC-like update)
        # Input: src_mem + dst_mem + edge_feat + time_enc
        gru_input_dim = self.memory_dim + self.memory_dim + self.edge_feat_dim + self.time_dim
        self.memory_gru = nn.GRUCell(input_size=gru_input_dim, hidden_size=self.memory_dim)
        logger.info("Memory GRU cell added for online updates (TAWRMAC style)")
        
        # Buffers for graph data
        self.edge_index = None
        self.edge_time = None
        self.node_raw_features = None
        self.edge_raw_features = None
        self.validation_step_outputs = []
        
        logger.info(f"HiCoSTv4 Initialized (TAWRMAC-beating mode): "
                   f"hidden_dim={config.hidden_dim}, memory_dim={config.memory_dim}, "
                   f"multi-scale={self.use_multi_scale_walks}, ST-ODE={self.use_st_ode}")

   
    def set_graph(self, edge_index: torch.Tensor, edge_time: torch.Tensor):
        self.edge_index = edge_index
        self.edge_time = edge_time
        self.walk_sampler.update_neighbors(edge_index, edge_time)
        logger.info(f"Graph initialized: {edge_index.shape[1]} edges")

    def set_raw_features(self, node_features: Optional[torch.Tensor], edge_features: Optional[torch.Tensor]):
        """Store node and edge features on the model's device."""
        device = next(self.parameters()).device
        if node_features is not None:
            self.node_raw_features = node_features.to(device)
            logger.info(f"Node raw features moved to {device}, shape: {self.node_raw_features.shape}")
        if edge_features is not None:
            self.edge_raw_features = edge_features.to(device)
            logger.info(f"Edge raw features moved to {device}, shape: {self.edge_raw_features.shape}")
    
    
    # def set_raw_features(self, node_features: Optional[torch.Tensor], edge_features: Optional[torch.Tensor]):
    #     if node_features is not None:
    #         self.node_raw_features = node_features.to(self.device)
    #     if edge_features is not None:
    #         self.edge_raw_features = edge_features.to(self.device)

    # def set_neighbor_finder(self, neighbor_finder):
    #     if hasattr(neighbor_finder, 'edge_index') and hasattr(neighbor_finder, 'edge_time'):
    #         device = next(self.parameters()).device
    #         edge_index = neighbor_finder.edge_index.to(device)
    #         edge_time = neighbor_finder.edge_time.to(device)
    #         self.walk_sampler.update_neighbors(edge_index, edge_time, force=True)
    #         logger.info("Neighbor finder passed to walk sampler.")
    #     else:
    #         logger.warning("Neighbor finder missing edge_index/edge_time; walk sampler not updated.")

    def set_neighbor_finder(self, neighbor_finder):
        """
        Extract edge_index and edge_time from neighbor_finder if possible.
        If not, log a warning and skip (will rely on later set_graph call).
        """
        edge_index = None
        edge_time = None

        # Case 1: neighbor_finder has direct attributes (our preferred NeighborFinder)
        if hasattr(neighbor_finder, 'edge_index') and hasattr(neighbor_finder, 'edge_time'):
            edge_index = neighbor_finder.edge_index
            edge_time = neighbor_finder.edge_time
        # Case 2: TAWRMAC-style NewNeighborFinder (no single edge_index, but we can reconstruct if needed)
        elif hasattr(neighbor_finder, 'node_to_edge_idxs'):
            logger.warning(
                "Neighbor finder is TAWRMAC-style (no edge_index). "
                "Walk sampler will not be updated here. Ensure set_graph is called later with explicit edge_index/edge_time."
            )
            return
        else:
            logger.warning("Neighbor finder does not provide edge_index/edge_time; walk sampler not updated.")
            return

        # Move to correct device and update walk sampler
        device = next(self.parameters()).device
        edge_index = edge_index.to(device)
        edge_time = edge_time.to(device)
        self.walk_sampler.update_neighbors(edge_index, edge_time, force=True)
        logger.info("Neighbor finder passed to walk sampler (edge_index/edge_time extracted).")
    
    
    
    # ---------- Core embedding method (modified to add node raw features) ----------
    def compute_temporal_embeddings(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        batch: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = src_nodes.device
        batch_size = src_nodes.size(0)
        
        with torch.no_grad():
            current_memory_snapshot = self.sam_module.raw_memory.clone()
        
        src_tensor = src_nodes.to(device).long()
        dst_tensor = dst_nodes.to(device).long()
        ts_tensor = timestamps.to(device).float()

        # Sample walks
        walk_data = self.walk_sampler(
            source_nodes=src_tensor,
            target_nodes=dst_tensor,
            current_times=ts_tensor,
            memory_states=current_memory_snapshot,
            edge_index=self.edge_index,
            edge_time=self.edge_time
        )
        
        # Encode walks (HCT or simple pooling)
        with torch.autograd.set_detect_anomaly(True):
            src_hct, dst_hct, _ = self._get_hct_embeddings(
                walk_data, current_memory_snapshot, batch_size
            )
        
        # Default memory output (with node raw features added)
        src_memory_out = current_memory_snapshot[src_nodes].clone()
        dst_memory_out = current_memory_snapshot[dst_nodes].clone()
        
        # --- Add node raw features (like TAWRMAC) ---
        if self.node_raw_features is not None:
            if self.node_raw_features.device != src_memory_out.device:
                self.node_raw_features = self.node_raw_features.to(src_memory_out.device)
            src_memory_out = src_memory_out + self.node_raw_features[src_nodes]
            dst_memory_out = dst_memory_out + self.node_raw_features[dst_nodes]
        
        # ST-ODE evolution (if enabled – disabled by default for speed)
        if self.use_st_ode and self.training and self.st_ode is not None:           
            self._ode_batch_counter += 1
            t_max = timestamps.max()
            dt = t_max - self.last_ode_update_time
            
            if self._ode_batch_counter >= self.st_ode_update_interval or dt > 100.0:
                self._ode_batch_counter = 0
                
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                if free_memory >= 2 * 1024**3:
                    obs_encodings = self._prepare_ode_observations(
                        walk_data, current_memory_snapshot, batch_size
                    )
                
                    if obs_encodings is not None:
                        observed_nodes = obs_encodings['node_map']
                        
                        # ST-ODE computation in its own scope
                        try:
                            with torch.no_grad():
                                observed_input = current_memory_snapshot[observed_nodes]
                            
                            # Forward through ST-ODE (may have gradients)
                            evolved_observed = self.st_ode(
                                node_states=observed_input,
                                walk_encodings=obs_encodings['encodings'],
                                walk_times=obs_encodings['times'],
                                walk_masks=obs_encodings['masks'],
                                adj_matrix=obs_encodings['adj']
                            )
                            
                            # CRITICAL: Update raw_memory with NO gradient tracking
                            # and NO view relationship to evolved_observed
                            with torch.no_grad():
                                evolved_detached = evolved_observed.detach().clone()
                                # Create completely new tensor for update
                                update_values = 0.9 * self.sam_module.raw_memory[observed_nodes] + 0.1 * evolved_detached
                                # Use scatter-like update instead of indexing assignment
                                indices = observed_nodes.unsqueeze(-1).expand(-1, self.memory_dim)
                                self.sam_module.raw_memory.scatter_(0, indices, update_values)
                                
                            self.last_ode_update_time = t_max
                            
                            # Get fresh read from updated memory (detached)
                            with torch.no_grad():
                                src_memory_out = self.sam_module.raw_memory[src_nodes].clone()
                                dst_memory_out = self.sam_module.raw_memory[dst_nodes].clone()
                            
                        except RuntimeError as e:
                            logger.warning(f"ST-ODE failed: {e}")
                            # Keep defaults
                        
                        del obs_encodings
                        torch.cuda.empty_cache()
        
        # Project to hidden_dim
        src_sam_proj = self.mem_to_hidden(src_memory_out)
        dst_sam_proj = self.mem_to_hidden(dst_memory_out)
        src_stode_proj = src_sam_proj  # placeholder when ST-ODE off
        dst_stode_proj = dst_sam_proj
        
        # Extract per-type embeddings (if multi-scale)
        src_per_type, src_masks = self._extract_per_type_embeds(walk_data['source'], current_memory_snapshot)
        dst_per_type, dst_masks = self._extract_per_type_embeds(walk_data['target'], current_memory_snapshot)
        
        # Refiner
        if self.use_gated_refinement:
            final_src, final_dst, _ = self.refiner(
                src_hct=src_hct.clone(), 
                dst_hct=dst_hct.clone(),
                src_sam=src_sam_proj.clone(), 
                dst_sam=dst_sam_proj.clone(),
                src_stode=src_stode_proj.clone(), 
                dst_stode=dst_stode_proj.clone(),
                src_per_type=src_per_type.clone() if src_per_type is not None else None,
                dst_per_type=dst_per_type.clone() if dst_per_type is not None else None,
                src_masks=src_masks, 
                dst_masks=dst_masks
            )
        else:
            stacked_src = torch.cat([src_sam_proj, src_hct, src_stode_proj], dim=-1)
            stacked_dst = torch.cat([dst_sam_proj, dst_hct, dst_stode_proj], dim=-1)
            final_src = self.refiner(stacked_src)
            final_dst = self.refiner(stacked_dst)
        
        return final_src, final_dst

    # ---------- Memory update (TAWRMAC style) ----------
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update memory using GRU after each batch (critical for beating TAWRMAC)."""
        if not self.training:
            return
        
        src = batch['src_nodes']
        dst = batch['dst_nodes']
        ts = batch['timestamps']
        edge_feat = batch.get('edge_features')
        if edge_feat is None:
            edge_feat = torch.zeros(len(src), self.edge_feat_dim, device=self.device)
        
        with torch.no_grad():
            # Get current memory for source and destination
            src_mem = self.sam_module.raw_memory[src]
            dst_mem = self.sam_module.raw_memory[dst]
            
            # Time delta encoding
            last_update = self.sam_module.last_update[src]
            time_delta = ts - last_update
            time_enc = self.time_encoder(time_delta.unsqueeze(1)).squeeze(1)
            
            # Build message (src_mem, dst_mem, edge_feat, time_enc)
            message = torch.cat([src_mem, dst_mem, edge_feat, time_enc], dim=1)
            
            # GRU update
            updated_mem = self.memory_gru(message, src_mem)
            self.sam_module.raw_memory[src] = updated_mem
            self.sam_module.last_update[src] = ts
            
            # Also update memory for destination? In TAWRMAC only source is updated because the edge is directed.
            # For undirected graphs you may update both. We'll follow TAWRMAC: only source.
        
        # Optional: cleanup
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # ---------- The rest of the methods (forward, training_step, validation_step, etc.) remain the same as original ----------
    # We keep them unchanged for brevity. However, we must ensure _get_hct_embeddings and _extract_per_type_embeds are present.
    # They are already in your original file. We'll include them below as they were.
    
    def _get_hct_embeddings(self, walk_data, node_memory, batch_size):
        # Same as original (copied from your code)
        if self.hct is None or not self.use_hct_hierarchical:
            all_nodes = torch.cat([walk_data['source']['short']['nodes'], walk_data['target']['short']['nodes']], dim=0)
            all_masks = torch.cat([walk_data['source']['short']['masks'], walk_data['target']['short']['masks']], dim=0)
            B_total = all_nodes.size(0)
            flat_n = all_nodes.view(-1)
            flat_f = node_memory[flat_n].view(B_total, all_nodes.size(1), all_nodes.size(2), -1)
            m_sum = all_masks.sum(dim=-1, keepdim=True).clamp(min=1)
            pooled = (flat_f * all_masks.unsqueeze(-1)).sum(dim=2) / m_sum
            w_sum = pooled.sum(dim=1) / pooled.size(1)
            src_simple = w_sum[:batch_size]
            dst_simple = w_sum[batch_size:]
            if src_simple.size(-1) != self.hidden_dim:
                src_simple = self.mem_to_hidden(src_simple)
                dst_simple = self.mem_to_hidden(dst_simple)
            return src_simple, dst_simple, walk_data
        else:
            types = ['short', 'long', 'tawr'] if self.use_multi_scale_walks else ['short']
            combined_walks = {}
            for wt in types:
                if wt not in walk_data['source']: continue
                combined_walks[wt] = {
                    'nodes': torch.cat([walk_data['source'][wt]['nodes'], walk_data['target'][wt]['nodes']], dim=0),
                    'nodes_anon': torch.cat([walk_data['source'][wt]['nodes_anon'], walk_data['target'][wt]['nodes_anon']], dim=0),
                    'masks': torch.cat([walk_data['source'][wt]['masks'], walk_data['target'][wt]['masks']], dim=0),
                    'times': torch.cat([walk_data['source'][wt]['times'], walk_data['target'][wt]['times']], dim=0)
                }
                if 'restart_flags' in walk_data['source'][wt]:
                    combined_walks[wt]['restart_flags'] = torch.cat([walk_data['source'][wt]['restart_flags'], walk_data['target'][wt]['restart_flags']], dim=0)
            hct_out = self.hct(walks_dict=combined_walks, node_memory=node_memory, return_all=False)
            hct_out = self.hct_proj(hct_out)
            src_hct = hct_out[:batch_size]
            dst_hct = hct_out[batch_size:]
            return src_hct, dst_hct, combined_walks

    def _extract_per_type_embeds(self, side_data, memory):
        # Same as original (keep your implementation)
        types = ['short', 'long', 'tawr'] if self.use_multi_scale_walks else ['short']
        max_w = max(self.config.num_walks_short, self.config.num_walks_long, self.config.num_walks_tawr)
        if max_w == 0:
            return None, None
        B = side_data['short']['nodes'].size(0) if 'short' in side_data else 0
        if B == 0:
            return None, None
        D = memory.size(-1)
        device = memory.device
        embeds, masks = [], []
        for t in types:
            if t not in side_data:
                # dummy
                dummy_nodes = torch.zeros(B, 1, 1, dtype=torch.long, device=device)
                dummy_masks = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
                flat_f = memory[dummy_nodes.view(-1)].view(B, 1, 1, D)
                pooled = torch.zeros(B, 1, D, device=device)
                m_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
            else:
                data_t = side_data[t]
                if 'masks' not in data_t or 'nodes' not in data_t:
                    dummy_nodes = torch.zeros(B, 1, 1, dtype=torch.long, device=device)
                    dummy_masks = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
                    flat_f = memory[dummy_nodes.view(-1)].view(B, 1, 1, D)
                    pooled = torch.zeros(B, 1, D, device=device)
                    m_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
                else:
                    nodes = data_t['nodes']
                    m = data_t['masks']
                    if nodes.size(1) == 0:
                        dummy_nodes = torch.zeros(B, 1, 1, dtype=torch.long, device=device)
                        dummy_masks = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
                        flat_f = memory[dummy_nodes.view(-1)].view(B, 1, 1, D)
                        pooled = torch.zeros(B, 1, D, device=device)
                        m_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
                    else:
                        W, L = nodes.shape[1], nodes.shape[2]
                        flat_n = nodes.view(-1).clamp(0, memory.size(0)-1)
                        flat_f = memory[flat_n].view(B, W, L, D)
                        m_sum = m.sum(dim=-1, keepdim=True).clamp(min=1)
                        pooled = (flat_f * m.unsqueeze(-1)).sum(dim=2) / m_sum
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
        if len(embeds) > 0:
            pooled_embeds = torch.stack(embeds, dim=1)
            mask_stack = torch.stack(masks, dim=1)
            if hasattr(self, 'mem_to_hidden') and not isinstance(self.mem_to_hidden, nn.Identity):
                B, T, W, D_mem = pooled_embeds.shape
                flat = pooled_embeds.reshape(-1, D_mem)
                projected = self.mem_to_hidden(flat)
                pooled_embeds = projected.view(B, T, W, -1)
            return pooled_embeds.clone(), mask_stack.clone()
        return None, None

    
    def forward(self, batch, return_probs=False):
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']
        src, dst = self.compute_temporal_embeddings(src_nodes, dst_nodes, timestamps, batch=batch)
        logits = self.link_predictor(src, dst).squeeze(-1)
        if return_probs:
            temp = self.link_predictor.temperature.clamp(0.1, 10.0)
            return logits, torch.sigmoid(logits / temp)
        return logits

    def training_step(self, batch, batch_idx):
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        aug_batch = self.hard_neg_miner(batch, memory=self.sam_module.raw_memory, cooccurrence_matrix=None)
        logits = self(aug_batch)
        loss = F.binary_cross_entropy_with_logits(logits, aug_batch['labels'])
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        weight_decay = float(self.config.weight_decay) if isinstance(self.config.weight_decay, str) else self.config.weight_decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=weight_decay)
        # Simple cosine scheduler (no complex steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def _shared_eval_step(self, batch, batch_idx, prefix: str):
        logits, probs = self(batch, return_probs=True)
        labels = batch['labels'].float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        probs_cpu = probs.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        ap = average_precision_score(labels_cpu, probs_cpu)
        auc = roc_auc_score(labels_cpu, probs_cpu)
        preds = (probs > 0.5).float()
        accuracy = (preds == labels).float().mean().item()
        metrics = {'ap': ap, 'auc': auc, 'accuracy': accuracy}
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_ap", ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss, metrics

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, "val")
        self.validation_step_outputs.append({'probs': torch.tensor(metrics['ap']), 'labels': torch.tensor(metrics['auc'])})
        return {"loss": loss, **metrics}

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, "test")
        if not hasattr(self, 'test_step_outputs'):
            self.test_step_outputs = []
        self.test_step_outputs.append({'probs': torch.tensor(metrics['ap']), 'labels': torch.tensor(metrics['auc'])})
        return {"loss": loss, **metrics}

    def on_validation_epoch_start(self):
        # Backup memory (same as original)
        if self.training and hasattr(self, 'sam_module') and self.sam_module is not None:
            raw_mem = getattr(self.sam_module, 'raw_memory', None)
            if raw_mem is not None:
                try:
                    self._sam_val_backup = raw_mem.clone().detach()
                except Exception as e:
                    logger.warning(f"Could not backup SAM memory: {e}")
                    self._sam_val_backup = None
            else:
                self._sam_val_backup = None
        else:
            self._sam_val_backup = None
        if self.use_st_ode and hasattr(self, 'st_ode') and self.st_ode is not None:
            node_states = getattr(self.st_ode, 'node_states', None)
            if node_states is not None:
                try:
                    self._stode_val_backup = node_states.clone().detach()
                except Exception as e:
                    logger.warning(f"Could not backup ST-ODE states: {e}")
                    self._stode_val_backup = None
            else:
                self._stode_val_backup = None
        else:
            self._stode_val_backup = None

    def on_validation_epoch_end(self):
        with torch.no_grad():
            if hasattr(self, '_sam_val_backup') and self._sam_val_backup is not None:
                if hasattr(self, 'sam_module') and self.sam_module is not None:
                    raw_mem = getattr(self.sam_module, 'raw_memory', None)
                    if raw_mem is not None and hasattr(raw_mem, 'data') and raw_mem.data is not None:
                        try:
                            raw_mem.data.copy_(self._sam_val_backup)
                        except Exception as e:
                            logger.warning(f"Could not restore SAM memory: {e}")
            if self.use_st_ode and hasattr(self, '_stode_val_backup') and self._stode_val_backup is not None and hasattr(self, 'st_ode') and self.st_ode is not None:
                node_states = getattr(self.st_ode, 'node_states', None)
                if node_states is not None and hasattr(node_states, 'data') and node_states.data is not None:
                    try:
                        node_states.data.copy_(self._stode_val_backup)
                    except Exception as e:
                        logger.warning(f"Could not restore ST-ODE states: {e}")
        for attr in ['_sam_val_backup', '_stode_val_backup']:
            if hasattr(self, attr):
                delattr(self, attr)

    def on_test_epoch_end(self):
        if hasattr(self, 'test_step_outputs'):
            self.test_step_outputs.clear()
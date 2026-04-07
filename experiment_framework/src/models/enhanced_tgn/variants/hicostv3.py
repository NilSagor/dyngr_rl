# === HiCoST v3  ===


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


# === HiCoST v3  ===
class HiCoSTv3(L.LightningModule):
    def __init__(self, config: HiCoSTConfig):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)        
        
        self.save_hyperparameters()
        self.config = config  
        
        # Extract all values from config 
        self.num_nodes = config.num_nodes
        self.hidden_dim = config.hidden_dim
        self.memory_dim = config.memory_dim
        self.time_dim = config.time_dim
        self.edge_feat_dim = config.edge_feat_dim
        self.dropout = config.dropout

        # === Projection layers for dimension alignment
        if config.memory_dim != config.hidden_dim:
            self.mem_to_hidden = nn.Linear(config.memory_dim, config.hidden_dim)
            logger.info(f"Added projection: memory_dim({config.memory_dim}) → hidden_dim({config.hidden_dim})")
        else:
            self.mem_to_hidden = nn.Identity()
        
        # YAML may load as strings
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
        
        # === 1. Time Encoder ===
        self.time_encoder = TimeEncoder(config.time_dim)
        
        # === 2. SAM Module ===
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
        logger.info(f"SAM: Using {effective_prototypes} prototype(s) "
                   f"({'attention' if self.use_prototype_attention else 'single'})")

        # === 3. Walk Sampler ===
        if self.use_multi_scale_walks:
            w_len_short = config.walk_length_short
            w_num_short = config.num_walks_short
            w_len_long = config.walk_length_long
            w_num_long = config.num_walks_long
            w_len_tawr = config.walk_length_tawr
            w_num_tawr = config.num_walks_tawr
        else:
            # Ablation: single scale only
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
        
        # === 4. HCT Module ===
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
            
        # === 5. ST-ODE ===
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
            logger.info("ST-ODE: Disabled")
            
        # === 6. Refiner (MRP) ===
        max_walks = max(w_num_short, w_num_long, w_num_tawr)
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
            logger.info(f"Refiner: Gated mutual refinement enabled "
                       f"(fusion={config.mrp_fusion_mode}, pool_heads={config.mrp_pool_attn_heads})")
        else:
            self.refiner = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2), 
                nn.LayerNorm(config.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim) 
            )
            logger.info("Refiner: Simple concat fusion (ablation mode)")
        
                
        
        # === 7. Link Predictor ===
        self.link_predictor = MergeLayer(
            input_dim1=config.hidden_dim,
            input_dim2=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            dropout=config.dropout,
            use_temperature=True
        )
        
        # === 8. Hard Negative Miner ===
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
        
        # Buffers for graph data
        self.edge_index = None
        self.edge_time = None
        self.node_raw_features = None
        self.edge_raw_features = None
        self.validation_step_outputs = []
        
        # Log final config summary
        logger.info(f"HiCoSTv3 Initialized: "
                   f"hidden_dim={config.hidden_dim}, memory_dim={config.memory_dim}, "
                   f"SAM-proto={self.use_prototype_attention}, "
                   f"HCT-hier={self.use_hct_hierarchical}, "
                   f"Gated-refine={self.use_gated_refinement}, "
                   f"Multi-scale={self.use_multi_scale_walks}, "
                   f"ST-ODE={self.use_st_ode}, "
                   f"HNM={self.use_hard_negative_mining}"
                )

    def set_graph(self, edge_index: torch.Tensor, edge_time: torch.Tensor):
        self.edge_index = edge_index
        self.edge_time = edge_time
        self.walk_sampler.update_neighbors(edge_index, edge_time)
        logger.info(f"Graph initialized: {edge_index.shape[1]} edges")

    def set_raw_features(self, node_features: Optional[torch.Tensor], edge_features: Optional[torch.Tensor]):
        if node_features is not None:
            self.node_raw_features = node_features.to(self.device)

    def set_neighbor_finder(self, neighbor_finder):
        if hasattr(neighbor_finder, 'edge_index') and hasattr(neighbor_finder, 'edge_time'):
            device = next(self.parameters()).device
            edge_index = neighbor_finder.edge_index.to(device)
            edge_time = neighbor_finder.edge_time.to(device)
            self.walk_sampler.update_neighbors(edge_index, edge_time, force=True)
            logger.info("Neighbor finder passed to walk sampler.")
        else:
            logger.warning(
                "Neighbor finder does not provide 'edge_index' and 'edge_time'. "
                "Walk sampler will not be initialized."
            )
    
    
    def _get_hct_embeddings(self, walk_data: Dict, node_memory: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        device = node_memory.device
        
        if self.hct is None or not self.use_hct_hierarchical:
             # === Simple pooling for ablation mode ===
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
            
            # Project to hidden_dim if needed
            if src_simple.size(-1) != self.hidden_dim:
                src_simple = self.mem_to_hidden(src_simple)
                dst_simple = self.mem_to_hidden(dst_simple)
            
            return src_simple, dst_simple, walk_data
        # === Full HCT hierarchical mode ===
        types = ['short', 'long', 'tawr'] if self.use_multi_scale_walks else ['short']
        combined_walks = {}
        
        for wt in types:
            if wt not in walk_data['source']: 
                continue
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
        
        hct_out = self.hct(
            walks_dict=combined_walks,
            node_memory=node_memory,
            return_all=False
        )
        hct_out = self.hct_proj(hct_out)
        
        src_hct = hct_out[:batch_size]
        dst_hct = hct_out[batch_size:]
        
        return src_hct, dst_hct, combined_walks
    
    # def _get_hct_embeddings(self, walk_data: Dict, node_memory: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    #     device = node_memory.device
        
    #     if self.hct is None:
    #         # Simple pooling logic for ablation
    #         all_nodes = torch.cat([walk_data['source']['short']['nodes'], walk_data['target']['short']['nodes']], dim=0)
    #         all_masks = torch.cat([walk_data['source']['short']['masks'], walk_data['target']['short']['masks']], dim=0)
            
    #         B_total = all_nodes.size(0)
    #         flat_n = all_nodes.view(-1)
    #         flat_f = node_memory[flat_n].view(B_total, all_nodes.size(1), all_nodes.size(2), -1)
            
    #         m_sum = all_masks.sum(dim=-1, keepdim=True).clamp(min=1)
    #         pooled = (flat_f * all_masks.unsqueeze(-1)).sum(dim=2) / m_sum
    #         w_sum = pooled.sum(dim=1) / pooled.size(1)
            
    #         src_simple = w_sum[:batch_size]
    #         dst_simple = w_sum[batch_size:]
    #         return src_simple, dst_simple, walk_data
        
    #     # Use self.use_multi_scale_walks instead of self.ablation_walk_multi_scale
    #     types = ['short', 'long', 'tawr'] if self.use_multi_scale_walks else ['short']
    #     combined_walks = {}
    #     for wt in types:
    #         if wt not in walk_data['source']: continue
    #         combined_walks[wt] = {
    #             'nodes': torch.cat([walk_data['source'][wt]['nodes'], walk_data['target'][wt]['nodes']], dim=0),
    #             'nodes_anon': torch.cat([walk_data['source'][wt]['nodes_anon'], walk_data['target'][wt]['nodes_anon']], dim=0),
    #             'masks': torch.cat([walk_data['source'][wt]['masks'], walk_data['target'][wt]['masks']], dim=0),
    #             'times': torch.cat([walk_data['source'][wt]['times'], walk_data['target'][wt]['times']], dim=0)
    #         }
    #         if 'restart_flags' in walk_data['source'][wt]:
    #             combined_walks[wt]['restart_flags'] = torch.cat([
    #                 walk_data['source'][wt]['restart_flags'],
    #                 walk_data['target'][wt]['restart_flags']
    #             ], dim=0)
        
    #     hct_out = self.hct(
    #         walks_dict=combined_walks,
    #         node_memory=node_memory,
    #         return_all=False
    #     )
    #     hct_out = self.hct_proj(hct_out)
        
    #     src_hct = hct_out[:batch_size]
    #     dst_hct = hct_out[batch_size:]
        
    #     return src_hct, dst_hct, combined_walks

    def compute_temporal_embeddings(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        batch: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = src_nodes.device
        batch_size = src_nodes.size(0)
        
        # logger.debug("[CTE-1] Getting current memory")
        
        with torch.no_grad():
            current_memory_snapshot = self.sam_module.raw_memory.clone()
        
        src_tensor = src_nodes.to(device).long()
        dst_tensor = dst_nodes.to(device).long()
        ts_tensor = timestamps.to(device).float()

        if ts_tensor.size(0) != batch_size:
            raise RuntimeError(f"Timestamps ({ts_tensor.size(0)}) do not match nodes ({batch_size})")
        
        # 2. Sample Multi-Scale Walks
        # logger.debug("[CTE-2] Sampling walks")
        walk_data = self.walk_sampler(
            source_nodes=src_tensor,
            target_nodes=dst_tensor,
            current_times=ts_tensor,
            memory_states=current_memory_snapshot,
            edge_index=self.edge_index,
            edge_time=self.edge_time
        )
        
        # 3. Encode Walks (HCT)
        # logger.debug("[CTE-3] Encoding walks (HCT)")
        with torch.autograd.set_detect_anomaly(True):
            src_hct, dst_hct, combined_walks = self._get_hct_embeddings(
                walk_data, current_memory_snapshot, batch_size
            )
        
        # 4. ST-ODE Memory Evolution (COMPLETELY DETACHED)
        # logger.debug("[CTE-4] ST-ODE evolution")
        
        # Default: use snapshot (no ST-ODE update)
        src_memory_out = current_memory_snapshot[src_nodes].clone()
        dst_memory_out = current_memory_snapshot[dst_nodes].clone()
        
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
        
        # 5. Prepare embeddings for refiner
        # logger.debug("[CTE-5] Projecting to hidden_dim")
        
        # These are all fresh clones, no view relationships
        src_sam = src_memory_out.clone()
        dst_sam = dst_memory_out.clone()
        src_stode = src_memory_out.clone()  # Same as sam when stode doesn't run, or fresh read
        dst_stode = dst_memory_out.clone()
        
        # Project to hidden_dim
        src_sam_proj = self.mem_to_hidden(src_sam)
        dst_sam_proj = self.mem_to_hidden(dst_sam)
        src_stode_proj = self.mem_to_hidden(src_stode)
        dst_stode_proj = self.mem_to_hidden(dst_stode)
        
        # Extract per-type embeddings
        src_per_type, src_masks = self._extract_per_type_embeds(
            walk_data['source'], current_memory_snapshot
        )
        dst_per_type, dst_masks = self._extract_per_type_embeds(
            walk_data['target'], current_memory_snapshot
        )
        
        # 6. Gated Mutual Refinement
        # logger.debug("[CTE-6] Calling refiner")
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
    
    
    # def compute_temporal_embeddings(
    #     self,
    #     src_nodes: torch.Tensor,
    #     dst_nodes: torch.Tensor,
    #     timestamps: torch.Tensor,
    #     batch: Optional[Dict] = None
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     device = src_nodes.device
    #     batch_size = src_nodes.size(0)
        
    #     logger.debug("[CTE-1] Getting current memory")
    #     current_memory = self.sam_module.raw_memory.detach().clone()
        
    #     src_tensor = src_nodes.to(device).long()
    #     dst_tensor = dst_nodes.to(device).long()
    #     ts_tensor = timestamps.to(device).float()

    #     if ts_tensor.size(0) != batch_size:
    #         raise RuntimeError(f"Internal Error: Timestamps ({ts_tensor.size(0)}) do not match nodes ({batch_size})")
        
    #     # 2. Sample Multi-Scale Walks
    #     logger.debug("[CTE-2] Sampling walks")
    #     walk_data = self.walk_sampler(
    #         source_nodes=src_tensor,
    #         target_nodes=dst_tensor,
    #         current_times=ts_tensor,
    #         memory_states=current_memory,
    #         edge_index=self.edge_index,
    #         edge_time=self.edge_time
    #     )
        
    #     # 3. Encode Walks (HCT)
    #     logger.debug("[CTE-3] Encoding walks (HCT)")
    #     src_hct, dst_hct, combined_walks = self._get_hct_embeddings(
    #         walk_data, current_memory, batch_size
    #     )
        
    #     # 4. Get SAM Memory Embeddings
    #     logger.debug("[CTE-4] Getting SAM embeddings")         
    #     src_sam = current_memory[src_nodes]
    #     dst_sam = current_memory[dst_nodes]
        
        
    #     # 5. Evolve Memory (ST-ODE)
    #     logger.debug("[CTE-5] ST-ODE evolution")
    #     src_stode, dst_stode = None, None
    #     # src_stode = current_memory[src_nodes]
    #     # dst_stode = current_memory[dst_nodes]
        
    #     if self.use_st_ode and self.training and self.st_ode is not None:
    #         self._ode_batch_counter += 1
    #         t_max = timestamps.max()
    #         dt = t_max - self.last_ode_update_time
            
    #         if self._ode_batch_counter >= self.st_ode_update_interval or dt > 100.0:
    #             self._ode_batch_counter = 0
                
    #             free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    #             if free_memory < 2 * 1024**3:
    #                 logger.warning(f"Skipping ST-ODE due to low memory ({free_memory/1024**3:.2f} GB free)")
    #                 src_stode = current_memory[src_nodes].clone()
    #                 dst_stode = current_memory[dst_nodes].clone()
    #             else:
    #                 obs_encodings = self._prepare_ode_observations(walk_data, current_memory, batch_size)
                
    #                 if obs_encodings is not None:
    #                     observed_nodes = obs_encodings['node_map']
    #                     observed_memory = current_memory[observed_nodes]
    #                     try:
    #                         evolved_observed = self.st_ode(
    #                             node_states=observed_memory,
    #                             walk_encodings=obs_encodings['encodings'],
    #                             walk_times=obs_encodings['times'],
    #                             walk_masks=obs_encodings['masks'],
    #                             adj_matrix=obs_encodings['adj']
    #                         )
                            
    #                         with torch.no_grad():
    #                             evolved_detached = evolved_observed.detach()
    #                             self.sam_module.raw_memory.data[observed_nodes] = \
    #                                 0.9 * self.sam_module.raw_memory.data[observed_nodes] + \
    #                                 0.1 * evolved_detached
                                
    #                         self.last_ode_update_time = t_max
                            
    #                         src_stode = self.sam_module.raw_memory[src_nodes].clone().detach()
    #                         dst_stode = self.sam_module.raw_memory[dst_nodes].clone().detach()
    #                     except RuntimeError as e:
    #                         logger.warning(f"ST-ODE failed: {e}")
    #                         src_stode = current_memory[src_nodes]
    #                         dst_stode = current_memory[dst_nodes]
                        
    #                     del obs_encodings
    #                     torch.cuda.empty_cache()

       
        
    #     # 6. Gated Mutual Refinement
    #     src_per_type, src_masks = self._extract_per_type_embeds(walk_data['source'], current_memory)
    #     dst_per_type, dst_masks = self._extract_per_type_embeds(walk_data['target'], current_memory)
        
    #     logger.debug("[CTE-5] Projecting to hidden_dim")
    #     src_sam_proj = self.mem_to_hidden(src_sam)
    #     dst_sam_proj = self.mem_to_hidden(dst_sam)
    #     src_stode_proj = self.mem_to_hidden(src_stode)
    #     dst_stode_proj = self.mem_to_hidden(dst_stode)
        
        
        
    #     # Use self.use_gated_refinement instead of self.ablation_mrp_gating
    #     # if self.use_gated_refinement:
    #     #     final_src, final_dst = self.refiner(
    #     #         src_hct=src_hct, dst_hct=dst_hct,
    #     #         src_sam=src_sam, dst_sam=dst_sam,
    #     #         src_stode=src_stode, dst_stode=dst_stode,
    #     #         src_per_type=src_per_type, dst_per_type=dst_per_type,
    #     #         src_masks=src_masks, dst_masks=dst_masks
    #     #     )
        
    #     # Use projected versions for refiner
    #     logger.debug("[CTE-6] Calling refiner")
    #     if self.use_gated_refinement:
    #         final_src, final_dst, _ = self.refiner(
    #             src_hct=src_hct,         
    #             dst_hct=dst_hct,
    #             src_sam=src_sam_proj,    
    #             dst_sam=dst_sam_proj,
    #             src_stode=src_stode_proj, 
    #             dst_stode=dst_stode_proj,
    #             src_per_type=src_per_type,
    #             dst_per_type=dst_per_type,
    #             src_masks=src_masks,
    #             dst_masks=dst_masks
    #         )
    #     else:
    #         stacked_src = torch.cat([src_sam, src_hct, src_stode], dim=-1)
    #         stacked_dst = torch.cat([dst_sam, dst_hct, dst_stode], dim=-1)
    #         final_src = self.refiner(stacked_src)
    #         final_dst = self.refiner(stacked_dst)
        
    #     return final_src, final_dst

    def _prepare_ode_observations(self, walk_data, memory, batch_size) -> Optional[Dict]:
        device = memory.device
        
        #  Use self.use_multi_scale_walks instead of self.ablation_walk_multi_scale
        key = 'short' if not self.use_multi_scale_walks else 'short'
        
        src_nodes = walk_data['source'][key]['nodes'].view(batch_size, -1)
        tgt_nodes = walk_data['target'][key]['nodes'].view(batch_size, -1)
        src_times = walk_data['source'][key]['times'].view(batch_size, -1)
        tgt_times = walk_data['target'][key]['times'].view(batch_size, -1)
        
        batch_nodes = torch.cat([src_nodes, tgt_nodes], dim=1)
        batch_times = torch.cat([src_times, tgt_times], dim=1)
        
        flat_nodes = batch_nodes.view(-1)
        flat_times = batch_times.view(-1)
        
        valid_mask = flat_nodes > 0
        valid_nodes = flat_nodes[valid_mask]
        valid_times = flat_times[valid_mask]
        
        if valid_nodes.numel() == 0:
            return None
        
        unique_nodes, node_inv = torch.unique(valid_nodes, return_inverse=True)
        unique_times, time_inv = torch.unique(valid_times, return_inverse=True)
        
        num_unique_nodes = unique_nodes.size(0)
        num_unique_times = unique_times.size(0)
        
        max_nodes = min(num_unique_nodes, 500)
        max_times = min(num_unique_times, 10)
        
        if num_unique_nodes > max_nodes:
            node_counts = torch.bincount(node_inv, minlength=num_unique_nodes)
            top_node_indices = torch.topk(node_counts, max_nodes).indices
            unique_nodes = unique_nodes[top_node_indices]
            node_map = torch.full((num_unique_nodes,), -1, dtype=torch.long, device=device)
            node_map[top_node_indices] = torch.arange(max_nodes, device=device)
            node_inv = node_map[node_inv]
            valid_mask = node_inv >= 0
            valid_nodes = valid_nodes[valid_mask]
            valid_times = valid_times[valid_mask]
            node_inv = node_inv[valid_mask]
            time_inv = time_inv[valid_mask]
            num_unique_nodes = max_nodes
        
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
        
        flat_idx = node_inv * num_unique_times + time_inv
        
        node_features = memory[unique_nodes]
        feats_expanded = node_features[node_inv]
        
        total_slots = num_unique_nodes * num_unique_times
        obs_flat = torch.zeros(total_slots, self.memory_dim, device=device)
        counts_flat = torch.zeros(total_slots, device=device)
        
        obs_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, self.memory_dim), feats_expanded)
        counts_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
        
        obs = obs_flat.view(num_unique_nodes, num_unique_times, self.memory_dim)
        counts = counts_flat.view(num_unique_nodes, num_unique_times)
        
        counts = counts.clamp(min=1.0).unsqueeze(-1)
        obs = obs / counts
        
        adj = torch.eye(num_unique_nodes, device=device)
        
        times_matrix = unique_times.view(1, num_unique_times, 1).expand(num_unique_nodes, -1, -1)
        masks = (counts.squeeze(-1) > 0).float().unsqueeze(-1)
        
        return {
            'encodings': obs.unsqueeze(2),
            'times': times_matrix,
            'masks': masks,
            'adj': adj,
            'node_map': unique_nodes,
        }

    def _extract_per_type_embeds(self, side_data, memory) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Use self.use_multi_scale_walks instead of self.ablation_walk_multi_scale
        types = ['short', 'long', 'tawr'] if self.use_multi_scale_walks else ['short']
        
        embeds = []
        masks = []

        # Calculate max_w based on AVAILABLE walks in this specific run
        # We look at the config hparams, but clamp to 0 if the sampler didn't generate them
        # try:
        #     max_w = max(
        #         self.config.num_walks_short if self.config.num_walks_short > 0 else 0,
        #         self.config.num_walks_long if self.config.num_walks_long > 0 else 0,
        #         self.config.num_walks_tawr if self.config.num_walks_tawr > 0 else 0
        #     )
        # except AttributeError:
        #     # Fallback if hparams not accessible yet
        #     max_w = 5 
        
        max_w = max(
            self.config.num_walks_short if self.config.num_walks_short > 0 else 0,
            self.config.num_walks_long if self.config.num_walks_long > 0 else 0,
            self.config.num_walks_tawr if self.config.num_walks_tawr > 0 else 0,
        )

        # max_w = max(self.config.num_walks_short, self.config.num_walks_long, self.config.num_walks_tawr)
        
        if max_w == 0:
            # Edge case: no walks at all
            return None, None
        
        B = side_data['short']['nodes'].size(0) if 'short' in side_data else 0
        if B == 0:
            return None, None
        
        D = memory.size(-1)
        device = memory.device
        
        for t in types:            
            if t not in side_data:
                logger.warning(f"Walk type '{t}' missing from side_data. Creating zero placeholders.")
                # Create zero tensors for missing types
                dummy_nodes = torch.zeros(B, 1, 1, dtype=torch.long, device=device)
                dummy_masks = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
                # Process dummy to match expected shape [B, 1, D] after pooling
                flat_f = memory[dummy_nodes.view(-1)].view(B, 1, 1, D)
                pooled = torch.zeros(B, 1, D, device=device)
                m_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
            else:
                data_t = side_data[t]
                
                #  Does 'masks' key exist
                if 'masks' not in data_t or 'nodes' not in data_t:
                    logger.warning(f"Walk type '{t}' missing 'masks' or 'nodes'. Creating zero placeholders.")
                    dummy_nodes = torch.zeros(B, 1, 1, dtype=torch.long, device=device)
                    dummy_masks = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
                    flat_f = memory[dummy_nodes.view(-1)].view(B, 1, 1, D)
                    pooled = torch.zeros(B, 1, D, device=device)
                    m_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
                else:
                    nodes = data_t['nodes']
                    m = data_t['masks']
                    
                    # Handle case where num_walks=0 results in empty tensor dim 1
                    if nodes.size(1) == 0:
                        # Create dummy single walk to avoid reshape errors downstream
                        dummy_nodes = torch.zeros(B, 1, 1, dtype=torch.long, device=device)
                        dummy_masks = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
                        flat_f = memory[dummy_nodes.view(-1)].view(B, 1, 1, D)
                        pooled = torch.zeros(B, 1, D, device=device)
                        m_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
                    else:
                        W, L = nodes.shape[1], nodes.shape[2]
                        
                        flat_n = nodes.view(-1)
                        # Clamp indices just in case
                        flat_n = torch.clamp(flat_n, 0, memory.size(0)-1)
                        flat_f = memory[flat_n].view(B, W, L, D)
                        
                        m_sum = m.sum(dim=-1, keepdim=True).clamp(min=1)
                        pooled = (flat_f * m.unsqueeze(-1)).sum(dim=2) / m_sum
                        
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
            
        # Stack only if we have valid data
        if len(embeds) > 0:
            pooled_embeds = torch.stack(embeds, dim=1) 
            mask_stack = torch.stack(masks, dim=1)            
            
            # Project if needed
            if hasattr(self, 'mem_to_hidden') and not isinstance(self.mem_to_hidden, nn.Identity):
                B, T, W, D_mem = pooled_embeds.shape
                flat = pooled_embeds.reshape(-1, D_mem)
                projected = self.mem_to_hidden(flat)
                pooled_embeds = projected.view(B, T, W, -1)
            
            # Clone to ensure no shared memory
            return pooled_embeds.clone(), mask_stack.clone()
        else:
            return None, None

    def forward(self, batch: Dict, return_probs: bool = False) -> torch.Tensor:
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']  
        
        src, dst = self.compute_temporal_embeddings(
            src_nodes, dst_nodes, timestamps, batch=batch
        )
        logits = self.link_predictor(src, dst).squeeze(-1)
        
        if return_probs:
            temp = self.link_predictor.temperature.clamp(0.1, 10.0)
            return logits, torch.sigmoid(logits / temp)
        
        return logits

    def training_step(self, batch, batch_idx):
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        aug_batch = self.hard_neg_miner(
            batch, 
            memory=self.sam_module.raw_memory,
            cooccurrence_matrix=None
        )
        
        logits = self(aug_batch)
        loss = F.binary_cross_entropy_with_logits(logits, aug_batch['labels'])
        
        # Aggressive cleanup every N steps
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()  # import gc at top
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # if self.training:
        #     src = batch['src_nodes']
        #     dst = batch['dst_nodes']
        #     ts = batch['timestamps']
        #     edge_feat = batch.get('edge_features', torch.zeros(len(src), self.edge_feat_dim, device=self.device))
            
        #     with torch.no_grad():
        #         self.sam_module.update_memory_batch(
        #             source_nodes=src.detach(),
        #             target_nodes=dst.detach(),
        #             edge_features=edge_feat.detach(),
        #             current_time=ts.detach(),
        #             node_features=self.node_raw_features.detach() if self.node_raw_features is not None else None
        #         )
                
        #         all_nodes = torch.unique(torch.cat([src, dst]))
        #         self.sam_module.reset_prototypes_if_needed(all_nodes)
        pass

    def configure_optimizers(self):        
        logger.debug(f"self.hparams type: {type(self.hparams)}")
        logger.debug(f"self.hparams keys: {list(self.hparams.keys()) if hasattr(self.hparams, 'keys') else 'N/A'}")
        logger.debug(f"self.config type: {type(self.config)}")
        logger.debug(f"self.config keys: {[k for k in dir(self.config) if not k.startswith('_')][:10]}")
        
        # Try accessing warmup_epochs
        try:
            we = self.hparams.warmup_epochs
            logger.debug(f"self.hparams.warmup_epochs = {we}")
        except AttributeError as e:
            logger.error(f"❌ self.hparams.warmup_epochs failed: {e}")
        
        try:
            we = self.config.warmup_epochs
            logger.debug(f"✓ self.config.warmup_epochs = {we}")
        except AttributeError as e:
            logger.error(f"❌ self.config.warmup_epochs failed: {e}")
        
        
        
        
        weight_decay = self.config.weight_decay
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
            
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=weight_decay
        )
        
        steps_per_epoch = None
        
        if hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader is not None:
            try:
                dl = self.trainer.train_dataloader
                if isinstance(dl, list):
                    dl = dl[0]
                if hasattr(dl, '__len__'):
                    steps_per_epoch = len(dl) // self.trainer.accumulate_grad_batches
            except (TypeError, AttributeError):
                pass
        
        if steps_per_epoch is None and hasattr(self.trainer, 'estimated_stepping_batches'):
            if self.trainer.estimated_stepping_batches is not None:
                total_steps = self.trainer.estimated_stepping_batches
                if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs > 0:
                    steps_per_epoch = total_steps // self.trainer.max_epochs
        
        if steps_per_epoch is None or steps_per_epoch == 0:
            logger.warning("Could not determine steps_per_epoch. Defaulting to 862.")
            steps_per_epoch = 862

        logger.info(f"Configuring LR Scheduler with steps_per_epoch={steps_per_epoch}")

        scheduler = self.hard_neg_miner.get_cosine_scheduler(
            optimizer,
            num_epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=self.config.warmup_epochs,
            min_lr_ratio=self.config.min_lr_ratio
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

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
        
        metrics = {
            'ap': ap,
            'auc': auc,
            'accuracy': accuracy
        }
        
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
        """Safely backup memory states before validation."""
       
        if self.training and hasattr(self, 'sam_module') and self.sam_module is not None:
            raw_mem = getattr(self.sam_module, 'raw_memory', None)
            if raw_mem is not None:
                try:
                    self._sam_val_backup = raw_mem.clone().detach()
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Could not backup SAM memory: {e}")
                    self._sam_val_backup = None
            else:
                self._sam_val_backup = None
        else:
            self._sam_val_backup = None
        
        # Backup ST-ODE states if enabled
        if (self.use_st_ode and 
            hasattr(self, 'st_ode') and self.st_ode is not None):
            node_states = getattr(self.st_ode, 'node_states', None)
            if node_states is not None:
                try:
                    self._stode_val_backup = node_states.clone().detach()
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Could not backup ST-ODE states: {e}")
                    self._stode_val_backup = None
            else:
                self._stode_val_backup = None
        else:
            self._stode_val_backup = None
    
    def on_validation_epoch_end(self):
        """Safely restore memory states after validation."""
        # Restore SAM memory
        with torch.no_grad():
            
            if hasattr(self, '_sam_val_backup') and self._sam_val_backup is not None:
                if hasattr(self, 'sam_module') and self.sam_module is not None:
                    raw_mem = getattr(self.sam_module, 'raw_memory', None)
                    if raw_mem is not None and hasattr(raw_mem, 'data') and raw_mem.data is not None:
                        try:
                            raw_mem.data.copy_(self._sam_val_backup)
                        except (AttributeError, RuntimeError) as e:
                            logger.warning(f"Could not restore SAM memory: {e}")
            
            # Restore ST-ODE states
            if (self.use_st_ode and 
                hasattr(self, '_stode_val_backup') and self._stode_val_backup is not None and
                hasattr(self, 'st_ode') and self.st_ode is not None):
                node_states = getattr(self.st_ode, 'node_states', None)
                if node_states is not None and hasattr(node_states, 'data') and node_states.data is not None:
                    try:
                        node_states.data.copy_(self._stode_val_backup)
                    except (AttributeError, RuntimeError) as e:
                        logger.warning(f"Could not restore ST-ODE states: {e}")
        
        # Clean up backup attributes (outside no_grad is fine)
        for attr in ['_sam_val_backup', '_stode_val_backup']:
            if hasattr(self, attr):
                delattr(self, attr)

    def on_test_epoch_end(self):
        if hasattr(self, 'test_step_outputs'):
            self.test_step_outputs.clear()

    def _ensure_dim(self, tensor, expected_dim, name):
        if tensor.size(-1) != expected_dim:
            logger.warning(f"{name} has wrong dim: {tensor.size(-1)}, expected {expected_dim}")
            # Project using mem_to_hidden
            return self.mem_to_hidden(tensor)
        return tensor
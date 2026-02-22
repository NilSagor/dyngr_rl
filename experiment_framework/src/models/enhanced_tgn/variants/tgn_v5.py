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


class TGNv5(BaseEnhancedTGN):
    """
    TGN + SAM + Multi-Scale Walk Sampler + Hierarchical Co-occurrence Transformer (HCT)
    
    Architecture:
    1. SAM maintains stable node memory via prototypes
    2. Multi-scale walk sampler generates short/long/TAWR walks with anonymization
    3. HCT processes walks: intra-walk -> co-occurrence -> inter-walk -> fusion
    4. Combine HCT walk embeddings with base TGN embeddings
    5. Link prediction on combined representations
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
        torch.autograd.set_detect_anomaly(True)
        
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
        
        # Fusion layer: combine HCT walk embeddings with base TGN embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._walk_cache = {}
        self._cache_walks = False  # Toggle

        logger.info(f"TGNv5 initialized: SAM({num_prototypes}p) + WalkSampler + "
                   f"HCT({hct_d_model}d, {hct_nhead}h)" if use_hct else "SimpleWalkEncoder")
    
    
    
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
        scores = self.link_predictor(
            torch.cat([source_emb, dest_emb], dim=-1)
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
                edge_index=None,
                edge_time=None
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
        # walk_data = self.walk_sampler(
        #     source_nodes=src_tensor,
        #     target_nodes=dst_tensor,
        #     current_times=ts_tensor,
        #     memory_states=self.sam_module.raw_memory,  # For TAWR restart prob
        #     edge_index=None,
        #     edge_time=None
        # )
        
        # 2. HCT encodes walks by looking up SAM memory
        # 2. HCT encodes walks by looking up SAM memory
        if self.use_hct:
            # Pass walks_dict with 'nodes' (for lookup) and 'nodes_anon' (for co-occurrence)
            hct_src_output = self.hct(
                walks_dict=walk_data['source'],
                node_memory=self.sam_module.raw_memory,
                memory_proj=self.memory_proj,
                return_all=True  # Return dict with all intermediate outputs
            )
            
            hct_dst_output = self.hct(
                walks_dict=walk_data['target'],
                node_memory=self.sam_module.raw_memory,
                memory_proj=self.memory_proj,
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
        
    
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Override to use SAM memory instead of GRU memory."""
        return self.sam_module.get_memory(node_ids)
       
    def _get_cache_key(self, src_tensor: torch.Tensor, ts_tensor: torch.Tensor) -> tuple:
        """Safe cache key generation."""
        return (src_tensor.data_ptr(), src_tensor.shape, ts_tensor[0].item())
    
    def on_fit_start(self):
        """Initialize SAM memory at start of training."""
        super().on_fit_start()
        if self.use_memory:
            self.sam_module.reset_memory()
            logger.info(" SAM memory initialized for training")
    
       
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.training and self.use_memory and hasattr(self, '_sam_batch_buffer'):
            buffer = self._sam_batch_buffer
            
            # Log L2 norm (more meaningful for symmetric distributions)
            mem_before_norm = torch.norm(self.sam_module.raw_memory).item()
            mem_before_std = self.sam_module.raw_memory.std().item()
            
            # FIX: Replace ... with actual arguments from buffer
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
            
            # Clear buffer
            # CRITICAL: Delete buffer to prevent memory leak
            delattr(self, '_sam_batch_buffer')
    
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


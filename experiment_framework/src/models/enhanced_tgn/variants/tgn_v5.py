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
        walk_length_short: int = 3,
        walk_length_long: int = 10,
        walk_length_tawr: int = 8,
        num_walks_short: int = 10,
        num_walks_long: int = 5,
        num_walks_tawr: int = 5,
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




        # Fusion layer: combine HCT walk embeddings with base TGN embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        logger.info(f"TGNv5 initialized: SAM({num_prototypes}p) + WalkSampler + "
                   f"HCT({hct_d_model}d, {hct_nhead}h)" if use_hct else "SimpleWalkEncoder")
    
    
    
      
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Single unified method
        source_emb, dest_emb = self.compute_temporal_embeddings(
            batch['src_nodes'],
            batch['dst_nodes'],
            batch['timestamps']
        )
        
        if self.training and self.use_memory:
            self._compute_and_store_messages(batch)
        
        scores = self.link_predictor(
            torch.cat([source_emb, dest_emb], dim=-1)
        ).squeeze(-1)
        
        if self.training and self.use_memory:
            self._aggregate_and_update_memory()  # Call here or in on_train_batch_end
            self.train_batch_counter += 1
        
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
        
        # 1. Generate walks (indices only)
        walk_data = self.walk_sampler(
            source_nodes=src_tensor,
            target_nodes=dst_tensor,
            current_times=ts_tensor,
            memory_states=self.sam_module.raw_memory,  # For TAWR restart prob
            edge_index=None,
            edge_time=None
        )
        
        # 2. HCT encodes walks by looking up SAM memory
        if self.use_hct:
            # Pass walks_dict with 'nodes' (for lookup) and 'nodes_anon' (for co-occurrence)
            hct_src_emb = self.hct(
                walks_dict=walk_data['source'],
                node_memory=self.sam_module.raw_memory,
                memory_proj=self.memory_proj,
                return_all=False
            )  # [B, d_model]
            
            hct_dst_emb = self.hct(
                walks_dict=walk_data['target'],
                node_memory=self.sam_module.raw_memory,
                memory_proj=self.memory_proj,
                return_all=False
            )
            
            # Project to TGN hidden dimension
            walk_src_emb = self.hct_to_tgn(hct_src_emb)  # [B, hidden_dim]
            walk_dst_emb = self.hct_to_tgn(hct_dst_emb)
        else:
            # Fallback: simple mean of walk node memories (no HCT)
            walk_src_emb = self._simple_walk_embed(walk_data['source'])
            walk_dst_emb = self._simple_walk_embed(walk_data['target'])
        
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

    def _compute_and_store_messages(self, batch):
        """SAM replaces message aggregation - store raw interactions for SAM update."""
        if not self.use_memory or self.memory is None:
            logger.warning("Trying to store messages but use_memory=False")
            return
        
        device = self.device
        src_nodes = batch['src_nodes'].to(device)
        dst_nodes = batch['dst_nodes'].to(device)
        timestamps = batch['timestamps'].to(device)
        
        # Get edge features (zero if missing)
        if 'edge_features' in batch and batch['edge_features'] is not None:
            edge_feats = batch['edge_features'].to(device)
        else:
            edge_feats = torch.zeros(len(src_nodes), self.edge_features_dim, device=device)
        
        #  Store interactions for SAM update (not GRU messages)
        node_id_to_interactions = {}
        for node, neighbor, ts, edge_feat in zip(
            src_nodes, dst_nodes, timestamps, edge_feats
        ):
            node_id_to_interactions.setdefault(node.item(), []).append(
                (neighbor.item(), ts.item(), edge_feat)
            )
        for node, neighbor, ts, edge_feat in zip(
            dst_nodes, src_nodes, timestamps, edge_feats
        ):
            node_id_to_interactions.setdefault(node.item(), []).append(
                (neighbor.item(), ts.item(), edge_feat)
            )
        
        # Store in memory.messages (SAM will process during _aggregate_and_update_memory)
        self.memory.store_raw_messages(
            list(node_id_to_interactions.keys()), 
            node_id_to_interactions
        )
    
    def _aggregate_and_update_memory(self):
        """Replace GRU update with SAM update using stored interactions."""
        if not self.use_memory or self.memory is None:
            return
        
        nodes_with_interactions = list(self.memory.messages.keys())
        if not nodes_with_interactions:
            return
        
        # Process SAM updates in batches to avoid OOM
        batch_size = 2048
        for i in range(0, len(nodes_with_interactions), batch_size):
            batch_nodes = nodes_with_interactions[i:i+batch_size]
            
            # Prepare batched inputs for SAM update
            src_nodes_list = []
            tgt_nodes_list = []
            timestamps_list = []
            edge_feats_list = []
            
            for node_id in batch_nodes:
                for neighbor_id, ts, edge_feat in self.memory.messages[node_id]:
                    src_nodes_list.append(node_id)
                    tgt_nodes_list.append(neighbor_id)
                    timestamps_list.append(ts)
                    edge_feats_list.append(edge_feat)
            
            if not src_nodes_list:
                continue
            
            # Convert to tensors
            src_tensor = torch.tensor(src_nodes_list, dtype=torch.long, device=self.device)
            tgt_tensor = torch.tensor(tgt_nodes_list, dtype=torch.long, device=self.device)
            ts_tensor = torch.tensor(timestamps_list, dtype=torch.float32, device=self.device)
            edge_tensor = torch.stack(edge_feats_list) if edge_feats_list else \
                torch.zeros(len(src_nodes_list), self.edge_features_dim, device=self.device)
            
            # SAM update replaces GRU update
            self.sam_module.update_memory_batch(
                source_nodes=src_tensor,
                target_nodes=tgt_tensor,
                edge_features=edge_tensor,
                current_time=ts_tensor,
                node_features=self.node_raw_features
            )
        
        # Clear processed interactions
        self.memory.clear_messages(nodes_with_interactions)
    
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Override to use SAM memory instead of GRU memory."""
        return self.sam_module.get_memory(node_ids)
       
    
    def on_fit_start(self):
        """Initialize SAM memory at start of training."""
        super().on_fit_start()
        if self.use_memory:
            # Reset SAM memory (not GRU memory)
            self.sam_module.reset_memory()
            logger.info(" SAM memory initialized for training")
            
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Override to update SAM memory after optimizer step and avoid
        calling base method which triggers detach_memory() on raw interactions.
        """
        if self.training and self.use_memory:
            self._aggregate_and_update_memory()

        # Optional: log gradient norms (copied from base class)
        if batch_idx % 100 == 0:
            total_grad_norm = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
            logger.debug(f"Batch {batch_idx}: Total gradient norm = {total_grad_norm:.4f}")

            # Optionally log SAM memory update norm
            if hasattr(self, '_memory_before_update'):
                mem_update = torch.norm(self.sam_module.raw_memory - self._memory_before_update)
                logger.debug(f"SAM memory update norm = {mem_update:.4f}")

   
    
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
    
    def on_test_epoch_start(self):
        """Reset SAM memory for cold-start test evaluation."""
        super().on_test_epoch_start()
        if self.use_memory:
            self.sam_module.reset_memory()
            logger.info("SAM memory reset for TEST (cold-start evaluation)")


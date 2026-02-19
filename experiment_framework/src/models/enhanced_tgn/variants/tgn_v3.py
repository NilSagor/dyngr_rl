import torch 
import torch.nn as nn
import numpy as np

from typing import Dict, Optional, Tuple
from loguru import logger

from ..base_enhance_tgn import BaseEnhancedTGN
from ..component.time_encoder import TimeEncoder
from ..component.sam_module import StabilityAugmentedMemory





class TGNv3(BaseEnhancedTGN):
    """
    TGN with Stability-Augmented Memory (SAM)
    Replaces GRU memory updater with prototype-based stable memory updates.
    Maintains full compatibility with TGN pipeline and evaluation protocols.
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
        memory_updater_type: str = "sam",  # Critical: use SAM instead of GRU
        num_prototypes: int = 5,
        similarity_metric: str = "cosine",
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
        
        logger.info(f"✓ SAM initialized: {num_prototypes} prototypes, {similarity_metric} similarity")
    
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
    
    def compute_temporal_embeddings_pair(self,
                                     source_nodes: torch.Tensor,
                                     destination_nodes: torch.Tensor,
                                     edge_times: torch.Tensor,
                                     n_neighbors: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device

        # Convert to tensors if needed
        if isinstance(source_nodes, np.ndarray):
            src_tensor = torch.from_numpy(source_nodes).long().to(device)
            dst_tensor = torch.from_numpy(destination_nodes).long().to(device)
            ts_tensor = torch.from_numpy(edge_times).float().to(device)
        else:
            src_tensor = source_nodes.to(device)
            dst_tensor = destination_nodes.to(device)
            ts_tensor = edge_times.to(device)

        if self.embedding_module is not None:
            # Use the full raw memory matrix (updated via SAM)
            source_emb = self.embedding_module.compute_embedding(
                memory=self.sam_module.raw_memory,          # <-- full matrix
                source_nodes=source_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            destination_emb = self.embedding_module.compute_embedding(
                memory=self.sam_module.raw_memory,          # <-- full matrix
                source_nodes=destination_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            return source_emb, destination_emb

        # Fallback for models without embedding module
        if self.node_raw_features is not None:
            src_feat = self.node_raw_features[src_tensor]
            dst_feat = self.node_raw_features[dst_tensor]
        else:
            src_feat = self.node_embedding(src_tensor)
            dst_feat = self.node_embedding(dst_tensor)

        return src_feat, dst_feat
    
    
    # def compute_temporal_embeddings_pair(self,
    #                                      source_nodes: torch.Tensor,
    #                                      destination_nodes: torch.Tensor,
    #                                      edge_times: torch.Tensor,
    #                                      n_neighbors: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Use SAM-stabilized memory for embeddings."""
    #     device = self.device
        
    #     # Convert to tensors if needed
    #     if isinstance(source_nodes, np.ndarray):
    #         src_tensor = torch.from_numpy(source_nodes).long().to(device)
    #         dst_tensor = torch.from_numpy(destination_nodes).long().to(device)
    #         ts_tensor = torch.from_numpy(edge_times).float().to(device)
    #     else:
    #         src_tensor = source_nodes.to(device)
    #         dst_tensor = destination_nodes.to(device)
    #         ts_tensor = edge_times.to(device)
        
    #     # ✅ CRITICAL FIX #4: Get SAM-stabilized memory (not raw GRU memory)
    #     if self.embedding_module is not None:
    #         # Get stabilized memories for current timestamp
    #         source_memory = self.sam_module.get_stabilized_memory(src_tensor, ts_tensor)
    #         destination_memory = self.sam_module.get_stabilized_memory(dst_tensor, ts_tensor)
            
    #         # Use stabilized memories in embedding computation
    #         source_emb = self.embedding_module.compute_embedding(
    #             memory=source_memory,  # SAM-stabilized
    #             source_nodes=source_nodes,
    #             timestamps=edge_times,
    #             n_layers=self.num_layers,
    #             n_neighbors=n_neighbors
    #         )
    #         destination_emb = self.embedding_module.compute_embedding(
    #             memory=destination_memory,  # SAM-stabilized
    #             source_nodes=destination_nodes,
    #             timestamps=edge_times,
    #             n_layers=self.num_layers,
    #             n_neighbors=n_neighbors
    #         )
    #         return source_emb, destination_emb
        
    #     # Fallback for models without embedding module
    #     if self.node_raw_features is not None:
    #         src_feat = self.node_raw_features[src_tensor]
    #         dst_feat = self.node_raw_features[dst_tensor]
    #     else:
    #         src_feat = self.node_embedding(src_tensor)
    #         dst_feat = self.node_embedding(dst_tensor)
        
    #     return src_feat, dst_feat
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = self.device

        if self.embedding_module is None:
            logger.error("CRITICAL: Embedding module is None! Using fallback features only.")

        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']

        source_emb, dest_emb = self.compute_temporal_embeddings_pair(
            source_nodes=src_nodes,
            destination_nodes=dst_nodes,
            edge_times=timestamps,
            n_neighbors=self.n_neighbors
        )

        if self.training and self.use_memory:
            # Store messages for this batch (to be used in on_after_backward)
            self._compute_and_store_messages(batch)
            # Memory update is NOT performed here – moved to on_after_backward

        if torch.all(source_emb == 0) or torch.all(dest_emb == 0):
            logger.warning("All-zero embeddings detected!")

        combined = torch.cat([source_emb, dest_emb], dim=-1)
        scores = self.link_predictor(combined).squeeze(-1)

        if self.training and self.use_memory and self.memory is not None:
            self._memory_initialized = True
            self.train_batch_counter += 1

        return scores
    
    def on_fit_start(self):
        """Initialize SAM memory at start of training."""
        super().on_fit_start()
        if self.use_memory:
            # Reset SAM memory (not GRU memory)
            self.sam_module.reset_memory()
            logger.info("✓ SAM memory initialized for training")
            
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
            logger.info("✓ Cloned SAM memory for validation")
    
    def on_validation_epoch_end(self):
        """Restore SAM memory after validation."""
        super().on_validation_epoch_end()
        if self.use_memory:
            self.sam_module.raw_memory.data.copy_(self._sam_validation_memory)
            self.sam_module.last_update.data.copy_(self._sam_validation_last_update)
            logger.info("✓ Restored SAM memory after validation")
    
    def on_test_epoch_start(self):
        """Reset SAM memory for cold-start test evaluation."""
        super().on_test_epoch_start()
        if self.use_memory:
            self.sam_module.reset_memory()
            logger.info("✓ SAM memory reset for TEST (cold-start evaluation)")





# class TGNv3(BaseEnhancedTGN):
#     """
#     TGN + Time Encoder + Stability Augmented Memory (SAM)
#     Replaces original TGN memory module with prototype-based stable memory.
#     """
    
#     def __init__(
#         self,
#         num_nodes: int,
#         hidden_dim: int = 172,
#         time_encoding_dim: int = 32,
#         memory_dim: int = 172,
#         edge_feat_dim: int = 172,
#         node_feat_dim: int = 0,
#         num_prototypes: int = 5,
#         similarity_metric: str = "cosine",
#         dropout: float = 0.1,
#         device: str = "cuda",
#         # ... other TGN params
#         **kwargs
#     ):
#         # Initialize base but skip original memory initialization
#         super().__init__(
#             num_nodes=num_nodes,
#             hidden_dim=hidden_dim,
#             time_encoding_dim=time_encoding_dim,
#             memory_dim=memory_dim,
#             use_memory=True,  # We use SAM instead
#             **kwargs
#         )
        
#         # Replace base time encoder if needed (SAM has its own)
#         # Or ensure they share the same dimension
#         assert time_encoding_dim == memory_dim, \
#             "SAM expects time_dim == memory_dim for concatenation"
        
#         # Initialize Stability Augmented Memory
#         self.sam_module = StabilityAugmentedMemory(
#             num_nodes=num_nodes,
#             memory_dim=memory_dim,
#             node_feat_dim=node_feat_dim,
#             edge_feat_dim=edge_feat_dim,
#             time_dim=time_encoding_dim,  # Must match!
#             num_prototypes=num_prototypes,
#             similarity_metric=similarity_metric,
#             dropout=dropout,
#             device=device
#         )
        
#         # Disable original TGN memory if base class created it
#         # This depends on your BaseEnhancedTGN implementation
#         self._use_sam = True
        
#     def forward(
#         self,
#         source_nodes: torch.Tensor,
#         target_nodes: torch.Tensor,
#         timestamps: torch.Tensor,
#         edge_features: Optional[torch.Tensor] = None,
#         node_features: Optional[torch.Tensor] = None,
#         **kwargs
#     ) -> Tuple[torch.Tensor, Dict]:
#         """
#         Forward pass using SAM for memory updates.
        
#         Returns:
#             predictions: [batch_size, num_classes] or [batch_size] for binary
#             aux_info: dict with attention weights for analysis
#         """
#         # Update memory via SAM
#         update_info = self.sam_module.update_memory_batch(
#             source_nodes=source_nodes,
#             target_nodes=target_nodes,
#             edge_features=edge_features if edge_features is not None else \
#                 torch.zeros(source_nodes.size(0), self.sam_module.edge_feat_dim, 
#                            device=source_nodes.device),
#             current_time=timestamps,
#             node_features=node_features
#         )
        
#         # Get stabilized memories for embedding generation
#         src_memory = update_info['source_memory']  # Already stabilized
#         tgt_memory = update_info['target_memory']
        
#         # ... rest of TGN forward (message passing, embedding, etc.)
#         # You need to integrate with your graph attention/embedding modules
        
#         # Example: compute edge representations
#         edge_repr = self.compute_edge_representation(
#             src_memory, tgt_memory, edge_features, timestamps
#         )
        
#         # Predict link probability
#         predictions = self.predictor(edge_repr)
        
#         return predictions, {
#             'source_attention': update_info['source_attention'],
#             'target_attention': update_info['target_attention'],
#             'attention_weights': update_info['source_attention']['attention_weights']
#         }
    
#     def compute_edge_representation(
#         self,
#         src_memory: torch.Tensor,
#         tgt_memory: torch.Tensor,
#         edge_features: Optional[torch.Tensor],
#         timestamps: torch.Tensor
#     ) -> torch.Tensor:
#         """Combine source/target memories into edge representation."""
#         # Implement based on your TGN architecture
#         # Common: concatenate + MLP, or Hadamard product, etc.
#         combined = torch.cat([src_memory, tgt_memory], dim=-1)
#         return self.edge_mlp(combined)  # You need to define this
    
#     def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
#         """Override to use SAM."""
#         return self.sam_module.get_memory(node_ids)
    
#     def reset_memory(self):
#         """Reset SAM memory."""
#         self.sam_module.reset_memory()
        
#     def get_stabilized_memory(
#         self, 
#         node_ids: torch.Tensor, 
#         current_time: torch.Tensor
#     ) -> torch.Tensor:
#         """Get memory without updating (for inference)."""
#         return self.sam_module.get_stabilized_memory(node_ids, current_time)
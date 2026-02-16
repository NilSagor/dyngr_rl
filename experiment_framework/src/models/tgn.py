"""Temporal Graph Networks (TGN) implementation."""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger
from collections import defaultdict

from .base_model import BaseDynamicGNN, TimeEncoder
from .tgn_module.temporal_attention import TemporalAttentionLayer, MergeLayer
from .tgn_module.memory import Memory
from .tgn_module.memory_updater import get_memory_updater
from .tgn_module.message_aggregator import get_message_aggregator
from .tgn_module.message_function import get_message_function
from .tgn_module.embedding_module import get_embedding_module


class TGN(BaseDynamicGNN):
    """Temporal Graph Networks for Deep Learning on Dynamic Graphs.
   
    """    
    def __init__(
        self,
        num_nodes: int,
        node_features: int=0,
        hidden_dim: int=172,
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
        memory_update_at_start: bool = True,
        embedding_module_type: str = "graph_attention",
        message_function_type: str = "mlp",
        aggregator_type: str = "last", 
        memory_updater_type: str = "gru",               
    ):        
        super().__init__(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=hidden_dim,
            time_encoding_dim=time_encoding_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,            
        )
        self.save_hyperparameters(ignore=['neighbor_finder'])
        # VALIDATE num_nodes immediately
        # if num_nodes < 1000:  # Wikipedia needs 9228
        #     raise ValueError(
        #         f"CRITICAL: TGN initialized with num_nodes={num_nodes}! "
        #         f"Expected >=1000 for real datasets. Check config/model reconstruction."
        #     )

        # RELAXED check - only catch obviously wrong values
        if num_nodes < 2:  # Minimum for a graph
            raise ValueError(f"CRITICAL: TGN initialized with num_nodes={num_nodes}! Must be >= 2")
        
        # Optional warning for small datasets
        if num_nodes < 100:
            logger.warning(f"Small dataset detected: num_nodes={num_nodes}. "
                        f"Verify this is expected (UNvote=201, USLegis=225, etc.)")


        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.edge_features_dim = edge_features_dim
        self.n_heads = n_heads
        self.n_neighbors = n_neighbors
        self.use_memory = use_memory
        self.memory_update_at_start = memory_update_at_start
        self.embedding_module_type = embedding_module_type
        self.message_function_type = message_function_type
        self.aggregator_type = aggregator_type
        self.memory_updater_type = memory_updater_type

        # STEP 3: Store dependencies BEFORE module initialization
        self.neighbor_finder = None
        self.embedding_module = None
        
        self.register_buffer('_num_nodes', torch.tensor(num_nodes, dtype=torch.long))
        # # Store raw features as buffers        
        # Register buffers (will be filled later)
        # self.register_buffer('_node_raw_features', None)
        # self.register_buffer('_edge_raw_features', None)      
        
        # self.node_raw_features = None  # Will point to buffer
        # self.edge_raw_features = None
        
        # Memory management for TGN         
        self._memory_initialized = False

               
        # Add layer norm to attention
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Always initialize node embedding layer (even when node_features>0)
        # Required for structural datasets and Wikipedia (no raw features)
        self.node_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        
        
        # Initialize modules
        self._init_modules()

        # Ensure memory is on correct device
        if self.use_memory and self.memory is not None:
            self.memory = self.memory.to(self.device)

        if self.use_memory and self.node_features == 0:
            self.node_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)

        #  Ensure time encoder matches config dimension
        self.time_encoding_dim = time_encoding_dim  # Store for message function
        
        # Initialize time encoder with EXACT config dimension
        self.time_encoder = TimeEncoder(time_dim=time_encoding_dim)
        logger.info(f" Time encoder initialized with dim={time_encoding_dim}")
        # self.link_predictor = MergeLayer()

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        

        # Initialize weights properly
        self._init_weights()

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Add training state tracking
        self.train_batch_counter = 0
        logger.info(f"TGN initialized with use_memory={self.use_memory}")       

      
    
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
             
        # Forward pass
        logits = self.forward(batch)
        
        labels = batch['labels'].float()


        # compute loss
        loss = self.loss_fn(logits, labels)
                
        # prepare message for memory update (but not update yet) 
        # with torch.no_grad():
        # self._update_memory_for_batch(batch)
        return loss
             
              
    def _compute_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        logits = self.forward(batch)
        
        labels = batch['labels'].float()
        
        probs = torch.sigmoid(logits)

        # Check for constant predictions
        if torch.all(probs == probs[0]):
            # print(f"WARNING: All predictions are {probs[0].item():.4f}")
            # Add small noise to break symmetry
            probs = probs + torch.randn_like(probs) * 1e-4

        predictions = (probs > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        
        # Average Precision
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels[sorted_indices]
        cumulative_positives = torch.cumsum(sorted_labels, dim=0)
        cumulative_predictions = torch.arange(1, len(labels) + 1, device=labels.device, dtype=torch.float)
        precisions = cumulative_positives / cumulative_predictions
        ap = precisions.mean()
        
        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels.cpu().numpy(), probs.detach().cpu().numpy())
            auc_tensor = torch.tensor(auc, device=self.device)
        except:
            auc_tensor = torch.tensor(0.0, device=self.device)
        
        loss = self.loss_fn(logits, labels)

        return {
            'accuracy': accuracy,
            'ap': ap,
            'auc': auc_tensor,
            'loss': loss
        }
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        required = ['find_neighbors', 'get_temporal_neighbor']
        for method in required:
            if not hasattr(neighbor_finder, method):
                raise ValueError(f"NeighborFinder missing: {method}")

        if self.edge_raw_features is not None and neighbor_finder is not None:
            self._init_embedding_module()
        else:
            logger.warning(f"Cannot initialize embedding module: "
                           f"edge_raw_features={self.edge_raw_features is not None}, "
                           f"neighbor_finder={neighbor_finder is not None}")
    
    def _init_embedding_module(self):
        device = self.device

        if self.node_raw_features is not None:
            node_feat_dim = self.node_raw_features.shape[1]
            node_features = self.node_raw_features
            logger.info(f" Using raw node features (dim={node_feat_dim})")
        else:
            node_feat_dim = self.hidden_dim
            node_features = self.node_embedding.weight
            logger.info(f" Using learned node embeddings (dim={node_feat_dim})")

        logger.info(f"Initializing embedding module with edge_features_dim={self.edge_features_dim}")

        self.embedding_module = get_embedding_module(
            module_type=self.embedding_module_type,
            node_features=node_features,
            edge_features=self.edge_raw_features,
            memory=self.memory,
            neighbor_finder=self.neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=self.num_layers,
            n_node_features=node_feat_dim,
            n_edge_features=self.edge_features_dim,
            n_time_features=self.time_encoding_dim,
            embedding_dimension=self.hidden_dim,
            device=device,
            n_heads=self.n_heads,
            dropout=self.dropout,
            n_neighbors=self.n_neighbors,
            use_memory=self.use_memory
        )
        logger.info(f"Embedding module initialized: node_dim={node_feat_dim}, edge_dim={self.edge_features_dim}")
    
    def _init_modules(self):
        device = self.device
        logger.debug(f"use_memory={self.use_memory}, memory_updater_type={self.memory_updater_type}")

        if self.use_memory:
            self.memory = Memory(
                n_nodes=self.num_nodes,
                memory_dimension=self.memory_dim,
                input_dimension=self.message_dim + self.time_encoding_dim,
                message_dimension=self.message_dim,
                device=device
            )
        else:
            self.memory = None

        self.message_aggregator = get_message_aggregator(
            aggregator_type=self.aggregator_type,
            device=device
        )

        if self.use_memory:
            self.memory_updater = get_memory_updater(
                module_type=self.memory_updater_type,
                memory=self.memory,
                message_dim=self.message_dim,
                memory_dim=self.memory_dim,
                device=device
            )
        else:
            self.memory = None
            self.memory_updater = None

        self.embedding_module = None
        self.message_fn = None
    
    
    def set_raw_features(self, node_raw_features: Optional[torch.Tensor], edge_raw_features: torch.Tensor):
        device = self.device

        if node_raw_features is not None:
            self.node_raw_features = node_raw_features.to(device)
        else:
            self.node_raw_features = None

        if edge_raw_features is not None:
            self.edge_raw_features = edge_raw_features.to(device)
        else:
            self.edge_raw_features = None

        logger.info(f" Set raw features on {device}: "
                    f"Node={self.node_raw_features.shape if self.node_raw_features is not None else None}, "
                    f"Edge={self.edge_raw_features.shape if self.edge_raw_features is not None else None}")

        self._init_message_function()
        if self.message_fn is None:
            raise RuntimeError("Failed to initialize message_fn in set_raw_features")
    
    
    def _init_message_function(self):
        if self.node_raw_features is not None and len(self.node_raw_features.shape) > 1:
            node_feat_dim = self.node_raw_features.shape[1]
        else:
            node_feat_dim = self.hidden_dim

        # Use ACTUAL edge feature dimension from data
        if self.edge_raw_features is not None:
            edge_feat_dim = self.edge_raw_features.shape[1]
        else:
            edge_feat_dim = self.edge_features_dim

        # Message dimension: [src_mem, dst_mem, time_enc, edge_feat] (NO node features per TGN paper)
        raw_message_dim = (
            self.memory_dim * 2 + 
            self.time_encoding_dim + 
            edge_feat_dim
        )
        
        logger.info(f" Message function input dim: {raw_message_dim} "
                f"(mem={self.memory_dim}*2, time={self.time_encoding_dim}, edge={edge_feat_dim})")

        if raw_message_dim <= 0:
            raw_message_dim = self.message_dim

        logger.info(f"Message function input dim: {raw_message_dim} "
                    f"(node={node_feat_dim}, mem={self.memory_dim}, "
                    f"time={self.time_encoding_dim}, edge={edge_feat_dim})")

        self.message_fn = get_message_function(
            module_type=self.message_function_type,
            raw_message_dimension=raw_message_dim,
            message_dimension=self.message_dim
        ).to(self.device)

        if self.message_fn is None:
            raise RuntimeError("get_message_function returned None")

        self.message_fn = self.message_fn.to(self.device)
    
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = self.device

        if self.embedding_module is None:
            logger.error("CRITICAL: Embedding module is None! Using fallback features only.")
            logger.error(f"node_raw_features: {self.node_raw_features is not None}")
            logger.error(f"edge_raw_features: {self.edge_raw_features is not None}")
            logger.error(f"neighbor_finder: {self.neighbor_finder is not None}")

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
            self._compute_and_store_messages(batch)
            self._aggregate_and_update_memory()

        if torch.all(source_emb == 0) or torch.all(dest_emb == 0):
            logger.warning("All-zero embeddings detected!")

        combined = torch.cat([source_emb, dest_emb], dim=-1)
        scores = self.link_predictor(combined).squeeze(-1)

        if self.training and self.use_memory and self.memory is not None:
            self._memory_initialized = True
            self.train_batch_counter += 1

        return scores
    
    def _compute_and_store_messages(self, batch):
        """Compute messages from this batch for NEXT memory update."""
        if not self.use_memory or self.memory is None:
            logger.warning("Trying to store messages but use_memory=False or memory=None")
            return

        device = self.device
        src_nodes = batch['src_nodes'].to(device)
        dst_nodes = batch['dst_nodes'].to(device)
        timestamps = batch['timestamps'].to(device)

        # if self.node_raw_features is not None:
        #     src_features = self.node_raw_features[src_nodes]
        #     dst_features = self.node_raw_features[dst_nodes]
        # else:
        #     src_features = self.node_embedding(src_nodes)
        #     dst_features = self.node_embedding(dst_nodes)

        # Get edge features with CORRECT dimension
        if 'edge_features' in batch and batch['edge_features'] is not None:
            edge_feats = batch['edge_features'].to(device)
        else:
            # Use dimension from initialization (prevents 100-dim leakage for UCI)
            edge_feats = torch.zeros(len(src_nodes), self.edge_features_dim, device=device)
            logger.debug(f"Created {self.edge_features_dim}-dim dummy edge features for batch")
        
        
        
        # Get memory states (NO node features in messages - TGN spec)
        src_memory = self.memory.get_memory(src_nodes)
        dst_memory = self.memory.get_memory(dst_nodes)
        time_enc = self.time_encoder(timestamps)

        #  Runtime validation BEFORE message computation
        expected_dim = self.memory_dim * 2 + self.time_encoding_dim + self.edge_features_dim
        actual_dim = src_memory.size(1) + dst_memory.size(1) + time_enc.size(1) + edge_feats.size(1)
        
        if actual_dim != expected_dim:
            # Diagnose root cause
            time_dim_mismatch = time_enc.size(1) != self.time_encoding_dim
            edge_dim_mismatch = edge_feats.size(1) != self.edge_features_dim
            
            error_msg = (
                f"CRITICAL: Message dimension mismatch! Expected {expected_dim}, got {actual_dim}.\n"
                f"  Memory: {src_memory.size(1)} + {dst_memory.size(1)} = {src_memory.size(1) + dst_memory.size(1)}\n"
                f"  Time: {time_enc.size(1)} (config: {self.time_encoding_dim}) {'← MISMATCH!' if time_dim_mismatch else ''}\n"
                f"  Edge: {edge_feats.size(1)} (config: {self.edge_features_dim}) {'← MISMATCH!' if edge_dim_mismatch else ''}\n"
            )
            
            # Most likely cause: checkpoint trained with different config
            if hasattr(self, 'hparams') and 'time_encoding_dim' in self.hparams:
                error_msg += (
                    f"\n  CHECKPOINT CONFIG: time_dim={self.hparams.time_encoding_dim}, "
                    f"edge_dim={self.hparams.get('edge_features_dim', 'N/A')}\n"
                    f"  CURRENT CONFIG: time_dim={self.time_encoding_dim}, edge_dim={self.edge_features_dim}\n"
                    f"\n  SOLUTION: Train from scratch OR use checkpoint with matching configuration."
                )
            
            raise RuntimeError(error_msg)
        
        
        
        src_msg_input = torch.cat([src_memory, dst_memory, time_enc, edge_feats], dim=-1)
        dst_msg_input = torch.cat([dst_memory, src_memory, time_enc, edge_feats], dim=-1)

        src_messages = self.message_fn(src_msg_input)
        dst_messages = self.message_fn(dst_msg_input)

        

        # Store messages
        node_id_to_messages = {}
        for node, msg, ts in zip(src_nodes, src_messages, timestamps):
            node_id_to_messages.setdefault(node.item(), []).append((msg, ts.item()))
        for node, msg, ts in zip(dst_nodes, dst_messages, timestamps):
            node_id_to_messages.setdefault(node.item(), []).append((msg, ts.item()))

        self.memory.store_raw_messages(list(node_id_to_messages.keys()), node_id_to_messages)
    
    
    
    def _aggregate_and_update_memory(self):
        device = self.device
        if not self.use_memory or self.memory is None:
            return

        if self.training:
            self._memory_before_update = self.memory.memory.clone().detach()

        nodes_with_messages = list(self.memory.messages.keys())
        if not nodes_with_messages:
            return

        node_ids = []
        messages_list = []
        timestamps_list = []

        for node in nodes_with_messages:
            node_msgs = self.memory.messages[node]
            if not node_msgs:
                continue
            msgs = torch.stack([msg for msg, _ in node_msgs]).to(device)
            ts = torch.tensor([ts for _, ts in node_msgs], dtype=torch.float32, device=device)
            node_ids.append(node)
            messages_list.append(msgs)
            timestamps_list.append(ts)

        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(
            node_ids=node_ids,
            messages=messages_list,
            timestamps=timestamps_list
        )

        if len(unique_nodes) > 0:
            unique_nodes_t = torch.tensor(unique_nodes, device=device)
            unique_messages = unique_messages.to(device)
            unique_timestamps = unique_timestamps.to(device)

            self.memory_updater.update_memory(
                unique_node_ids=unique_nodes_t,
                unique_messages=unique_messages,
                timestamps=unique_timestamps
            )

        self.memory.clear_messages(nodes_with_messages)
    
    def compute_temporal_embeddings_pair(self,
                                         source_nodes: np.ndarray,
                                         destination_nodes: np.ndarray,
                                         edge_times: np.ndarray,
                                         n_neighbors: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device

        if isinstance(source_nodes, np.ndarray):
            src_tensor = torch.from_numpy(source_nodes).long().to(device)
            dst_tensor = torch.from_numpy(destination_nodes).long().to(device)
        else:
            src_tensor = source_nodes.to(device)
            dst_tensor = destination_nodes.to(device)

        if self.embedding_module is not None:
            source_emb = self.embedding_module.compute_embedding(
                memory=self.memory.memory,
                source_nodes=source_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            destination_emb = self.embedding_module.compute_embedding(
                memory=self.memory.memory,
                source_nodes=destination_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            return source_emb, destination_emb

        if self.node_raw_features is not None:
            src_feat = self.node_raw_features[src_tensor]
            dst_feat = self.node_raw_features[dst_tensor]
        else:
            src_feat = self.node_embedding(src_tensor)
            dst_feat = self.node_embedding(dst_tensor)

        return src_feat, dst_feat
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        labels = batch['labels'].float()
        loss = self.loss_fn(logits, labels)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        probs = torch.sigmoid(logits)
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels[sorted_indices]
        cumulative_positives = torch.cumsum(sorted_labels, dim=0)
        cumulative_predictions = torch.arange(1, len(labels) + 1, device=self.device, dtype=torch.float)
        precisions = cumulative_positives / cumulative_predictions
        ap = precisions.mean()
        self.log('val_ap', ap, prog_bar=True, sync_dist=True)
        return loss
    
    
    def on_fit_start(self):
        super().on_fit_start()
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()

        device = self.device
        if self.node_raw_features is not None and self.node_raw_features.device != device:
            self.node_raw_features = self.node_raw_features.to(device)
            logger.info(f" Moved node_raw_features to {device} in on_fit_start()")
        if self.edge_raw_features is not None and self.edge_raw_features.device != device:
            self.edge_raw_features = self.edge_raw_features.to(device)
            logger.info(f" Moved edge_raw_features to {device} in on_fit_start()")

    def on_train_batch_start(self, batch, batch_idx):
        if self.use_memory and self.memory is not None:
            mem_mean = self.memory.memory.mean().item()
            mem_std = self.memory.memory.std().item()
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: memory mean={mem_mean:.4f}, std={mem_std:.4f}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_memory and self.memory is not None:
            self.memory.detach_memory()

        if batch_idx % 100 == 0:
            total_grad_norm = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
            logger.debug(f"Batch {batch_idx}: Total gradient norm = {total_grad_norm:.4f}")
            if hasattr(self, '_memory_before_update'):
                mem_update = torch.norm(self.memory.memory - self._memory_before_update)
                logger.debug(f"Memory update norm = {mem_update:.4f}")
    

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train_batch_counter = 0
        logger.info(f" Training epoch start - Memory state preserved (size={self.memory.memory.shape[0]})")

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if self.use_memory:
            self._validation_memory = self.memory.memory.clone().detach()
            self._validation_last_update = self.memory.last_update.clone().detach()
            logger.info(" Cloned memory for validation (preserves training state)")

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.use_memory:
            self.memory.memory.data.copy_(self._validation_memory)
            self.memory.last_update.data.copy_(self._validation_last_update)
            logger.info(" Restored training memory after validation")   
    
  
    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        if self.use_memory:
            self.memory.__init_memory__()
        logger.info(" Memory reset for TEST (cold-start evaluation - prevents temporal leakage)")
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)

        device = None
        if args:
            device = args[0] if isinstance(args[0], (torch.device, str)) else None
        elif 'device' in kwargs:
            device = kwargs['device']

        if device is not None:
            if self.node_raw_features is not None:
                self.node_raw_features = self.node_raw_features.to(device)
                logger.debug(f" Moved node_raw_features to {device}")
                if self.embedding_module is not None and hasattr(self.embedding_module, 'node_features'):
                    self.embedding_module.node_features = self.node_raw_features
            if self.edge_raw_features is not None:
                self.edge_raw_features = self.edge_raw_features.to(device)
                logger.debug(f" Moved edge_raw_features to {device}")
                if self.embedding_module is not None and hasattr(self.embedding_module, 'edge_features'):
                    self.embedding_module.edge_features = self.edge_raw_features

        return self



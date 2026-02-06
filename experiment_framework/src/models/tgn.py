"""Temporal Graph Networks (TGN) implementation."""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger
from collections import defaultdict

from .base_model import BaseDynamicGNN
from .tgn_module.temporal_attention import TemporalAttentionLayer, MergeLayer
from .tgn_module.memory import Memory
from .tgn_module.memory_updater import get_memory_updater
from .tgn_module.message_aggregator import get_message_aggregator
from .tgn_module.message_function import get_message_function
from .tgn_module.embedding_module import get_embedding_module


# def scatter_mean(src, index, dim=0, dim_size=None):
#     if dim_size is None:
#         dim_size = index.max().item() + 1
#     count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
#     count.scatter_add_(dim, index, torch.ones_like(index, dtype=src.dtype))
#     out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
#     out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
#     return out / count.unsqueeze(1).clamp(min=1)



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
        

        # # Store raw features as buffers        
        # Register buffers (will be filled later)
        self.register_buffer('_node_raw_features', None)
        self.register_buffer('_edge_raw_features', None)      
        
        self.node_raw_features = None  # Will point to buffer
        self.edge_raw_features = None
        
        # Memory management for TGN (lagged update)
        self._pending_messages = None
        self._memory_initialized = False

        # Add message store as in original TGN
        self.message_store = defaultdict(list)
        
        # Add layer norm to attention
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize modules
        self._init_modules()

        # Ensure memory is on correct device
        if self.use_memory and self.memory is not None:
            self.memory = self.memory.to(self.device)

        if self.use_memory and self.node_features == 0:
            self.node_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)

       
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add normalization
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
        # if self.memory_updater is not None:
        #     print("Memory updater: OK")
        # else:
        #     print("Memory updater: NONE!")

        
        # self.last_batch_time = None
        # self._memory_initialized = False

        

    # Add these methods to your TGN class (after __init__)
    def __getstate__(self):
        """Preserve critical parameters during pickling/reconstruction."""
        state = self.__dict__.copy()
        # Backup critical parameters that Lightning often corrupts
        state['_num_nodes_backup'] = self.num_nodes
        state['_edge_features_dim_backup'] = self.edge_features_dim
        state.pop('_memory_initialized', None)
        return state

    def __setstate__(self, state):
        """Restore state with validation to prevent corruption."""
        # Extract backups BEFORE updating state
        num_nodes_backup = state.pop('_num_nodes_backup', None)
        edge_dim_backup = state.pop('_edge_features_dim_backup', None)

        # if num_nodes_backup is not None:
        #     self.num_nodes = num_nodes_backup
        # if edge_dim_backup is not None:
        #     self.edge_features_dim = edge_dim_backup
        
        # Update instance dictionary
        self.__dict__.update(state)
        self._pending_messages = None
        self._memory_initialized = False
        
        # # CRITICAL: Detect and repair corrupted num_nodes (<1000 indicates reconstruction failure)
        # if num_nodes_backup is not None and (not hasattr(self, 'num_nodes') or self.num_nodes < 1000):
        #     print(f"RECONSTRUCTION CORRUPTION DETECTED! "
        #         f"Restoring num_nodes from {self.num_nodes} → {num_nodes_backup}")
        #     self.num_nodes = num_nodes_backup
            
        #     # Reinitialize memory with CORRECT size
        #     if hasattr(self, 'memory') and self.memory is not None and self.use_memory:
        #         device = self.memory.memory.device
        #         # Create new memory tensor with correct size
        #         new_memory = torch.zeros(
        #             num_nodes_backup, 
        #             self.memory.memory_dimension, 
        #             device=device
        #         )
        #         # Preserve existing values where possible
        #         min_size = min(self.memory.memory.shape[0], num_nodes_backup)
        #         new_memory[:min_size] = self.memory.memory[:min_size]
        #         self.memory.memory = new_memory
        #         self.memory.last_update = torch.zeros(
        #             num_nodes_backup, 
        #             dtype=torch.float32, 
        #             device=device
        #         )
        #         print(f"Memory reinitialized with size {num_nodes_backup}")
        

        if num_nodes_backup is not None and getattr(self, 'num_nodes', 0) < 2:
            self.num_nodes = num_nodes_backup
            if self.use_memory and hasattr(self, 'memory'):
                self._init_modules()


        # Restore edge dimension if corrupted
        # if edge_dim_backup is not None and hasattr(self, 'edge_features_dim'):
        #     if self.edge_features_dim != edge_dim_backup:
        #         print(f"Restoring edge_features_dim from {self.edge_features_dim} → {edge_dim_backup}")
        #         self.edge_features_dim = edge_dim_backup

        if edge_dim_backup is not None:
            self.edge_features_dim = edge_dim_backup

        # self._memory_initialized = False
        # self._init_modules()
        # if self.node_raw_features is not None and self.edge_raw_features is not None:
        #     self.set_raw_features(self.node_raw_features, self.edge_raw_features)
        
        



    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def set_neighbor_finder(self, neighbor_finder):
        """Initialize embedding module AFTER model reconstruction (critical fix)."""
        self.neighbor_finder = neighbor_finder
        
        # Only initialize embedding module if features are already set
        # if self.embedding_module is None and neighbor_finder is not None and self.node_raw_features is not None:
        #     self._init_embedding_module()
        # Validate interface
        required = ['find_neighbors', 'get_temporal_neighbor']
        for method in required:
            if not hasattr(neighbor_finder, method):
                raise ValueError(f"NeighborFinder missing: {method}")
        
        # FIXED: Initialize even without node features
        if self.edge_raw_features is not None and neighbor_finder is not None:
            self._init_embedding_module()
        else:
            logger.warning(f"Cannot initialize embedding module: "
                        f"edge_raw_features={self.edge_raw_features is not None}, "
                        f"neighbor_finder={neighbor_finder is not None}")
        
        # if self.node_raw_features is not None and self.edge_raw_features is not None:
        #     self._init_embedding_module()

    def _init_embedding_module(self):
        """Initialize embedding module with CORRECT feature dimensions."""
        device = self.device
        
        logger.info(f"Initializing embedding module with edge_features_dim={self.edge_features_dim}")
        
        # Handle Wikipedia (no node features)
        # CRITICAL FIX: Handle case where node features are None (Wikipedia)
        if self.node_raw_features is None:
            logger.warning("Node features are None! Using node embedding layer for embedding module.")
            # Use the node embedding layer's weight as node features
            node_features = self.node_embedding.weight  # Shape: [num_nodes, hidden_dim]
        else:
            node_features = self.node_raw_features
        logger.info(f"Initializing embedding module with edge_features_dim={self.edge_features_dim}")

        self.embedding_module = get_embedding_module(
            module_type=self.embedding_module_type,
            node_features=node_features,
            edge_features=self.edge_raw_features,
            memory=self.memory,
            neighbor_finder=self.neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=self.num_layers,
            n_node_features=self.hidden_dim,
            n_edge_features=self.edge_features_dim,  # CORRECT dimension (4 for MOOC, 172 for Wikipedia)
            n_time_features=self.time_encoding_dim,
            embedding_dimension=self.hidden_dim,
            device=device,
            n_heads=self.n_heads,
            dropout=self.dropout,
            n_neighbors=self.n_neighbors,
            use_memory=self.use_memory
        )
        logger.info(f"Embedding module initialized: {self.embedding_module is not None}")

    def _init_modules(self):
        """Initialize TGN modules"""
        device = self.device
        
        # CRITICAL DEBUG: Log actual num_nodes during initialization
        # print(f"DEBUG: _init_modules() called with num_nodes={self.num_nodes}, "
        #   f"device={device}, has_neighbor_finder={self.neighbor_finder is not None}")
        
        
        # if self.num_nodes < 1000:
        #     raise RuntimeError(
        #         f"TGN._init_modules() called with num_nodes={self.num_nodes}! "
        #         f"This indicates BaseDynamicGNN still calls save_hyperparameters(). "
        #         f"FIX: Remove self.save_hyperparameters() from BaseDynamicGNN.__init__(). "
        #         f"Current hparams: {getattr(self, 'hparams', 'MISSING')}"
        #     )
        
        # CRITICAL DEBUG
        print(f"DEBUG: use_memory={self.use_memory}, memory_updater_type={self.memory_updater_type}")

        # Memory module
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

                
        # Message aggregator
        self.message_aggregator = get_message_aggregator(
            aggregator_type=self.aggregator_type,
            device=device
        )
        
        # Memory updater
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
        # Message function will be initialized after raw features are set
        self.message_fn = None

    def set_raw_features(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor):
        # """Set raw features from dataset."""        
        """Set raw features WITHOUT reinitializing modules (critical fix)."""
        device = self.device
        
        # self.node_raw_features = node_raw_features.to(device) if node_raw_features is not None else None
        # self.edge_raw_features = edge_raw_features.to(device)
        # Direct assignment works for registered buffers
        # Store in buffers for proper device handling
        if node_raw_features is not None:
            self._node_features_buffer = node_raw_features.to(device)
            self.node_raw_features = self._node_features_buffer
        else:
            self.node_raw_features = None
            
        if edge_raw_features is not None:
            self._edge_features_buffer = edge_raw_features.to(device)
            self.edge_raw_features = self._edge_features_buffer
        else:
            self.edge_raw_features = None
        
        logger.info(f"Set raw features: Node={self.node_raw_features.shape if self.node_raw_features is not None else None}, "
                   f"Edge={self.edge_raw_features.shape if self.edge_raw_features is not None else None}")
        
        # CRITICAL FIX: NEVER call _init_modules() here - causes double initialization with corrupted state
        # Update embedding module features directly if it exists
        # if hasattr(self, 'embedding_module') and self.embedding_module is not None:
        #     self.embedding_module.node_features = self.node_raw_features
        #     self.embedding_module.edge_features = self.edge_raw_features
        
        # Initialize message function (safe - doesn't reinit modules)
        self._init_message_function()

        # Verify it worked
        if self.message_fn is None:
            raise RuntimeError("Failed to initialize message_fn in set_raw_features")

    def _init_message_function(self):
        """Initialize message function with correct dimensions."""        
        # Calculate actual message dimension
        if self.node_raw_features is not None and len(self.node_raw_features.shape) > 1:
            node_feat_dim = self.node_raw_features.shape[1]
        else:
            node_feat_dim = self.hidden_dim # changed from self.node_features

        # Edge feature dimension (now dynamic!)
        edge_feat_dim = self.edge_features_dim    
        
        # Raw message dimension: [src_features, dst_features, src_memory, dst_memory, time_enc]
        raw_message_dim = (
            node_feat_dim * 2 + # src + dst node features
            self.memory_dim * 2 + # src + dst memory
            self.time_encoding_dim + # time encoding
            edge_feat_dim # edge features
        )
        
        # Ensure raw_message_dim is positive
        if raw_message_dim <= 0:
            raw_message_dim = self.message_dim

        # logger.info(f"Message function: input_dim={raw_message_dim}")
        
        
        # If it prints something like Message function input dim: 360, then node_feat_dim=0 → features not loaded.
        logger.info(f"Message function input dim: {raw_message_dim} "
          f"(node={node_feat_dim}, mem={self.memory_dim}, "
          f"time={self.time_encoding_dim}, edge={edge_feat_dim})")
        
        
        # if raw_message_dim <= 0:
        #     raw_message_dim = self.message_dim


        self.message_fn = get_message_function(
            module_type=self.message_function_type,
            raw_message_dimension=raw_message_dim,
            message_dimension=self.message_dim
        )

        if self.message_fn is None:
            raise RuntimeError("get_message_function returned None")

        self.message_fn = self.message_fn.to(self.device)
        # print(f" Message function initialized: {self.message_fn}")        

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of TGN."""       
        device = self.device
        # if self.training and self.use_memory and self.memory is not None:
        #     # self._update_memory_for_batch(batch)
        
        # DEBUG: Check if embedding module exists
        if self.embedding_module is None:
            logger.error("CRITICAL: Embedding module is None! Using fallback features only.")
            logger.error(f"node_raw_features: {self.node_raw_features is not None}")
            logger.error(f"edge_raw_features: {self.edge_raw_features is not None}")
            logger.error(f"neighbor_finder: {self.neighbor_finder is not None}")
        
        
        # PHASE 0: Update memory from previous batch's messages
        if self.training and self.use_memory and self._pending_messages is not None:
            with torch.no_grad():
                self._apply_pending_memory_update()

        # # PHASE 1: Compute messages for CURRENT batch
        # if self.training and self.use_memory:
        #     self._compute_and_store_messages(batch)
        
        
        # # PHASE 1: Apply pending memory updates from PREVIOUS batch
        # if self.use_memory and self.memory is not None:
        #     if self._memory_initialized and self._pending_messages is not None:
        #         with torch.no_grad():
        #             self._apply_pending_memory_update()
        
        
        
        # if hasattr(self, 'node_raw_features') and self.node_raw_features is not None:
        #     if self.node_raw_features.device != device:
        #         self.node_raw_features = self.node_raw_features.to(device)
        # if hasattr(self, 'edge_raw_features') and self.edge_raw_features is not None:
        #     if self.edge_raw_features.device != device:
        #         self.edge_raw_features = self.edge_raw_features.to(device)
        
        
        
        # Guard: ensure message_fn is initialized
        # if self.message_fn is None:
        #     raise RuntimeError(
        #         "message_fn is not initialized! "
        #         "Ensure set_raw_features() is called before forward(). "
        #         f"node_raw_features: {self.node_raw_features is not None}, "
        #         f"edge_raw_features: {self.edge_raw_features is not None}"
        #     )
        
        # PHASE 2: Compute embeddings with current memory
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']
        
        source_emb, dest_emb = self.compute_temporal_embeddings_pair(
            source_nodes=src_nodes.cpu().numpy(),
            destination_nodes=dst_nodes.cpu().numpy(),
            edge_times=timestamps.cpu().numpy(),
            n_neighbors=self.n_neighbors
        )
        
        # DEBUG: Check embeddings
        if torch.all(source_emb == 0) or torch.all(dest_emb == 0):
            logger.warning("All-zero embeddings detected!")
        
        # PHASE 3: Predict
        combined = torch.cat([source_emb, dest_emb], dim=-1)
        scores = self.link_predictor(combined).squeeze(-1)
        
        # PHASE 4: Store messages for NEXT batch (don't apply yet)
        if self.training and self.use_memory and self.memory is not None:
            with torch.no_grad():
                self._compute_and_store_messages(batch)
            self._memory_initialized = True
            self.train_batch_counter += 1

        
        # # Update memory from PREVIOUS batch's interactions
        # if self.use_memory and self.memory is not None and self._memory_initialized:
        #     with torch.no_grad():  # Memory update is not part of gradient computation
        #         self._update_memory_for_batch(batch)


        # src_nodes = batch['src_nodes']
        # dst_nodes = batch['dst_nodes']
        # timestamps = batch['timestamps']       
        
        # # Compute embeddings
        # source_embedding, destination_embedding = self.compute_temporal_embeddings_pair(
        #     source_nodes=src_nodes.cpu().numpy(),
        #     destination_nodes=dst_nodes.cpu().numpy(),            
        #     edge_times = timestamps.cpu().numpy(),
        #     n_neighbors=self.n_neighbors
        # )
        
        # # Concatenate embeddings for link prediction
        # combined = torch.cat([source_embedding, destination_embedding], dim=-1)
        # scores = self.link_predictor(combined).squeeze(-1)

        # # Mark memory as initialized AFTER first forward completes
        # if not self._memory_initialized:
        #     self._memory_initialized = True
        
        # # Increment batch counter for debugging
        # if self.training:
        #     self.train_batch_counter += 1
        

        return scores

    def _compute_and_store_messages(self, batch):
        """Compute messages from this batch for NEXT memory update."""
        if not self.use_memory or self.memory is None:
            print("WARNING: Trying to store messages but use_memory=False or memory=None")
            return
        
        device = self.device
        
        src_nodes = batch['src_nodes'].to(device)
        dst_nodes = batch['dst_nodes'].to(device)
        timestamps = batch['timestamps'].to(device)
        
        # print(f"DEBUG: Storing messages for {len(src_nodes)} source nodes")
        # print(f"DEBUG: Message function exists: {self.message_fn is not None}")
        # print(f"DEBUG: Memory exists: {self.memory is not None}")
        
        
        # Get features
        if self.node_raw_features is not None:
            src_features = self.node_raw_features[src_nodes]
            dst_features = self.node_raw_features[dst_nodes]
        else:
            src_features = self.node_embedding(src_nodes)
            dst_features = self.node_embedding(dst_nodes)
        
        # Use SAME edge features for both directions (TGN standard)
        if 'edge_features' in batch and batch['edge_features'] is not None:
            edge_feats = batch['edge_features']
        else:
            edge_feats = torch.zeros(len(src_nodes), self.edge_features_dim, device=device)
        
        # Get memory states
        src_memory = self.memory.get_memory(src_nodes)
        dst_memory = self.memory.get_memory(dst_nodes)
        time_enc = self.time_encoder(timestamps)
        
        # Compute messages
        src_msg_input = torch.cat([src_features, dst_features, src_memory, dst_memory, time_enc, edge_feats], dim=-1)
        dst_msg_input = torch.cat([dst_features, src_features, dst_memory, src_memory, time_enc, edge_feats], dim=-1)
        
        src_messages = self.message_fn(src_msg_input)
        dst_messages = self.message_fn(dst_msg_input)
        
        # Store for next batch
        self._pending_messages = {
            'nodes': torch.cat([src_nodes, dst_nodes]),
            'messages': torch.cat([src_messages, dst_messages]),
            'timestamps': torch.cat([timestamps, timestamps])
        }   
    
    def _apply_pending_memory_update(self):
        """Apply stored messages to memory."""
        if self._pending_messages is None or self.memory_updater is None:
            return
        
        self.memory_updater.update_memory(
            self._pending_messages['nodes'],
            self._pending_messages['messages'],
            self._pending_messages['timestamps']
        )
        self._pending_messages = None
    
    def compute_temporal_embeddings_pair(self, source_nodes: np.ndarray,
                                   destination_nodes: np.ndarray,
                                   edge_times: np.ndarray,
                                   n_neighbors: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        
        n_samples = len(source_nodes)
        src_tensor = torch.from_numpy(source_nodes).long().to(device)
        dst_tensor = torch.from_numpy(destination_nodes).long().to(device)
        # edge_time_tensor = torch.from_numpy(edge_times).float().to(device)

        # node_raw_features = self.node_raw_features.to(device) if self.node_raw_features is not None else None
        # edge_raw_features = self.edge_raw_features.to(device) if self.edge_raw_features is not None else None
        
        
        # Get raw features
        if self.node_features > 0 and hasattr(self, 'node_raw_features') and self.node_raw_features is not None:
            src_feat = self.node_raw_features[src_tensor]
            dst_feat = self.node_raw_features[dst_tensor]            
        else:
            src_feat = self.node_embedding(src_tensor)
            dst_feat = self.node_embedding(dst_tensor)

        
        if self.embedding_module is not None:           
            
            # Ensure we're passing tensors on the correct device
            # source_nodes_tensor = torch.from_numpy(source_nodes).long().to(device)
            # destination_nodes_tensor = torch.from_numpy(destination_nodes).long().to(device)
            
            # Get memory for source and destination nodes
            # source_memory = self.memory.get_memory(source_nodes_tensor)
            # destination_memory = self.memory.get_memory(destination_nodes_tensor)

            full_memory = self.memory.memory if self.memory is not None else None   
            
            # Use full attention-based embeddings with neighbor sampling
            source_emb = self.embedding_module.compute_embedding(
                memory= full_memory,
                # memory= source_memory,
                source_nodes=source_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            destination_emb = self.embedding_module.compute_embedding(
                memory= full_memory,
                # memory= destination_memory,
                source_nodes=destination_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            return source_emb, destination_emb
        else:
            # Fallback: use features only
            src_emb = src_feat
            dst_emb = dst_feat
            if src_emb.shape[-1] != self.hidden_dim:
                src_emb = torch.zeros(n_samples, self.hidden_dim, device=self.device)
                dst_emb = torch.zeros(n_samples, self.hidden_dim, device=self.device)

        return src_emb, dst_emb
    
        
    # def _generate_negative_samples(self, src_nodes: np.ndarray, 
    #                               dst_nodes: np.ndarray) -> np.ndarray:
    #     """Generate negative samples for link prediction."""
    #     n_samples = len(src_nodes)
    #     negative_nodes = np.random.choice(self.num_nodes, size=n_samples, replace=True)
        
    #     for i in range(n_samples):
    #         if negative_nodes[i] == dst_nodes[i]:
    #             negative_nodes[i] = (negative_nodes[i] + 1) % self.num_nodes
                
    #     return negative_nodes  
    
    
    
    def _update_memory_for_batch(self, batch: Dict[str, torch.Tensor]):
        """Update memory immediately after forward pass."""
        device = self.device
        
        if not self.use_memory or self.memory is None:
            return
            
        with torch.no_grad():
            src_nodes = batch['src_nodes'].to(device)
            dst_nodes = batch['dst_nodes'].to(device)
            timestamps = batch['timestamps'].to(device)

            # Get features
            if hasattr(self, 'node_raw_features') and self.node_raw_features is not None:
                src_features = self.node_raw_features[src_nodes]
                dst_features = self.node_raw_features[dst_nodes]
            else:
                src_features = self.node_embedding(src_nodes)
                dst_features = self.node_embedding(dst_nodes)

            # Get edge features from batch
            if 'edge_features' in batch and batch['edge_features'] is not None:
                edge_feats = batch['edge_features']
            else:
                # FIXED: Use configured dimension
                edge_feats = torch.zeros(
                    len(src_nodes), 
                    self.edge_features_dim,  # NOT 172
                    device=self.device
                )

            # Get current memory states
            src_memory = self.memory.get_memory(src_nodes)
            dst_memory = self.memory.get_memory(dst_nodes)
            time_enc = self.time_encoder(timestamps)
            
            # Create message inputs
            src_msg_input = torch.cat([src_features, dst_features, src_memory, dst_memory, time_enc, edge_feats], dim=-1)
            dst_msg_input = torch.cat([dst_features, src_features, dst_memory, src_memory, time_enc, edge_feats], dim=-1)
            
            # Generate messages
            src_messages = self.message_fn(src_msg_input)
            dst_messages = self.message_fn(dst_msg_input)
            
            # Update memory
            all_messages = torch.cat([src_messages, dst_messages], dim=0)
            all_nodes = torch.cat([src_nodes, dst_nodes], dim=0)
            all_timestamps = torch.cat([timestamps, timestamps], dim=0)
            
            # if self.memory_updater is not None:
            self.memory_updater.update_memory(all_nodes, all_messages, all_timestamps)
     
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
    
    def on_train_batch_start(self, batch, batch_idx):
        """Verify memory is updating."""
        if self.use_memory and self.memory is not None:
            mem_mean = self.memory.memory.mean().item()
            mem_std = self.memory.memory.std().item()
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: memory mean={mem_mean:.4f}, std={mem_std:.4f}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Check if gradients are flowing."""
        if batch_idx % 100 == 0:
            total_grad_norm = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
            
            print(f"Batch {batch_idx}: Total gradient norm = {total_grad_norm:.4f}")
            
            # Check memory updates
            if self.use_memory and self.memory is not None:
                mem_update = torch.norm(self.memory.memory - self.memory.memory.clone().detach())
                print(f"Memory update norm = {mem_update:.4f}")  
    
    def _reset_memory(self, phase: str):
        """Reset memory state."""
        if not self.use_memory or self.memory is None:
            return
        
        self.memory.__init_memory__()
        self._pending_messages = None
        self._memory_initialized = False
        logger.info(f"Memory reset for {phase}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train_batch_counter = 0
        logger.info(f"✓ Training epoch start - Memory state preserved (size={self.memory.memory.shape[0]})")

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        # self._reset_memory("validation")
        if self.use_memory:
            self.memory.__init_memory__()
            logger.info(" Memory reset for validation (prevents training leakage)")

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        # self._reset_memory("test")
        if self.use_memory:
            self.memory.__init_memory__()
            logger.info(" Memory reset for test (prevents training leakage)")

    



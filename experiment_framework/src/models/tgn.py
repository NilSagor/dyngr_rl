"""Temporal Graph Networks (TGN) implementation."""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger


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
        # neighbor_finder: Optional[nn.Module] = None,        
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
        if num_nodes < 1000:  # Wikipedia needs 9228
            raise ValueError(
                f"CRITICAL: TGN initialized with num_nodes={num_nodes}! "
                f"Expected >=1000 for real datasets. Check config/model reconstruction."
            )

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
        self.node_raw_features = None
        self.edge_raw_features = None 

        # Time shift normalization (not used in basic TGN)
        # self.register_buffer("mean_time_shift_src", torch.tensor(0.0))
        # self.register_buffer("std_time_shift_src", torch.tensor(1.0))
        # self.register_buffer("mean_time_shift_dst", torch.tensor(0.0))
        # self.register_buffer("std_time_shift_dst", torch.tensor(1.0))

        
        
        # Initialize modules
        self._init_modules()

        # Ensure memory is on correct device
        if self.use_memory and self.memory is not None:
            self.memory = self.memory.to(self.device)

       
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

        print(f"TGN initialized with use_memory={self.use_memory}")
        if self.memory_updater is not None:
            print("Memory updater: OK")
        else:
            print("Memory updater: NONE!")

        # Add training state tracking
        self.train_batch_counter = 0
        self.last_batch_time = None

        # self._pending_messages = None

    # Add these methods to your TGN class (after __init__)
    def __getstate__(self):
        """Preserve critical parameters during pickling/reconstruction."""
        state = self.__dict__.copy()
        # Backup critical parameters that Lightning often corrupts
        state['_num_nodes_backup'] = self.num_nodes
        state['_edge_features_dim_backup'] = self.edge_features_dim
        return state

    def __setstate__(self, state):
        """Restore state with validation to prevent corruption."""
        # Extract backups BEFORE updating state
        num_nodes_backup = state.pop('_num_nodes_backup', None)
        edge_dim_backup = state.pop('_edge_features_dim_backup', None)
        
        # Update instance dictionary
        self.__dict__.update(state)
        
        # CRITICAL: Detect and repair corrupted num_nodes (<1000 indicates reconstruction failure)
        if num_nodes_backup is not None and (not hasattr(self, 'num_nodes') or self.num_nodes < 1000):
            print(f"RECONSTRUCTION CORRUPTION DETECTED! "
                f"Restoring num_nodes from {self.num_nodes} → {num_nodes_backup}")
            self.num_nodes = num_nodes_backup
            
            # Reinitialize memory with CORRECT size
            if hasattr(self, 'memory') and self.memory is not None and self.use_memory:
                device = self.memory.memory.device
                # Create new memory tensor with correct size
                new_memory = torch.zeros(
                    num_nodes_backup, 
                    self.memory.memory_dimension, 
                    device=device
                )
                # Preserve existing values where possible
                min_size = min(self.memory.memory.shape[0], num_nodes_backup)
                new_memory[:min_size] = self.memory.memory[:min_size]
                self.memory.memory = new_memory
                self.memory.last_update = torch.zeros(
                    num_nodes_backup, 
                    dtype=torch.float32, 
                    device=device
                )
                print(f"Memory reinitialized with size {num_nodes_backup}")
        
        # Restore edge dimension if corrupted
        if edge_dim_backup is not None and hasattr(self, 'edge_features_dim'):
            if self.edge_features_dim != edge_dim_backup:
                print(f"Restoring edge_features_dim from {self.edge_features_dim} → {edge_dim_backup}")
                self.edge_features_dim = edge_dim_backup



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
        if self.embedding_module is None and neighbor_finder is not None and self.node_raw_features is not None:
            self._init_embedding_module()

    def _init_embedding_module(self):
        """Initialize embedding module with CORRECT feature dimensions."""
        device = self.device
        
        logger.info(f"Initializing embedding module with edge_features_dim={self.edge_features_dim}")
        
        self.embedding_module = get_embedding_module(
            module_type=self.embedding_module_type,
            node_features=self.node_raw_features,
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



    def _init_modules(self):
        """Initialize TGN modules"""
        device = self.device
        
        # CRITICAL DEBUG: Log actual num_nodes during initialization
        print(f"DEBUG: _init_modules() called with num_nodes={self.num_nodes}, "
          f"device={device}, has_neighbor_finder={self.neighbor_finder is not None}")
        
        
        if self.num_nodes < 1000:
            raise RuntimeError(
                f"TGN._init_modules() called with num_nodes={self.num_nodes}! "
                f"This indicates BaseDynamicGNN still calls save_hyperparameters(). "
                f"FIX: Remove self.save_hyperparameters() from BaseDynamicGNN.__init__(). "
                f"Current hparams: {getattr(self, 'hparams', 'MISSING')}"
            )
        
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
            self.memory_updater = None
        
        # # Embedding module
        # if self.neighbor_finder is not None:

        #     # Validate neighbor_finder is actual object, not string
        #     assert hasattr(self.neighbor_finder, 'get_temporal_neighbor'), \
        #         f"neighbor_finder must be NeighborFinder object, got {type(self.neighbor_finder)}"
            

        #     self.embedding_module = get_embedding_module(
        #         module_type=self.embedding_module_type,
        #         node_features=self.node_raw_features,
        #         edge_features=self.edge_raw_features,
        #         memory=self.memory,
        #         neighbor_finder=self.neighbor_finder,
        #         time_encoder=self.time_encoder,
        #         n_layers=self.num_layers,
        #         n_node_features=self.hidden_dim, # changed from self.node_features
        #         n_edge_features=172,
        #         n_time_features=self.time_encoding_dim,
        #         embedding_dimension=self.hidden_dim,
        #         device=device,
        #         n_heads=self.n_heads,
        #         dropout=self.dropout,
        #         n_neighbors=self.n_neighbors,
        #         use_memory=self.use_memory
        #     )
        # else:
        #     self.embedding_module = None

        self.embedding_module = None
        # Message function will be initialized after raw features are set
        self.message_fn = None

    def set_raw_features(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor):
        # """Set raw features from dataset."""
        # device = self.device
        
        # self.node_raw_features = node_raw_features.to(device) if node_raw_features is not None else None
        # self.edge_raw_features = edge_raw_features.to(device)
        
        # actual_edge_dim = self.edge_raw_features.shape[1]
        # print(f"Setting raw features - Node: {self.node_raw_features.shape if self.node_raw_features is not None else None}, Edge: {self.edge_raw_features.shape}")
        
        # # Check if embedding module needs reinitialization due to dimension mismatch
        # reinit_needed = False
        # if hasattr(self, 'embedding_module') and self.embedding_module is not None:
        #     # Check current configured edge dim
        #     if len(self.embedding_module.attention_models) > 0:
        #         attn_layer = self.embedding_module.attention_models[0]
        #         # key_dim = n_neighbors_features + n_edge_features + time_dim
        #         configured_edge_dim = attn_layer.key_dim - attn_layer.feat_dim - attn_layer.time_dim
                
        #         if configured_edge_dim != actual_edge_dim:
        #             print(f"Edge dim mismatch: model expects {configured_edge_dim}, got {actual_edge_dim}. Reinitializing...")
        #             reinit_needed = True
        
        # # Update or reinitialize embedding module
        # if reinit_needed or not hasattr(self, 'embedding_module') or self.embedding_module is None:
        #     # Update config for reinitialization
        #     self.n_edge_features = actual_edge_dim
            
        #     # Reinitialize modules with correct dimensions
        #     self._init_modules()
            
        #     # Set features again after reinit
        #     if self.embedding_module is not None:
        #         self.embedding_module.node_features = self.node_raw_features
        #         self.embedding_module.edge_features = self.edge_raw_features
        # else:
        #     # Just update features
        #     self.embedding_module.node_features = self.node_raw_features
        #     self.embedding_module.edge_features = self.edge_raw_features
        
        # # Create message function with correct dimensions
        # self._init_message_function()
        """Set raw features WITHOUT reinitializing modules (critical fix)."""
        device = self.device
        
        self.node_raw_features = node_raw_features.to(device) if node_raw_features is not None else None
        self.edge_raw_features = edge_raw_features.to(device)
        
        print(f"✓ Setting raw features - Node: {self.node_raw_features.shape if self.node_raw_features is not None else None}, Edge: {self.edge_raw_features.shape}")
        
        # CRITICAL FIX: NEVER call _init_modules() here - causes double initialization with corrupted state
        # Update embedding module features directly if it exists
        if hasattr(self, 'embedding_module') and self.embedding_module is not None:
            self.embedding_module.node_features = self.node_raw_features
            self.embedding_module.edge_features = self.edge_raw_features
        
        # Initialize message function (safe - doesn't reinit modules)
        self._init_message_function()

    def _init_message_function(self):
        """Initialize message function with correct dimensions."""
        # Calculate actual message dimension
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
        # If it prints something like Message function input dim: 360, then node_feat_dim=0 → features not loaded.
        print(f"Message function input dim: {raw_message_dim} "
          f"(node={node_feat_dim}, mem={self.memory_dim}, "
          f"time={self.time_encoding_dim}, edge={edge_feat_dim})")
        
        # Ensure raw_message_dim is positive
        if raw_message_dim <= 0:
            raw_message_dim = self.message_dim


        self.message_fn = get_message_function(
            module_type=self.message_function_type,
            raw_message_dimension=raw_message_dim,
            message_dimension=self.message_dim
        )
        self.message_fn = self.message_fn.to(self.device)

        # self.add_module("message_fn", message_fn)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of TGN."""
        
        # # Apply pending messages from PREVIOUS batch
        # if self._pending_messages is not None:
        #     with torch.no_grad():
        #         nodes, messages, timestamps = self._pending_messages
        #         if self.memory_updater is not None:
        #             self.memory_updater.update_memory(nodes, messages, timestamps)
        #     self._pending_messages = None
        
        # DEBUG: Check initial memory state
        # if self.train_batch_counter == 0 and self.training:
        #     print(f"First training batch - Memory stats: "
        #           f"mean={self.memory.memory.mean():.6f}, "
        #           f"std={self.memory.memory.std():.6f}")
        
        # Update memory BEFORE forward pass for this batch
        # (using previous batch's messages if needed)
        if self.training and self.use_memory and self.memory is not None:
            self._update_memory_for_batch(batch)
        
        src_nodes = batch['src_nodes']
        dst_nodes = batch['dst_nodes']
        timestamps = batch['timestamps']

        # # Generate negative samples
        # negative_nodes = self._generate_negative_samples(
        #     src_nodes.cpu().numpy(),
        #     dst_nodes.cpu().numpy(),
        # )

        # CRITICAL: Update memory from previous batch's messages before computing embeddings
        # if self.use_memory and self.memory is not None:
        #     self._process_pending_messages()

        
        # Compute embeddings
        source_embedding, destination_embedding = self.compute_temporal_embeddings_pair(
            source_nodes=src_nodes.cpu().numpy(),
            destination_nodes=dst_nodes.cpu().numpy(),            
            edge_times = timestamps.cpu().numpy(),
            n_neighbors=self.n_neighbors
        )

        # # DEBUG CHECKS (remove after fixing)
        # if self.training and batch_idx == 0:  # First batch only
        #     print(f"Source embedding stats: mean={source_embedding.mean():.4f}, std={source_embedding.std():.4f}")
        #     print(f"Embedding range: [{source_embedding.min():.4f}, {source_embedding.max():.4f}]")
        #     if source_embedding.std() < 0.01:
        #         print("WARNING: Embeddings have near-zero variance! Check projection layer.")
            
        # Link prediction
        # pos_scores = self.link_predictor(source_embedding, destination_embedding)
        # neg_scores = self.link_predictor(source_embedding, negative_embedding)
        # Concatenate embeddings for link prediction
        combined = torch.cat([source_embedding, destination_embedding], dim=-1)
        scores = self.link_predictor(combined).squeeze(-1)

        # Increment batch counter for debugging
        if self.training:
            self.train_batch_counter += 1

        return scores
        
    def compute_temporal_embeddings_pair(self, source_nodes: np.ndarray,
                                   destination_nodes: np.ndarray,
                                   edge_times: np.ndarray,
                                   n_neighbors: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        n_samples = len(source_nodes)
        src_tensor = torch.from_numpy(source_nodes).long().to(self.device)
        dst_tensor = torch.from_numpy(destination_nodes).long().to(self.device)
        edge_time_tensor = torch.from_numpy(edge_times).long().to(self.device)

        # Get raw features
        if self.node_features > 0 and hasattr(self, 'node_raw_features') and self.node_raw_features is not None:
            src_feat = self.node_raw_features[src_tensor]
            dst_feat = self.node_raw_features[dst_tensor]
            
            # # DEBUG: Check feature values
            # if self.training and self.train_batch_counter < 3:
            #     print(f"Features - src mean: {src_feat.mean():.6f}, dst mean: {dst_feat.mean():.6f}")

        else:
            src_feat = self.node_embedding(src_tensor)
            dst_feat = self.node_embedding(dst_tensor)

        # CRITICAL: Incorporate memory WITHOUT detaching during training
        # if self.use_memory and self.memory is not None:
        #     src_mem = self.memory.get_memory(src_tensor)
        #     dst_mem = self.memory.get_memory(dst_tensor)
            
        #     # DEBUG: Check memory values at this point
        #     if self.training and self.train_batch_counter < 3:
        #         print(f"Memory in embeddings - src mean: {src_mem.mean():.6f}, "
        #               f"std: {src_mem.std():.6f}")
            
        #     # Only warn if memory is truly zero for many batches
        #     if self.training and self.train_batch_counter > 10 and src_mem.abs().sum() < 1e-6:
        #         print("WARNING: Memory is still all zeros after 10 batches!")

        #     # Concatenate features + memory
        #     src_emb = torch.cat([src_feat, src_mem], dim=-1)
        #     dst_emb = torch.cat([dst_feat, dst_mem], dim=-1)
        #     # Project to hidden_dim
        #     if not hasattr(self, 'mem_proj'):
        #         self.mem_proj = nn.Linear(src_emb.shape[-1], self.hidden_dim).to(self.device)
        #     src_emb = self.mem_proj(src_emb)
        #     dst_emb = self.mem_proj(dst_emb)
        if self.embedding_module is not None:
            
            
            # Ensure we're passing tensors on the correct device
            source_nodes_tensor = torch.from_numpy(source_nodes).long().to(self.device)
            destination_nodes_tensor = torch.from_numpy(destination_nodes).long().to(self.device)
            
            # Get memory for source and destination nodes
            source_memory = self.memory.get_memory(source_nodes_tensor)
            destination_memory = self.memory.get_memory(destination_nodes_tensor)

            # full_memory = self.memory.memory    
            
            # Use full attention-based embeddings with neighbor sampling
            source_emb = self.embedding_module.compute_embedding(
                # memory= self.memory,
                memory= source_memory,
                source_nodes=source_nodes,
                timestamps=edge_times,
                n_layers=self.num_layers,
                n_neighbors=n_neighbors
            )
            destination_emb = self.embedding_module.compute_embedding(
                # memory= self.memory,
                memory= destination_memory,
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
    
    
    

    
    def _generate_negative_samples(self, src_nodes: np.ndarray, 
                                  dst_nodes: np.ndarray) -> np.ndarray:
        """Generate negative samples for link prediction."""
        n_samples = len(src_nodes)
        negative_nodes = np.random.choice(self.num_nodes, size=n_samples, replace=True)
        
        for i in range(n_samples):
            if negative_nodes[i] == dst_nodes[i]:
                negative_nodes[i] = (negative_nodes[i] + 1) % self.num_nodes
                
        return negative_nodes  
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
             
        # Forward pass
        logits = self.forward(batch)
        
        labels = batch['labels'].float()

        # # Debug: Check label distribution
        # if self.training:
        #     pos_ratio = labels.mean().item()
        #     if pos_ratio < 0.4 or pos_ratio > 0.6:
        #         print(f"WARNING: Label imbalance detected. Positive ratio: {pos_ratio:.2f}")
         

        # compute loss
        loss = self.loss_fn(logits, labels)
                
        # prepare message for memory update (but not update yet) 
        # with torch.no_grad():
        self._update_memory_for_batch(batch)
        return loss
    
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
                edge_feats = torch.zeros(len(src_nodes), 172, device=self.device) 

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
            
            # # DEBUG: Check if memory is updating
            # if self.training and self.train_batch_counter < 5:
            #     updated_memory = self.memory.get_memory(src_nodes[:1])
            #     print(f"Batch {self.train_batch_counter}: "
            #         f"Memory updated - mean={updated_memory.mean():.6f}, "
            #         f"std={updated_memory.std():.6f}")
            #     print(f"Messages - mean={all_messages.mean():.6f}, "
            #         f"std={all_messages.std():.6f}")
       
    
    def _process_messages_for_batch(self, batch: Dict[str, torch.Tensor]):
        """Update node memories after interaction."""
        if not self.use_memory or self.memory is None:
            return
        
        with torch.no_grad():
            src_nodes = batch['src_nodes'].to(self.device)
            dst_nodes = batch['dst_nodes'].to(self.device)
            timestamps = batch['timestamps'].to(self.device)

            # # DEBUG: Check memory BEFORE update
            # mem_before = self.memory.memory[src_nodes].mean().item()
            # print(f"Memory before update: {mem_before:.6f}")
        # Debug: Check node ID bounds
        # max_node_id = self.node_raw_features.size(0) - 1  # 9228 for Wikipedia
        # assert src_nodes.max() <= max_node_id, f"Source node {src_nodes.max()} > {max_node_id}"
        # assert dst_nodes.max() <= max_node_id, f"Destination node {dst_nodes.max()} > {max_node_id}"
        # assert src_nodes.min() >= 1, f"Source node {src_nodes.min()} < 1"
        # assert dst_nodes.min() >= 1, f"Destination node {dst_nodes.min()} < 1"

        # if src_nodes.min() < 1:
        #     print(f"WARNING: Invalid source node {src_nodes.min()}, filtering...")
        #     valid_mask = src_nodes >= 1
        #     if valid_mask.sum() == 0:
        #         return  # Skip invalid batch
        #     src_nodes = src_nodes[valid_mask]
        #     dst_nodes = dst_nodes[valid_mask]
        #     timestamps = timestamps[valid_mask]


            # Get node features
            if hasattr(self, 'node_raw_features') and self.node_raw_features is not None:
                src_features = self.node_raw_features[src_nodes]
                dst_features = self.node_raw_features[dst_nodes]
            else:
                # src_features = torch.zeros(len(src_nodes), self.hidden_dim, device=self.device)
                # dst_features = torch.zeros(len(dst_nodes), self.hidden_dim, device=self.device)
                # Use hidden_dim for embedding lookup
                src_features = self.node_embedding(src_nodes)
                dst_features = self.node_embedding(dst_nodes)

            # Get edge features if available
            if 'edge_features' in batch and batch['edge_features'] is not None:
                edge_feats = batch['edge_features']
                # Expand to match source/destination
                src_edge_feats = edge_feats
                dst_edge_feats = edge_feats
            else:
                src_edge_feats = torch.zeros(len(src_nodes), self.message_dim - (src_features.size(-1) * 2 + src_memory.size(-1) * 2 + time_enc.size(-1)), device=self.device)
                dst_edge_feats = src_edge_feats
            
            
            # Get memories
            # if self.use_memory:
            #     src_memory = self.memory.get_memory(src_nodes).detach()
            #     dst_memory = self.memory.get_memory(dst_nodes).detach()
            # else:
            #     src_memory = torch.zeros(len(src_nodes), self.memory_dim, device=self.device)
            #     dst_memory = torch.zeros(len(dst_nodes), self.memory_dim, device=self.device)

            src_memory = self.memory.get_memory(src_nodes)
            dst_memory = self.memory.get_memory(dst_nodes)
            
            
            
            
            # Time encoding - should be [batch_size, time_dim]
            # Time encoding
            time_enc = self.time_encoder(timestamps)
            
      
            # Debug: Print shapes
            # print(f"src_features: {src_features.shape}")
            # print(f"dst_features: {dst_features.shape}")  
            # print(f"src_memory: {src_memory.shape}")
            # print(f"dst_memory: {dst_memory.shape}")
            # print(f"time_enc: {time_enc.shape}")
            
            
            
            # Create message inputs
            src_msg_input = torch.cat([src_features, dst_features, src_memory, dst_memory, time_enc, src_edge_feats], dim=-1)
            dst_msg_input = torch.cat([dst_features, src_features, dst_memory, src_memory, time_enc, dst_edge_feats], dim=-1)
            
            # Generate messages        
            src_messages = self.message_fn(src_msg_input)
            dst_messages = self.message_fn(dst_msg_input)
            
            # Combine messages
            all_messages = torch.cat([src_messages, dst_messages], dim=0)
            all_nodes = torch.cat([src_nodes, dst_nodes], dim=0)
            all_timestamps = torch.cat([timestamps, timestamps], dim=0)

            # print(f"Messages shape: {all_messages.shape}, expected last dim: {self.message_dim}")
            
            with torch.no_grad():
                if self.memory_updater is not None:
                    self.memory_updater.update_memory(all_nodes, all_messages, all_timestamps)
            
            # # Debug: Check memory values
            # if self.training and torch.rand(1).item() < 0.01:  # 1% of batches
            #     memory_mean = self.memory.memory.mean().item()
            #     memory_std = self.memory.memory.std().item()
            #     print(f"Memory stats - mean: {memory_mean:.4f}, std: {memory_std:.4f}")
                
            #     # Should NOT be zero after first few batches
            #     if memory_std < 1e-6:
            #         print("WARNING: Memory has near-zero variance! Not updating properly.")

            if self.memory_updater is not None:
                self.memory_updater.update_memory(all_nodes, all_messages, all_timestamps)
                
                # # DEBUG: Check memory AFTER update
                # mem_after = self.memory.memory[src_nodes].mean().item()
                # print(f"Memory after update: {mem_after:.6f}")
                # if abs(mem_after - mem_before) < 1e-8:
                #     print("WARNING: Memory not changing! Updater not working.")        

    def _compute_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        logits = self.forward(batch)
        
        labels = batch['labels'].float()
        
        probs = torch.sigmoid(logits)
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

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     super().on_train_batch_end(outputs, batch, batch_idx)
    #     if hasattr(self, '_pending_messages'):
    #         nodes, messages, timestamps = self._pending_messages
    #         with torch.no_grad():
    #             if self.memory_updater is not None:
    #                 self.memory_updater.update_memory(nodes, messages, timestamps)
    #         delattr(self, '_pending_messages')
            
    #         # Debug: Check if memory is updating
    #         if batch_idx == 0 and self.current_epoch == 0:
    #             mem_std = self.memory.memory.std().item()
    #             print(f"Memory std after first batch: {mem_std:.6f}")
        

    # def on_validation_batch_end(self, outputs, batch, batch_idx):
    #     """Update memory after backward pass."""
    #     # super().on_validation_batch_end(outputs, batch, batch_idx)
    #     with torch.no_grad():
    #         self._process_messages_for_batch(batch)
    #         self._process_pending_messages()

    # def on_test_batch_end(self, outputs, batch, batch_idx):
    #     """Update memory after backward pass."""
    #     super().on_test_batch_end(outputs, batch, batch_idx)
    #     with torch.no_grad():
    #         self._process_pending_messages()
    
    
    def on_train_epoch_start(self):
        """DO NOT reset memory at start of train epoch."""
        super().on_train_epoch_start()
        # Memory should persist across training batches
        self.train_batch_counter = 0  # Reset counter for debugging
        print(f"Starting training epoch - Memory initialized: "
              f"mean={self.memory.memory.mean():.6f}")
        if self.use_memory:
            # self.memory.__init_memory__()
            # self.pending_messages = [] # clear any stale message
            self.memory.memory.data.zero_()
            self.memory.last_update.data.zero_()

    def on_validation_epoch_start(self):
        """Reset memory at start of validation epoch."""
        super().on_validation_epoch_start()
        # if self.use_memory and self.memory is not None:
        if self.use_memory:
            self.memory.__init_memory__()
            # self.pending_messages = [] # clear any stale message
            self.memory.memory.data.zero_()
            self.memory.last_update.data.zero_()
            print("Memory reset for validation")

    def on_test_epoch_start(self):
        """Reset memory at start of test epoch."""
        super().on_test_epoch_start()
        # if self.use_memory and self.memory is not None:
        if self.use_memory:
            self.memory.__init_memory__()
            # self.pending_messages = []
            print("Memory reset for test")

    



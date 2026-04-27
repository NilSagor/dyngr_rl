# model.py
import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import AUROC, AveragePrecision, Accuracy
from dataclasses import asdict
from loguru import logger


from src.models.hicost_dev.components.time_encoding import TimeEncode

from src.models.hicost_dev.components.memory import (
    Memory, GRUMemoryUpdater, LastMessageAggregator,
    IdentityMessageFunction, GraphAttentionEmbedding,
)
from src.models.hicost_dev.components.walk import WalkEncoder, PositionEncoder
from src.models.hicost_dev.components.cooccurrence import NeighborCooccurrenceEncoder
from src.models.hicost_dev.components.merge_layer import AffinityMergeLayer
from src.models.hicost_dev.components.mlp_module import RestartMLP

from src.models.hicost_dev.v1.co_gnn import CoGNN

from .config import HiCoSTConfig


def build_cooccurrence_graph(edge_index, num_nodes):
    """Construct co‑occurrence graph from adjacency (Jaccard weights)."""
    adj = [set() for _ in range(num_nodes)]
    ei = edge_index.cpu().numpy() if torch.is_tensor(edge_index) else edge_index
    for u, v in zip(ei[0], ei[1]):
        adj[u].add(v)
        adj[v].add(u)

    rows, cols, weights = [], [], []
    for u in range(num_nodes):
        if not adj[u]:
            continue
        for v in range(u + 1, num_nodes):
            if not adj[v]:
                continue
            inter = len(adj[u] & adj[v])
            if inter == 0:
                continue
            uni = len(adj[u] | adj[v])
            jaccard = inter / uni
            rows.extend([u, v])
            cols.extend([v, u])
            weights.extend([jaccard, jaccard])

    dev = edge_index.device if torch.is_tensor(edge_index) else None
    return (
        torch.tensor([rows, cols], dtype=torch.long, device=dev),
        torch.tensor(weights, dtype=torch.float, device=dev),
    )

class HiCoSTdev1(L.LightningModule):
    def __init__(self, config: HiCoSTConfig):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.cfg = config

        self._device = config.device if isinstance(config.device, torch.device) else torch.device(config.device)


         # ── Raw node & edge features ──
        if isinstance(config.node_features, np.ndarray):
            self.node_raw_features = torch.from_numpy(config.node_features.astype(np.float32)).to(self._device)
        else:
            self.node_raw_features = config.node_features.to(self._device)

        if isinstance(config.edge_features, np.ndarray):
            self.edge_raw_features = torch.from_numpy(config.edge_features.astype(np.float32)).to(self._device)
        else:
            self.edge_raw_features = config.edge_features.to(self._device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features

        # Store all config flags as attributes for easy access
        self.use_memory = self.cfg.use_memory
        self.memory_update_at_start = self.cfg.memory_update_at_start
        self.enable_walk = self.cfg.enable_walk
        self.enable_restart = self.cfg.enable_restart
        self.pick_new_neighbors = self.cfg.pick_new_neighbors
        self.walk_length = self.cfg.walk_length
        self.num_walks = self.cfg.num_walks
        self.n_neighbors = self.cfg.n_neighbors
        self.neighbor_cooc = self.cfg.enable_neighbor_cooc
        self.use_explicit_co_gnn = self.cfg.use_explicit_co_gnn
        self.co_gnn_out_dim = self.cfg.co_gnn_out_dim
        self.time_feat_dim = self.cfg.time_dim
        if self.cfg.enable_neighbor_cooc:
            self.max_input_sequence_length = self.cfg.max_input_seq_length

        # ── Time encodings ──
        self.time_encoder = TimeEncode(dimension=self.cfg.time_dim)

        # ── Co‑occurrence setup ──
        if self.cfg.enable_neighbor_cooc:
            self.max_input_sequence_length = self.cfg.max_input_seq_length
            self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
                neighbor_co_occurrence_feat_dim=50, device=self._device
            )
            self.neighbor_cooc_proj = nn.Linear(50, 10)

        # ── Walk setup ──
        if self.cfg.enable_walk:
            self.position_encoder = PositionEncoder(
                position_feat_dim=self.cfg.position_feat_dim,
                walk_length=self.cfg.walk_length,
                device=self._device,
            )
            self.walk_encoder = WalkEncoder(
                input_dim=self.n_node_features + self.n_edge_features + self.cfg.time_dim + self.cfg.position_feat_dim,
                position_feat_dim=self.cfg.position_feat_dim,
                output_dim=self.cfg.walk_emb_dim,
                num_walk_heads=self.cfg.num_walk_heads,
                dropout=self.cfg.dropout,
            )
            if self.cfg.enable_restart:
                self.restart_prob = RestartMLP(dim=self.n_node_features)
        
        
        
        # ── Memory setup ──
        if self.cfg.use_memory:
            self.fixed_time_encoder = TimeEncode(dimension=self.cfg.fixed_time_dim, parameter_requires_grad=False)
            raw_msg_dim = 2 * self.n_node_features + self.n_edge_features + self.time_encoder.dimension
            self.memory = Memory(
                n_nodes=self.n_nodes,
                memory_dimension=self.n_node_features,
                input_dimension=raw_msg_dim,
                message_dimension=raw_msg_dim,
                device=self._device,
            )
            self.message_aggregator = LastMessageAggregator(device=self._device)
            self.message_function = IdentityMessageFunction()
            self.memory_updater = GRUMemoryUpdater(
                memory=self.memory,
                message_dimension=raw_msg_dim,
                memory_dimension=self.n_node_features,
                device=self._device,
            )
            self.embedding_module = GraphAttentionEmbedding(
                node_features=self.node_raw_features,
                edge_features=self.edge_raw_features,
                memory=self.memory,
                neighbor_finder=self.cfg.neighbor_finder,
                time_encoder=self.time_encoder,
                fixed_time_encoder=self.fixed_time_encoder,
                n_layers=self.cfg.n_layers,
                n_node_features=self.n_node_features,
                n_edge_features=self.n_edge_features,
                n_time_features=self.cfg.time_dim,
                embedding_dimension=self.embedding_dimension,
                device=self._device,
                n_heads=self.cfg.n_heads,
                dropout=self.cfg.dropout,
                use_memory=True,
                n_fixed_time_features=self.cfg.fixed_time_dim,
            )
        
        
        # ── Compute final embedding size ──
        self.final_emb_dim = 0
        if self.cfg.use_memory:
            self.final_emb_dim += self.n_node_features
        if self.cfg.enable_walk:
            self.final_emb_dim += self.cfg.walk_emb_dim
            if self.cfg.enable_restart:
                self.final_emb_dim += 1
        if self.cfg.enable_neighbor_cooc:
            self.final_emb_dim += (self.cfg.max_input_seq_length + 1) * 10   # proj_out=10
        
        
        # ── Proposed Co‑GNN ──
        self.use_explicit_co_gnn = self.cfg.use_explicit_co_gnn
        if self.use_explicit_co_gnn:
            self.co_gnn = CoGNN(
                in_dim=self.n_node_features,
                hidden_dim=self.cfg.co_gnn_hidden_dim,
                out_dim=self.cfg.co_gnn_out_dim,
                dropout=self.cfg.dropout,
            )
            self.final_emb_dim += self.cfg.co_gnn_out_dim
        else:
            self.co_gnn = None
        
        
        # ── Affinity merge layer ──
        self.affinity_score = AffinityMergeLayer(
            self.final_emb_dim, self.final_emb_dim,
            self.n_node_features, 1,
        )
        
        # ── Metrics ──
        self.val_auroc = AUROC(task='binary')
        self.val_ap = AveragePrecision(task='binary')
        self.test_auroc = AUROC(task='binary')
        self.test_ap = AveragePrecision(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')

    def forward(self, sources, destinations, timestamps, edge_idxs, negative_sources=None, negative_destinations=None):
        return self.compute_edge_probabilities(
            sources, destinations, negative_sources, negative_destinations,
            timestamps, edge_idxs, self.cfg.n_neighbors,
        )
    
    def training_step(self, batch, batch_idx):
        sources = batch['sources']
        destinations = batch['destinations']
        timestamps = batch['timestamps']
        edge_idxs = batch['edge_idxs']
        neg_sources = batch['negative_sources']
        neg_destinations = batch['negative_destinations']

        batch_size = len(sources)

        pos_prob, neg_prob = self.compute_edge_probabilities(
            sources, destinations, neg_sources, neg_destinations,
            timestamps, edge_idxs, self.n_neighbors
        )

        pos_label = torch.ones_like(pos_prob, dtype=torch.long)
        neg_label = torch.zeros_like(neg_prob, dtype=torch.long)
        loss = F.binary_cross_entropy(pos_prob, pos_label.float()) + \
            F.binary_cross_entropy(neg_prob, neg_label.float())
        
        if self.use_memory and self.memory is not None:
            self.memory.detach_memory()
        
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Process a validation batch."""
        return self._shared_eval_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        """Process a test batch."""
        return self._shared_eval_step(batch, 'test')
    
    def _shared_eval_step(self, batch, prefix):
        sources = batch['sources']
        destinations = batch['destinations']
        timestamps = batch['timestamps']
        edge_idxs = batch['edge_idxs']
        neg_sources = batch['negative_sources']
        neg_destinations = batch['negative_destinations']
        batch_size = len(sources)

        pos_prob, neg_prob = self.compute_edge_probabilities(
            sources, destinations, neg_sources, neg_destinations,
            timestamps, edge_idxs, self.n_neighbors
        )

                
        pos_label = torch.ones_like(pos_prob, dtype=torch.long)
        neg_label = torch.zeros_like(neg_prob, dtype=torch.long)

        loss = F.binary_cross_entropy(pos_prob, pos_label.float()) + \
            F.binary_cross_entropy(neg_prob, neg_label.float())

        preds = torch.cat([pos_prob, neg_prob])
        targets = torch.cat([pos_label, neg_label])

        device = preds.device
        if prefix == 'val':
            auroc_metric = self.val_auroc.to(device)
            ap_metric = self.val_ap.to(device)
            accuracy_metric = self.val_accuracy.to(device)
        else:
            auroc_metric = self.test_auroc.to(device)
            ap_metric = self.test_ap.to(device)
            accuracy_metric = self.test_accuracy.to(device)        
            
        
        auc = auroc_metric(preds, targets)
        ap = ap_metric(preds, targets)
        accuracy = accuracy_metric(preds, targets)

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f'{prefix}_auc', auc, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'{prefix}_ap', ap, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'{prefix}_accuracy', accuracy, on_step=False, on_epoch=True, batch_size=batch_size)

        return {'preds': preds, 'targets': targets}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def on_train_epoch_start(self):
        # Build co-occurrence graph for explicit Co-GNN
        if self.use_explicit_co_gnn:
            if hasattr(self.neighbor_finder, 'edge_index'):
                edge_index = self.neighbor_finder.edge_index
                if edge_index is not None:
                    # Move to model device before building graph
                    edge_index = edge_index.to(self._device)
                    self.co_gnn_edge_index, self.co_gnn_edge_weight = build_cooccurrence_graph(edge_index, self.n_nodes)
                    self.update_co_gnn_embeddings()
                else:
                    logger.warning("edge_index is None; explicit Co-GNN disabled")
                    self._c_emb = None
            else:
                logger.warning("No edge_index in neighbor_finder; explicit Co-GNN disabled")
                self._c_emb = None

        if self.neighbor_finder is not None:
            self.neighbor_finder.clear_cache()
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()

    def update_co_gnn_embeddings(self):
        """Compute Co-GNN embeddings for all nodes and store in self._c_emb."""
        if not self.use_explicit_co_gnn:
            return
        
        # Use the stored co_gnn_edge_index/weight from on_train_epoch_start
        if not hasattr(self, 'co_gnn_edge_index') or self.co_gnn_edge_index is None:
            logger.warning("co_gnn_edge_index not set; skipping Co-GNN update")
            return

        # Build base node features: node_raw_features + memory (if available)
        base_x = self.node_raw_features
        if self.use_memory and hasattr(self, 'memory') and self.memory is not None:
            base_x = base_x + self.memory.get_memory(list(range(self.n_nodes)))

        # Ensure co_gnn is on the same device as base_x
        self.co_gnn = self.co_gnn.to(base_x.device)
        
        # Move co-occurrence graph to the same device as base_x
        edge_index = self.co_gnn_edge_index.to(base_x.device)
        edge_weight = self.co_gnn_edge_weight.to(base_x.device)

        with torch.no_grad():
            c_emb = self.co_gnn(base_x, edge_index, edge_weight)   # [N, out_dim]
        self.register_buffer('_c_emb', c_emb)

    def get_node_embedding_dim(self):
        return self.final_emb_dim

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_sources, negative_destinations,
                                   edge_times, edge_idxs, n_neighbors=20):

        n_samples = len(source_nodes)

        source_node_embedding, destination_node_embedding, neg_source_node_embedding, neg_destination_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_sources, negative_destinations, edge_times, edge_idxs,
            n_neighbors)


        score = self.affinity_score(torch.cat([source_node_embedding, neg_source_node_embedding], dim=0),
                                    torch.cat([destination_node_embedding,
                                               neg_destination_node_embedding])
                                    ).squeeze(dim=0)

        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_sources, negative_destinations,
                                    edge_times, edge_idxs, n_neighbors=20):


        n_samples = len(source_nodes)
        if negative_sources is not None:
            nodes = np.concatenate([source_nodes, destination_nodes, negative_sources, negative_destinations])
            timestamps = np.concatenate([edge_times, edge_times, edge_times, edge_times])
        else:
            nodes = np.concatenate([source_nodes, destination_nodes])
            timestamps = np.concatenate([edge_times, edge_times])
        positives = np.concatenate([source_nodes, destination_nodes])

        self.neighbor_finder.find_all_first_hop(nodes, timestamps)

        memory = None

        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches

                node_list = list(range(self.n_nodes))
                memory, last_update = self.get_updated_memory(node_list, self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            # Compute the embeddings using the embedding module
            node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                     source_nodes=nodes,
                                                                     timestamps=timestamps,
                                                                     n_layers=self.n_layers,
                                                                     n_neighbors=n_neighbors,
                                                                     time_diffs=None)

            source_node_embedding = node_embedding[:n_samples]
            destination_node_embedding = node_embedding[n_samples: 2 * n_samples]

            neg_source_node_embedding = None
            neg_destination_node_embedding = None

            if negative_sources is not None:
                neg_source_node_embedding = node_embedding[2 * n_samples:3 * n_samples]
                neg_destination_node_embedding = node_embedding[3 * n_samples:]

            src_restart_emb = source_node_embedding
            dst_restart_emb = destination_node_embedding
            neg_src_restart_emb = neg_source_node_embedding
            neg_dst_restart_emb = neg_destination_node_embedding
        else:
            src_restart_emb = torch.nn.Parameter(
                torch.empty((n_samples, self.n_node_features), requires_grad=True)).to(self._device)
            dst_restart_emb = torch.nn.Parameter(
                torch.empty((n_samples, self.n_node_features), requires_grad=True)).to(self._device)
            neg_src_restart_emb = torch.nn.Parameter(
                torch.empty((n_samples, self.n_node_features), requires_grad=True)).to(self._device)
            neg_dst_restart_emb = torch.nn.Parameter(
                torch.empty((n_samples, self.n_node_features), requires_grad=True)).to(self._device)

            # Initialize the tensor with Xavier uniform
            torch.nn.init.xavier_uniform_(src_restart_emb)
            torch.nn.init.xavier_uniform_(dst_restart_emb)
            torch.nn.init.xavier_uniform_(neg_src_restart_emb)
            torch.nn.init.xavier_uniform_(neg_dst_restart_emb)

        if self.enable_walk:
            walk_restarts = None

            if self.enable_restart:

                src_walk_restart = self.restart_prob(src_restart_emb)
                dst_walk_restart = self.restart_prob(dst_restart_emb)
                if negative_sources is not None:
                    neg_src_walk_restart = self.restart_prob(neg_src_restart_emb)
                    neg_dst_walk_restart = self.restart_prob(neg_dst_restart_emb)
                    walk_restarts = torch.cat(
                        [src_walk_restart, dst_walk_restart, neg_src_walk_restart, neg_dst_walk_restart])
                else:
                    walk_restarts = torch.cat(
                        [src_walk_restart, dst_walk_restart])

            src_walk_embedding, dst_walk_embedding, neg_src_walk_embedding, neg_dst_walk_embedding = self.compute_walk_embeddings(
                nodes, timestamps, n_samples, self.num_walks,
                source_nodes, destination_nodes, negative_sources, negative_destinations, edge_times, walk_restarts)

            if self.use_memory:
                source_node_embedding = torch.cat([source_node_embedding, src_walk_embedding], dim=1)
                destination_node_embedding = torch.cat([destination_node_embedding, dst_walk_embedding], dim=1)
                if negative_sources is not None:
                    neg_source_node_embedding = torch.cat([neg_source_node_embedding, neg_src_walk_embedding], dim=1)
                    neg_destination_node_embedding = torch.cat([neg_destination_node_embedding, neg_dst_walk_embedding],
                                                               dim=1)
            else:
                source_node_embedding = src_walk_embedding
                destination_node_embedding = dst_walk_embedding
                if negative_sources is not None:
                    neg_source_node_embedding = neg_src_walk_embedding
                    neg_destination_node_embedding = neg_dst_walk_embedding

            if self.enable_restart:
                source_node_embedding = torch.cat([source_node_embedding, src_walk_restart.view(-1, 1)], dim=1)
                destination_node_embedding = torch.cat([destination_node_embedding, dst_walk_restart.view(-1, 1)],
                                                       dim=1)
                if negative_sources is not None:
                    neg_source_node_embedding = torch.cat([neg_source_node_embedding, neg_src_walk_restart.view(-1, 1)],
                                                          dim=1)

                    neg_destination_node_embedding = torch.cat(
                        [neg_destination_node_embedding, neg_dst_walk_restart.view(-1, 1)], dim=1)

        if self.neighbor_cooc:
            src_cooc_embedding, dst_cooc_embedding, neg_src_cooc_embedding, neg_dst_cooc_embedding = self.compute_cooc_embeddings(
                nodes, timestamps, n_samples,
                source_nodes, destination_nodes, negative_sources, negative_destinations, edge_times)

            source_node_embedding = torch.cat([source_node_embedding, src_cooc_embedding], dim=1)
            destination_node_embedding = torch.cat([destination_node_embedding, dst_cooc_embedding], dim=1)
            if negative_sources is not None:
                neg_source_node_embedding = torch.cat([neg_source_node_embedding, neg_src_cooc_embedding], dim=1)
                neg_destination_node_embedding = torch.cat([neg_destination_node_embedding, neg_dst_cooc_embedding],
                                                           dim=1)

        source_node_embedding = F.normalize(source_node_embedding)
        destination_node_embedding = F.normalize(destination_node_embedding)

        if negative_sources is not None:
            neg_source_node_embedding = F.normalize(neg_source_node_embedding)
            neg_destination_node_embedding = F.normalize(neg_destination_node_embedding)

        if self.use_memory:
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                self.update_memory(positives, self.memory.messages)

                assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-3), \
                    "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)


            unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                          source_node_embedding,
                                                                          destination_nodes,
                                                                          destination_node_embedding,
                                                                          edge_times, edge_idxs)
            unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                    destination_node_embedding,
                                                                                    source_nodes,
                                                                                    source_node_embedding,
                                                                                    edge_times, edge_idxs)
            if self.memory_update_at_start:

                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)

            if self.use_explicit_co_gnn:
                out_dim = self.co_gnn_out_dim
                if hasattr(self, '_c_emb') and self._c_emb is not None:
                    src_c = self._c_emb[source_nodes]
                    dst_c = self._c_emb[destination_nodes]
                    if negative_sources is not None:
                        neg_src_c = self._c_emb[negative_sources]
                        neg_dst_c = self._c_emb[negative_destinations]
                else:
                    src_c = torch.zeros(len(source_nodes), out_dim, device=self._device)
                    dst_c = torch.zeros(len(destination_nodes), out_dim, device=self._device)
                    if negative_sources is not None:
                        neg_src_c = torch.zeros(len(negative_sources), out_dim, device=self._device)
                        neg_dst_c = torch.zeros(len(negative_destinations), out_dim, device=self._device)
                    # logger.debug("Co-GNN embeddings not available; using zeros")

            source_node_embedding = torch.cat([source_node_embedding, src_c], dim=1)
            destination_node_embedding = torch.cat([destination_node_embedding, dst_c], dim=1)
            if negative_sources is not None:
                neg_source_node_embedding = torch.cat([neg_source_node_embedding, neg_src_c], dim=1)
                neg_destination_node_embedding = torch.cat([neg_destination_node_embedding, neg_dst_c], dim=1)
        
        # Continue with normalization and memory update 
        source_node_embedding = F.normalize(source_node_embedding)
        destination_node_embedding = F.normalize(destination_node_embedding)
        if negative_sources is not None:
            neg_source_node_embedding = F.normalize(neg_source_node_embedding)
            neg_destination_node_embedding = F.normalize(neg_destination_node_embedding)
        
        
        
        # logger.debug(f"[DEBUG] source_node_embedding shape: {source_node_embedding.shape}")
        # logger.debug(f"[DEBUG] self.final_emb_dim: {self.final_emb_dim}")

        return (
            source_node_embedding, destination_node_embedding, neg_source_node_embedding,
            neg_destination_node_embedding)  # , memory[-1].view(1, -1))

    def compute_cooc_embeddings(self, nodes, timestamps, n_samples, source_nodes, destination_nodes, negative_sources,
                                negative_destinations,
                                edge_times):
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = \
            self.neighbor_finder.get_all_first_hop_neighbors(node_ids=nodes,
                                                             node_interact_times=timestamps)

        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = nodes_neighbor_ids_list[
                                                                                              :n_samples], nodes_edge_ids_list[
                                                                                                           :n_samples], nodes_neighbor_times_list[
                                                                                                                        :n_samples]

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = nodes_neighbor_ids_list[
                                                                                              n_samples: 2 * n_samples], nodes_edge_ids_list[
                                                                                                                         n_samples: 2 * n_samples], nodes_neighbor_times_list[
                                                                                                                                                    n_samples: 2 * n_samples]

        # pad the sequences of first-hop neighbors for source and destination nodes
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=source_nodes, node_interact_times=edge_times,
                               nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list,
                               nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=destination_nodes, node_interact_times=edge_times,
                               nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list,
                               nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        src_nodes_neighbor_co_occurrence_features = self.neighbor_cooc_proj(
            src_padded_nodes_neighbor_co_occurrence_features)

        dst_nodes_neighbor_co_occurrence_features = self.neighbor_cooc_proj(
            dst_padded_nodes_neighbor_co_occurrence_features)

        src_cooc_embedding = src_nodes_neighbor_co_occurrence_features.flatten(1, 2)
        dst_cooc_embedding = dst_nodes_neighbor_co_occurrence_features.flatten(1, 2)
        neg_src_cooc_embedding = None
        neg_dst_cooc_embedding = None
        if negative_sources is not None:
            # get the first-hop neighbors of source and destination nodes
            # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
            neg_src_nodes_neighbor_ids_list, neg_src_nodes_edge_ids_list, neg_src_nodes_neighbor_times_list = nodes_neighbor_ids_list[
                                                                                                              2 * n_samples: 3 * n_samples], nodes_edge_ids_list[
                                                                                                                                             2 * n_samples: 3 * n_samples], nodes_neighbor_times_list[
                                                                                                                                                                            2 * n_samples: 3 * n_samples]

            # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
            neg_dst_nodes_neighbor_ids_list, neg_dst_nodes_edge_ids_list, neg_dst_nodes_neighbor_times_list = nodes_neighbor_ids_list[
                                                                                                              3 * n_samples:], nodes_edge_ids_list[
                                                                                                                               3 * n_samples:], nodes_neighbor_times_list[
                                                                                                                                                3 * n_samples:]
            # pad the sequences of first-hop neighbors for source and destination nodes
            neg_src_padded_nodes_neighbor_ids, neg_src_padded_nodes_edge_ids, neg_src_padded_nodes_neighbor_times = \
                self.pad_sequences(node_ids=negative_sources, node_interact_times=edge_times,
                                   nodes_neighbor_ids_list=neg_src_nodes_neighbor_ids_list,
                                   nodes_edge_ids_list=neg_src_nodes_edge_ids_list,
                                   nodes_neighbor_times_list=neg_src_nodes_neighbor_times_list,
                                   max_input_sequence_length=self.max_input_sequence_length)

            neg_dst_padded_nodes_neighbor_ids, neg_dst_padded_nodes_edge_ids, neg_dst_padded_nodes_neighbor_times = \
                self.pad_sequences(node_ids=negative_destinations, node_interact_times=edge_times,
                                   nodes_neighbor_ids_list=neg_dst_nodes_neighbor_ids_list,
                                   nodes_edge_ids_list=neg_dst_nodes_edge_ids_list,
                                   nodes_neighbor_times_list=neg_dst_nodes_neighbor_times_list,
                                   max_input_sequence_length=self.max_input_sequence_length)

            # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
            # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
            neg_src_padded_nodes_neighbor_co_occurrence_features, neg_dst_padded_nodes_neighbor_co_occurrence_features = \
                self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=neg_src_padded_nodes_neighbor_ids,
                                                    dst_padded_nodes_neighbor_ids=neg_dst_padded_nodes_neighbor_ids)
            neg_src_nodes_neighbor_co_occurrence_features = self.neighbor_cooc_proj(
                neg_src_padded_nodes_neighbor_co_occurrence_features)

            neg_dst_nodes_neighbor_co_occurrence_features = self.neighbor_cooc_proj(
                neg_dst_padded_nodes_neighbor_co_occurrence_features)
            neg_src_cooc_embedding = neg_src_nodes_neighbor_co_occurrence_features.flatten(1, 2)
            neg_dst_cooc_embedding = neg_dst_nodes_neighbor_co_occurrence_features.flatten(1, 2)

        return src_cooc_embedding, dst_cooc_embedding, neg_src_cooc_embedding, neg_dst_cooc_embedding

    def compute_walk_embeddings(self, nodes, timestamps, n_samples, n_neighbors, source_nodes, destination_nodes,
                                negative_sources, negative_destinations,
                                edge_times, walk_restarts):

        nodes_multi_hop_graphs = self.neighbor_finder.get_multi_hop_neighbors(num_hops=self.walk_length,
                                                                              source_nodes=nodes,
                                                                              timestamps=timestamps,
                                                                              n_neighbors=self.num_walks,
                                                                              walk_restart=walk_restarts,
                                                                              pick_new_neighbors=self.pick_new_neighbors)

        neighbors = [arr.reshape(nodes.shape[0] // n_samples, n_samples, n_neighbors) for arr in
                     nodes_multi_hop_graphs[0]]
        edges = [arr.reshape(nodes.shape[0] // n_samples, n_samples, n_neighbors) for arr in nodes_multi_hop_graphs[1]]
        times = [arr.reshape(nodes.shape[0] // n_samples, n_samples, n_neighbors) for arr in nodes_multi_hop_graphs[2]]

        # src_node_multi_hop_graphs = ([neighbors[0][0], neighbors[1][0]], edges[0][0], times[0][0])
        # dst_node_multi_hop_graphs = ([neighbors[0][1], neighbors[1][1]], edges[0][1], times[0][1])
        # neg_src_multi_hop_graphs = ([neighbors[0][2], neighbors[1][2]], edges[0][2], times[0][2])
        # neg_dst_multi_hop_graphs = ([neighbors[0][3], neighbors[1][3]], edges[0][3], times[0][3])

        src_node_multi_hop_graphs = ([neighbors[i][0] for i in range(self.walk_length)],
                                     [edges[i][0] for i in range(self.walk_length)],
                                     [times[i][0] for i in range(self.walk_length)])

        dst_node_multi_hop_graphs = ([neighbors[i][1] for i in range(self.walk_length)],
                                     [edges[i][1] for i in range(self.walk_length)],
                                     [times[i][1] for i in range(self.walk_length)])

        if negative_sources is not None:
            neg_src_multi_hop_graphs = ([neighbors[i][2] for i in range(self.walk_length)],
                                        [edges[i][2] for i in range(self.walk_length)],
                                        [times[i][2] for i in range(self.walk_length)])

            neg_dst_multi_hop_graphs = ([neighbors[i][3] for i in range(self.walk_length)],
                                        [edges[i][3] for i in range(self.walk_length)],
                                        [times[i][3] for i in range(self.walk_length)])

        # count the appearances appearances of nodes in the multi-hop graphs that are generated by random walks that
        # start from src node in src_node_ids and dst node in dst_node_ids
        self.position_encoder.count_nodes_appearances(src_node_ids=source_nodes, dst_node_ids=destination_nodes,
                                                      node_interact_times=edge_times,
                                                      src_node_multi_hop_graphs=src_node_multi_hop_graphs,
                                                      dst_node_multi_hop_graphs=dst_node_multi_hop_graphs)

        # Tensor, shape (batch_size, node_feat_dim)
        src_walk_embedding = self.compute_node_walk_embeddings(node_ids=source_nodes,
                                                               node_interact_times=edge_times,
                                                               node_multi_hop_graphs=src_node_multi_hop_graphs,
                                                               num_neighbors=self.num_walks)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_walk_embedding = self.compute_node_walk_embeddings(node_ids=destination_nodes,
                                                               node_interact_times=edge_times,
                                                               node_multi_hop_graphs=dst_node_multi_hop_graphs,
                                                               num_neighbors=self.num_walks)
        neg_src_walk_embedding = None
        neg_dst_walk_embedding = None
        if negative_sources is not None:
            neg_source_nodes = negative_sources

            self.position_encoder.count_nodes_appearances(src_node_ids=neg_source_nodes,
                                                          dst_node_ids=negative_destinations,
                                                          node_interact_times=edge_times,
                                                          src_node_multi_hop_graphs=neg_src_multi_hop_graphs,
                                                          dst_node_multi_hop_graphs=neg_dst_multi_hop_graphs)

            # if negative_sources is not None is not None:
            #     # Tensor, shape (batch_size, node_feat_dim)
            neg_src_walk_embedding = self.compute_node_walk_embeddings(node_ids=neg_source_nodes,
                                                                       node_interact_times=edge_times,
                                                                       node_multi_hop_graphs=neg_src_multi_hop_graphs,
                                                                       num_neighbors=self.num_walks)

            # Tensor, shape (batch_size, node_feat_dim)
            neg_dst_walk_embedding = self.compute_node_walk_embeddings(node_ids=negative_destinations,
                                                                       node_interact_times=edge_times,
                                                                       node_multi_hop_graphs=neg_dst_multi_hop_graphs,
                                                                       num_neighbors=self.num_walks)

        return src_walk_embedding, dst_walk_embedding, neg_src_walk_embedding, neg_dst_walk_embedding

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list,
                      nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, max_input_sequence_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(
                nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        max_seq_length = max_input_sequence_length
        # include the target node itself
        max_seq_length += 1

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = \
                    nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def compute_node_walk_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                     node_multi_hop_graphs: tuple, num_neighbors: int = 20):
        """
        given node interaction time node_interact_times and node multi-hop graphs node_multi_hop_graphs,
        return the temporal embeddings of nodes
        :param node_interact_times: ndarray, shape (batch_size, )
        :param node_multi_hop_graphs: tuple of three ndarrays, each array with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        :return:
        """
        # three ndarrays, each array with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = \
            self.convert_format_from_tree_to_array(node_ids=node_ids, node_interact_times=node_interact_times,
                                                   node_multi_hop_graphs=node_multi_hop_graphs,
                                                   num_neighbors=num_neighbors)

        # get raw features of nodes in the multi-hop graphs
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, node_feat_dim)
        neighbor_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]

        # ndarray, shape (batch_size, num_neighbors ** self.walk_length), record the valid length of each walk
        walks_valid_lengths = (nodes_neighbor_ids != 0).sum(axis=-1)

        walks_valid_lengths = np.maximum(walks_valid_lengths, 1)
        # get time features of nodes in the multi-hop graphs
        # check that the time of start node in each walk should be identical to the node in the batch
        assert (nodes_neighbor_times[:, :, 0] == node_interact_times.repeat(repeats=num_neighbors,
                                                                            axis=0).
                reshape(len(node_interact_times), num_neighbors)).all()
        # ndarray, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        nodes_neighbor_delta_times = nodes_neighbor_times[:, :, 0][:, :, np.newaxis] - nodes_neighbor_times
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, time_feat_dim)
        neighbor_time_features = self.time_encoder(
            torch.from_numpy(nodes_neighbor_delta_times).float().to(self._device).flatten(start_dim=1)) \
            .reshape(nodes_neighbor_delta_times.shape[0], nodes_neighbor_delta_times.shape[1],
                     nodes_neighbor_delta_times.shape[2], self.time_feat_dim)

        # get edge features of nodes in the multi-hop graphs
        # ndarray, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        # check that the edge ids of the target node is denoted by zeros
        assert (nodes_edge_ids[:, :, 0] == 0).all()
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, edge_feat_dim)
        edge_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]

        # get position features of nodes in the multi-hop graphs
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, position_feat_dim)
        neighbor_position_features = self.position_encoder(nodes_neighbor_ids=nodes_neighbor_ids)

        # encode the random walks by walk encoder
        # Tensor, shape (batch_size, self.output_dim)
        final_node_embeddings = self.walk_encoder(neighbor_raw_features=neighbor_raw_features,
                                                  neighbor_time_features=neighbor_time_features,
                                                  edge_features=edge_features,
                                                  neighbor_position_features=neighbor_position_features,
                                                  walks_valid_lengths=walks_valid_lengths)
        return final_node_embeddings

    def convert_format_from_tree_to_array(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                          node_multi_hop_graphs: tuple, num_neighbors: int = 20):
        """
        convert the multi-hop graphs from tree-like data format to aligned array-like format
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param node_multi_hop_graphs: tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = node_multi_hop_graphs

        # add the target node to the list to generate random walks in array-like format
        nodes_neighbor_ids = [node_ids[:, np.newaxis]] + nodes_neighbor_ids
        # follow the CAWN official implementation, the edge ids of the target node is denoted by zeros
        nodes_edge_ids = [np.zeros((len(node_ids), 1)).astype(np.longlong)] + nodes_edge_ids
        nodes_neighbor_times = [node_interact_times[:, np.newaxis]] + nodes_neighbor_times

        array_format_data_list = []
        for tree_format_data in [nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times]:
            # num_last_hop_neighbors equals to num_neighbors ** self.walk_length
            batch_size, num_last_hop_neighbors, walk_length_plus_1, dtype = \
                tree_format_data[0].shape[0], tree_format_data[-1].shape[-1], len(tree_format_data), tree_format_data[
                    0].dtype
            assert batch_size == len(
                node_ids) and num_last_hop_neighbors == num_neighbors and walk_length_plus_1 == self.walk_length + 1
            # record the information of random walks with num_last_hop_neighbors paths, where each path has length walk_length_plus_1 (include the target node)
            # ndarray, shape (batch_size, num_last_hop_neighbors, walk_length_plus_1)
            array_format_data = np.empty((batch_size, num_last_hop_neighbors, walk_length_plus_1), dtype=dtype)
            for hop_idx, hop_data in enumerate(tree_format_data):
                assert (num_last_hop_neighbors % hop_data.shape[-1] == 0)
                # pad the data at each hop to be the same shape with the last hop data (which has the most number of neighbors)
                # repeat the traversed nodes in tree_format_data to get the aligned array-like format
                array_format_data[:, :, hop_idx] = np.repeat(hop_data,
                                                             repeats=num_last_hop_neighbors // hop_data.shape[-1],
                                                             axis=1)
            array_format_data_list.append(array_format_data)
        # three ndarrays with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        return array_format_data_list[0], array_format_data_list[1], array_format_data_list[2]

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)


    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)

        return updated_memory, updated_last_update


    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                         destination_node_embedding, edge_times, edge_idxs):
        edge_times = torch.from_numpy(edge_times).float().to(self._device)
        edge_features = self.edge_raw_features[edge_idxs]

        source_memory = self.memory.get_memory(source_nodes)
        destination_memory = self.memory.get_memory(destination_nodes)

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
            source_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                    source_time_delta_encoding],
                                   dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages  
    
    def set_neighbor_finder(self, neighbor_finder):
        """Called by training pipeline to inject neighbor finder."""
        self.neighbor_finder = neighbor_finder
        if self.use_memory and hasattr(self, 'embedding_module'):
            self.embedding_module.neighbor_finder = neighbor_finder

        # --- Co-GNN initialisation ---
        if self.use_explicit_co_gnn:
            if hasattr(neighbor_finder, 'edge_index') and neighbor_finder.edge_index is not None:
                edge_index = neighbor_finder.edge_index.to(self._device)
                self.co_gnn_edge_index, self.co_gnn_edge_weight = build_cooccurrence_graph(edge_index, self.n_nodes)
                self.update_co_gnn_embeddings()
                logger.info("Co-GNN graph built and embeddings initialised")
            else:
                logger.warning("Neighbor finder has no edge_index; Co-GNN embeddings will be zero (fallback)")
                # self.use_explicit_co_gnn = False
                self._c_emb = None
    
    
    def set_graph(self, edge_index, edge_time):
        self.edge_index = edge_index
        self.edge_time = edge_time
        if self.use_explicit_co_gnn:
            edge_index = edge_index.to(self._device)
            self.co_gnn_edge_index, self.co_gnn_edge_weight = build_cooccurrence_graph(edge_index, self.n_nodes)
            self.update_co_gnn_embeddings()
            logger.info("Co-GNN graph rebuilt from set_graph")

    def set_raw_features(self, node_features, edge_features):
        """Called by training pipeline to update raw features (optional)."""
        # Already set during init; can ignore or update if needed
        pass
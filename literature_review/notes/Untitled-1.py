==========================
#reproduce SOTA
# tawrmac_v1.py 

import logging
from collections import defaultdict
from gc import enable
from loguru import logger

import lightning as L

import numpy as np
import torch
import torch.nn.functional as F
from src.models.tawrmac_module.time_encoding import TimeEncode
from src.models.tawrmac_module.memory import Memory, GRUMemoryUpdater, LastMessageAggregator, IdentityMessageFunction, GraphAttentionEmbedding
from src.models.tawrmac_module.walk import WalkEncoder, PositionEncoder
from src.models.tawrmac_module.cooccurrence import NeighborCooccurrenceEncoder
from src.models.tawrmac_module.merg_layer import AffinityMergeLayer
from src.models.tawrmac_module.mlp_module import RestartMLP
from torchmetrics import AUROC, AveragePrecision
from torchmetrics import Accuracy

from .tawrmac_config import TAWRMACConfig
from dataclasses import asdict


class TAWRMACv1(L.LightningModule):
    def __init__(self, config:TAWRMACConfig):
        super(TAWRMACv1, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.save_hyperparameters(asdict(config))
        self.cfg = config
        
        self.n_layers = config.n_layers
        self.neighbor_finder = config.neighbor_finder
        self._device = config.device if isinstance(config.device, torch.device) else torch.device(config.device)
        
        if isinstance(config.node_features, np.ndarray):
            self.node_raw_features = torch.from_numpy(config.node_features.astype(np.float32)).to(self._device)
        else:
            self.node_raw_features = config.node_features.to(self._device)

        if isinstance(config.edge_features, np.ndarray):
            self.edge_raw_features = torch.from_numpy(config.edge_features.astype(np.float32)).to(self._device)
        else:
            self.edge_raw_features = config.edge_features.to(self._device)
        
        
        
        # self.node_raw_features = torch.from_numpy(config.node_features.astype(np.float32)).to(config.device)
        # self.edge_raw_features = torch.from_numpy(config.edge_features.astype(np.float32)).to(config.device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = config.n_neighbors

        
        
        
        # self.use_memory = use_memory
        # self.time_feat_dim = time_dim
        # self.time_encoder = TimeEncode(dimension=self.time_feat_dim)
        # self.memory = None
        # self.enable_walk = enable_walk
        # self.enable_restart = enable_restart
        # self.pick_new_neighbors = pick_new_neighbors  # True if approach 2 (picks new neighbors in restart)
        # self.num_walks = num_walks
        # self.neighbor_cooc = enable_neighbor_cooc
        # self.fixed_time_encoder = None

        self.use_memory = config.use_memory
        self.time_feat_dim = config.time_dim
        self.time_encoder = TimeEncode(dimension=self.time_feat_dim)
        self.memory = None
        self.enable_walk = config.enable_walk
        self.enable_restart = config.enable_restart
        self.pick_new_neighbors = config.pick_new_neighbors
        self.num_walks = config.num_walks
        self.neighbor_cooc = config.enable_neighbor_cooc
        self.fixed_time_encoder = None
        
        
        
        
        # if self.neighbor_cooc:
        #     self.max_input_sequence_length = max_input_seq_length
        #     self.neighbor_cooc_proj_out = 10
        #     self.neighbor_co_occurrence_feat_dim = 50
        #     self.neighbor_cooc_proj = torch.nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim,
        #                                               out_features=self.neighbor_cooc_proj_out, bias=True)
        #     self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
        #         neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=device)

         # Co-occurrence setup
        if self.neighbor_cooc:
            self.max_input_sequence_length = config.max_input_seq_length
            self.neighbor_cooc_proj_out = 10
            self.neighbor_co_occurrence_feat_dim = 50
            self.neighbor_cooc_proj = torch.nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_cooc_proj_out, bias=True)
            self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
                neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
                device=self._device)
        

        # Walk setup
        if self.enable_walk:
            self.walk_emb_dim = config.walk_emb_dim
            self.position_feat_dim = config.position_feat_dim
            self.walk_length = config.walk_length
            self.num_walk_heads = config.num_walk_heads

            self.position_encoder = PositionEncoder(
                position_feat_dim=self.position_feat_dim,
                walk_length=self.walk_length,
                device=self._device)

            self.walk_encoder = WalkEncoder(
                input_dim=self.n_node_features + self.n_edge_features + self.time_feat_dim + self.position_feat_dim,
                position_feat_dim=self.position_feat_dim,
                output_dim=self.walk_emb_dim,
                num_walk_heads=self.num_walk_heads,
                dropout=config.dropout)
            if self.enable_restart:
                self.restart_prob = RestartMLP(dim=self.n_node_features)
            
       
        # Memory setup
        if self.use_memory:
            self.fixed_time_dim = config.fixed_time_dim
            self.fixed_time_encoder = TimeEncode(dimension=self.fixed_time_dim, parameter_requires_grad=False)
            self.memory_dimension = self.n_node_features
            self.memory_update_at_start = config.memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + self.time_encoder.dimension
            message_dimension = raw_message_dimension
            self.memory = Memory(
                n_nodes=self.n_nodes,
                memory_dimension=self.memory_dimension,
                input_dimension=message_dimension,
                message_dimension=message_dimension,
                device=self._device
            )
            self.message_aggregator = LastMessageAggregator(device=self._device)
            self.message_function = IdentityMessageFunction()
            self.memory_updater = GRUMemoryUpdater(
                memory=self.memory,
                message_dimension=message_dimension,
                memory_dimension=self.memory_dimension,
                device=self._device
            )

            self.embedding_module = GraphAttentionEmbedding(
                node_features=self.node_raw_features,
                edge_features=self.edge_raw_features,
                memory=self.memory,
                neighbor_finder=self.neighbor_finder,
                time_encoder=self.time_encoder,
                fixed_time_encoder=self.fixed_time_encoder,
                n_layers=self.n_layers,
                n_node_features=self.n_node_features,
                n_edge_features=self.n_edge_features,
                n_time_features=self.time_feat_dim,
                embedding_dimension=self.embedding_dimension,
                device=self._device,
                n_heads=config.n_heads,
                dropout=config.dropout,
                use_memory=True,
                n_fixed_time_features=self.fixed_time_dim
            )


        # Final embedding dimension
        self.final_emb_dim = 0
        if self.use_memory:
            self.final_emb_dim += self.n_node_features
        if self.enable_walk:
            self.final_emb_dim += self.walk_emb_dim
            if self.enable_restart:
                self.final_emb_dim += 1
        if self.neighbor_cooc:
            self.final_emb_dim += (self.max_input_sequence_length + 1) * self.neighbor_cooc_proj_out
        self.affinity_score = AffinityMergeLayer(
            self.final_emb_dim, 
            self.final_emb_dim,
            self.n_node_features, 1
        )

        self.val_auroc = AUROC(task='binary')
        self.val_ap = AveragePrecision(task='binary')
        self.test_auroc = AUROC(task='binary')
        self.test_ap = AveragePrecision(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')

    def forward(self, sources, destinations, timestamps, edge_idxs, negative_sources=None, negative_destinations=None):
        """Wrapper for edge probability computation."""
       
        return self.compute_edge_probabilities(
            sources, 
            destinations, 
            negative_sources, 
            negative_destinations,
            timestamps, 
            edge_idxs, 
            self.n_neighbors
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
        if self.neighbor_finder is not None:
            self.neighbor_finder.clear_cache()
        # Also reinitialize memory if needed
        if self.use_memory and self.memory is not None:
            self.memory.__init_memory__()
    
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


    
    
    # def set_neighbor_finder(self, neighbor_finder):
    #     self.neighbor_finder = neighbor_finder
    #     if self.use_memory:
    #         self.embedding_module.neighbor_finder = neighbor_finder

    def set_neighbor_finder(self, neighbor_finder):
        """Called by training pipeline to inject neighbor finder."""
        self.neighbor_finder = neighbor_finder
        if self.use_memory and hasattr(self, 'embedding_module'):
            self.embedding_module.neighbor_finder = neighbor_finder

    def set_graph(self, edge_index, edge_time):
        """Called by training pipeline; TAWRMAC uses neighbor_finder instead."""
        # Not needed, neighbor_finder already holds graph info
        pass

    def set_raw_features(self, node_features, edge_features):
        """Called by training pipeline to update raw features (optional)."""
        # Already set during init; can ignore or update if needed
        pass
============
# time_encoding.py
import torch
import numpy as np


class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension, parameter_requires_grad: bool = True):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))

        return output
======================
#memory.py
import torch
from torch import nn
import numpy as np
from collections import defaultdict
from copy import deepcopy
from src.models.tawrmac_module.temporal_attention import TemporalAttentionLayer


class Memory(nn.Module):

    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
                 device="cpu", combination_method='sum'):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device

        self.combination_method = combination_method

        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model
        # n_nodes = self.n_nodes

        # self.memory = nn.Parameter(torch.zeros((n_nodes, self.memory_dimension)).to(self.device),
        #                            requires_grad=False)
        # self.last_update = nn.Parameter(torch.zeros(n_nodes).to(self.device),
        #                                 requires_grad=False)

        # self.messages = defaultdict(list)
        current_device = self.memory.device if hasattr(self, 'memory') else self.device
        # Use small random values instead of zeros
        memory_tensor = torch.randn(self.n_nodes, self.memory_dimension, device=current_device) * 0.01
        last_update_tensor = torch.zeros(self.n_nodes, device=current_device)
        self.memory = nn.Parameter(memory_tensor, requires_grad=False)
        self.last_update = nn.Parameter(last_update_tensor, requires_grad=False)
        self.messages = defaultdict(list)

    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])


    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self.memory.detach()

        # if isinstance(self.last_update, torch.Tensor):
        #     self.last_update = self.last_update.detach()
        
        # Detach all stored messages
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))

            self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []


class GRUMemoryUpdater(nn.Module):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__()
        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return

        # assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
        #                                                                                   "update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps

        updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)


    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        #
        # assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
        #                                                                                   "update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update



class IdentityMessageFunction(nn.Module):

    def compute_message(self, raw_messages):
        return raw_messages


class LastMessageAggregator(nn.Module):
    def __init__(self, device):
        super(LastMessageAggregator, self).__init__()
        self.device = device

    def aggregate(self, node_ids, messages):
        """Only keep the last message for each node"""
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = []

        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_messages.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps

    def group_by_id(self, node_ids, messages, timestamps):
        node_id_to_messages = defaultdict(list)

        for i, node_id in enumerate(node_ids):
            node_id_to_messages[node_id].append((messages[i], timestamps[i]))

        return node_id_to_messages


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout, fixed_time_encoder, n_fixed_time_features):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.fixed_time_encoder = fixed_time_encoder
        self.n_fixed_time_features = n_fixed_time_features
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        return NotImplemented


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, fixed_time_encoder=None, n_fixed_time_features=0):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout, fixed_time_encoder,
                                             n_fixed_time_features)

        self.use_memory = use_memory
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))

        source_nodes_fixed_time_embedding = None
        if self.fixed_time_encoder is not None:
            source_nodes_fixed_time_embedding = self.fixed_time_encoder(torch.zeros_like(
                timestamps_torch))

        source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:

            source_node_conv_embeddings = self.compute_embedding(memory,
                                                                 source_nodes,
                                                                 timestamps,
                                                                 n_layers=n_layers - 1,
                                                                 n_neighbors=n_neighbors)

            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors)

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps[:, np.newaxis] - edge_times

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            neighbor_embeddings = self.compute_embedding(memory,
                                                         neighbors,
                                                         np.repeat(timestamps, n_neighbors),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_fixed_time_embeddings = None
            if self.fixed_time_encoder is not None:
                edge_fixed_time_embeddings = self.fixed_time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            mask = neighbors_torch == 0

            source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                              source_nodes_time_embedding,
                                              source_nodes_fixed_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_fixed_time_embeddings,
                                              edge_features,
                                              mask)

            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding, source_nodes_fixed_time_embedding, neighbor_embeddings, edge_time_embeddings, edge_fixed_time_embeddings, edge_features,
                  mask):
        return NotImplemented


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, fixed_time_encoder=None, n_fixed_time_features=0):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout,
                                                      use_memory, fixed_time_encoder, n_fixed_time_features)

        if fixed_time_encoder is None:
            n_fixed_time_features = 0
        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            fixed_time_dim=n_fixed_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding, source_nodes_fixed_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_time_fixed_embeddings, edge_features,
                  mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              source_nodes_fixed_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_time_fixed_embeddings,
                                              edge_features,
                                              mask)

        return source_embedding
=====================
#walk.py
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class PositionEncoder(torch.nn.Module):

    def __init__(self, position_feat_dim: int, walk_length: int, device: str = 'cpu'):
        """
        Position encoder that computes each node position features.
        :param position_feat_dim: int, dimension of position features (encodings)
        :param walk_length: int, length of each random walk
        :param device: str, device
        """
        super(PositionEncoder, self).__init__()
        self.position_feat_dim = position_feat_dim
        self.walk_length = walk_length
        self.device = device

        # two-layered feed forward network with ReLU activation
        self.position_encode_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.walk_length + 1, out_features=self.position_feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.position_feat_dim, out_features=self.position_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                node_interact_times: np.ndarray,
                                src_node_multi_hop_graphs: tuple, dst_node_multi_hop_graphs: tuple):
        """
        count the appearances of nodes in the multi-hop graphs that are generated by random walks starting from src and dst nodes
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param src_node_multi_hop_graphs: tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        :param dst_node_multi_hop_graphs: tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        :return:
        """
        # use node id and interaction timestamp to identify a node in the multi-hop graph
        # src_nodes_neighbor_ids and src_nodes_neighbor_times are lists, each list contains self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        src_nodes_neighbor_ids, _, src_nodes_neighbor_times = src_node_multi_hop_graphs
        # dst_nodes_neighbor_ids and dst_nodes_neighbor_times are lists, each list contains self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        dst_nodes_neighbor_ids, _, dst_nodes_neighbor_times = dst_node_multi_hop_graphs

        # dictionary, {node_identity (key): ndarray with shape (2, self.walk_length + 1) (value)}
        # store the appearances of all the nodes in the multi-hop graphs that are generated by random walks starting from src and dst nodes
        self.nodes_appearances = {}
        # get the multi-hop information for each node
        for idx, (src_node_id, dst_node_id, node_interact_time) in enumerate(
                zip(src_node_ids, dst_node_ids, node_interact_times)):
            # src_node_neighbor_ids, list of ndarrays, each ndarray with shape (num_neighbors ** current_hop)
            src_node_neighbor_ids = [src_nodes_single_hop_neighbor_ids[idx] for src_nodes_single_hop_neighbor_ids in
                                     src_nodes_neighbor_ids]
            src_node_neighbor_times = [src_nodes_single_hop_neighbor_times[idx] for src_nodes_single_hop_neighbor_times
                                       in src_nodes_neighbor_times]
            dst_node_neighbor_ids = [dst_nodes_single_hop_neighbor_ids[idx] for dst_nodes_single_hop_neighbor_ids in
                                     dst_nodes_neighbor_ids]
            dst_node_neighbor_times = [dst_nodes_single_hop_neighbor_times[idx] for dst_nodes_single_hop_neighbor_times
                                       in dst_nodes_neighbor_times]

            # dictionary, {node_identity (key): ndarray with shape (2, self.walk_length + 1) (value)}
            # store the appearances of nodes in the multi-hop graphs that are generated by random walks starting from src_node_id and dst_node_id
            tmp_nodes_appearances = {}
            # add the information of src_node and dst_node to the lists
            src_node_neighbor_ids, src_node_neighbor_times = [[src_node_id]] + src_node_neighbor_ids, [
                [node_interact_time]] + src_node_neighbor_times
            dst_node_neighbor_ids, dst_node_neighbor_times = [[dst_node_id]] + dst_node_neighbor_ids, [
                [node_interact_time]] + dst_node_neighbor_times
            for current_hop in range(self.walk_length + 1):
                for src_node_neighbor_id, src_node_neighbor_time, dst_node_neighbor_id, dst_node_neighbor_time in \
                        zip(src_node_neighbor_ids[current_hop], src_node_neighbor_times[current_hop],
                            dst_node_neighbor_ids[current_hop], dst_node_neighbor_times[current_hop]):

                    # follow the CAWN official implementation, use the batch index and node id to represent the node key
                    src_node_key = '-'.join([str(idx), str(src_node_neighbor_id)])
                    dst_node_key = '-'.join([str(idx), str(dst_node_neighbor_id)])

                    if src_node_key not in tmp_nodes_appearances:
                        # create a ndarray with shape (2, self.walk_length + 1) for the src node to record its appearances
                        tmp_nodes_appearances[src_node_key] = np.zeros((2, self.walk_length + 1), dtype=np.float32)
                    if dst_node_key not in tmp_nodes_appearances:
                        # create a ndarray with shape (2, self.walk_length + 1) for the dst node to record its appearances
                        tmp_nodes_appearances[dst_node_key] = np.zeros((2, self.walk_length + 1), dtype=np.float32)

                    # count the appearances of each node in the multi-hop graphs that are generated by random walks starting from src_node_id and dst_node_id
                    # for each node, tmp_nodes_appearances[node_key][0, :] records the node appearances in the random walks starting from src_node_id
                    # while tmp_nodes_appearances[node_key][1, :] records the node appearances in the random walks starting from dst_node_id
                    # number of neighbors at the current hop
                    num_current_hop_neighbors = len(src_node_neighbor_ids[current_hop])
                    # convert into landing probabilities by normalizing with k hop sampling number
                    tmp_nodes_appearances[src_node_key][0, current_hop] += 1 / num_current_hop_neighbors
                    tmp_nodes_appearances[dst_node_key][1, current_hop] += 1 / num_current_hop_neighbors
            # set the appearances of the padded node (with zero index) to zeros
            tmp_nodes_appearances['-'.join([str(idx), str(0)])] = np.zeros((2, self.walk_length + 1), dtype=np.float32)
            self.nodes_appearances.update(tmp_nodes_appearances)

    def forward(self, nodes_neighbor_ids: np.ndarray):
        """
        compute the position features of nodes in nodes_neighbor_ids
        :param nodes_neighbor_ids: ndarray, shape shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        :return:
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        """
        # batch_indices -> array([[[0, ..., 0,], ..., [0, ..., 0,]], [[1, ..., 1], ..., [1, ..., 1]] ..., [[batch - 1, ..., batch - 1], ..., [batch - 1, ..., batch - 1]]])
        batch_indices = np.arange(nodes_neighbor_ids.shape[0]).repeat(
            nodes_neighbor_ids.shape[1] * nodes_neighbor_ids.shape[2]).reshape(nodes_neighbor_ids.shape)

        # list of string keys, shape (batch_size * (num_neighbors ** self.walk_length) * (self.walk_length + 1))
        batch_keys = ['-'.join([str(batch_indices[i][j][k]), str(nodes_neighbor_ids[i][j][k])])
                      for i in range(batch_indices.shape[0]) for j in range(batch_indices.shape[1]) for k in
                      range(batch_indices.shape[2])]

        # unique_keys, ndarray, shape (num_unique_keys, )
        # inverse_indices, ndarray, shape (batch_size * (num_neighbors ** self.walk_length) * (self.walk_length + 1))
        # we can use unique_keys[inverse_indices] to reconstruct the original input
        unique_keys, inverse_indices = np.unique(batch_keys, return_inverse=True)
        # self.nodes_appearances, dictionary, {node_identity (key): ndarray with shape (2, self.walk_length + 1) (value)}
        # unique_node_appearances, ndarray, shape (num_unique_keys, 2, self.walk_length + 1)
        unique_node_appearances = np.array([self.nodes_appearances[unique_key] for unique_key in unique_keys])
        # the appearances of nodes in nodes_neighbor_ids, ndarray, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, 2, self.walk_length + 1)
        node_appearances = unique_node_appearances[inverse_indices, :].reshape(nodes_neighbor_ids.shape[0],
                                                                               nodes_neighbor_ids.shape[1],
                                                                               nodes_neighbor_ids.shape[2], 2,
                                                                               self.walk_length + 1)

        # encode the node appearances in the random walks by MLPs
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, 2, position_feat_dim)
        position_features = self.position_encode_layer(torch.Tensor(node_appearances).float().to(self.device))
        # add the position features of each node in random walks generated by src and dst nodes by summing over the second last dimension, Equation (6) in CAWN paper
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, position_feat_dim)
        position_features = position_features.sum(dim=-2)
        return position_features


class WalkEncoder(torch.nn.Module):

    def __init__(self, input_dim: int, position_feat_dim: int, output_dim: int, num_walk_heads: int,
                 dropout: float = 0.1):
        """
        Walk encoder that first encodes each random walk by BiLSTM and then aggregates all the walks by the self-attention in Transformer
        :param input_dim: int, dimension of the input
        :param position_feat_dim: int, dimension of position features (encodings)
        :param output_dim: int, dimension of the output
        :param num_walk_heads: int, number of attention heads to aggregate random walks
        :param dropout: float, dropout rate
        """
        super(WalkEncoder, self).__init__()
        self.input_dim = input_dim
        self.position_feat_dim = position_feat_dim
        # follow the CAWN official implementation, take half of the model dimension to save computation cost for attention
        self.attention_dim = self.input_dim // 2
        self.output_dim = output_dim
        self.num_walk_heads = num_walk_heads
        self.dropout = dropout
        # make sure that the attention dimension can be divided by number of walk heads
        if self.attention_dim % self.num_walk_heads != 0:
            self.attention_dim += (self.num_walk_heads - self.attention_dim % self.num_walk_heads)

        # BiLSTM Encoders, encode the node features along each random walk
        self.feature_encoder = BiLSTMEncoder(input_dim=self.input_dim, hidden_dim=self.input_dim)
        # encode position features along each temporal walk
        self.position_encoder = BiLSTMEncoder(input_dim=self.position_feat_dim, hidden_dim=self.position_feat_dim)

        self.transformer_encoder = TransformerEncoder(attention_dim=self.attention_dim, num_heads=self.num_walk_heads,
                                                      dropout=self.dropout)

        # due to the usage of BiLSTM, self.feature_encoder.modell_dim may not be equal to self.input_dim, since self.input_dim may not be an even number
        # also, self.position_encoder.model_dim may not be equal to self.input_dim, since self.input_dim may not be an even number
        # projection layers for 1) combination of outputs from self.feature_encoder and self.position_encoder; and 2) final output
        self.projection_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=self.feature_encoder.model_dim + self.position_encoder.model_dim,
                            out_features=self.attention_dim),
            torch.nn.Linear(in_features=self.attention_dim, out_features=self.output_dim)
        ])

    def forward(self, neighbor_raw_features: torch.Tensor, neighbor_time_features: torch.Tensor,
                edge_features: torch.Tensor,
                neighbor_position_features: torch.Tensor, walks_valid_lengths: np.ndarray):
        """
        first encode each random walk by BiLSTM and then aggregate all the walks by the self-attention in Transformer
        :param neighbor_raw_features: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, node_feat_dim)
        :param neighbor_time_features: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, time_feat_dim)
        :param edge_features: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, edge_feat_dim)
        :param neighbor_position_features: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, position_feat_dim)
        :param walks_valid_lengths: ndarray, shape (batch_size, num_neighbors ** self.walk_length), record the valid length of each walk
        :return:
        """
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, node_feat_dim + time_feat_dim + edge_feat_dim + position_feat_dim)
        combined_features = torch.cat(
            [neighbor_raw_features, neighbor_time_features, edge_features, neighbor_position_features], dim=-1)
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.feature_encoder.model_dim), feed the combined features to BiLSTM
        combined_features = self.feature_encoder(inputs=combined_features, lengths=walks_valid_lengths)
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.position_encoder.model_dim), feed the position features to BiLSTM
        neighbor_position_features = self.position_encoder(inputs=neighbor_position_features,
                                                           lengths=walks_valid_lengths)
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.feature_encoder.model_dim + self.position_encoder.model_dim)
        combined_features = torch.cat([combined_features, neighbor_position_features], dim=-1)
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.attention_dim)
        combined_features = self.projection_layers[0](combined_features)
        # Tensor, shape (batch_size, self.attention_dim), feed into Transformer and then perform mean pooling over multiple random walks
        combined_features = self.transformer_encoder(inputs_query=combined_features).mean(dim=-2)
        # Tensor, shape (batch_size, self.output_dim)
        outputs = self.projection_layers[1](combined_features)
        return outputs


class BiLSTMEncoder(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        BiLSTM encoder.
        :param input_dim: int, dimension of the input
        :param hidden_dim: int, dimension of the hidden state
        """
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim_one_direction = hidden_dim // 2
        self.model_dim = self.hidden_dim_one_direction * 2
        self.bilstm_encoder = torch.nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim_one_direction,
                                            batch_first=True,
                                            bidirectional=True)

    def forward(self, inputs: torch.Tensor, lengths: np.ndarray):
        """
        encode the inputs by BiLSTM encoder based on lengths
        :param inputs: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, input_dim)
        :param lengths: ndarray, shape (batch_size, num_neighbors ** self.walk_length), record the valid length of each walk
        :return:
        """
        
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length), self.walk_length + 1, input_dim), which corresponds to the LSTM input (batch_size, seq_len, input_dim)
        inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
        # a PackedSequence object, pack the padded sequence for efficient computation and avoid the errors of computing padded value, set enforce_sorted to False
        inputs = pack_padded_sequence(inputs, lengths.flatten(), batch_first=True, enforce_sorted=False)
        # the outputs of LSTM are output, (h_n, c_n), and we only use the output and do not use hidden states
        encoded_features, _ = self.bilstm_encoder(inputs)
        # encoded_features, Tensor, shape (batch_size * (num_neighbors ** self.walk_length), self.walk_length + 1, self.model_dim), pad the packed sequence
        # seq_lengths, Tensor, shape (batch_size * (num_neighbors ** self.walk_length), )
        encoded_features, seq_lengths = pad_packed_sequence(encoded_features, batch_first=True)
        assert (seq_lengths.numpy() == lengths.flatten()).all()
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length), ), the shifted sequence lengths
        shifted_seq_lengths = seq_lengths + torch.tensor(
            [i * encoded_features.shape[1] for i in range(encoded_features.shape[0])])
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length) * (self.walk_length + 1), self.model_dim)
        encoded_features = encoded_features.reshape(encoded_features.shape[0] * encoded_features.shape[1],
                                                    encoded_features.shape[2])
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.model_dim), get the encodings of each walk at the last position
        # note that we need to use shifted_seq_lengths - 1 to get the shifted indices
        encoded_features = encoded_features[shifted_seq_lengths - 1].reshape(lengths.shape[0], lengths.shape[1],
                                                                             self.model_dim)

        return encoded_features


class TransformerEncoder(torch.nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = torch.nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads,
                                                                dropout=dropout)

        self.dropout = torch.nn.Dropout(dropout)

        self.linear_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            torch.nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = torch.nn.ModuleList([
            torch.nn.LayerNorm(attention_dim),
            torch.nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0,
                                                                                                         1), inputs_key.transpose(
            0, 1), inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[
            0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs
    
============================
#cooccurrence.py
import torch
import numpy as np

class NeighborCooccurrenceEncoder(torch.nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim,
                            out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_nodes_neighbor_ids,
                                                                              dst_nodes_neighbor_ids):
            # src_unique_keys, ndarray, shape (num_src_unique_keys, )
            # src_inverse_indices, ndarray, shape (src_max_seq_length, )
            # src_counts, ndarray, shape (num_src_unique_keys, )
            # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids,
                                                                         return_inverse=True, return_counts=True)
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(
                self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
            # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
            # dst_counts, ndarray, shape (num_dst_unique_keys, )
            # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids,
                                                                         return_inverse=True, return_counts=True)
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(
                self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(
                lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(
                torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

            # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(
                lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(
                torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))

        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_nodes_appearances[torch.from_numpy(src_nodes_neighbor_ids == 0)] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_nodes_appearances[torch.from_numpy(dst_nodes_neighbor_ids == 0)] = 0.0

        return src_nodes_appearances, dst_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(
            src_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features
===================
#merge_layer.py
import torch
import torch.nn as nn

class AffinityMergeLayer(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3 * 2)
        self.fc2 = nn.Linear(dim3 * 2, dim3)
        self.fc3 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop, inplace=False)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.act(x)
        return self.fc3(x)


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

=======================
#mlp_module.py
import torch 
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(dim, 250)
        self.fc_2 = nn.Linear(250, 50)
        self.fc_3 = nn.Linear(50, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class RestartMLP(nn.Module):
    def __init__(self, dim, drop=0.2):
        super().__init__()
        self.fc_1 = nn.Linear(dim, 80)
        self.fc_2 = nn.Linear(80, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.fc_2(x)
        x = torch.sigmoid(x)
        return x
    
====================
#temporal_attention.py
import torch
from torch import nn

from src.models.tawrmac_module.merg_layer import MergeLayer


class TemporalAttentionLayer(torch.nn.Module):
    """
    Temporal attention layer. Return the temporal embedding of a node given the node itself,
     its neighbors and the edge timestamps.
    """

    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim, fixed_time_dim,
                 output_dimension, n_head=2,
                 dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()

        self.n_head = n_head

        self.feat_dim = n_node_features
        self.time_dim = time_dim
        self.fixed_time_dim = fixed_time_dim
        self.query_dim = n_node_features + time_dim + fixed_time_dim

        self.key_dim = n_neighbors_features + time_dim + n_edge_features + fixed_time_dim

        self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

        self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                       kdim=self.key_dim,
                                                       vdim=self.key_dim,
                                                       num_heads=n_head,
                                                       dropout=dropout)

    def forward(self, src_node_features, src_time_features, src_fixed_time_features, neighbors_features,
                neighbors_time_features, neighbors_fixed_time_features, edge_features, neighbors_padding_mask):
        """
        "Temporal attention model
        :param src_node_features: float Tensor of shape [batch_size, n_node_features]
        :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
        :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
        :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
        time_dim]
        :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
        :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
        :return:
        attn_output: float Tensor of shape [1, batch_size, n_node_features]
        attn_output_weights: [batch_size, 1, n_neighbors]
        """

        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)

        query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
        if src_fixed_time_features is not None:
            query = torch.cat([query, src_fixed_time_features], dim=2)
        key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)
        if neighbors_fixed_time_features is not None:
            key = torch.cat([key, neighbors_fixed_time_features], dim=2)

        # print(neighbors_features.shape, edge_features.shape, neighbors_time_features.shape)
        # Reshape tensors so to expected shape by multi head attention
        query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
        key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

        # Compute mask of which source nodes have no valid neighbors
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
        # force the attention to just 'attend' on this neighbor (which has the same features as all
        # the others since they are fake neighbors) and will produce an equivalent result to the
        # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

        # print(query.shape, key.shape)

        attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                                  key_padding_mask=neighbors_padding_mask)

        # mask = torch.unsqueeze(neighbors_padding_mask, dim=2)  # mask [B, N, 1]
        # mask = mask.permute([0, 2, 1])
        # attn_output, attn_output_weights = self.multi_head_target(q=query, k=key, v=key,
        #                                                           mask=mask)

        attn_output = attn_output.squeeze()
        attn_output_weights = attn_output_weights.squeeze()

        # Source nodes with no neighbors have an all zero attention output. The attention output is
        # then added or concatenated to the original source node features and then fed into an MLP.
        # This means that an all zero vector is not used.
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

        # Skip connection with temporal attention over neighborhood and the features of the node itself
        attn_output = self.merger(attn_output, src_node_features)

        return attn_output, attn_output_weights
========================


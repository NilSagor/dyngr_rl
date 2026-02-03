import logging
from collections import defaultdict
from gc import enable

import numpy as np
import torch
import torch.nn.functional as F
from model.time_encoding import TimeEncode
from model.memory import Memory, GRUMemoryUpdater, LastMessageAggregator, IdentityMessageFunction, GraphAttentionEmbedding
from model.walk import WalkEncoder, PositionEncoder
from model.cooccurrence import NeighborCooccurrenceEncoder
from utils.utils import AffinityMergeLayer
from utils.utils import RestartMLP


class TAWRMAC(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True,
                 memory_dimension=500,
                 n_neighbors=None,
                 enable_walk=False,
                 enable_restart=False,
                 pick_new_neighbors=False,
                 enable_neighbor_cooc=False,
                 walk_emb_dim=172,
                 time_dim=172,
                 fixed_time_dim=20,
                 max_input_seq_length=32,
                 position_feat_dim=100,
                 walk_length=4,
                 num_walks=10,
                 num_walk_heads=4):
        super(TAWRMAC, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors

        self.use_memory = use_memory
        self.time_feat_dim = time_dim
        self.time_encoder = TimeEncode(dimension=self.time_feat_dim)
        self.memory = None
        self.enable_walk = enable_walk
        self.enable_restart = enable_restart
        self.pick_new_neighbors = pick_new_neighbors  # True if approach 2 (picks new neighbors in restart)
        self.num_walks = num_walks
        self.neighbor_cooc = enable_neighbor_cooc
        self.fixed_time_encoder = None

        if self.neighbor_cooc:
            self.max_input_sequence_length = max_input_seq_length
            self.neighbor_cooc_proj_out = 10
            self.neighbor_co_occurrence_feat_dim = 50
            self.neighbor_cooc_proj = torch.nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim,
                                                      out_features=self.neighbor_cooc_proj_out, bias=True)
            self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
                neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=device)

        if self.enable_walk:
            self.walk_emb_dim = walk_emb_dim
            self.position_feat_dim = position_feat_dim
            self.walk_length = walk_length
            self.num_walk_heads = num_walk_heads


            self.position_encoder = PositionEncoder(position_feat_dim=self.position_feat_dim,
                                                    walk_length=self.walk_length,
                                                    device=device)

            self.walk_encoder = WalkEncoder(
                input_dim=self.n_node_features + self.n_edge_features + self.time_feat_dim + self.position_feat_dim,
                position_feat_dim=self.position_feat_dim, output_dim=self.walk_emb_dim,
                num_walk_heads=self.num_walk_heads,
                dropout=dropout)
            if self.enable_restart:
                self.restart_prob = RestartMLP(dim=self.n_node_features)

        if self.use_memory:
            self.fixed_time_dim = fixed_time_dim
            self.fixed_time_encoder = TimeEncode(dimension=self.fixed_time_dim, parameter_requires_grad=False)

            self.memory_dimension = self.n_node_features
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension


            message_dimension = raw_message_dimension
            self.memory = Memory(n_nodes=self.n_nodes,
                                 memory_dimension=self.memory_dimension,
                                 input_dimension=message_dimension,
                                 message_dimension=message_dimension,
                                 device=device)
            self.message_aggregator = LastMessageAggregator(device=device)
            self.message_function = IdentityMessageFunction()
            self.memory_updater = GRUMemoryUpdater(memory=self.memory,
                                                   message_dimension=message_dimension,
                                                   memory_dimension=self.memory_dimension,
                                                   device=device)


            self.embedding_module = GraphAttentionEmbedding(node_features=self.node_raw_features,
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
                                                         device=self.device,
                                                         n_heads=n_heads, dropout=dropout,
                                                         use_memory=True,
                                                         n_fixed_time_features=self.fixed_time_dim)


        self.final_emb_dim = 0
        if self.use_memory:
            self.final_emb_dim += self.n_node_features
        if self.enable_walk:
            self.final_emb_dim += self.walk_emb_dim
            if self.enable_restart:
                self.final_emb_dim += 1
        if self.neighbor_cooc:
            self.final_emb_dim += (self.max_input_sequence_length + 1) * self.neighbor_cooc_proj_out
        self.affinity_score = AffinityMergeLayer(self.final_emb_dim, self.final_emb_dim,
                                                 self.n_node_features, 1)

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
                torch.empty((n_samples, self.n_node_features), requires_grad=True)).to(self.device)
            dst_restart_emb = torch.nn.Parameter(
                torch.empty((n_samples, self.n_node_features), requires_grad=True)).to(self.device)
            neg_src_restart_emb = torch.nn.Parameter(
                torch.empty((n_samples, self.n_node_features), requires_grad=True)).to(self.device)
            neg_dst_restart_emb = torch.nn.Parameter(
                torch.empty((n_samples, self.n_node_features), requires_grad=True)).to(self.device)

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

        # get time features of nodes in the multi-hop graphs
        # check that the time of start node in each walk should be identical to the node in the batch
        assert (nodes_neighbor_times[:, :, 0] == node_interact_times.repeat(repeats=num_neighbors,
                                                                            axis=0).
                reshape(len(node_interact_times), num_neighbors)).all()
        # ndarray, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        nodes_neighbor_delta_times = nodes_neighbor_times[:, :, 0][:, :, np.newaxis] - nodes_neighbor_times
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, time_feat_dim)
        neighbor_time_features = self.time_encoder(
            torch.from_numpy(nodes_neighbor_delta_times).float().to(self.device).flatten(start_dim=1)) \
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
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
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
        self.neighbor_finder = neighbor_finder
        if self.use_memory:
            self.embedding_module.neighbor_finder = neighbor_finder





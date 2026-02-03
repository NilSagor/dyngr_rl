import torch
from torch import nn
import numpy as np
from collections import defaultdict
from copy import deepcopy
from model.temporal_attention import TemporalAttentionLayer


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
        n_nodes = self.n_nodes

        self.memory = nn.Parameter(torch.zeros((n_nodes, self.memory_dimension)).to(self.device),
                                   requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(n_nodes).to(self.device),
                                        requires_grad=False)

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
        self.memory.detach_()

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
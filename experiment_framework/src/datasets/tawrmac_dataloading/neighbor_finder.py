import numpy as np
import torch
import random

PRECISION = 5


# TAWRMAC NeighborFinder implementation  from
# https://anonymous.4open.science/r/tawrmac-A253/utils/utils.py

def get_neighbor_finder(data, uniform, max_node_idx=None, seed=None, use_cache=False):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs,
                                                        data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NewNeighborFinder(adj_list, sample_neighbor_strategy='uniform' if uniform else 'recent', seed=seed,
                             time_scaling_factor=1e-6, use_cache=use_cache)


class NewNeighborFinder:
    def __init__(self, adj_list, use_cache=False, sample_neighbor_strategy='uniform', seed=None,
                 time_scaling_factor=0.0):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        self.sample_neighbor_strategy = sample_neighbor_strategy

        # if self.sample_neighbor_strategy == 'time_interval_aware':
        self.nodes_neighbor_sampled_probabilities = []
        self.time_scaling_factor = time_scaling_factor

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            # if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities.append(
                self.compute_sampled_probabilities(np.array([x[2] for x in sorted_neighhbors])))

        self.seed = seed

        self.use_cache = use_cache
        self.cache = {}
        if seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        sampled_probabilities = np.exp(self.time_scaling_factor * node_neighbor_times)
        # sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_before(self, src_idx, cut_time, return_sampled_probabilities: bool = False):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph.
        The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """

        if self.use_cache:
            result = self.check_cache(src_idx, cut_time)
            if result is not None:
                return result[0], result[1], result[2], result[3]
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        if return_sampled_probabilities:
            result = (self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i],
                self.node_to_edge_timestamps[src_idx][:i], self.nodes_neighbor_sampled_probabilities[src_idx][:i])

        else:
            result = (self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i],
                      self.node_to_edge_timestamps[src_idx][:i], None)

        if self.use_cache:
            self.update_cache(src_idx, cut_time, result)
        return result

    def update_cache(self, node, ts, results):
        ts_str = str(round(ts, PRECISION))
        key = (node, ts_str)
        if key not in self.cache:
            self.cache[key] = results

    def check_cache(self, node, ts):
        ts_str = str(round(ts, PRECISION))
        key = (node, ts_str)
        return self.cache.get(key)

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        assert len(source_nodes) == len(timestamps)

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.int32)
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.float32)
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.int32)

        for i, (src, ts) in enumerate(zip(source_nodes, timestamps)):
            # Directly query neighbors before ts
            src_neighbors, src_edge_idxs, src_edge_times, _ = self.find_before(src, ts)

            if len(src_neighbors) > 0 and n_neighbors > 0:
                if self.sample_neighbor_strategy == 'uniform':
                    # Uniform sampling with probabilities (softmax if needed)
                    _, _, _, probs = self.find_before(src, ts, return_sampled_probabilities=True)
                    if probs is not None and len(probs) > 0:
                        # Convert to probabilities
                        probs = torch.softmax(torch.from_numpy(probs).float(), dim=0).numpy()
                    else:
                        probs = None
                    if self.seed is None:
                        sampled_idx = np.random.choice(len(src_neighbors), size=n_neighbors, p=probs)
                    else:
                        sampled_idx = self.random_state.choice(len(src_neighbors), size=n_neighbors, p=probs)
                    neighbors[i, :] = src_neighbors[sampled_idx]
                    edge_times[i, :] = src_edge_times[sampled_idx]
                    edge_idxs[i, :] = src_edge_idxs[sampled_idx]
                    # Sort by time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    # Take most recent n_neighbors
                    src_edge_times = src_edge_times[-n_neighbors:]
                    src_neighbors = src_neighbors[-n_neighbors:]
                    src_edge_idxs = src_edge_idxs[-n_neighbors:]
                    neighbors[i, n_neighbors - len(src_neighbors):] = src_neighbors
                    edge_times[i, n_neighbors - len(src_edge_times):] = src_edge_times
                    edge_idxs[i, n_neighbors - len(src_edge_idxs):] = src_edge_idxs
        return neighbors, edge_idxs, edge_times
    
    
    # def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    #     """
    #     Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    #     Params
    #     ------
    #     src_idx_l: List[int]
    #     cut_time_l: List[float],
    #     num_neighbors: int
    #     """
    #     assert (len(source_nodes) == len(timestamps))

    #     tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    #     # NB! All interactions described in these matrices are sorted in each row by time
    #     neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
    #         np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    #     edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
    #         np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    #     edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
    #         np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    #     assert len(self.all_source_neighbors) > 0
    #     for i in range(len(source_nodes)):
    #         # source_neighbors, source_edge_idxs, source_edge_times, node_neighbor_sampled_probabilities = self.find_before(
    #         #     source_node, timestamp,
    #         #     return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
    #         source_neighbors, source_edge_idxs, source_edge_times, node_neighbor_sampled_probabilities = \
    #             self.all_source_neighbors[i], self.all_source_edge_idx[i], self.all_source_edge_times[i], \
    #                 None
    #         if len(source_neighbors) > 0 and n_neighbors > 0:
    #             # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None

    #             if self.sample_neighbor_strategy == 'uniform':  # if we are applying uniform sampling, shuffles the data above before sampling
    #                 # sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

    #                 if self.seed is None:
    #                     sampled_idx = np.random.choice(a=len(source_neighbors), size=n_neighbors,
    #                                                    p=node_neighbor_sampled_probabilities)
    #                 else:
    #                     sampled_idx = self.random_state.choice(a=len(source_neighbors), size=n_neighbors,
    #                                                            p=node_neighbor_sampled_probabilities)

    #                 neighbors[i, :] = source_neighbors[sampled_idx]
    #                 edge_times[i, :] = source_edge_times[sampled_idx]
    #                 edge_idxs[i, :] = source_edge_idxs[sampled_idx]

    #                 # re-sort based on time
    #                 pos = edge_times[i, :].argsort()
    #                 neighbors[i, :] = neighbors[i, :][pos]
    #                 edge_times[i, :] = edge_times[i, :][pos]
    #                 edge_idxs[i, :] = edge_idxs[i, :][pos]
    #             else:
    #                 # Take most recent interactions
    #                 source_edge_times = source_edge_times[-n_neighbors:]
    #                 source_neighbors = source_neighbors[-n_neighbors:]
    #                 source_edge_idxs = source_edge_idxs[-n_neighbors:]

    #                 assert (len(source_neighbors) <= n_neighbors)
    #                 assert (len(source_edge_times) <= n_neighbors)
    #                 assert (len(source_edge_idxs) <= n_neighbors)

    #                 neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
    #                 edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
    #                 edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    #     return neighbors, edge_idxs, edge_times

    def find_all_first_hop(self, source_nodes, timestamps):
        self.all_source_neighbors = []
        self.all_source_edge_idx = []
        self.all_source_edge_times = []
        self.all_node_neighbor_sampled_probabilities = []
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times, node_neighbor_sampled_probabilities = self.find_before(
                source_node, timestamp,
                return_sampled_probabilities=True)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            self.all_source_neighbors.append(source_neighbors)
            self.all_source_edge_idx.append(source_edge_idxs)
            self.all_source_edge_times.append(source_edge_times)
            self.all_node_neighbor_sampled_probabilities.append(node_neighbor_sampled_probabilities)

    def get_walk_temporal_neighbors(self, source_nodes, timestamps, n_neighbors=20, source=False):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):

            if source:
                source_neighbors, source_edge_idxs, source_edge_times, node_neighbor_sampled_probabilities = \
                    self.all_source_neighbors[i], self.all_source_edge_idx[i], self.all_source_edge_times[i], \
                        self.all_node_neighbor_sampled_probabilities[i]
            else:
                source_neighbors, source_edge_idxs, source_edge_times, node_neighbor_sampled_probabilities = self.find_before(
                    source_node, timestamp,
                    return_sampled_probabilities=True)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if len(source_neighbors) > 0 and n_neighbors > 0:

                # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1

                node_neighbor_sampled_probabilities = torch.softmax(
                    torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()

                if self.seed is None:
                    sampled_idx = np.random.choice(a=len(source_neighbors), size=n_neighbors,
                                                   p=node_neighbor_sampled_probabilities)
                else:
                    sampled_idx = self.random_state.choice(a=len(source_neighbors), size=n_neighbors,
                                                           p=node_neighbor_sampled_probabilities)
                # sampled_idx = seq_binary_sample(node_neighbor_sampled_probabilities, n_neighbors)
                neighbors[i, :] = source_neighbors[sampled_idx]
                edge_times[i, :] = source_edge_times[sampled_idx]
                edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                # re-sort based on time
                pos = edge_times[i, :].argsort()
                neighbors[i, :] = neighbors[i, :][pos]
                edge_times[i, :] = edge_times[i, :][pos]
                edge_idxs[i, :] = edge_idxs[i, :][pos]

        return neighbors, edge_idxs, edge_times

    def get_multi_hop_neighbors(self, num_hops: int, source_nodes: np.ndarray, timestamps: np.ndarray, walk_restart,
                                n_neighbors: int = 20, pick_new_neighbors=False):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        neighbors, edge_idxs, edge_times = self.get_walk_temporal_neighbors(source_nodes=source_nodes,
                                                                            timestamps=timestamps,
                                                                            n_neighbors=n_neighbors, source=True)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [neighbors]
        nodes_edge_ids_list = [edge_idxs]
        nodes_neighbor_times_list = [edge_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * 1, 1)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_walk_temporal_neighbors(
                source_nodes=nodes_neighbor_ids_list[-1].flatten(),
                timestamps=nodes_neighbor_times_list[-1].flatten(),
                n_neighbors=1)

            # three ndarrays with shape (batch_size, 1)
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(source_nodes), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(source_nodes), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(source_nodes), -1)
            if walk_restart is not None:
                if (num_hops == 2) or (hop == 2):
                    p = walk_restart

                    if pick_new_neighbors:
                        neighbors, edge_idxs, edge_times = self.get_walk_temporal_neighbors(source_nodes=source_nodes,
                                                                                            timestamps=timestamps,
                                                                                            n_neighbors=n_neighbors,
                                                                                            source=True)
                    if torch.is_tensor(p):
                        mask = (torch.rand(*nodes_neighbor_ids.shape) < p.repeat(1, n_neighbors).cpu()).numpy()
                    else:
                        mask = np.random.rand(*nodes_neighbor_ids.shape) < p
                    nodes_neighbor_ids[mask] = neighbors[mask]
                    nodes_edge_ids[mask] = edge_idxs[mask]
                    nodes_neighbor_times[mask] = edge_times[mask]

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):

        return self.all_source_neighbors, self.all_source_edge_idx, self.all_source_edge_times
    
    def clear_cache(self):
        self.all_source_neighbors = []
        self.all_source_edge_idx = []
        self.all_source_edge_times = []
        self.all_node_neighbor_sampled_probabilities = []
        if self.use_cache:
            self.cache.clear()
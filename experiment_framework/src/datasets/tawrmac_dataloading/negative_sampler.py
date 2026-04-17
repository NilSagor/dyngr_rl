import numpy as np
import torch 

# TAWRMAC NegativeEdgeSample implementation  from
# https://anonymous.4open.science/r/tawrmac-A253/utils/utils.py

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None,
                 last_observed_time: float = None,
                 negative_sample_strategy: str = 'random', seed: int = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
            self.possible_edges = set(
                (src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in
                self.unique_dst_node_ids)

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time,
                                                                               self.last_observed_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in
                   zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

    def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size,
                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                  current_batch_start_time=current_batch_start_time,
                                                                                  current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size,
                                                                                 batch_src_node_ids=batch_src_node_ids,
                                                                                 batch_dst_node_ids=batch_dst_node_ids,
                                                                                 current_batch_start_time=current_batch_start_time,
                                                                                 current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[
            random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray,
                                           batch_dst_node_ids: np.ndarray):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in
                          zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size,
                                                       replace=len(possible_random_edges) < size)
        return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
            np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time,
                                                                        end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time,
                                                                           end_time=current_batch_end_time)
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size,
                                                                           replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time,
                                                                        end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time,
                                                                           end_time=current_batch_end_time)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size,
                                                                          replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)
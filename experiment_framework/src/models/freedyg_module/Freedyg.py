import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from src.models.freedyg_module.TimeEncoder import TimeEncoder
from utils.utils import NeighborSampler
from src.models.freedyg_module.FeedForwardNet import FeedForwardNet
from src.models.freedyg_module.FilterLayer import FilterLayer
from src.models.freedyg_module.MLPMixer import MLPMixer
from src.models.freedyg_module.NIFEncoder import NIFEncoder






class FreeDyG(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, num_layers: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 128, device: str = 'cpu'):
        """
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(FreeDyG, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim,parameter_requires_grad=False)
        
        self.nif_feat_dim = self.channel_embedding_dim

        self.nif_encoder = NIFEncoder(nif_feat_dim=self.nif_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'nif': nn.Linear(in_features=self.nif_feat_dim, out_features=self.edge_feat_dim, bias=True)
        })
        self.reduce_layer = nn.Linear(4*self.edge_feat_dim, self.edge_feat_dim)
       
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=max_input_sequence_length, num_channels=self.edge_feat_dim,
                     token_dim_expansion_factor=0.5,
                     channel_dim_expansion_factor=4, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        # self.transformers = nn.ModuleList([
        #     TransformerEncoder(seq_len=max_input_sequence_length,attention_dim=self.edge_feat_dim, num_heads=self.num_heads, dropout=self.dropout)
        #     for _ in range(self.num_layers)
        # ])
       
        self.weightagg = nn.Linear(self.edge_feat_dim,1)
       

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        src_nodes_neighbor_ids, src_nodes_edge_ids, src_nodes_neighbor_times = \
           self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

       
        dst_nodes_neighbor_ids, dst_nodes_edge_ids, dst_nodes_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

        # src_nodes_nif_features, Tensor, shape (batch_size, src_max_seq_length, nif_feat_dim)
        # dst_nodes_nif_features, Tensor, shape (batch_size, dst_max_seq_length, nif_feat_dim)
        src_nodes_nif_features, dst_nodes_nif_features = \
            self.nif_encoder(src_node_ids=src_node_ids,dst_node_ids=dst_node_ids,src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                                                dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

        # get the features of the sequence of source and destination nodes
        # src_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_nodes_neighbor_ids,
                              nodes_edge_ids=src_nodes_edge_ids, nodes_neighbor_times=src_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=dst_nodes_neighbor_ids,
                              nodes_edge_ids=dst_nodes_edge_ids, nodes_neighbor_times=dst_nodes_neighbor_times, time_encoder=self.time_encoder)
        
        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)
        src_nodes_nif_features = self.projection_layer['nif'](src_nodes_nif_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_nodes_neighbor_node_raw_features)
        dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        dst_nodes_neighbor_time_features = self.projection_layer['time'](dst_nodes_neighbor_time_features)
        dst_nodes_nif_features = self.projection_layer['nif'](dst_nodes_nif_features)

        src_combined_features = torch.cat([src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features,src_nodes_nif_features], dim=-1)
        dst_combined_features = torch.cat([dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features,dst_nodes_nif_features], dim=-1)
      
        src_combined_features = self.reduce_layer(src_combined_features)
        dst_combined_features = self.reduce_layer(dst_combined_features)
       
        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            src_combined_features = mlp_mixer(input_tensor=src_combined_features)
        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            dst_combined_features = mlp_mixer(input_tensor=dst_combined_features)
        
        # for transformer in self.transformers:
        #     src_combined_features = transformer(src_combined_features)
        # for transformer in self.transformers:
        #     dst_combined_features = transformer(dst_combined_features)
    

        src_weight = self.weightagg(src_combined_features).transpose(1, 2)
        dst_weight = self.weightagg(dst_combined_features).transpose(1, 2)
       
        src_combined_features = src_weight.matmul(src_combined_features).squeeze(dim=1)
        dst_combined_features = dst_weight.matmul(dst_combined_features).squeeze(dim=1)
      
       
        return src_combined_features, dst_combined_features


    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        

        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[torch.from_numpy(nodes_neighbor_ids == 0)] = 0.0
    

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features


    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()




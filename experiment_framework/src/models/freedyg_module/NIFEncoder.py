import torch 
import torch.nn as nn
import numpy as np


class NIFEncoder(nn.Module):

    def __init__(self, nif_feat_dim: int, device: str = 'cpu'):
       
        super(NIFEncoder, self).__init__()

        self.nif_feat_dim = nif_feat_dim
        self.device = device

        self.nif_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.nif_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.nif_feat_dim, out_features=self.nif_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
       
        # two lists to store the appearances of source and destination nodes
        src_nodes_appearances, dst_nodes_appearances = [], []
        # src_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for i in range(len(src_node_ids)):
            src_node_id = src_node_ids[i]
            dst_node_id = dst_node_ids[i]
            src_node_neighbor_ids = src_nodes_neighbor_ids[i]
            dst_node_neighbor_ids = dst_nodes_neighbor_ids[i]

            # Calculate unique keys and counts for source and destination
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_node_neighbor_ids, return_inverse=True, return_counts=True)

            # Create mappings from node IDs to their counts
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # Adjust counts specifically for the cases where src_node_id appears in dst's neighbors and vice versa
            if src_node_id in dst_mapping_dict:
                src_count_in_dst = dst_mapping_dict[src_node_id]
                src_mapping_dict[src_node_id] = src_count_in_dst
                dst_mapping_dict[src_node_id] = src_count_in_dst
            if dst_node_id in src_mapping_dict:
                dst_count_in_src = src_mapping_dict[dst_node_id]
                src_mapping_dict[dst_node_id] = dst_count_in_src
                dst_mapping_dict[dst_node_id] = dst_count_in_src

        # Calculate appearances in each other's lists
            src_node_neighbor_counts_in_dst = torch.tensor([dst_mapping_dict.get(neighbor_id, 0) for neighbor_id in src_node_neighbor_ids]).float().to(self.device)
            dst_node_neighbor_counts_in_src = torch.tensor([src_mapping_dict.get(neighbor_id, 0) for neighbor_id in dst_node_neighbor_ids]).float().to(self.device)

            # Stack counts to get a two-column tensor for each node list
            src_nodes_appearances.append(torch.stack([torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device), src_node_neighbor_counts_in_dst], dim=1))
            dst_nodes_appearances.append(torch.stack([dst_node_neighbor_counts_in_src, torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)], dim=1))

    # Stack to form batch tensors
        src_nodes_appearances = torch.stack(src_nodes_appearances, dim=0)
        dst_nodes_appearances = torch.stack(dst_nodes_appearances, dim=0)

        return src_nodes_appearances, dst_nodes_appearances


    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_nodes_neighbor_ids and dst_nodes_neighbor_ids
        :param src_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_nodes_appearances, dst_nodes_appearances = self.count_nodes_appearances(src_node_ids=src_node_ids,dst_node_ids=dst_node_ids,src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                                                                                                  dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

       
        # Tensor, shape (batch_size, src_max_seq_length, nif_feat_dim)
        
        # Tensor, shape (batch_size, dst_max_seq_length, nif_feat_dim)

        src_nodes_nif_features =  (src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features =  (dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)


        src_nodes_nif_features = self.nif_encode_layer(src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = self.nif_encode_layer(dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        
        # src_nodes_nif_features, Tensor, shape (batch_size, src_max_seq_length, nif_feat_dim)
        # dst_nodes_nif_features, Tensor, shape (batch_size, dst_max_seq_length, nif_feat_dim)
        return src_nodes_nif_features, dst_nodes_nif_features
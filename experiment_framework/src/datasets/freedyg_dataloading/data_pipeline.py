import torch 
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from src.models.freedyg_module.NeighborSampler import get_neighbor_sampler, NegativeEdgeSampler
from src.datasets.freedyg_dataloading.data_utils import get_link_prediction_data




class FreeDyGDataPipeline:
    def __init__(self, config):
        self.config = config
        self._load_data()

    def _load_data(self):
        # Load raw data splits
        (self.node_raw_features, self.edge_raw_features,
         self.full_data, self.train_data, self.val_data, self.test_data,
         self.new_node_val_data, self.new_node_test_data) = get_link_prediction_data(
            dataset_name=self.config.dataset_name,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio
        )
        # Convert to tensors
        self.node_raw_features = torch.from_numpy(self.node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(self.edge_raw_features.astype(np.float32))

        # Build neighbor samplers
        self.train_neighbor_sampler = get_neighbor_sampler(
            data=self.train_data,
            sample_neighbor_strategy=self.config.sample_neighbor_strategy,
            time_scaling_factor=self.config.time_scaling_factor,
            seed=self.config.seed
        )
        self.full_neighbor_sampler = get_neighbor_sampler(
            data=self.full_data,
            sample_neighbor_strategy=self.config.sample_neighbor_strategy,
            time_scaling_factor=self.config.time_scaling_factor,
            seed=self.config.seed + 1
        )

        # Negative edge samplers
        self.train_neg_sampler = NegativeEdgeSampler(
            src_node_ids=self.train_data.src_node_ids,
            dst_node_ids=self.train_data.dst_node_ids
        )
        self.val_neg_sampler = NegativeEdgeSampler(
            src_node_ids=self.full_data.src_node_ids,
            dst_node_ids=self.full_data.dst_node_ids,
            seed=0
        )
        self.test_neg_sampler = NegativeEdgeSampler(
            src_node_ids=self.full_data.src_node_ids,
            dst_node_ids=self.full_data.dst_node_ids,
            seed=2
        )

        # Create DataLoaders
        self.loaders = {
            'train': self._make_loader(self.train_data, shuffle=True),
            'val': self._make_loader(self.val_data, shuffle=False),
            'test': self._make_loader(self.test_data, shuffle=False)
        }

    def _make_loader(self, data, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(data.src_node_ids),
            torch.from_numpy(data.dst_node_ids),
            torch.from_numpy(data.node_interact_times),
            torch.from_numpy(data.edge_ids)
        )
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, num_workers=4)

    def get_features(self):
        return {
            'node_raw_features': self.node_raw_features,
            'edge_raw_features': self.edge_raw_features,
        }

    @property
    def num_nodes(self):
        return self.node_raw_features.shape[0]

    @property
    def neighbor_finder(self):
        return self.train_neighbor_sampler  # used by model setup
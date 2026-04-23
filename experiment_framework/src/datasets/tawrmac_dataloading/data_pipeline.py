#
import torch
import numpy as np
from typing import Dict, Optional, List
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.datasets.sam_dataloading.data_loader import load_dataset 
from src.datasets.tawrmac_dataloading.neighbor_finder import NewNeighborFinder
from src.datasets.tawrmac_dataloading.negative_sampler import NegativeEdgeSampler


class TAWRMACTemporalDataset(Dataset):
    """Dataset that yields batches in the format expected by TAWRMACv1."""
    
    def __init__(
        self,
        edges: np.ndarray,
        timestamps: np.ndarray,
        edge_idxs: np.ndarray,
        negative_sampler: NegativeEdgeSampler,
        split: str,
        batch_size: int,
    ):
        self.edges = edges  # shape (2, N)
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.negative_sampler = negative_sampler
        self.split = split
        self.batch_size = batch_size
        
        # Precompute batches for consistent iteration
        self.num_samples = edges.shape[1]
        self.indices = np.arange(self.num_samples)
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.num_samples)
        batch_indices = self.indices[start:end]
        
        sources = self.edges[0, batch_indices]
        destinations = self.edges[1, batch_indices]
        timestamps = self.timestamps[batch_indices]
        edge_idxs = self.edge_idxs[batch_indices]
        
        # Sample negatives using TAWRMAC's NegativeEdgeSampler
        neg_sources, neg_destinations = self.negative_sampler.sample(
            size=len(sources),
            batch_src_node_ids=sources,
            batch_dst_node_ids=destinations,
            current_batch_start_time=timestamps.min(),
            current_batch_end_time=timestamps.max(),
        )
        
        return {
            'sources': sources,
            'destinations': destinations,
            'timestamps': timestamps,
            'edge_idxs': edge_idxs,
            'negative_sources': neg_sources,
            'negative_destinations': neg_destinations,
        }
    
    @staticmethod
    def collate_fn(batch):
        # batch is a list of dicts (one per batch in the DataLoader)
        # Since __getitem__ already returns a full batch, we simply return the first element
        return batch[0]


class TAWRMACDataPipeline:
    """Data pipeline for TAWRMAC models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data: Optional[Dict] = None
        self.neighbor_finder: Optional[NewNeighborFinder] = None
        self.samplers: Dict[str, NegativeEdgeSampler] = {}
        self.datasets: Dict[str, TAWRMACTemporalDataset] = {}
        self.loaders: Dict[str, DataLoader] = {}
    
    def load(self) -> 'TAWRMACDataPipeline':
        """Load raw dataset using existing loader."""
        logger.info(f"Loading dataset: {self.config['data']['dataset']}")
        
        eval_type = self.config['data']['evaluation_type']
        sampling_strategy = self.config['data']['negative_sampling_strategy']
        
        self.data = load_dataset(
            dataset_name=self.config['data']['dataset'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            inductive=(eval_type == 'inductive'),
            unseen_ratio=self.config['data'].get('unseen_ratio', 0.1),
            seed=self.config['experiment']['seed'],
        )
        
        logger.info(f"Loaded: {self.data['num_nodes']} nodes, {self.data['statistics']['num_edges']} edges")
        return self
    
    def build_neighbor_finder(self) -> 'TAWRMACDataPipeline':
        """Build TAWRMAC NewNeighborFinder from training edges only."""
        train_mask = self.data['train_mask']
        train_edges = self.data['edges'][train_mask]  # shape: (num_train, 2)
        train_src = train_edges[:, 0].cpu().numpy()
        train_dst = train_edges[:, 1].cpu().numpy()
        train_ts = self.data['timestamps'][train_mask].cpu().numpy()
        train_eid = np.arange(len(train_src))

        # Use total number of nodes, not just max in training
        max_node_idx = self.data['num_nodes'] - 1
        adj_list = [[] for _ in range(max_node_idx + 1)]

        for src, dst, eid, ts in zip(train_src, train_dst, train_eid, train_ts):
            adj_list[src].append((dst, eid, ts))
            adj_list[dst].append((src, eid, ts))

        self.neighbor_finder = NewNeighborFinder(
            adj_list,
            sample_neighbor_strategy='uniform' if self.config['data'].get('uniform', False) else 'recent',
            seed=self.config['experiment']['seed']
        )
        
        self.train_edge_index = torch.tensor([train_src, train_dst], dtype=torch.long)
        self.train_edge_time = torch.tensor(train_ts, dtype=torch.float)
        
        self.neighbor_finder.edge_index = self.train_edge_index
        self.neighbor_finder.edge_time = self.train_edge_time
        
        logger.info(f"Built TAWRMAC NewNeighborFinder from {len(train_src)} training edges")
        return self
    
    # def build_neighbor_finder(self) -> 'TAWRMACDataPipeline':
    #     """Build TAWRMAC NewNeighborFinder from training edges only."""
    #     train_mask = self.data['train_mask']
    #     train_src = self.data['edges'][0, train_mask]
    #     train_dst = self.data['edges'][1, train_mask]
    #     train_ts = self.data['timestamps'][train_mask]
    #     train_eid = np.arange(len(train_src))  # Edge indices within training set
        
    #     # Create adjacency list for NewNeighborFinder
    #     max_node_idx = max(train_src.max(), train_dst.max())
    #     adj_list = [[] for _ in range(max_node_idx + 1)]
    #     for src, dst, eid, ts in zip(train_src, train_dst, train_eid, train_ts):
    #         adj_list[src].append((dst, eid, ts))
    #         adj_list[dst].append((src, eid, ts))
        
    #     self.neighbor_finder = NewNeighborFinder(
    #         adj_list,
    #         sample_neighbor_strategy='uniform' if self.config['data'].get('uniform', False) else 'recent',
    #         seed=self.config['experiment']['seed']
    #     )
        
    #     logger.info(f"Built TAWRMAC NewNeighborFinder from {len(train_src)} training edges")
    #     return self
    
    def build_samplers(self) -> 'TAWRMACDataPipeline':
        splits = ['train', 'val', 'test']
        masks = ['train_mask', 'val_mask', 'test_mask']
        strategy = self.config['data']['negative_sampling_strategy']

        for split, mask_key in zip(splits, masks):
            mask = self.data[mask_key]
            split_edges = self.data['edges'][mask]  # shape: (num_split, 2)
            src = split_edges[:, 0].cpu().numpy()
            dst = split_edges[:, 1].cpu().numpy()
            ts = self.data['timestamps'][mask].cpu().numpy()

            last_observed_time = None
            if split == 'val':
                last_observed_time = self.data['timestamps'][self.data['train_mask']].max().item()
            elif split == 'test':
                last_observed_time = self.data['timestamps'][self.data['val_mask']].max().item()

            self.samplers[split] = NegativeEdgeSampler(
                src_node_ids=src,
                dst_node_ids=dst,
                interact_times=ts,
                last_observed_time=last_observed_time,
                negative_sample_strategy=strategy if split != 'train' else 'random',
                seed=self.config['experiment']['seed']
            )
        logger.info(f"Built TAWRMAC samplers with strategy '{strategy}'")
        return self
    
    
    # def build_samplers(self) -> 'TAWRMACDataPipeline':
    #     """Build TAWRMAC NegativeEdgeSamplers for each split."""
    #     splits = ['train', 'val', 'test']
    #     masks = ['train_mask', 'val_mask', 'test_mask']
    #     strategy = self.config['data']['negative_sampling_strategy']
        
    #     for split, mask_key in zip(splits, masks):
    #         mask = self.data[mask_key]
    #         src = self.data['edges'][0, mask]
    #         dst = self.data['edges'][1, mask]
    #         ts = self.data['timestamps'][mask]
            
    #         last_observed_time = None
    #         if split == 'val':
    #             last_observed_time = self.data['timestamps'][self.data['train_mask']].max()
    #         elif split == 'test':
    #             last_observed_time = self.data['timestamps'][self.data['val_mask']].max()
            
    #         self.samplers[split] = NegativeEdgeSampler(
    #             src_node_ids=src,
    #             dst_node_ids=dst,
    #             interact_times=ts,
    #             last_observed_time=last_observed_time,
    #             negative_sample_strategy=strategy if split != 'train' else 'random',
    #             seed=self.config['experiment']['seed']
    #         )
        
    #     logger.info(f"Built TAWRMAC samplers with strategy '{strategy}'")
    #     return self
    
    def build_datasets(self) -> 'TAWRMACDataPipeline':
        splits = ['train', 'val', 'test']
        masks = ['train_mask', 'val_mask', 'test_mask']
        batch_size = self.config['training']['batch_size']

        for split, mask_key in zip(splits, masks):
            mask = self.data[mask_key]
            split_edges = self.data['edges'][mask]  # (num_split, 2)
            # Transpose to (2, num_split) for dataset compatibility
            edges = split_edges.T.cpu().numpy()  # shape (2, num_split)
            timestamps = self.data['timestamps'][mask].cpu().numpy()
            edge_idxs = np.arange(edges.shape[1])

            self.datasets[split] = TAWRMACTemporalDataset(
                edges=edges,
                timestamps=timestamps,
                edge_idxs=edge_idxs,
                negative_sampler=self.samplers[split],
                split=split,
                batch_size=batch_size,
            )
        logger.info(f"Built datasets: { {k: len(v) for k, v in self.datasets.items()} }")
        return self
    
    
    # def build_datasets(self) -> 'TAWRMACDataPipeline':
    #     """Build TAWRMAC datasets for each split."""
    #     splits = ['train', 'val', 'test']
    #     masks = ['train_mask', 'val_mask', 'test_mask']
    #     batch_size = self.config['training']['batch_size']
        
    #     for split, mask_key in zip(splits, masks):
    #         mask = self.data[mask_key]
    #         edges = self.data['edges'][:, mask]
    #         timestamps = self.data['timestamps'][mask]
    #         # Create sequential edge indices for the split
    #         edge_idxs = np.arange(edges.shape[1])
            
    #         self.datasets[split] = TAWRMACTemporalDataset(
    #             edges=edges,
    #             timestamps=timestamps,
    #             edge_idxs=edge_idxs,
    #             negative_sampler=self.samplers[split],
    #             split=split,
    #             batch_size=batch_size,
    #         )
        
    #     logger.info(f"Built datasets: { {k: len(v) for k, v in self.datasets.items()} }")
    #     return self
    
    def build_loaders(self) -> 'TAWRMACDataPipeline':
        """Wrap datasets in DataLoaders."""
        num_workers = self.config['hardware'].get('num_workers', 0)
        
        for split, dataset in self.datasets.items():
            self.loaders[split] = DataLoader(
                dataset,
                batch_size=1,  # Each dataset __getitem__ already returns a full batch
                shuffle=False,
                num_workers=num_workers,
                collate_fn=TAWRMACTemporalDataset.collate_fn,
                pin_memory=self.config['hardware'].get('pin_memory', False),
            )
        
        logger.info("DataLoaders built for TAWRMAC")
        return self
    
    def get_features(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return node/edge features as expected by ModelFactory."""
        # TAWRMAC expects raw numpy arrays for node/edge features
        node_feats = self.data.get('node_features')
        edge_feats = self.data.get('edge_features')
        
        # Edge features should be full matrix; TAWRMAC will index by edge_idxs
        return {
            'node_features': node_feats,
            'edge_features': edge_feats,
            'num_nodes': self.data['num_nodes'],
            'edge_feat_dim': edge_feats.shape[1] if edge_feats is not None else 0,
            'node_feat_dim': node_feats.shape[1] if node_feats is not None else 0,
        }
    
    @property
    def num_nodes(self) -> int:
        return self.data['num_nodes']
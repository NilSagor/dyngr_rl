# ============================================================================
# DATA PIPELINE - MODIFIED FOR SHUFFLED TRAINING
# ============================================================================

import torch 
from typing import Dict, List, Optional, Any
from loguru import logger
from torch.utils.data import DataLoader

from src.datasets.sam_dataloading.data_loader import load_dataset
from src.datasets.sam_dataloading.negative_sample import NegativeSampler
# from src.datasets.sam_dataloading.neighbor_finder import NeighborFinder
from src.datasets.continue_temporal.neighbor_finder import NeighborFinder
from src.datasets.sam_dataloading.temporal_data import TemporalDataset


class DataPipeline:
    """Encapsulates all data-related setup with shuffled training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data: Optional[Dict] = None
        self.neighbor_finder: Optional[NeighborFinder] = None
        self.samplers: Dict[str, NegativeSampler] = {}
        self.datasets: Dict[str, TemporalDataset] = {}
        self.loaders: Dict[str, DataLoader] = {}
    
    def load(self) -> 'DataPipeline':
        # Same as before – unchanged
        logger.info(f"Loading dataset: {self.config['data']['dataset']}")
        eval_type = self.config['data']['evaluation_type']
        sampling_strategy = self.config['data']['negative_sampling_strategy']
        if sampling_strategy == 'inductive':
            if eval_type != 'inductive':
                raise ValueError(f"Inductive sampling requires inductive evaluation. Got evaluation_type='{eval_type}'.")
            if self.config['data'].get('unseen_ratio', 0.1) <= 0:
                raise ValueError("Inductive sampling requires unseen_ratio > 0.")
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
    
   
    
    def build_neighbor_finder(self) -> 'DataPipeline':
        """Build neighbor finder from training edges only (leakage-proof)."""
        train_edges = self.data['edges'][self.data['train_mask']]
        train_ts = self.data['timestamps'][self.data['train_mask']]        
        
        self.neighbor_finder = NeighborFinder(
            train_edges=train_edges,
            train_timestamps=train_ts,
            max_neighbors=self.config['data']['max_neighbors']
        )
        logger.info(f"Built leakage-proof NeighborFinder from {len(train_edges)} training edges")
        return self
    
    def build_samplers(self) -> 'DataPipeline':
        # Same as before – unchanged
        splits = ['train', 'val', 'test']
        masks = ['train_mask', 'val_mask', 'test_mask']
        for split, mask_key in zip(splits, masks):
            edges = self.data['edges'][self.data[mask_key]]
            timestamps = self.data['timestamps'][self.data[mask_key]]
            self.samplers[split] = NegativeSampler(
                edges=edges,
                timestamps=timestamps,
                num_nodes=self.data['num_nodes'],
                neighbor_finder=self.neighbor_finder,
                seed=self.config['experiment']['seed']
            )
        logger.info(f"Built samplers: train=random (TGN standard), val/test={self.config['data']['negative_sampling_strategy']}")
        return self
    
    def build_datasets(self) -> 'DataPipeline':
        # Same as before – unchanged
        splits = ['train', 'val', 'test']
        masks = ['train_mask', 'val_mask', 'test_mask']
        is_inductive = self.config['data']['evaluation_type'] == 'inductive'
        for split, mask_key in zip(splits, masks):
            mask = self.data[mask_key]
            split_edge_features = self.data['edge_features'][mask] if self.data['edge_features'] is not None else None
            unseen_nodes = self.data['unseen_nodes'] if (is_inductive and split != 'train') else None
            sampling_strategy = 'random' if split == 'train' else self.config['data']['negative_sampling_strategy']
            self.datasets[split] = TemporalDataset(
                edges=self.data['edges'][mask],
                timestamps=self.data['timestamps'][mask],
                edge_features=split_edge_features,
                num_nodes=self.data['num_nodes'],
                split=split,
                negative_sampler=self.samplers[split],
                negative_sampling_strategy=sampling_strategy,
                unseen_nodes=unseen_nodes,
                seed=self.config['experiment']['seed']
            )
        if self.config['data'].get('validate_batches', True):
            self._validate_evaluation_batches()
        else:
            logger.info("Skipping evaluation batch validation")
        logger.info(f"Built datasets: { {k: len(v) for k, v in self.datasets.items()} }")
        return self
    
    def _validate_evaluation_batches(self):
        # Same as before – unchanged
        for split in ['val', 'test']:
            if split not in self.datasets:
                continue
            dataset = self.datasets[split]
            batch_size = self.config['training']['batch_size']
            for i in range(min(10, len(dataset) // batch_size)):
                start = i * batch_size
                end = min(start + batch_size, len(dataset))
                batch_labels = [dataset.samples[j]['label'] for j in range(start, end)]
                if 0.0 not in batch_labels or 1.0 not in batch_labels:
                    raise ValueError(f"Single-class batch in {split} split (batch {i})! Labels: {set(batch_labels)}.")
        logger.info("All evaluation batches contain both classes.")
    
    def build_loaders(self) -> 'DataPipeline':
        """Build DataLoaders with SHUFFLED training (critical for beating TAWRMAC)."""
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['hardware'].get('num_workers', 0)
        pin_memory = self.config['hardware'].get('pin_memory', False)
        
        for split, dataset in self.datasets.items():
            # SHUFFLE ONLY TRAINING SPLIT
            shuffle = (split == 'train')
            self.loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=TemporalDataset.collate_fn,
                pin_memory=pin_memory,
                drop_last=(split == 'train')  # optional, helps with batch norm
            )
            logger.info(f"{split} loader: shuffle={shuffle}, batch_size={batch_size}")
        return self
    
    def get_features(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get node/edge features with structural dataset detection (unchanged)."""
        dataset = self.config['data']['dataset'].lower()
        IS_STRUCTURAL = dataset in {'untrade', 'uslegis', 'canparl', 'unvote'}
        if IS_STRUCTURAL:
            node_features = None
            num_edges = self.data['train_mask'].sum().item()
            edge_features = torch.ones(num_edges, 1)
            logger.info(f"Structural dataset {dataset}: using 1-dim dummy edge features")
            return {
                'node_features': node_features,
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': 1,
                'node_feat_dim': 0,
            }
        if dataset == "enron":
            train_mask = self.data['train_mask']
            edge_features = self.data['edge_features'][train_mask]
            return {
                'node_features': None,
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': edge_features.shape[1],
                'node_feat_dim': 0,
            }
        if dataset == "uci":
            train_mask = self.data['train_mask']
            edge_features = self.data['edge_features'][train_mask]
            if edge_features.shape[1] != 2:
                edge_features = edge_features[:, :2]
            return {
                'node_features': self.data.get('node_features'),
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': 2,
                'node_feat_dim': 100 if self.data.get('node_features') is not None else 0,
            }
        node_features = self.data.get("node_features")
        train_edge_features = self.data['edge_features'][self.data['train_mask']] if self.data['edge_features'] is not None else None
        return {
            'node_features': node_features,
            'edge_features': train_edge_features,
            'num_nodes': self.data['num_nodes'],
            'edge_feat_dim': train_edge_features.shape[1] if train_edge_features is not None else 0,
            'node_feat_dim': node_features.shape[1] if node_features is not None else 0,
        }
    
    @property
    def num_nodes(self) -> int:
        return self.data['num_nodes']
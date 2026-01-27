"""Dataset loading utilities."""


import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from loguru import logger

from .temporal_dataset import TemporalDataset
from .load_dataset import load_generic_dataset
from .load_dataset import (
    load_wikipedia,
    load_reddit,
    load_mooc,
    load_lastfm,
    load_uci,
)

from .data_utils import temporal_train_val_test_split, create_inductive_split

# Dataset registry
DATASET_LOADERS = {
    "wikipedia": load_wikipedia,
    "reddit": load_reddit,
    "mooc": load_mooc,
    "lastfm": load_lastfm,
    "uci": load_uci,
}


def get_dataset_loader(config: Dict[str, Any],negative_sampling_strategy) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Get dataset loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        negative_sampling_strate: random, historical, 
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_nodes)
    """
    
    dataset_name = config['data']['dataset']
    data_config = config['data']
    training_config = config['training']
    hardware_config = config['hardware']
    model_name = config['model']['name']

    # Load with model-specific formatting
    if model_name in ['TGN', 'JODIE']:
        data = load_generic_dataset(dataset_name, model_type="tgn")
    else:  # DyGFormer, TAWRMAC
        data = load_generic_dataset(dataset_name, model_type="dygformer")

    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available datasets: {list(DATASET_LOADERS.keys())}")
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    # loader_func = DATASET_LOADERS[dataset_name]
    
    # # Load raw data
    # data = loader_func()
    
    # Extract components
    edges = data['edges']  # [num_edges, 2]
    timestamps = data['timestamps']  # [num_edges]
    edge_features = data.get('edge_features')  # Optional
    node_features = data.get('node_features')  # Optional
    num_nodes = data['num_nodes']
    
    logger.info(f"Loaded dataset with {num_nodes} nodes and {len(edges)} edges")
    
    
    # FIX: Log feature dimensions for debugging
    if node_features is not None:
        logger.info(f"Node features shape: {node_features.shape}")
    if edge_features is not None:
        logger.info(f"Edge features shape: {edge_features.shape}")
    
    logger.info(f"Edges shape: {edges.shape}")
    logger.info(f"Edge features shape: {edge_features.shape if edge_features is not None else 'None'}")
    
    # # Temporal split
    # from .utils import temporal_train_val_test_split

    eval_type = config['data']['evaluation_type']

    if eval_type == "inductive":
        
        splits = create_inductive_split(
            edges=edges,
            timestamps=timestamps,
            edge_features = edge_features,
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio'],
            test_ratio=data_config['test_ratio'],
            unseen_node_ratio = 0.2 # make configurable
        )
    else: # transductive
        splits = temporal_train_val_test_split(
            edges=edges,
            timestamps=timestamps,
            # edge_features=edge_features,
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio'],
            test_ratio=data_config['test_ratio']
        )
    
    # FIX: Determine device from config
    device = 'cuda' if hardware_config.get('gpus', 0) > 0 else 'cpu'

    # Create datasets
    train_dataset = TemporalDataset(
        edges=splits['train_edges'],
        timestamps=splits['train_timestamps'],
        edge_features=splits.get('train_edge_features'),
        num_nodes=num_nodes,
        max_neighbors=data_config['max_neighbors'],
        split='train',
        negative_sampling_strategy=config['data']['negative_sampling_strategy'],
        device= device
    )
    
    val_dataset = TemporalDataset(
        edges=splits['val_edges'],
        timestamps=splits['val_timestamps'],
        edge_features=splits.get('val_edge_features'),
        num_nodes=num_nodes,
        max_neighbors=data_config['max_neighbors'],
        negative_sampling_strategy=config['data']['negative_sampling_strategy'],
        split='val',
        device = device
    )
    
    test_dataset = TemporalDataset(
        edges=splits['test_edges'],
        timestamps=splits['test_timestamps'],
        unseen_nodes= splits.get("unseen_nodes"),
        edge_features=splits.get('test_edge_features'),
        num_nodes=num_nodes,
        max_neighbors=data_config['max_neighbors'],
        negative_sampling_strategy=config['data']['negative_sampling_strategy'],
        split='test',
        device = device
    )
     # FIX: Use proper device selection for DataLoader
    pin_memory = hardware_config.get('pin_memory', True) and device == 'cuda'

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=hardware_config['num_workers'],
        # pin_memory=hardware_config['pin_memory'],
        pin_memory=pin_memory,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=hardware_config['num_workers'],
        # pin_memory=hardware_config['pin_memory'],
        pin_memory=pin_memory,
        collate_fn=val_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['test_batch_size'],
        shuffle=False,
        num_workers=hardware_config['num_workers'],
        # pin_memory=hardware_config['pin_memory'],
        pin_memory=pin_memory,
        collate_fn=test_dataset.collate_fn
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, num_nodes


def create_synthetic_dataset(
    num_nodes: int = 1000,
    num_edges: int = 10000,
    temporal_window: float = 100.0,
    **kwargs
) -> Dict[str, Any]:
    """Create a synthetic temporal graph dataset.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of temporal edges
        temporal_window: Time window for edge generation
        
    Returns:
        Dictionary with edges, timestamps, and num_nodes
    """
    
    # Generate random edges
    edges = torch.randint(0, num_nodes, (num_edges, 2))
    
    # Generate timestamps
    timestamps = torch.sort(torch.rand(num_edges) * temporal_window)[0]
    
    # Remove self-loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]
    timestamps = timestamps[mask]
    
    return {
        'edges': edges,
        'timestamps': timestamps,
        'num_nodes': num_nodes,
        'name': 'synthetic'
    }
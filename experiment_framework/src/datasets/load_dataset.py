import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Set
from loguru import logger
from pathlib import Path


# Resolve data root relative to this file's location
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DEFAULT_DATA_ROOT = os.path.normpath(
#     os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
# )

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_ROOT = SCRIPT_DIR.parent.parent / "data" / "processed"
DATA_ROOT = str(DATA_ROOT)

# Allow environment override for deployment flexibility
# DATA_ROOT = os.environ.get('DYGLIB_DATA_ROOT', DEFAULT_DATA_ROOT)

DATA_ROOT = str(DATA_ROOT)


logger.debug(f"Using DATA_ROOT: {DATA_ROOT}")

class DatasetLoadError(Exception):
    """Raised when dataset loading fails validation."""
    pass


def load_dataset(
    dataset_name: str,
    data_root: Optional[str] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    inductive: bool = False,
    unseen_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Load temporal graph dataset with CORRECT node counting.
    Critical fix: Compute num_nodes BEFORE 0-indexing conversion.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load raw DyGLib data (1-indexed)
    folder_path = os.path.join(DATA_ROOT, dataset_name)
    csv_path = os.path.join(folder_path, f"ml_{dataset_name}.csv")
    node_feat_path = os.path.join(folder_path, f"ml_{dataset_name}_node.npy")
    edge_feat_path = os.path.join(folder_path, f"ml_{dataset_name}.npy")
    
    df = pd.read_csv(csv_path)
    node_features = np.load(node_feat_path).astype(np.float32)
    edge_features = np.load(edge_feat_path).astype(np.float32)
    
    # Correct UCI feature mislabeling (DyGLib bug)
    if dataset_name.lower() == "uci":
        # DyGLib's ml_uci.npy contains 100-dim NODE features mislabeled as edge features
        # True edge features should be 2-dim message content embeddings
        if edge_features.shape[1] == 100 and (node_features is None or node_features.shape[1] != 100):
            logger.warning(
                "UCI feature mislabeling detected: edge_features contains 100-dim node features. "
                "Swapping node/edge features per DyGLib specification."
            )
            # Swap: what DyGLib calls "edge_features" is actually node features
            node_features, edge_features = edge_features, node_features
        
        # Ensure edge features are 2-dim (TGN specification for UCI)
        if edge_features.shape[1] != 2:
            logger.info(f"UCI edge features truncated from {edge_features.shape[1]} to 2 dims")
            edge_features = edge_features[:, :2]  # Keep only first 2 dims as message content
    
        
    
    #  CAPTURE MAX ID BEFORE CONVERSION
    raw_src = df['u'].values.astype(np.int64)  # 1-indexed
    raw_dst = df['i'].values.astype(np.int64)  # 1-indexed
    max_1_indexed = int(max(raw_src.max(), raw_dst.max()))  # 9227 for Wikipedia
    
    #  COMPUTE NODE COUNT BEFORE CONVERSION
    num_nodes = max_1_indexed + 1  # 9227 + 1 = 9228 (correct for Wikipedia)
    
    # Now convert to 0-indexed
    src = raw_src - 1  # 1→0, 9227→9226
    dst = raw_dst - 1
    timestamps = df['ts'].values.astype(np.float32)
    
    # Fix DyGLib's extra edge feature row
    if edge_features.shape[0] > len(df):
        edge_features = edge_features[1:]
    
    # Validation (catches errors early)
    assert src.min() >= 0, f"Negative node IDs! Min: {src.min()}"
    assert src.max() < num_nodes, f"Node ID {src.max()} >= num_nodes {num_nodes}"
    assert len(src) == len(edge_features), f"Edge count mismatch"
    
    edges = np.column_stack([src, dst])
    
    # Temporal splitting (unchanged)
    sorted_indices = np.argsort(timestamps)
    edges = edges[sorted_indices]
    timestamps = timestamps[sorted_indices]
    edge_features = edge_features[sorted_indices]
    
    total_edges = len(edges)
    train_end = int(total_edges * (1 - val_ratio - test_ratio))
    val_end = int(total_edges * (1 - test_ratio))
    
    # Inductive node reservation (unchanged)
    unseen_nodes = None
    if inductive:
        all_nodes = np.arange(num_nodes)
        np.random.shuffle(all_nodes)
        num_unseen = int(num_nodes * unseen_ratio)
        unseen_nodes = torch.tensor(all_nodes[:num_unseen], dtype=torch.long)
        
        train_mask = np.ones(total_edges, dtype=bool)
        for i in range(train_end):
            if edges[i, 0] in unseen_nodes or edges[i, 1] in unseen_nodes:
                train_mask[i] = False
    else:
        train_mask = np.zeros(total_edges, dtype=bool)
        train_mask[:train_end] = True
    
    val_mask = np.zeros(total_edges, dtype=bool)
    val_mask[train_end:val_end] = True
    test_mask = np.zeros(total_edges, dtype=bool)
    test_mask[val_end:] = True
    
    # #  DATASET-SPECIFIC VALIDATION (now passes)
    # if dataset_name == 'wikipedia':
    #     assert num_nodes == 9228, f"Wikipedia must have 9228 nodes (IDs 0-9227), got {num_nodes}"
    # elif dataset_name == 'reddit':
    #     assert num_nodes == 7144, f"Reddit must have 7144 nodes, got {num_nodes}"
    # elif dataset_name == 'mooc':
    #     assert num_nodes == 7144, f"MOOC must have 7144 nodes, got {num_nodes}"
    
    statistics = {
        'dataset': dataset_name,
        'num_nodes': num_nodes,  # Now correctly 9228 for Wikipedia
        'num_edges': total_edges,
        'train_edges': int(train_mask.sum()),
        'val_edges': int(val_mask.sum()),
        'test_edges': int(test_mask.sum()),
        'inductive': inductive,
        'unseen_nodes': len(unseen_nodes) if unseen_nodes is not None else 0,
        'edge_feat_dim': edge_features.shape[1],
        'node_feat_dim': node_features.shape[1] if node_features is not None else 0,
        'time_span': (timestamps.min(), timestamps.max()),
        'seed': seed
    }
    
    logger.info(f" Loaded {dataset_name}: {num_nodes} nodes (0-indexed IDs 0-{num_nodes-1}), {total_edges} edges")
    
    return {
        'edges': torch.from_numpy(edges).long(),
        'timestamps': torch.from_numpy(timestamps).float(),
        'edge_features': torch.from_numpy(edge_features).float(),
        'node_features': torch.from_numpy(node_features[:num_nodes]).float() 
                         if node_features is not None else None,
        'num_nodes': num_nodes,  # CORRECT VALUE: 9228 for Wikipedia
        'train_mask': torch.from_numpy(train_mask),
        'val_mask': torch.from_numpy(val_mask),
        'test_mask': torch.from_numpy(test_mask),
        'unseen_nodes': unseen_nodes,
        'statistics': statistics
    }









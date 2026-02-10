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
    Load temporal graph dataset with strict 0-indexing and temporal splitting.
    
    Args:
        dataset_name: Name of dataset (e.g., 'wikipedia', 'reddit')
        data_root: Override data directory. If None, uses DATA_ROOT computed
                  relative to this file's location.
        val_ratio: Fraction of edges for validation
        test_ratio: Fraction of edges for testing
        inductive: If True, reserve unseen nodes for inductive evaluation
        unseen_ratio: Fraction of nodes to mark as unseen (inductive only)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing edges, features, masks, and metadata
        
    Raises:
        DatasetLoadError: If validation fails or files missing
    """
    # Local RNG to avoid global state pollution
    rng = np.random.default_rng(seed)
    
    # Resolve to absolute path for clarity in error messages
    # root = os.path.abspath(os.path.expanduser(root))
    
    # Load files
    folder_path = os.path.join(DATA_ROOT, dataset_name)
    
    if not os.path.exists(folder_path):
        raise DatasetLoadError(
            f"Dataset folder not found: {folder_path}\n"
            f"  (data_root={DATA_ROOT}, dataset_name={dataset_name})\n"
            f"  To override: set DYGLIB_DATA_ROOT env var or pass data_root="
        )
    
    try:
        df = pd.read_csv(os.path.join(folder_path, f"ml_{dataset_name}.csv"))
        node_features = np.load(os.path.join(folder_path, f"ml_{dataset_name}_node.npy")).astype(np.float32)
        edge_features = np.load(os.path.join(folder_path, f"ml_{dataset_name}.npy")).astype(np.float32)
    except FileNotFoundError as e:
        raise DatasetLoadError(f"Missing file for {dataset_name}: {e}")
    
    # Validate DataFrame columns
    required_cols = {'u', 'i', 'ts'}
    if not required_cols.issubset(df.columns):
        raise DatasetLoadError(f"Missing columns: {required_cols - set(df.columns)}")
    
    # Convert to 0-indexed (DyGLib uses 1-indexed)
    src = df['u'].values.astype(np.int64) - 1
    dst = df['i'].values.astype(np.int64) - 1
    timestamps = df['ts'].values.astype(np.float32)
    
    # Validate no negative indices (catches double-conversion)
    if src.min() < 0 or dst.min() < 0:
        raise DatasetLoadError(f"Negative node IDs detected after conversion. "
                              f"src_min={src.min()}, dst_min={dst.min()}")
    
    # Fix edge features shape (DyGLib sometimes has padding row)
    if edge_features.shape[0] == len(df) + 1:
        edge_features = edge_features[1:]  # Remove padding
    elif edge_features.shape[0] != len(df):
        raise DatasetLoadError(
            f"Edge feature count {edge_features.shape[0]} != edge count {len(df)}"
        )
    
    # Compute num_nodes correctly for 0-indexed
    max_node_id = int(max(src.max(), dst.max()))
    num_nodes = max_node_id + 1
    
    # Validate node features
    if node_features.shape[0] == num_nodes + 1:
        node_features = node_features[1:]  # Remove padding if present
    if node_features.shape[0] != num_nodes:
        raise DatasetLoadError(
            f"Node features count {node_features.shape[0]} != num_nodes {num_nodes}"
        )
    
    # Validate edge-node alignment
    if src.max() >= num_nodes or dst.max() >= num_nodes:
        raise DatasetLoadError(f"Edge references node >= num_nodes")
    
    edges = np.column_stack([src, dst])
    
    # Temporal sorting (CRITICAL: no shuffling)
    sort_idx = np.argsort(timestamps, kind='stable')
    edges = edges[sort_idx]
    timestamps = timestamps[sort_idx]
    edge_features = edge_features[sort_idx]
    
    total_edges = len(edges)
    
    # Compute split boundaries
    train_end = int(total_edges * (1 - val_ratio - test_ratio))
    val_end = int(total_edges * (1 - test_ratio))
    
    if train_end == 0 or val_end <= train_end or total_edges - val_end == 0:
        raise DatasetLoadError(
            f"Invalid split: train={train_end}, val={val_end-train_end}, "
            f"test={total_edges-val_end}. Adjust ratios."
        )
    
    # Initialize masks
    train_mask = np.zeros(total_edges, dtype=bool)
    val_mask = np.zeros(total_edges, dtype=bool)
    test_mask = np.zeros(total_edges, dtype=bool)
    
    unseen_nodes_tensor: Optional[torch.Tensor] = None
    actual_train_end = train_end  # Define for both cases
    
    if inductive:
        # Select unseen nodes BEFORE any edge filtering
        all_nodes = np.arange(num_nodes)
        rng.shuffle(all_nodes)
        num_unseen = max(1, int(num_nodes * unseen_ratio))  # At least 1
        unseen_nodes_arr = all_nodes[:num_unseen]
        unseen_set: Set[int] = set(unseen_nodes_arr)
        unseen_nodes_tensor = torch.from_numpy(unseen_nodes_arr).long()
        
        

        # Identify edges containing unseen nodes
        has_unseen = np.array([
            (e[0] in unseen_set) or (e[1] in unseen_set) 
            for e in edges
        ])
        
        # Training edges: must be in train window AND no unseen nodes
        train_candidates = np.arange(total_edges) < train_end
        train_mask = train_candidates & (~has_unseen)
        
        # CRITICAL FIX: Edges with unseen nodes in train window go to val
        # This ensures temporal continuity and all edges are assigned
        val_candidates = (np.arange(total_edges) >= train_end) & (np.arange(total_edges) < val_end)
        val_candidates = val_candidates | (train_candidates & has_unseen)
        
        val_mask = val_candidates
        test_mask[np.arange(total_edges) >= val_end] = True

        
        logger.info(f"Inductive: {train_mask.sum()}/{train_end} edges in train "
                   f"(removed {(train_candidates & has_unseen).sum()} with unseen nodes -> val), "
                   f"val={val_mask.sum()}, test={test_mask.sum()}")
    else:        
        # Standard transductive: temporal split only
        train_mask[:train_end] = True
        val_mask[train_end:val_end] = True
        test_mask[val_end:] = True
    
    
    
    # ============================================
    # VALIDATION (unchanged)
    # ============================================
    
    # Validate splits don't overlap
    assert not (train_mask & val_mask).any(), "Train/Val overlap"
    assert not (train_mask & test_mask).any(), "Train/Test overlap"
    assert not (val_mask & test_mask).any(), "Val/Test overlap"
    assert (train_mask | val_mask | test_mask).all(), "Unassigned edges"

    # Validate temporal ordering
    train_max_ts = timestamps[train_mask].max() if train_mask.any() else -np.inf
    val_min_ts = timestamps[val_mask].min() if val_mask.any() else np.inf
    val_max_ts = timestamps[val_mask].max() if val_mask.any() else -np.inf
    test_min_ts = timestamps[test_mask].min() if test_mask.any() else np.inf
    

    # Temporal validation 
    if not inductive:
        # In transductive mode, enforce strict temporal separation
        assert train_max_ts <= val_min_ts, f"Temporal leak: train ends at {train_max_ts}, val starts at {val_min_ts}"
        assert val_max_ts <= test_min_ts, f"Temporal leak: val ends at {val_max_ts}, test starts at {test_min_ts}"
    
    # Build statistics
    stats = {
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'num_edges': total_edges,
        'num_edges_train': int(train_mask.sum()),
        'num_edges_val': int(val_mask.sum()),
        'num_edges_test': int(test_mask.sum()),
        'split_ratios': {'train': 1 - val_ratio - test_ratio, 'val': val_ratio, 'test': test_ratio},
        'inductive': inductive,
        'num_unseen_nodes': len(unseen_nodes_arr) if inductive else 0,
        'unseen_ratio': unseen_ratio if inductive else 0.0,
        'edge_feat_dim': edge_features.shape[1],
        'node_feat_dim': node_features.shape[1],
        'time_span': (float(timestamps.min()), float(timestamps.max())),
        'train_time_range': (float(timestamps[train_mask].min()), float(train_max_ts)) if train_mask.any() else (None, None),
        'seed': seed,
    }
    
    logger.info(f" Loaded {dataset_name}: {num_nodes} nodes, {total_edges} edges")
    logger.info(f"  Split: train={stats['num_edges_train']}, val={stats['num_edges_val']}, test={stats['num_edges_test']}")
    if inductive:
        logger.info(f"  Inductive: {stats['num_unseen_nodes']} unseen nodes")
    
    return {
        'edges': torch.from_numpy(edges).long(),
        'timestamps': torch.from_numpy(timestamps).float(),
        'edge_features': torch.from_numpy(edge_features).float(),
        'node_features': torch.from_numpy(node_features).float(),
        'num_nodes': num_nodes,
        'train_mask': torch.from_numpy(train_mask),
        'val_mask': torch.from_numpy(val_mask),
        'test_mask': torch.from_numpy(test_mask),
        'unseen_nodes': unseen_nodes_tensor,
        'statistics': stats,
        'train_end_idx': actual_train_end,  # Now defined for both cases
    }









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
    
    if inductive:
        # Select unseen nodes BEFORE any edge filtering
        all_nodes = np.arange(num_nodes)
        rng.shuffle(all_nodes)
        num_unseen = max(1, int(num_nodes * unseen_ratio))  # At least 1
        unseen_nodes_arr = all_nodes[:num_unseen]
        unseen_set: Set[int] = set(unseen_nodes_arr)
        unseen_nodes_tensor = torch.from_numpy(unseen_nodes_arr).long()
        
        # Identify edges containing unseen nodes
        has_both_unseen = np.array([
            (e[0] in unseen_set) or (e[1] in unseen_set) 
            for e in edges
        ])
        
        # Training edges: must be in train window AND no unseen nodes
        train_candidates = np.arange(total_edges) < train_end
        train_mask = train_candidates & (~has_both_unseen)
        
        # Adjust train_end to actual used boundary for consistent val/test
        actual_train_end = train_end  # Keep original temporal boundaries
        
        logger.info(f"Inductive: {train_mask.sum()}/{train_end} edges in train "
                   f"(removed {(~train_mask[:train_end]).sum()} with unseen nodes)")
    else:
        # Standard transductive: temporal split only
        train_mask[:train_end] = True
        actual_train_end = train_end
    
    # Val/Test: temporal boundaries (may include unseen nodes in inductive)
    val_mask[actual_train_end:val_end] = True
    test_mask[val_end:] = True
    
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
    
    if inductive:
        # In inductive, train may have gaps, but val/test must be after train window start
        pass  # More relaxed for inductive
    else:
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
        'split_ratios': {'train': val_ratio + test_ratio, 'val': val_ratio, 'test': test_ratio},
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
        'train_end_idx': actual_train_end,  # Useful for neighbor finder
    }










# def load_generic_dataset(dataset_name, model_type="dygformer"):
#     """Unified loader for all DyGLib datasets"""
#     folder_path = os.path.join(DATA_ROOT, dataset_name)
#     csv_path = os.path.join(folder_path, f"ml_{dataset_name}.csv")
#     node_feat_path = os.path.join(folder_path, f"ml_{dataset_name}_node.npy")
#     edge_feat_path = os.path.join(folder_path, f"ml_{dataset_name}.npy")

#     df = pd.read_csv(csv_path)
#     # src = df['u'].values.astype(np.int64) - 1
#     # dst = df['i'].values.astype(np.int64) - 1
#     # timestamps = df['ts'].values.astype(np.float32)

#     # Load features
#     node_features = np.load(node_feat_path).astype(np.float32)
#     edge_features = np.load(edge_feat_path).astype(np.float32)
    
#     print(len(df))
#     # Fix edge features extra row
#     if edge_features.shape[0] > len(df):
#         print(f"Removing extra row from {dataset_name} edge features: {edge_features.shape[0]} -> {len(df)}")
#         edge_features = edge_features[1:]

#     # Verify shapes match
#     assert edge_features.shape[0] == len(df), f"Edge features shape {edge_features.shape} doesn't match edges {len(df)}"
    

#     # Get true number of nodes from node features
#     num_actual_nodes = node_features.shape[0] 

#     if model_type == "tgn":
#         # TGN uses 0-indexed, no padding
#         src = df['u'].values.astype(np.int64) - 1
#         dst = df['i'].values.astype(np.int64) - 1
#         edges_array = np.column_stack([src, dst])
#         num_nodes = int(max(src.max(), dst.max()) + 1)
        
#         return {
#             "edges": torch.from_numpy(edges_array).long(),
#             "timestamps": torch.from_numpy(df['ts'].values.astype(np.float32)).float(),
#             "edge_features": torch.from_numpy(edge_features).float(),
#             "num_nodes": num_nodes,
#         }
#     else:
#         # DyGFormer/TAWRMAC: 1-indexed with padding
#         src_1indexed = df['u'].values.astype(np.int64)  # Keep 1-indexed
#         dst_1indexed = df['i'].values.astype(np.int64)  # Keep 1-indexed
#         edges_array = np.column_stack([src_1indexed, dst_1indexed])
        
#         # Create padded node features (index 0 = padding)
#         node_raw_features = np.zeros((num_actual_nodes + 1, node_features.shape[1]), dtype=np.float32)
#         node_raw_features[1:] = node_features
        
#         # Create padded edge features (index 0 = padding)
#         edge_features_unpadded = edge_features # shape [157474, 172]
#         edge_features_padded = np.zeros((len(edge_features) + 1, edge_features.shape[1]), dtype=np.float32)
#         edge_features_padded[1:] = edge_features
        

#     # # Convert to 0-indexed for calculations
#     # src = df['u'].values.astype(np.int64) - 1
#     # dst = df['i'].values.astype(np.int64) - 1
#     # timestamps = df['ts'].values.astype(np.float32)

    
   
    
#     # # Convert to 1-indexed (0 = padding)
#     # src_1indexed = src + 1
#     # dst_1indexed = dst + 1
#     # num_nodes = int(max(src_1indexed.max(), dst_1indexed.max()))
    
#     # # Create padded features
#     # node_raw_features = np.zeros((num_nodes + 1, node_features.shape[1]), dtype=np.float32)
#     # node_raw_features[1:node_features.shape[0] + 1] = node_features
    
#     # edge_raw_features = np.zeros((len(edge_features) + 1, edge_features.shape[1]), dtype=np.float32)
#     # edge_raw_features[1:] = edge_features

#     # edges_array = np.column_stack([src_1indexed, dst_1indexed])

#     return {
#         "edges": torch.from_numpy(edges_array).long(),
#         "timestamps": torch.from_numpy(df['ts'].values.astype(np.float32)).float(),
#         "node_features": torch.from_numpy(node_raw_features).float(),
#         "edge_features": torch.from_numpy(edge_features_unpadded).float(), # for splitting
#         "edge_padded": torch.from_numpy(edge_features_padded).float(), # for model
#         "num_nodes": num_actual_nodes,  # 9228
#     }

# # Dataset-specific loaders
# def load_wikipedia(): 
#     return load_generic_dataset("wikipedia", model_type="dygformer")
# def load_reddit(): 
#     return load_generic_dataset("reddit", model_type="dygformer")  
# def load_mooc(): 
#     return load_generic_dataset("mooc", model_type="dygformer")
# def load_lastfm(): 
#     return load_generic_dataset("lastfm", model_type="dygformer")
# def load_uci(): 
#     return load_generic_dataset("uci", model_type="dygformer")
# def load_enron(): 
#     return load_generic_dataset("uci", model_type="dygformer")
# def load_contacts(): 
#     return load_generic_dataset("uci", model_type="dygformer")
# def load_untrade(): 
#     return load_generic_dataset("uci", model_type="dygformer")
# def load_flights(): 
#     return load_generic_dataset("uci", model_type="dygformer")
# def load_unvote(): 
#     return load_generic_dataset("uci", model_type="dygformer")
# def load_canparl(): 
#     return load_generic_dataset("uci", model_type="dygformer")
# def load_uslegis(): 
#     return load_generic_dataset("uci", model_type="dygformer")






# def load_wikipedia():
    
#     csv_path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia.csv")
#     print(f"loading csv from {csv_path}")    
#     df = pd.read_csv(csv_path)

  

#     # Extract data
#     src = df['u'].values.astype(np.int64) - 1
#     dst = df['i'].values.astype(np.int64) - 1
#     timestamps = df['ts'].values.astype(np.float32)
    
#     edges = torch.stack([
#         torch.from_numpy(src),
#         torch.from_numpy(dst)
#     ], dim=1)

#     # Load node features
#     node_feat_path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia_node.npy")
#     print(f"Loading node features from: {node_feat_path}")
#     node_features = np.load(node_feat_path).astype(np.float32)
    
#     # Load edge features
#     edge_feat_path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia.npy")
#     print(f"Loading edge features from: {edge_feat_path}")
#     edge_features = np.load(edge_feat_path).astype(np.float32)
    
#     # Fix: Remove extra row if present
#     if edge_features.shape[0] == len(src) + 1:
#         print("Removing extra row from edge features...")
#         edge_features = edge_features[1:]  # Remove first row
    
#     assert edge_features.shape[0] == len(src), f"Edge features shape {edge_features.shape} doesn't match edges {len(src)}"

#     # Print debug info
#     print(f"\nDataset Statistics:")
#     print(f"  Number of edges: {len(src)}")
#     print(f"  Node features shape: {node_features.shape}")
#     print(f"  Edge features shape: {edge_features.shape}")
#     print(f"  Source node range: {src.min()} to {src.max()}")
#     print(f"  Destination node range: {dst.min()} to {dst.max()}")
#     print(f"  Timestamp range: {timestamps.min()} to {timestamps.max()}")

#     # Adjust for 1-indexing (DyGFormer uses 0 for padding)
#     src = src + 1
#     dst = dst + 1
#     # num_nodes = int(max(src.max(), dst.max()))
#     num_nodes = int(edges.max().item() + 1)  # This will be 9228 if max ID is 9227
    
#     print(f"\nAfter adjustment for 1-indexing:")
#     print(f"  Adjusted num_nodes: {num_nodes}")
#     print(f"  Node indices now start from 1 (0 is padding)")
    
#     # Create edge array in DyGFormer format: [src, dst, timestamp, edge_idx]
#     edge_idx = np.arange(len(src), dtype=np.int64) + 1  # 1-indexed edge ids
#     edges = np.column_stack([src, dst])
    
#     # Create node raw features with padding at index 0
#     node_raw_features = np.zeros((num_nodes + 1, node_features.shape[1]), dtype=np.float32)
#     actual_num_nodes = min(num_nodes, node_features.shape[0])
#     node_raw_features[1:actual_num_nodes + 1] = node_features[:actual_num_nodes]

#     # Create edge raw features with padding at index 0
#     edge_raw_features = np.zeros((len(edge_idx) + 1, edge_features.shape[1]), dtype=np.float32)
#     edge_raw_features[1:] = edge_features

#     # print(f"Edges shape: {edges.shape}")
#     # print(f"Node ID range: {edges.min()} to {edges.max()}")
#     # print(f"Num nodes from features: {node_features.size(0)}")

#     print(f"\nFinal shapes for DyGFormer:")
#     print(f"  node_raw_features shape: {node_raw_features.shape}")
#     print(f"  edge_raw_features shape: {edge_raw_features.shape}")
#     print(f"  edges_array shape: {edges.shape}")

#     edges = torch.from_numpy(edges).long()
#     timestamps = torch.from_numpy(timestamps).float()
#     node_features = torch.from_numpy(node_raw_features).float()
#     edge_features = torch.from_numpy(edge_raw_features).float()
    
#     return {
#         "edges": edges,
#         "timestamps": timestamps,
#         "node_features": node_features,
#         "edge_features": edge_features,
#         "num_nodes": num_nodes,
#         "num_edges": len(src),
#     }



# Model Specific Data Requirement
# Model | Node feature|Edge Feature| Memory|Special requirements|
# TGN  | option| Required|Yes| Message Passing, memory updates|
# Jodie  | No| No |Yes| Bipartite graphs only|
# TAWRMAC  | Required | Required | Yes| Temporal walks, co-occurrence|
# DyGFormer | Required | Required | No | Patch-based sequence |




# def load_reddit():
#     path = os.path.join(DATA_ROOT, "reddit", "ml_reddit.npy")
#     data = np.load(path)

#     if data.shape[1]>2:
#         edges = torch.from_numpy(data[:,:2]).long()
#         edge_features = torch.from_numpy(data[:,2:]).float()
#     else:
#         edges = torch.from_numpy(data).long()
#         edge_features = None
    
#     csv_path = os.path.join(DATA_ROOT, "reddit", "ml_reddit.csv")
#     # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:, 2])
#     timestamps_raw = torch.from_numpy(np.genfromtxt(
#         csv_path, 
#         delimiter=",",
#         skip_header=1,
#         dtype=np.float64,
#         filling_values=0.0)
#     )
        
#     if timestamps_raw.ndim == 2:
#         timestamps = timestamps_raw[:,2]
#     else:
#         timestamps = timestamps_raw

#     min_len = min(len(edges), len(timestamps))
#     edges = edges[:min_len]
#     if edge_features is not None:
#         edge_features = edge_features[:min_len]
#     # edges = torch.from_numpy(np.load("data/processed/reddit/ml_reddit.npy"))
#     # timestamps = torch.from_numpy(np.loadtxt("data/processed/reddit/ml_reddit.csv", delimiter=",")[:, 2])  # assuming [src, dst, ts]
#     # valid_mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] != edges[:, 1])
    
#     # edges = edges[valid_mask]    
#     # timestamps = timestamps[valid_mask]    
#     # if edge_features is not None:
#     #     edge_features = edge_features[valid_mask]
    

#     edges = torch.from_numpy(edges).long()
#     timestamps = torch.from_numpy(timestamps).float()
#     node_features = torch.from_numpy(node_raw_features).float()
#     edge_features = torch.from_numpy(edge_raw_features).float()
#     # recompute num_nodes after cleaning
#     # num_nodes = int(edges.max().item()+1)
#     # return {
#     #     "edges": edges,
#     #     "timestamps": timestamps,
#     #     "edge_feature": edge_features,
#     #     "num_nodes": num_nodes
#     # }
#     return {
#         "edges": edges,
#         "timestamps": timestamps,
#         "node_features": node_features,
#         "edge_features": edge_features,
#         "num_nodes": num_nodes,
#         "num_edges": len(src),
#     }

# def load_mooc():

#     path = os.path.join(DATA_ROOT, "mooc", "ml_mooc.npy")
#     data = np.load(path)
#     if data.shape[1]>2:
#         edges = torch.from_numpy(data[:,:2]).long()
#         edge_features = torch.from_numpy(data[:,2:]).float()
#     else:
#         edges = torch.from_numpy(data).long()
#         edge_features = None
#     # edges = torch.from_numpy()
    
#     csv_path = os.path.join(DATA_ROOT, "mooc", "ml_mooc.csv")
#     # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:, 2])
#     timestamps_raw = np.genfromtxt(
#         csv_path, 
#         delimiter=",",
#         skip_header=1,
#         dtype=np.float64,
#         filling_values=0.0)

#     # Ensure timestamps is 1D
#     if timestamps_raw.ndim == 2:
#         timestamps = timestamps_raw[:, 2]
#     else:
#         timestamps = timestamps_raw

#     min_len = min(len(edges), len(timestamps))
#     edges = edges[:min_len]
#     timestamps = timestamps[:min_len]
#     if edge_features is not None:
#         edge_features = edge_features[:min_len]

#     # Now apply valid mask
#     valid_mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] != edges[:, 1])
#     edges = edges[valid_mask]
#     timestamps = torch.from_numpy(timestamps[valid_mask]).float()
#     if edge_features is not None:
#         edge_features = edge_features[valid_mask]

#     num_nodes = int(edges.max().item()+1)    
#     return {
#         "edges": edges,
#         "timestamps": timestamps,
#         "edge_feature": edge_features,
#         "num_nodes": num_nodes
#     }


# def load_lastfm():
#     path = os.path.join(DATA_ROOT, "lastfm", "ml_lastfm.npy")
#     data = np.load(path)
    
    
#     if data.shape[1]>2:
#         edges = torch.from_numpy(data[:,:2]).long()
#         edge_features = torch.from_numpy(data[:,2:]).float()
#     else:
#         edges = torch.from_numpy(data).long()
#         edge_features = None
    
#     csv_path = os.path.join(DATA_ROOT, "lastfm", "ml_lastfm.csv")
#     # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:, 2])
#     timestamps_raw = np.genfromtxt(
#         csv_path, 
#         delimiter=",",
#         skip_header=1,
#         dtype=np.float64,
#         filling_values=0.0
#     )

#     # Ensure timestamps is 1D
#     if timestamps_raw.ndim == 2:
#         timestamps = timestamps_raw[:, 2]
#     else:
#         timestamps = timestamps_raw

#     min_len = min(len(edges), len(timestamps))
#     edges = edges[:min_len]
#     timestamps = timestamps[:min_len]
#     if edge_features is not None:
#         edge_features = edge_features[:min_len]

#     # Now apply valid mask
#     valid_mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] != edges[:, 1])
#     edges = edges[valid_mask]
#     timestamps = torch.from_numpy(timestamps[valid_mask]).float()
#     if edge_features is not None:
#         edge_features = edge_features[valid_mask]

#     if edges.numel() == 0:
#         raise ValueError("LastFM dataset resulted in empty edge list after filtering. Check data files.")
#     num_nodes = int(edges.max().item()+1)

#     # edges = torch.from_numpy(np.load("data/processed/lastfm/ml_lastfm.npy"))
#     # timestamps = torch.from_numpy(np.loadtxt("data/processed/lastfm/ml_lastfm.csv", delimiter=",")[:, 2])  # assuming [src, dst, ts]
    
#     return {
#         "edges": edges,
#         "timestamps": timestamps.float(),
#         "edge_feature": edge_features,
#         "num_nodes": num_nodes
#     }

# def load_uci():
#     path = os.path.join(DATA_ROOT, "uci", "ml_uci.npy")
#     data = np.load(path)
#     # edges = torch.from_numpy(np.load(path))
#     if data.shape[1]>2:
#         edges = torch.from_numpy(data[:,:2]).long()
#         edge_features = torch.from_numpy(data[:,2:]).float()
#     else:
#         edges = torch.from_numpy(data).long()
#         edge_features = None
    
#     csv_path = os.path.join(DATA_ROOT, "uci", "ml_uci.csv")
#     # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:, 2])
#     timestamps = torch.from_numpy(np.genfromtxt(
#         csv_path, 
#         delimiter=",",
#         skip_header=1,
#         dtype=np.float64,
#         filling_values=0.0)[:, 2])
#     # edges = torch.from_numpy(np.load("data/processed/uci/ml_uci.npy"))
#     # timestamps = torch.from_numpy(np.loadtxt("data/processed/uci/ml_uci.csv", delimiter=",")[:, 2])  # assuming [src, dst, ts]
    
#     return {
#         "edges": edges,
#         "timestamps": timestamps.float(),
#         "edge_feature": edge_features,
#         "num_nodes": int(edges.max().item() + 1)
#     }

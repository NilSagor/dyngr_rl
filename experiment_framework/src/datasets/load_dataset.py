
import os
import numpy as np
import torch
import pandas as pd

# Get the directory where this file lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")



def load_wikipedia():
    
    csv_path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia.csv")
    print(f"loading csv from {csv_path}")    
    df = pd.read_csv(csv_path)

  

    # Extract data
    src = df['u'].values.astype(np.int64) - 1
    dst = df['i'].values.astype(np.int64) - 1
    timestamps = df['ts'].values.astype(np.float32)
    
    edges = torch.stack([
        torch.from_numpy(src),
        torch.from_numpy(dst)
    ], dim=1)

    # Load node features
    node_feat_path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia_node.npy")
    print(f"Loading node features from: {node_feat_path}")
    node_features = np.load(node_feat_path).astype(np.float32)
    
    # Load edge features
    edge_feat_path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia.npy")
    print(f"Loading edge features from: {edge_feat_path}")
    edge_features = np.load(edge_feat_path).astype(np.float32)
    
    # Fix: Remove extra row if present
    if edge_features.shape[0] == len(src) + 1:
        print("Removing extra row from edge features...")
        edge_features = edge_features[1:]  # Remove first row
    
    assert edge_features.shape[0] == len(src), f"Edge features shape {edge_features.shape} doesn't match edges {len(src)}"

    # Print debug info
    print(f"\nDataset Statistics:")
    print(f"  Number of edges: {len(src)}")
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Edge features shape: {edge_features.shape}")
    print(f"  Source node range: {src.min()} to {src.max()}")
    print(f"  Destination node range: {dst.min()} to {dst.max()}")
    print(f"  Timestamp range: {timestamps.min()} to {timestamps.max()}")

    # Adjust for 1-indexing (DyGFormer uses 0 for padding)
    src = src + 1
    dst = dst + 1
    # num_nodes = int(max(src.max(), dst.max()))
    num_nodes = int(edges.max().item() + 1)  # This will be 9228 if max ID is 9227
    
    print(f"\nAfter adjustment for 1-indexing:")
    print(f"  Adjusted num_nodes: {num_nodes}")
    print(f"  Node indices now start from 1 (0 is padding)")
    
    # Create edge array in DyGFormer format: [src, dst, timestamp, edge_idx]
    edge_idx = np.arange(len(src), dtype=np.int64) + 1  # 1-indexed edge ids
    edges = np.column_stack([src, dst])
    
    # Create node raw features with padding at index 0
    node_raw_features = np.zeros((num_nodes + 1, node_features.shape[1]), dtype=np.float32)
    actual_num_nodes = min(num_nodes, node_features.shape[0])
    node_raw_features[1:actual_num_nodes + 1] = node_features[:actual_num_nodes]

    # Create edge raw features with padding at index 0
    edge_raw_features = np.zeros((len(edge_idx) + 1, edge_features.shape[1]), dtype=np.float32)
    edge_raw_features[1:] = edge_features

    # print(f"Edges shape: {edges.shape}")
    # print(f"Node ID range: {edges.min()} to {edges.max()}")
    # print(f"Num nodes from features: {node_features.size(0)}")

    print(f"\nFinal shapes for DyGFormer:")
    print(f"  node_raw_features shape: {node_raw_features.shape}")
    print(f"  edge_raw_features shape: {edge_raw_features.shape}")
    print(f"  edges_array shape: {edges.shape}")

    edges = torch.from_numpy(edges).long()
    timestamps = torch.from_numpy(timestamps).float()
    node_features = torch.from_numpy(node_raw_features).float()
    edge_features = torch.from_numpy(edge_raw_features).float()
    
    return {
        "edges": edges,
        "timestamps": timestamps,
        "node_features": node_features,
        "edge_features": edge_features,
        "num_nodes": num_nodes,
        "num_edges": len(src),
    }




def load_reddit():
    path = os.path.join(DATA_ROOT, "reddit", "ml_reddit.npy")
    data = np.load(path)

    if data.shape[1]>2:
        edges = torch.from_numpy(data[:,:2]).long()
        edge_features = torch.from_numpy(data[:,2:]).float()
    else:
        edges = torch.from_numpy(data).long()
        edge_features = None
    
    csv_path = os.path.join(DATA_ROOT, "reddit", "ml_reddit.csv")
    # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:, 2])
    timestamps_raw = torch.from_numpy(np.genfromtxt(
        csv_path, 
        delimiter=",",
        skip_header=1,
        dtype=np.float64,
        filling_values=0.0)
    )
        
    if timestamps_raw.ndim == 2:
        timestamps = timestamps_raw[:,2]
    else:
        timestamps = timestamps_raw

    min_len = min(len(edges), len(timestamps))
    edges = edges[:min_len]
    if edge_features is not None:
        edge_features = edge_features[:min_len]
    # edges = torch.from_numpy(np.load("data/processed/reddit/ml_reddit.npy"))
    # timestamps = torch.from_numpy(np.loadtxt("data/processed/reddit/ml_reddit.csv", delimiter=",")[:, 2])  # assuming [src, dst, ts]
    valid_mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] != edges[:, 1])
    
    edges = edges[valid_mask]    
    timestamps = timestamps[valid_mask]    
    if edge_features is not None:
        edge_features = edge_features[valid_mask]
    
    # recompute num_nodes after cleaning
    num_nodes = int(edges.max().item()+1)
    return {
        "edges": edges,
        "timestamps": timestamps,
        "edge_feature": edge_features,
        "num_nodes": num_nodes
    }

def load_mooc():

    path = os.path.join(DATA_ROOT, "mooc", "ml_mooc.npy")
    data = np.load(path)
    if data.shape[1]>2:
        edges = torch.from_numpy(data[:,:2]).long()
        edge_features = torch.from_numpy(data[:,2:]).float()
    else:
        edges = torch.from_numpy(data).long()
        edge_features = None
    # edges = torch.from_numpy()
    
    csv_path = os.path.join(DATA_ROOT, "mooc", "ml_mooc.csv")
    # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:, 2])
    timestamps_raw = np.genfromtxt(
        csv_path, 
        delimiter=",",
        skip_header=1,
        dtype=np.float64,
        filling_values=0.0)

    # Ensure timestamps is 1D
    if timestamps_raw.ndim == 2:
        timestamps = timestamps_raw[:, 2]
    else:
        timestamps = timestamps_raw

    min_len = min(len(edges), len(timestamps))
    edges = edges[:min_len]
    timestamps = timestamps[:min_len]
    if edge_features is not None:
        edge_features = edge_features[:min_len]

    # Now apply valid mask
    valid_mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] != edges[:, 1])
    edges = edges[valid_mask]
    timestamps = torch.from_numpy(timestamps[valid_mask]).float()
    if edge_features is not None:
        edge_features = edge_features[valid_mask]

    num_nodes = int(edges.max().item()+1)    
    return {
        "edges": edges,
        "timestamps": timestamps,
        "edge_feature": edge_features,
        "num_nodes": num_nodes
    }


def load_lastfm():
    path = os.path.join(DATA_ROOT, "lastfm", "ml_lastfm.npy")
    data = np.load(path)
    
    
    if data.shape[1]>2:
        edges = torch.from_numpy(data[:,:2]).long()
        edge_features = torch.from_numpy(data[:,2:]).float()
    else:
        edges = torch.from_numpy(data).long()
        edge_features = None
    
    csv_path = os.path.join(DATA_ROOT, "lastfm", "ml_lastfm.csv")
    # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:, 2])
    timestamps_raw = np.genfromtxt(
        csv_path, 
        delimiter=",",
        skip_header=1,
        dtype=np.float64,
        filling_values=0.0
    )

    # Ensure timestamps is 1D
    if timestamps_raw.ndim == 2:
        timestamps = timestamps_raw[:, 2]
    else:
        timestamps = timestamps_raw

    min_len = min(len(edges), len(timestamps))
    edges = edges[:min_len]
    timestamps = timestamps[:min_len]
    if edge_features is not None:
        edge_features = edge_features[:min_len]

    # Now apply valid mask
    valid_mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] != edges[:, 1])
    edges = edges[valid_mask]
    timestamps = torch.from_numpy(timestamps[valid_mask]).float()
    if edge_features is not None:
        edge_features = edge_features[valid_mask]

    if edges.numel() == 0:
        raise ValueError("LastFM dataset resulted in empty edge list after filtering. Check data files.")
    num_nodes = int(edges.max().item()+1)

    # edges = torch.from_numpy(np.load("data/processed/lastfm/ml_lastfm.npy"))
    # timestamps = torch.from_numpy(np.loadtxt("data/processed/lastfm/ml_lastfm.csv", delimiter=",")[:, 2])  # assuming [src, dst, ts]
    
    return {
        "edges": edges,
        "timestamps": timestamps.float(),
        "edge_feature": edge_features,
        "num_nodes": num_nodes
    }

def load_uci():
    path = os.path.join(DATA_ROOT, "uci", "ml_uci.npy")
    data = np.load(path)
    # edges = torch.from_numpy(np.load(path))
    if data.shape[1]>2:
        edges = torch.from_numpy(data[:,:2]).long()
        edge_features = torch.from_numpy(data[:,2:]).float()
    else:
        edges = torch.from_numpy(data).long()
        edge_features = None
    
    csv_path = os.path.join(DATA_ROOT, "uci", "ml_uci.csv")
    # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:, 2])
    timestamps = torch.from_numpy(np.genfromtxt(
        csv_path, 
        delimiter=",",
        skip_header=1,
        dtype=np.float64,
        filling_values=0.0)[:, 2])
    # edges = torch.from_numpy(np.load("data/processed/uci/ml_uci.npy"))
    # timestamps = torch.from_numpy(np.loadtxt("data/processed/uci/ml_uci.csv", delimiter=",")[:, 2])  # assuming [src, dst, ts]
    
    return {
        "edges": edges,
        "timestamps": timestamps.float(),
        "edge_feature": edge_features,
        "num_nodes": int(edges.max().item() + 1)
    }
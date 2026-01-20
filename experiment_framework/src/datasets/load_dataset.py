
import os
import numpy as np
import torch

# Get the directory where this file lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")



def load_wikipedia():

    path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia.npy")
    data = np.load(path)
    
    if data.shape[1]>2:
        edges = torch.from_numpy(data[:,:2]).long()
        edge_features = torch.from_numpy(data[:,2:]).float()
    else:
        edges = torch.from_numpy(data).long()
        edge_features = None
    # edges = torch.from_numpy(np.load(path))
    
    csv_path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia.csv")
    timestamps = torch.from_numpy(np.genfromtxt(
        csv_path, 
        delimiter=",",
        skip_header=1,
        dtype=np.float64,
        filling_values=0.0)[:, 2].astype(np.float32)).float()
    

    # path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia.npy")
    # edges = torch.from_numpy(np.load(path))
    # # edges = torch.from_numpy(np.load("data/processed/wikipedia/ml_wikipedia.npy"))
    # csv_path = os.path.join(DATA_ROOT, "wikipedia", "ml_wikipedia.csv") 
    # timestamps = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")[:,2])
    # # timestamps = torch.from_numpy(np.loadtxt("data/processed/wikipedia/ml_wikipedia.csv", delimiter=",")[:, 2])  # assuming [src, dst, ts]
    # valid_mask = (edges[:, 0]>=0)&(edges[:,1]>=0) & (edges[:, 0]!=edges[:,1])
    # edges = edges[valid_mask]
    # timestamps = timestamps[valid_mask]
    print("Edges shape:", edges.shape)
    # # After applying valid_mask
    # if len(edges) == 0:
    #     raise ValueError(f"No valid edges found in {dataset_name}. Check data integrity.")
    num_nodes = int(edges.max().item()+1)
    return {
        "edges": edges,
        "timestamps": timestamps.float(),
        "edge_feature": edge_features,
        "num_nodes": num_nodes
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
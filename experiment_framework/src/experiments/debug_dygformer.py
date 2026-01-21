# experiment_framework/src/experiments/debug_dygformer.py
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Fix the path - go up two levels from experiments directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

from src.models.dygformer import DyGFormer

def load_wikipedia_fixed():
    """Load Wikipedia dataset in the format expected by DyGFormer."""
    import pandas as pd
    from pathlib import Path
    
    DATA_ROOT = Path("experiment_framework/data/processed")
    
    # Load from CSV first
    csv_path = DATA_ROOT / "wikipedia" / "ml_wikipedia.csv"
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract data
    src = df['u'].values.astype(np.int64)
    dst = df['i'].values.astype(np.int64)
    timestamps = df['ts'].values.astype(np.float32)
    
    # Load node features
    node_feat_path = DATA_ROOT / "wikipedia" / "ml_wikipedia_node.npy"
    print(f"Loading node features from: {node_feat_path}")
    node_features = np.load(node_feat_path).astype(np.float32)
    
    # Load edge features
    edge_feat_path = DATA_ROOT / "wikipedia" / "ml_wikipedia.npy"
    print(f"Loading edge features from: {edge_feat_path}")
    edge_features = np.load(edge_feat_path).astype(np.float32)
    
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
    num_nodes = int(max(src.max(), dst.max()))
    
    print(f"\nAfter adjustment for 1-indexing:")
    print(f"  Adjusted num_nodes: {num_nodes}")
    print(f"  Node indices now start from 1 (0 is padding)")
    
    # Create edge array in DyGFormer format: [src, dst, timestamp, edge_idx]
    edge_idx = np.arange(len(src), dtype=np.int64) + 1  # 1-indexed edge ids
    edges_array = np.column_stack([src, dst, timestamps, edge_idx])
    
    # Create node raw features with padding at index 0
    node_raw_features = np.zeros((num_nodes + 1, node_features.shape[1]), dtype=np.float32)
    # Copy existing node features, starting from index 1
    node_raw_features[1:node_features.shape[0] + 1] = node_features[:num_nodes]
    
    # Create edge raw features with padding at index 0
    edge_raw_features = np.zeros((len(edge_idx) + 1, edge_features.shape[1]), dtype=np.float32)
    edge_raw_features[1:] = edge_features
    
    print(f"\nFinal shapes for DyGFormer:")
    print(f"  node_raw_features shape: {node_raw_features.shape}")
    print(f"  edge_raw_features shape: {edge_raw_features.shape}")
    print(f"  edges_array shape: {edges_array.shape}")
    
    return {
        "node_raw_features": node_raw_features,
        "edge_raw_features": edge_raw_features,
        "edges_array": edges_array,
        "num_nodes": num_nodes,
        "num_edges": len(src)
    }

def test_dygformer_small():
    """Test DyGFormer with a small batch to catch errors early"""
    
    print("=== Testing DyGFormer Implementation ===")
    
    # Load a small subset of data
    data = load_wikipedia_fixed()
    
    # Take only first 100 edges for testing
    num_test_edges = min(100, data["num_edges"])
    test_edges = data["edges_array"][:num_test_edges]
    
    print(f"\nUsing {num_test_edges} edges for testing")
    
    # Initialize model with small dimensions for debugging
    model = DyGFormer(
        num_nodes=data["num_nodes"],
        node_features=data["node_raw_features"].shape[1],
        hidden_dim=64,  # Smaller for testing
        time_encoding_dim=32,
        num_layers=1,  # Fewer layers
        num_heads=2,
        dropout=0.1,
        max_neighbors=10,  # Smaller neighborhood
        patch_size=1,
        max_sequence_length=64,
        channel_embedding_dim=32,
        neighbor_co_occurrence=True
    )
    
    # Set raw features
    model.set_raw_features(
        torch.from_numpy(data["node_raw_features"]),
        torch.from_numpy(data["edge_raw_features"])
    )
    
    # Create a simple test batch
    batch_size = 4
    print(f"\nCreating test batch with size {batch_size}")
    
    # Take first few edges
    test_batch = {
        'src_nodes': torch.from_numpy(test_edges[:batch_size, 0]).long(),
        'dst_nodes': torch.from_numpy(test_edges[:batch_size, 1]).long(),
        'timestamps': torch.from_numpy(test_edges[:batch_size, 2]).float(),
        'labels': torch.randint(0, 2, (batch_size,)).float(),
    }
    
    # Add neighbor information (simplified)
    max_neighbors = 10
    test_batch['src_neighbors'] = torch.randint(0, data["num_nodes"], (batch_size, max_neighbors))
    test_batch['dst_neighbors'] = torch.randint(0, data["num_nodes"], (batch_size, max_neighbors))
    test_batch['src_neighbor_times'] = torch.rand(batch_size, max_neighbors) * 1000
    test_batch['dst_neighbor_times'] = torch.rand(batch_size, max_neighbors) * 1000
    
    print(f"\nBatch keys: {list(test_batch.keys())}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            model.eval()
            output = model(test_batch)
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: {output.min().item():.4f} to {output.max().item():.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    print("\nTesting loss computation...")
    try:
        loss = model._compute_loss(test_batch)
        print(f"✓ Loss computation successful!")
        print(f"  Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Loss computation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test metrics computation
    print("\nTesting metrics computation...")
    try:
        metrics = model._compute_metrics(test_batch)
        print(f"✓ Metrics computation successful!")
        for key, value in metrics.items():
            print(f"  {key}: {value.item() if torch.is_tensor(value) else value:.4f}")
    except Exception as e:
        print(f"✗ Metrics computation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== All Tests Passed ===")
    return True

def check_memory_allocation():
    """Check for potential memory allocation issues"""
    print("\n=== Checking Memory Allocation ===")
    
    # Test different tensor sizes that might cause issues
    test_sizes = [
        (1000, 172),    # Node features
        (10000, 172),   # Edge features
        (1000, 256),    # Hidden states
        (1000, 64, 4),  # Patches
    ]
    
    for size in test_sizes:
        try:
            if len(size) == 2:
                tensor = torch.randn(size)
            else:
                tensor = torch.randn(size)
            print(f"✓ Allocated tensor of size {size}: {tensor.numel()} elements")
            del tensor
        except Exception as e:
            print(f"✗ Failed to allocate tensor of size {size}: {e}")

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Check memory allocation first
    check_memory_allocation()
    
    # Run the test
    success = test_dygformer_small()
    
    if success:
        print("\n✅ Model implementation looks correct!")
        print("The issue might be in the data loading or training pipeline.")
    else:
        print("\n❌ Model has issues that need to be fixed.")
# experiment_framework/src/experiments/train_simple.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import lightning as L
from loguru import logger

from src.models.dygformer import DyGFormer
from src.datasets.load_dataset_fixed import load_wikipedia_fixed
from src.datasets.temporal_dataset import TemporalDataset
from torch.utils.data import DataLoader

def create_simple_data_loaders(data, batch_size=32):
    """Create simple data loaders for testing"""
    
    # Use the edges_array to create temporal dataset
    edges = torch.from_numpy(data["edges_array"][:, :2]).long()  # src, dst
    timestamps = torch.from_numpy(data["edges_array"][:, 2]).float()
    
    # Create a simple dataset
    dataset = TemporalDataset(
        edges=edges,
        timestamps=timestamps,
        edge_features=None,  # We'll use raw features in model
        num_nodes=data["num_nodes"],
        max_neighbors=10,
        split='train'
    )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for debugging
        collate_fn=dataset.collate_fn
    )
    
    return loader

def run_simple_test():
    """Run a simple test to isolate the issue"""
    
    print("=== Running Simple Test ===")
    
    # Load data
    data = load_wikipedia_fixed()
    
    # Initialize model
    model = DyGFormer(
        num_nodes=data["num_nodes"],
        node_features=data["node_raw_features"].shape[1],
        hidden_dim=64,
        time_encoding_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.1,
        max_neighbors=10,
        patch_size=1,
        max_sequence_length=64,
        channel_embedding_dim=32,
        neighbor_co_occurrence=True,
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    
    # Set raw features
    model.set_raw_features(
        torch.from_numpy(data["node_raw_features"]),
        torch.from_numpy(data["edge_raw_features"])
    )
    
    # Create data loader
    train_loader = create_simple_data_loaders(data, batch_size=16)
    
    print(f"\nCreated data loader with {len(train_loader)} batches")
    
    # Test one batch
    for batch_idx, batch in enumerate(train_loader):
        print(f"\n=== Testing Batch {batch_idx} ===")
        print(f"Batch keys: {list(batch.keys())}")
        
        # Print batch statistics
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        
        # Test forward pass
        try:
            print("\nTesting forward pass...")
            with torch.no_grad():
                model.eval()
                output = model(batch)
                print(f"✓ Forward pass successful!")
                print(f"Output shape: {output.shape}")
                
                # Test loss
                loss = model._compute_loss(batch)
                print(f"Loss: {loss.item():.4f}")
                
                # Test metrics
                metrics = model._compute_metrics(batch)
                print("Metrics:")
                for k, v in metrics.items():
                    if torch.is_tensor(v):
                        print(f"  {k}: {v.item():.4f}")
                
                break  # Only test first batch
                
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    # Set up logging
    logger.add("debug.log", level="DEBUG")
    
    # Run test
    run_simple_test()
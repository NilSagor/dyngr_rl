# minimal_test.py (place in project root: /home/nilsagor/Documents/haiyang_exp/)
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiment_framework"))

print(f"Current directory: {os.getcwd()}")

# Try to import DyGFormer
try:
    from src.models.dygformer import DyGFormer
    print("✓ Successfully imported DyGFormer")
except ImportError as e:
    print(f"✗ Import error: {e}")
    # Try to find the module
    import importlib.util
    spec = importlib.util.find_spec("src.models.dygformer")
    if spec is None:
        print("Module not found in path")
        sys.exit(1)

def test_model_creation():
    """Test if we can create the model without data"""
    print("\n=== Testing Model Creation ===")
    
    # Minimal model with small dimensions
    model = DyGFormer(
        num_nodes=100,
        node_features=10,
        hidden_dim=32,
        time_encoding_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.1,
        max_neighbors=5,
        patch_size=1,
        max_sequence_length=16,
        channel_embedding_dim=16,
        neighbor_co_occurrence=False  # Disable for minimal test
    )
    
    print(f"✓ Model created")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set dummy raw features
    model.set_raw_features(
        torch.randn(101, 10),  # node_raw_features
        torch.randn(101, 10)   # edge_raw_features
    )
    
    return model

def test_forward_pass(model):
    """Test forward pass with minimal data"""
    print("\n=== Testing Forward Pass ===")
    
    # Create minimal batch
    batch = {
        'src_nodes': torch.tensor([1, 2, 3, 4], dtype=torch.long),
        'dst_nodes': torch.tensor([5, 6, 7, 8], dtype=torch.long),
        'timestamps': torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float32),
        'labels': torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        'src_neighbors': torch.randint(0, 100, (4, 5), dtype=torch.long),
        'dst_neighbors': torch.randint(0, 100, (4, 5), dtype=torch.long),
        'src_neighbor_times': torch.rand(4, 5) * 1000,
        'dst_neighbor_times': torch.rand(4, 5) * 1000,
    }
    
    print(f"Batch created with keys: {list(batch.keys())}")
    
    try:
        with torch.no_grad():
            model.eval()
            output = model(batch)
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading without model"""
    print("\n=== Testing Data Loading ===")
    
    data_path = "experiment_framework/data/processed/wikipedia"
    
    files_to_check = [
        "ml_wikipedia.csv",
        "ml_wikipedia.npy", 
        "ml_wikipedia_node.npy"
    ]
    
    for file in files_to_check:
        full_path = os.path.join(data_path, file)
        if os.path.exists(full_path):
            print(f"✓ {file} exists")
            try:
                if file.endswith('.npy'):
                    data = np.load(full_path)
                    print(f"  Shape: {data.shape}, Dtype: {data.dtype}")
                    
                    # Check for inf/nan
                    if np.any(np.isnan(data)):
                        print(f"  WARNING: Contains NaN values")
                    if np.any(np.isinf(data)):
                        print(f"  WARNING: Contains Inf values")
                    
                    # Check memory size
                    size_mb = data.nbytes / (1024 * 1024)
                    print(f"  Size: {size_mb:.2f} MB")
                    
                elif file.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(full_path)
                    print(f"  Shape: {df.shape}")
                    print(f"  Columns: {df.columns.tolist()}")
                    
            except Exception as e:
                print(f"  ERROR loading {file}: {e}")
        else:
            print(f"✗ {file} not found")

def test_memory_allocation_directly():
    """Test memory allocation directly"""
    print("\n=== Direct Memory Allocation Test ===")
    
    # Test the sizes that might be problematic
    test_shapes = [
        (9229, 172),  # node_raw_features
        (157475, 172),  # edge_raw_features
        (4, 256, 172),  # batch x seq x feat
        (4, 16, 4, 32),  # batch x patches x channels x dim
    ]
    
    for shape in test_shapes:
        try:
            if len(shape) == 2:
                tensor = torch.randn(shape, dtype=torch.float32)
            elif len(shape) == 3:
                tensor = torch.randn(shape, dtype=torch.float32)
            elif len(shape) == 4:
                tensor = torch.randn(shape, dtype=torch.float32)
                
            print(f"✓ Allocated shape {shape}: {tensor.numel():,} elements")
            del tensor
        except Exception as e:
            print(f"✗ Failed to allocate shape {shape}: {e}")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # Test 1: Data loading
    test_data_loading()
    
    # Test 2: Direct memory allocation
    test_memory_allocation_directly()
    
    # Test 3: Model creation (skip if previous tests fail)
    try:
        model = test_model_creation()
        
        # Test 4: Forward pass
        success = test_forward_pass(model)
        
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Forward pass failed")
    except Exception as e:
        print(f"\n❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
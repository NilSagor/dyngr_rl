# experiment_framework/src/experiments/debug_fixed.py
import torch
import sys
import os

# Fix the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

# Import the fixed model
from src.models.dygformer_fixed import DyGFormerFixed

def test_model_safely():
    print("\n=== Testing DyGFormerFixed Safely ===")
    
    try:
        # Create minimal model
        model = DyGFormerFixed(
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
            neighbor_co_occurrence=False
        )
        
        print(f"\n✓ Model created successfully")
        
        # Set raw features
        model.set_raw_features(
            torch.randn(101, 10),
            torch.randn(101, 10)
        )
        
        # Create test batch
        batch = {
            'src_nodes': torch.tensor([1, 2, 3, 4]),
            'dst_nodes': torch.tensor([5, 6, 7, 8]),
            'timestamps': torch.tensor([100.0, 200.0, 300.0, 400.0]),
            'labels': torch.tensor([1.0, 0.0, 1.0, 0.0]),
            'src_neighbors': torch.randint(0, 100, (4, 5)),
            'dst_neighbors': torch.randint(0, 100, (4, 5)),
            'src_neighbor_times': torch.rand(4, 5) * 1000,
            'dst_neighbor_times': torch.rand(4, 5) * 1000,
        }
        
        # Test forward pass
        with torch.no_grad():
            output = model(batch)
            print(f"\n✓ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # Test loss
        loss = model._compute_loss(batch)
        print(f"\n✓ Loss computation successful!")
        print(f"  Loss: {loss.item():.4f}")
        
        # Test metrics
        metrics = model._compute_metrics(batch)
        print(f"\n✓ Metrics computation successful!")
        for key, value in metrics.items():
            print(f"  {key}: {value.item() if torch.is_tensor(value) else value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    success = test_model_safely()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed")
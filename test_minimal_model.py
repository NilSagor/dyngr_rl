# test_minimal_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalModel(nn.Module):
    """Minimal model that replicates your structure but simpler"""
    def __init__(self):
        super().__init__()
        print("Initializing MinimalModel...")
        
        # Test register_buffer with large tensors
        try:
            # Try the size that might be causing issues
            self.register_buffer('node_raw_features', torch.zeros(10000, 172))
            print(f"✓ Registered node_raw_features: (10000, 172)")
        except Exception as e:
            print(f"✗ Failed to register node_raw_features: {e}")
            # Try smaller
            self.register_buffer('node_raw_features', torch.zeros(1000, 100))
        
        try:
            self.register_buffer('edge_raw_features', torch.zeros(1000, 172))
            print(f"✓ Registered edge_raw_features: (1000, 172)")
        except Exception as e:
            print(f"✗ Failed to register edge_raw_features: {e}")
            self.register_buffer('edge_raw_features', torch.zeros(100, 100))
        
        # Simple layers
        self.embedding = nn.Embedding(1000, 128)
        self.linear = nn.Linear(128, 64)
        
        print(f"✓ MinimalModel initialized")
    
    def forward(self, x):
        return self.linear(self.embedding(x))

# Test
print("Testing minimal model...")
try:
    model = MinimalModel()
    print(f"✓ Model created")
    
    # Test forward
    x = torch.tensor([1, 2, 3, 4])
    y = model(x)
    print(f"✓ Forward pass: {x.shape} -> {y.shape}")
    
    print("\n✅ Minimal model test passed!")
except Exception as e:
    print(f"\n❌ Minimal model test failed: {e}")
    import traceback
    traceback.print_exc()
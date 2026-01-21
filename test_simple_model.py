# test_simple_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing SimpleModel...")
        
        # Similar components to your DyGFormer but minimal
        self.embedding = nn.Embedding(1000, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        
        # Try to allocate tensors similar to your model
        self.register_buffer('dummy_features', torch.zeros(1001, 172))
        
        print("✓ SimpleModel initialized")
    
    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Test
print("Testing simple model...")
try:
    model = SimpleModel()
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    input = torch.tensor([1, 2, 3, 4])
    output = model(input)
    print(f"✓ Forward pass: {input.shape} -> {output.shape}")
    
    print("\n✅ Simple model test passed!")
except Exception as e:
    print(f"\n❌ Simple model test failed: {e}")
    import traceback
    traceback.print_exc()
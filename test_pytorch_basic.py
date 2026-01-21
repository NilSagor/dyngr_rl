# test_pytorch_large.py
import torch
import sys

def test_large_allocation():
    print("Testing large tensor allocations...")
    
    # Test sizes similar to your model
    test_cases = [
        (100, 172),       # Small node features
        (1000, 172),      # Medium node features  
        (10000, 172),     # Large node features
        (4, 256, 172),    # Batch x seq x feat
        (4, 256, 4, 32),  # Batch x patches x channels x dim
    ]
    
    for i, shape in enumerate(test_cases):
        try:
            print(f"\nTest {i+1}: shape {shape}")
            if len(shape) == 2:
                tensor = torch.empty(shape, dtype=torch.float32)
            elif len(shape) == 3:
                tensor = torch.empty(shape, dtype=torch.float32)
            elif len(shape) == 4:
                tensor = torch.empty(shape, dtype=torch.float32)
            
            print(f"  ✓ Allocated: {tensor.numel():,} elements")
            print(f"  ✓ Memory: {tensor.numel() * 4 / 1024 / 1024:.2f} MB")
            del tensor
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    success = test_large_allocation()
    if success:
        print("\n✅ All large allocations successful!")
    else:
        print("\n❌ Large allocation test failed!")
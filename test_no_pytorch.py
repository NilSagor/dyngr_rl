# test_no_pytorch.py
import sys
import os
print(f"Python: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Test basic Python
print("\n=== Basic Python Test ===")
try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
    
    # Test array creation
    arr = np.zeros((10000, 172))
    print(f"Created NumPy array: {arr.shape}")
    
    # Test large array
    large_arr = np.zeros((100000, 172))
    print(f"Created large NumPy array: {large_arr.shape}")
    
    print("✓ NumPy OK")
except Exception as e:
    print(f"✗ NumPy failed: {e}")

# Test PyTorch without NN
print("\n=== PyTorch (No NN) Test ===")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    
    # Test tensor creation
    x = torch.zeros((10000, 172))
    print(f"Created tensor: {x.shape}")
    
    # Test operations
    y = x + 1
    print(f"Tensor operations OK")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print(f"CUDA available, testing...")
        x_cuda = x.cuda()
        print(f"Moved to CUDA: {x_cuda.shape}")
        x_cpu = x_cuda.cpu()
        print(f"Moved back to CPU: {x_cpu.shape}")
    
    print("✓ PyTorch tensor operations OK")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")
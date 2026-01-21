# test_with_different_backend.py
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

# Try MPS if on Mac
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

# Force CPU
torch.set_default_device('cpu')
torch.set_default_dtype(torch.float32)

print("\nTesting with CPU backend...")
try:
    # Test simple model
    model = torch.nn.Linear(10, 5)
    x = torch.randn(4, 10)
    y = model(x)
    print(f"✓ CPU test passed: {x.shape} -> {y.shape}")
except Exception as e:
    print(f"✗ CPU test failed: {e}")
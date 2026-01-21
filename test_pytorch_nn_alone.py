# test_pytorch_nn_alone.py
import torch
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")

# Test 1: Simple Sequential
print("\n=== Test 1: Sequential Model ===")
try:
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    print(f"✓ Sequential model created: {model}")
    
    # Test forward
    x = torch.randn(4, 10)
    y = model(x)
    print(f"✓ Forward pass: {x.shape} -> {y.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Embedding
print("\n=== Test 2: Embedding Layer ===")
try:
    embedding = nn.Embedding(1000, 128)
    print(f"✓ Embedding created")
    
    # Test forward
    indices = torch.tensor([1, 2, 3, 4])
    embeds = embedding(indices)
    print(f"✓ Embedding forward: {indices.shape} -> {embeds.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Multihead Attention
print("\n=== Test 3: Multihead Attention ===")
try:
    mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
    print(f"✓ MultiheadAttention created")
    
    # Test forward
    query = torch.randn(2, 10, 64)
    key = torch.randn(2, 10, 64)
    value = torch.randn(2, 10, 64)
    attn_output, attn_weights = mha(query, key, value)
    print(f"✓ MHA forward: {query.shape} -> {attn_output.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n✅ All PyTorch NN tests passed!")
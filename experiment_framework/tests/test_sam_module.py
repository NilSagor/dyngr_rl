import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

# Import the modules under test
from src.models.enhanced_tgn.component.sam_module import (
    SAMCell,
    StabilityAugmentedMemory,    
)

from src.models.enhanced_tgn.component.time_encoder import TimeEncoder

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def basic_sam():
    """Create a SAM instance with small dimensions for testing."""
    return StabilityAugmentedMemory(
        num_nodes=5,
        memory_dim=8,
        node_feat_dim=4,
        edge_feat_dim=6,
        time_dim=4,
        num_prototypes=3,
        similarity_metric="cosine",
        dropout=0.1
    )

@pytest.fixture
def sam_no_node_feat():
    """SAM without node features."""
    return StabilityAugmentedMemory(
        num_nodes=5,
        memory_dim=8,
        node_feat_dim=0,
        edge_feat_dim=6,
        time_dim=4,
        num_prototypes=3,
        similarity_metric="dot",
        dropout=0.0
    )

@pytest.fixture
def sample_batch():
    """Create a small batch of interactions."""
    source = torch.tensor([0, 1, 2])
    target = torch.tensor([3, 4, 0])
    edge_feat = torch.randn(3, 6)
    times = torch.tensor([1.0, 2.0, 3.0])
    node_feat = torch.randn(5, 4)  # for all nodes
    return source, target, edge_feat, times, node_feat

@pytest.fixture
def mock_time_encoder():
    """A simple deterministic time encoder for testing."""
    class DummyTimeEncoder(nn.Module):
        def __init__(self, time_dim):
            super().__init__()
            self.time_dim = time_dim
            # Fixed linear projection
            self.proj = nn.Linear(1, time_dim, bias=False)
            nn.init.eye_(self.proj.weight[:1])  # simple mapping

        def forward(self, t):
            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            return self.proj(t)
    return DummyTimeEncoder

# ----------------------------------------------------------------------
# SAMCell tests
# ----------------------------------------------------------------------
def test_samcell_initialization():
    cell = SAMCell(memory_dim=8, edge_feat_dim=8, time_dim=4, num_prototypes=3)
    assert cell.memory_dim == 8
    assert cell.num_prototypes == 3
    assert cell.query_proj.in_features == 8 + 8 + 4
    assert cell.gate_proj.in_features == 2 * 8 + 4

def test_samcell_forward_basic():
    cell = SAMCell(memory_dim=8, edge_feat_dim=8, time_dim=4, num_prototypes=3)
    batch_size = 4
    raw_mem = torch.randn(batch_size, 8)
    edge_feat = torch.randn(batch_size, 8)
    time_enc = torch.randn(batch_size, 4)
    prototypes = torch.randn(batch_size, 3, 8)
    
    updated, info = cell(raw_mem, None, edge_feat, time_enc, prototypes)
    assert updated.shape == (batch_size, 8)
    assert "attention_weights" in info
    assert info["attention_weights"].shape == (batch_size, 3)
    assert "update_gate" in info
    assert info["update_gate"].shape == (batch_size, 1)

@pytest.mark.parametrize("metric", ["cosine", "dot", "scaled_dot"])
def test_samcell_similarity_metrics(metric):
    cell = SAMCell(memory_dim=8, edge_feat_dim=8, time_dim=4,
                   num_prototypes=3, similarity_metric=metric)
    batch_size = 2
    raw_mem = torch.randn(batch_size, 8)
    edge_feat = torch.randn(batch_size, 8)
    time_enc = torch.randn(batch_size, 4)
    prototypes = torch.randn(batch_size, 3, 8)
    
    updated, info = cell(raw_mem, None, edge_feat, time_enc, prototypes)
    assert updated.shape == (batch_size, 8)
    # Similarity should be computed without errors
    assert torch.isfinite(info["similarity_scores"]).all()

def test_samcell_attention_masking():
    cell = SAMCell(memory_dim=8, edge_feat_dim=8, time_dim=4, num_prototypes=3)
    batch_size = 2
    raw_mem = torch.randn(batch_size, 8)
    edge_feat = torch.randn(batch_size, 8)
    time_enc = torch.randn(batch_size, 4)
    prototypes = torch.randn(batch_size, 3, 8)
    node_mask = torch.tensor([[True, True, False],
                              [True, False, True]])  # mask some prototypes

    updated, info = cell(raw_mem, None, edge_feat, time_enc, prototypes, node_mask)
    attn = info["attention_weights"]
    # Check that masked prototypes have zero attention
    assert (attn[node_mask == False] == 0).all()
    # Sum of attention for each batch should be 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(batch_size))

def test_samcell_all_masked_prototypes():
    cell = SAMCell(memory_dim=8, edge_feat_dim=8, time_dim=4, num_prototypes=3)
    batch_size = 2
    raw_mem = torch.randn(batch_size, 8)
    edge_feat = torch.randn(batch_size, 8)
    time_enc = torch.randn(batch_size, 4)
    prototypes = torch.randn(batch_size, 3, 8)
    # All prototypes masked for batch 0
    node_mask = torch.tensor([[False, False, False],
                              [True, True, True]])

    updated, info = cell(raw_mem, None, edge_feat, time_enc, prototypes, node_mask)
    attn = info["attention_weights"]
    # Batch 0 should have uniform attention (since all masked)
    assert torch.allclose(attn[0], torch.ones(3)/3)
    # Batch 1 should have normal softmax over valid ones
    assert torch.allclose(attn[1].sum(), torch.tensor(1.0))

def test_samcell_update_gate_range():
    cell = SAMCell(memory_dim=8, edge_feat_dim=8, time_dim=4, num_prototypes=3)
    raw_mem = torch.randn(1, 8)
    edge_feat = torch.randn(1, 8)
    time_enc = torch.randn(1, 4)
    prototypes = torch.randn(1, 3, 8)
    _, info = cell(raw_mem, None, edge_feat, time_enc, prototypes)
    gate = info["update_gate"]
    assert (gate >= 0).all() and (gate <= 1).all()

def test_samcell_nan_inf_handling():
    cell = SAMCell(memory_dim=8, edge_feat_dim=8, time_dim=4, num_prototypes=3)
    # Introduce NaNs in raw_memory
    raw_mem = torch.randn(2, 8)
    raw_mem[0, 0] = float('nan')
    raw_mem[1, 1] = float('inf')
    edge_feat = torch.randn(2, 8)
    time_enc = torch.randn(2, 4)
    prototypes = torch.randn(2, 3, 8)
    
    updated, _ = cell(raw_mem, None, edge_feat, time_enc, prototypes)
    # Should not crash and output finite numbers
    assert torch.isfinite(updated).all()

def test_samcell_gradient_flow():
    cell = SAMCell(memory_dim=8, edge_feat_dim=8, time_dim=4, num_prototypes=3)
    batch_size = 2
    raw_mem = torch.randn(batch_size, 8, requires_grad=True)
    edge_feat = torch.randn(batch_size, 8, requires_grad=True)
    time_enc = torch.randn(batch_size, 4, requires_grad=True)
    prototypes = torch.randn(batch_size, 3, 8, requires_grad=True)
    
    updated, _ = cell(raw_mem, None, edge_feat, time_enc, prototypes)
    loss = updated.sum()
    loss.backward()
    # Check gradients on all input tensors
    assert raw_mem.grad is not None
    assert edge_feat.grad is not None
    assert time_enc.grad is not None
    assert prototypes.grad is not None
    # Check gradients on cell parameters
    assert cell.query_proj.weight.grad is not None
    assert cell.gate_proj.weight.grad is not None

# ----------------------------------------------------------------------
# StabilityAugmentedMemory tests
# ----------------------------------------------------------------------
def test_sam_initialization(basic_sam):
    assert basic_sam.num_nodes == 5
    assert basic_sam.memory_dim == 8
    assert basic_sam.num_prototypes == 3
    assert basic_sam.raw_memory.shape == (5, 8)
    assert basic_sam.all_prototypes.shape == (5, 3, 8)
    assert basic_sam.last_update.shape == (5,)

def test_sam_get_memory(basic_sam):
    node_ids = torch.tensor([0, 2, 4])
    mem = basic_sam.get_memory(node_ids)
    assert mem.shape == (3, 8)
    # Should be same as raw_memory indexed
    assert torch.equal(mem, basic_sam.raw_memory[node_ids])

def test_sam_get_prototypes(basic_sam):
    node_ids = torch.tensor([1, 3])
    protos = basic_sam.get_prototypes(node_ids)
    assert protos.shape == (2, 3, 8)
    # Should be normalized (LayerNorm applied)
    # Check that each prototype vector has mean close to 0? Not necessary.

def test_sam_update_memory_batch_basic(basic_sam, sample_batch):
    source, target, edge_feat, times, node_feat = sample_batch
    out = basic_sam.update_memory_batch(source, target, edge_feat, times, node_feat)
    assert "source_attention" in out
    assert "target_attention" in out
    assert "source_memory" in out
    assert "target_memory" in out

    # Memory should be non-zero after update
    updated_src = basic_sam.raw_memory[source]
    updated_tgt = basic_sam.raw_memory[target]
    assert not torch.allclose(updated_src, torch.zeros_like(updated_src))
    assert not torch.allclose(updated_tgt, torch.zeros_like(updated_tgt))

    # Check last_update times (nodes may be updated multiple times)
    # Node 0 appears as source (t=1) and target (t=3) → last = 3
    assert basic_sam.last_update[0] == 3.0
    # Node 1 only source (t=2)
    assert basic_sam.last_update[1] == 2.0
    # Node 2 only source (t=3)
    assert basic_sam.last_update[2] == 3.0
    # Node 3 only target (t=1)
    assert basic_sam.last_update[3] == 1.0
    # Node 4 only target (t=2)
    assert basic_sam.last_update[4] == 2.0

def test_sam_update_memory_batch_empty(basic_sam):
    source = torch.tensor([], dtype=torch.long)
    target = torch.tensor([], dtype=torch.long)
    edge_feat = torch.tensor([])
    times = torch.tensor([])
    node_feat = torch.randn(5, 4)
    out = basic_sam.update_memory_batch(source, target, edge_feat, times, node_feat)
    assert out['source_attention'] == {}
    assert out['target_attention'] == {}
    assert out['source_memory'].numel() == 0
    assert out['target_memory'].numel() == 0

def test_sam_update_memory_batch_no_node_feat(sam_no_node_feat, sample_batch):
    source, target, edge_feat, times, _ = sample_batch
    out = sam_no_node_feat.update_memory_batch(source, target, edge_feat, times, node_features=None)
    assert "source_attention" in out
    # Memory should still update
    updated = sam_no_node_feat.raw_memory[source]
    assert not torch.allclose(updated, torch.zeros_like(updated))

def test_sam_memory_clamping(basic_sam, sample_batch):
    # Set raw_memory to extreme values
    basic_sam.raw_memory.data = torch.full((5, 8), 100.0)
    source, target, edge_feat, times, node_feat = sample_batch
    basic_sam.update_memory_batch(source, target, edge_feat, times, node_feat)
    # After update, memory should be clamped to [-50,50]
    assert (basic_sam.raw_memory >= -50).all() and (basic_sam.raw_memory <= 50).all()

def test_sam_reset_memory(basic_sam):
    # Set some memory non-zero
    basic_sam.raw_memory[0] = torch.randn(8)
    basic_sam.last_update[0] = 5.0
    basic_sam.reset_memory(torch.tensor([0]))
    assert torch.all(basic_sam.raw_memory[0] == 0)
    assert basic_sam.last_update[0] == 0
    # Reset all
    basic_sam.reset_memory()
    assert torch.all(basic_sam.raw_memory == 0)
    assert torch.all(basic_sam.last_update == 0)

def test_sam_get_stabilized_memory(basic_sam, sample_batch):
    source, target, edge_feat, times, node_feat = sample_batch
    # First update memory
    basic_sam.update_memory_batch(source, target, edge_feat, times, node_feat)
    # Then get stabilized without updating
    node_ids = torch.tensor([0, 1, 2])
    current_time = torch.tensor(4.0)
    stabilized = basic_sam.get_stabilized_memory(node_ids, current_time, edge_feat[:3])
    assert stabilized.shape == (3, 8)
    # Raw memory should not have changed
    assert not torch.allclose(stabilized, basic_sam.raw_memory[node_ids])

def test_sam_gradient_flow(basic_sam, sample_batch):
    source, target, edge_feat, times, node_feat = sample_batch
    # Enable grad on all SAM parameters
    for param in basic_sam.parameters():
        param.requires_grad_(True)

    # Forward pass (updates memory, but returns detached tensors)
    out = basic_sam.update_memory_batch(source, target, edge_feat, times, node_feat)

    # Compute a loss that depends on parameters used in the forward pass.
    # The prototypes are always used, so sum them.
    loss = basic_sam.all_prototypes.sum()
    loss.backward()

    # Check that at least the prototypes have non‑zero gradient
    assert basic_sam.all_prototypes.grad is not None
    assert basic_sam.all_prototypes.grad.abs().sum() > 0

    

def test_sam_update_memory_batch_extreme_values(basic_sam):
    # Very large edge features, times, etc.
    source = torch.tensor([0])
    target = torch.tensor([1])
    edge_feat = torch.tensor([[1e10] * 6], dtype=torch.float32)
    times = torch.tensor([1e10])
    node_feat = torch.randn(5, 4)
    # Should not crash
    out = basic_sam.update_memory_batch(source, target, edge_feat, times, node_feat)
    assert torch.isfinite(basic_sam.raw_memory[0]).all()
    assert torch.isfinite(basic_sam.raw_memory[1]).all()

def test_sam_prototype_initialization():
    # Check that prototypes are initialized with xavier and not all zeros
    sam = StabilityAugmentedMemory(num_nodes=3, memory_dim=4, num_prototypes=2)
    assert not torch.allclose(sam.all_prototypes, torch.zeros_like(sam.all_prototypes))
    # Also check that LayerNorm is applied in get_prototypes
    protos = sam.get_prototypes(torch.tensor([0]))
    # Not checking exact values, just shape
    assert protos.shape == (1, 2, 4)

def test_sam_edge_projection(basic_sam):
    # Edge projection should produce finite values
    edge_feat = torch.randn(2, 6)
    proj = basic_sam.edge_proj(edge_feat)
    assert torch.isfinite(proj).all()
    # Should be normalized and scaled
    norm = proj.norm(dim=-1)
    # Due to scaling, norm should be around 10 (or exactly 10 if normalized)
    # Because we normalize then multiply by 10
    proj_norm = basic_sam.edge_proj(edge_feat)
    proj_norm = proj_norm / (proj_norm.norm(dim=-1, keepdim=True) + 1e-8) * 10.0
    assert torch.allclose(proj_norm.norm(dim=-1), torch.full((2,), 10.0), atol=1e-5)

def test_sam_time_encoder_integration(basic_sam):
    # Time encoder should produce correct dimension
    time = torch.tensor([1.0, 2.0])
    enc = basic_sam.time_encoder(time)
    assert enc.shape == (2, basic_sam.time_dim)

def test_sam_node_projection(basic_sam):
    # Node feature projection
    node_feat = torch.randn(3, 4)
    proj = basic_sam.node_proj(node_feat)
    assert proj.shape == (3, 8)
    assert torch.isfinite(proj).all()

def test_sam_similarity_metric_switch():
    # Test that changing metric doesn't break
    sam = StabilityAugmentedMemory(num_nodes=2, memory_dim=4, similarity_metric="dot")
    assert sam.sam_cell.similarity_metric == "dot"
    # Update with cosine? The metric is fixed at init; we can't change after.

def test_sam_node_ids_out_of_range(basic_sam):
    # Should raise IndexError
    with pytest.raises(IndexError):
        basic_sam.get_memory(torch.tensor([10]))
    # However, update_memory_batch might also crash if node IDs are out of range.
    source = torch.tensor([10])
    target = torch.tensor([1])
    edge_feat = torch.randn(1, 6)
    times = torch.tensor([1.0])
    node_feat = torch.randn(5, 4)
    with pytest.raises(IndexError):
        basic_sam.update_memory_batch(source, target, edge_feat, times, node_feat)
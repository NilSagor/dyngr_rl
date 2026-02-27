import pytest
import torch
import torch.nn as nn
from src.models.enhanced_tgn.component.co_transformer import (
    IntraWalkEncoder,
    CooccurrenceMatrix,
    InterWalkTransformer,
    HierarchicalCooccurrenceTransformer
)
from src.models.enhanced_tgn.component.multi_swalk import MultiScaleWalkSampler


# ----------------------------------------------------------------------
# IntraWalkEncoder tests
# ----------------------------------------------------------------------
def test_intra_walk_encoder_shapes():
    encoder = IntraWalkEncoder(d_model=64, num_layers=0)  # no transformer for simplicity
    batch, walks, length = 4, 3, 8
    emb = torch.randn(batch, walks, length, 64)
    mask = torch.ones(batch, walks, length).bool()
    mask[:, :, -2:] = 0

    encoded, summaries = encoder(emb, mask)
    assert encoded.shape == (batch, walks, length, 64)
    assert summaries.shape == (batch, walks, 64)

def test_intra_walk_encoder_pooling():
    # Use num_layers=0 to avoid transformation
    encoder = IntraWalkEncoder(d_model=64, num_layers=0)
    emb = torch.ones(2, 2, 5, 64) * 0.5
    mask = torch.ones(2, 2, 5).bool()
    _, summaries = encoder(emb, mask)

    # With constant input and no transformer, the summary should be the constant after projection and norm.
    # But norm layer shifts mean to 0, so we only check shape and non-zero.
    assert summaries.shape == (2, 2, 64)
    assert not torch.allclose(summaries, torch.zeros_like(summaries))

# ----------------------------------------------------------------------
# CooccurrenceMatrix tests
# ----------------------------------------------------------------------
def test_cooccurrence_matrix_naive_vs_sparse():
    torch.manual_seed(42)  # fixed seed for reproducibility
    B, W, L = 2, 3, 4
    nodes = torch.randint(1, 5, (B, W, L))
    mask = torch.randint(0, 2, (B, W, L)).bool()
    # Force some matches
    nodes[0, 0, 0] = 10
    nodes[0, 1, 2] = 10  # same node in different walks

    def naive_cooccurrence(nodes, mask, kernel):
        B, W, L = nodes.shape
        cooc = torch.zeros(B, W, W)
        for b in range(B):
            for r in range(W):
                for s in range(W):
                    match = (nodes[b, r] == nodes[b, s].unsqueeze(-1)) & mask[b, r].unsqueeze(-1) & mask[b, s].unsqueeze(-2)
                    val = (match.float() * kernel[:L, :L]).sum()
                    norm = (mask[b, r].sum() * mask[b, s].sum()).clamp_min(1e-8)
                    cooc[b, r, s] = val / norm
        return cooc

    kernel = torch.exp(-((torch.arange(L).float().unsqueeze(0) - torch.arange(L).float().unsqueeze(1))**2) / 2.0**2)
    mat = CooccurrenceMatrix(max_walk_length=L, sigma=2.0)
    mat.kernel = kernel  # inject for test

    sparse_result = mat(nodes, mask.float())
    naive_result = naive_cooccurrence(nodes, mask, kernel)

    # Print differences for debugging
    diff = (sparse_result - naive_result).abs()
    print("Max diff:", diff.max().item())
    print("Indices with diff > 1e-4:", torch.nonzero(diff > 1e-4))
    print("Sparse result:\n", sparse_result)
    print("Naive result:\n", naive_result)

    # Use a slightly larger tolerance if needed, but first check the print output
    torch.testing.assert_close(sparse_result, naive_result, rtol=1e-3, atol=1e-3)

def test_cooccurrence_matrix_edge_cases():
    mat = CooccurrenceMatrix(max_walk_length=5, sigma=2.0)
    # All masks zero -> co-occurrence should be zero
    nodes = torch.randint(0, 3, (1, 2, 5))
    mask = torch.zeros(1, 2, 5)
    result = mat(nodes, mask)
    assert (result == 0).all()

    # Single walk, single occurrence -> co-occurrence with itself should be 1.0
    nodes = torch.tensor([[[1, 0, 0, 0, 0]]])  # [1,1,5]
    mask = torch.tensor([[[1, 0, 0, 0, 0]]])
    result = mat(nodes, mask)
    assert result[0, 0, 0] == 1.0

# ----------------------------------------------------------------------
# InterWalkTransformer tests
# ----------------------------------------------------------------------
def test_inter_walk_attention_bias():
    d_model, nhead = 16, 2
    transformer = InterWalkTransformer(d_model=d_model, nhead=nhead, cooccurrence_gamma=0.5)
    B, W = 2, 3
    summaries = torch.randn(B, W, d_model)
    cooc = torch.rand(B, W, W)
    masks = torch.ones(B, W).bool()

    out = transformer(summaries, cooc, masks)
    assert out.shape == (B, W, d_model)

    loss = out.sum()
    loss.backward()
    assert transformer.gamma.grad is not None

# ----------------------------------------------------------------------
# HierarchicalCooccurrenceTransformer full forward test
# ----------------------------------------------------------------------
def test_hct_full_forward():
    hct = HierarchicalCooccurrenceTransformer(
        d_model=16,
        memory_dim=16,
        max_walk_length=5,
        max_num_walks=4
    )
    batch = 2
    walks_dict = {
        'short': {
            'nodes': torch.randint(0, 10, (batch, 2, 5)),
            'nodes_anon': torch.randint(1, 5, (batch, 2, 5)),
            'masks': torch.ones(batch, 2, 5).bool()
        },
        'long': {
            'nodes': torch.randint(0, 10, (batch, 2, 5)),
            'nodes_anon': torch.randint(1, 5, (batch, 2, 5)),
            'masks': torch.ones(batch, 2, 5).bool()
        },
        'tawr': {
            'nodes': torch.randint(0, 10, (batch, 2, 5)),
            'nodes_anon': torch.randint(1, 5, (batch, 2, 5)),
            'masks': torch.ones(batch, 2, 5).bool(),
            'restart_flags': torch.randint(0, 2, (batch, 2, 5))
        }
    }
    node_memory = torch.randn(10, 16, requires_grad=True)

    # Test with return_all=False
    fused = hct(walks_dict, node_memory)
    assert fused.shape == (batch, 16)

    # Test with return_all=True
    output_dict = hct(walks_dict, node_memory, return_all=True)
    assert 'fused' in output_dict
    for t in ['short', 'long', 'tawr']:
        assert t in output_dict
        assert 'encoded_walks' in output_dict[t]
        assert 'walk_summaries' in output_dict[t]
        assert 'refined_walks' in output_dict[t]
        assert 'cooccurrence' in output_dict[t]
        assert 'walk_masks' in output_dict[t]

    # Gradient check
    fused.sum().backward()
    assert node_memory.grad is not None

# ----------------------------------------------------------------------
# End-to-end integration test
# ----------------------------------------------------------------------
def test_end_to_end():
    num_nodes = 100
    sampler = MultiScaleWalkSampler(num_nodes=num_nodes)
    hct = HierarchicalCooccurrenceTransformer(memory_dim=128, d_model=64)
    # Create synthetic graph and memory
    edge_index = torch.randint(0, num_nodes, (2, 500))
    edge_time = torch.rand(500) * 10
    sampler.update_neighbors(edge_index, edge_time)
    memory = torch.randn(num_nodes, 128, requires_grad=True)
    # Dummy source/target
    src = torch.randint(0, num_nodes, (32,))
    tgt = torch.randint(0, num_nodes, (32,))
    times = torch.rand(32) * 10

    walks = sampler(src, tgt, times, memory)
    # HCT expects walks for a single node (e.g., source)
    out = hct(walks['source'], memory)
    loss = out.sum()
    loss.backward()
    assert memory.grad is not None
    print("Integration test passed")
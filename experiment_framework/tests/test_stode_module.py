import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# Import the modules under test (adjust import paths as needed)
from src.models.enhanced_tgn.component.stode_module import (
    GRUODECell,
    ODEFunc,
    SpectralRegularizer,
    STODEFunc,
    FlattenedSTODEFunc,
    STODEIntegrator,
    STODELayer,
    SpectralTemporalODE,
    STODEOutput,
    create_dummy_data,  # helper for consistent test data
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def hidden_dim():
    return 8

@pytest.fixture
def num_nodes():
    return 5

@pytest.fixture
def batch_size():
    return 3

@pytest.fixture
def time_precision():
    return 4

@pytest.fixture
def base_data(device, hidden_dim, num_nodes, time_precision):
    """Basic dummy data for all tests."""
    data = create_dummy_data(
        num_nodes=num_nodes,
        num_walks=2,
        walk_len=3,
        hidden_dim=hidden_dim,
        num_obs_times=2,
        device=device
    )
    return data

@pytest.fixture
def simple_hidden(device, hidden_dim, num_nodes):
    return torch.randn(num_nodes, hidden_dim, device=device)

@pytest.fixture
def simple_adj(device, num_nodes):
    adj = torch.eye(num_nodes, device=device)
    # add random edges
    rand = torch.rand(num_nodes, num_nodes, device=device) > 0.5
    adj = adj + rand.float()
    adj = (adj > 0).float()
    return adj

# ----------------------------------------------------------------------
# GRUODECell tests
# ----------------------------------------------------------------------

def test_gruodecell_init(hidden_dim):
    cell = GRUODECell(hidden_dim)
    assert cell.hidden_dim == hidden_dim
    assert hasattr(cell, 'W_update')
    assert hasattr(cell, 'W_reset')
    assert hasattr(cell, 'W_candidate')
    assert hasattr(cell, 'W_time')

def test_gruodecell_forward(device, hidden_dim):
    cell = GRUODECell(hidden_dim).to(device)
    batch_size = 4
    h = torch.randn(batch_size, hidden_dim, device=device)
    t = torch.tensor(1.5, device=device)
    dh = cell(t, h)
    assert dh.shape == (batch_size, hidden_dim)
    assert torch.isfinite(dh).all()

def test_gruodecell_batch_time(device, hidden_dim):
    cell = GRUODECell(hidden_dim).to(device)
    batch_size = 4
    h = torch.randn(batch_size, hidden_dim, device=device)
    t = torch.randn(batch_size, 1, device=device)  # different time per batch
    dh = cell(t, h)
    assert dh.shape == (batch_size, hidden_dim)

def test_gruodecell_nan_handling(device, hidden_dim):
    cell = GRUODECell(hidden_dim).to(device)
    h = torch.randn(2, hidden_dim, device=device)
    # Introduce NaN in input
    h[0, 0] = float('nan')
    # Should not crash, but output may have NaN
    t = torch.tensor(1.0, device=device)
    dh = cell(t, h)
    # The cell doesn't sanitize input, so NaN propagates
    assert torch.isnan(dh).any()

def test_gruodecell_gradient(device, hidden_dim):
    cell = GRUODECell(hidden_dim).to(device)
    h = torch.randn(2, hidden_dim, device=device, requires_grad=True)
    t = torch.tensor(1.0, device=device)
    dh = cell(t, h)
    loss = dh.sum()
    loss.backward()
    assert h.grad is not None
    assert h.grad.abs().sum() > 0
    for param in cell.parameters():
        assert param.grad is not None

# ----------------------------------------------------------------------
# ODEFunc tests
# ----------------------------------------------------------------------

def test_odefunc_init_gru(hidden_dim):
    func = ODEFunc(hidden_dim, use_gru_ode=True)
    assert isinstance(func.dynamics, GRUODECell)

def test_odefunc_init_mlp(hidden_dim):
    func = ODEFunc(hidden_dim, use_gru_ode=False, hidden_layers=2, activation='relu')
    assert isinstance(func.dynamics, nn.Sequential)
    # Input dim should be hidden_dim if time_invariant, else hidden_dim+1
    assert func.dynamics[0].in_features == hidden_dim + (0 if func.time_invariant else 1)

def test_odefunc_forward_gru(device, hidden_dim):
    func = ODEFunc(hidden_dim, use_gru_ode=True).to(device)
    h = torch.randn(3, hidden_dim, device=device)
    t = torch.tensor(2.0, device=device)
    dh = func(t, h)
    assert dh.shape == (3, hidden_dim)

def test_odefunc_forward_mlp_time_variant(device, hidden_dim):
    func = ODEFunc(hidden_dim, use_gru_ode=False, time_invariant=False).to(device)
    h = torch.randn(3, hidden_dim, device=device)
    t = torch.tensor(2.0, device=device)
    dh = func(t, h)
    assert dh.shape == (3, hidden_dim)

def test_odefunc_forward_mlp_time_invariant(device, hidden_dim):
    func = ODEFunc(hidden_dim, use_gru_ode=False, time_invariant=True).to(device)
    h = torch.randn(3, hidden_dim, device=device)
    t = torch.tensor(2.0, device=device)
    dh = func(t, h)
    assert dh.shape == (3, hidden_dim)

def test_odefunc_batch_time(device, hidden_dim):
    func = ODEFunc(hidden_dim, use_gru_ode=False, time_invariant=False).to(device)
    h = torch.randn(3, hidden_dim, device=device)
    t = torch.randn(3, 1, device=device)  # batch of times
    dh = func(t, h)
    assert dh.shape == (3, hidden_dim)

# ----------------------------------------------------------------------
# SpectralRegularizer tests
# ----------------------------------------------------------------------

def test_spectral_reg_init(hidden_dim):
    reg = SpectralRegularizer(hidden_dim, mu=0.1, num_eigenvectors=2, adaptive_mu=True)
    assert reg.mu.requires_grad if isinstance(reg.mu, nn.Parameter) else not reg.adaptive_mu

def test_spectral_reg_eigen_decomposition(device, hidden_dim, num_nodes, simple_adj):
    reg = SpectralRegularizer(hidden_dim, num_eigenvectors=2).to(device)
    vals, vecs = reg._eigen_decomposition(simple_adj)
    assert vals.shape == (2,)
    assert vecs.shape == (num_nodes, 2)
    # Test caching
    vals2, vecs2 = reg._eigen_decomposition(simple_adj)
    assert (vals == vals2).all() and (vecs == vecs2).all()

def test_spectral_reg_forward(device, hidden_dim, num_nodes, simple_adj):
    reg = SpectralRegularizer(hidden_dim, mu=0.5, num_eigenvectors=2).to(device)
    H = torch.randn(num_nodes, hidden_dim, device=device)
    force = reg(H, simple_adj)
    assert force.shape == (num_nodes, hidden_dim)

    # Create a new regularizer with mu=0
    reg_zero = SpectralRegularizer(hidden_dim, mu=0.0, num_eigenvectors=2).to(device)
    force_zero = reg_zero(H, simple_adj)
    assert torch.allclose(force_zero, torch.zeros_like(force_zero))

def test_spectral_reg_dirichlet_energy(device, hidden_dim, num_nodes, simple_adj):
    reg = SpectralRegularizer(hidden_dim, num_eigenvectors=2).to(device)
    H = torch.randn(num_nodes, hidden_dim, device=device)
    energy = reg.dirichlet_energy(H, simple_adj)
    assert energy.dim() == 0  # scalar
    assert torch.isfinite(energy)

def test_spectral_reg_gradient(device, hidden_dim, num_nodes, simple_adj):
    reg = SpectralRegularizer(hidden_dim, mu=0.1, adaptive_mu=True).to(device)
    H = torch.randn(num_nodes, hidden_dim, device=device, requires_grad=True)
    force = reg(H, simple_adj)
    loss = force.sum()
    loss.backward()
    assert H.grad is not None
    if isinstance(reg.mu, nn.Parameter):
        assert reg.mu.grad is not None

# ----------------------------------------------------------------------
# STODEFunc tests
# ----------------------------------------------------------------------

def test_stodefunc_init(device, hidden_dim):
    odefunc = ODEFunc(hidden_dim).to(device)
    spectral_reg = SpectralRegularizer(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, spectral_reg)
    assert stode.hidden_dim == hidden_dim

def test_stodefunc_forward(device, hidden_dim, num_nodes, simple_adj):
    odefunc = ODEFunc(hidden_dim).to(device)
    spectral_reg = SpectralRegularizer(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, spectral_reg).to(device)
    H = torch.randn(num_nodes, hidden_dim, device=device)
    t = torch.tensor(1.0, device=device)
    dh, info = stode(t, H, simple_adj, return_diagnostics=True)
    assert dh.shape == (num_nodes, hidden_dim)
    assert 'spectral_force_norm' in info
    assert 'total_dhdt_norm' in info

def test_stodefunc_no_spectral(device, hidden_dim, num_nodes):
    odefunc = ODEFunc(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, None).to(device)
    H = torch.randn(num_nodes, hidden_dim, device=device)
    t = torch.tensor(1.0, device=device)
    dh = stode(t, H)
    assert dh.shape == (num_nodes, hidden_dim)

# ----------------------------------------------------------------------
# FlattenedSTODEFunc tests
# ----------------------------------------------------------------------

def test_flattened_init(device, hidden_dim):
    odefunc = ODEFunc(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, None).to(device)
    flat = FlattenedSTODEFunc(stode)
    assert flat.core is stode
    assert flat._num_nodes == 0

def test_flattened_set_adj(device, hidden_dim, num_nodes, simple_adj):
    odefunc = ODEFunc(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, None).to(device)
    flat = FlattenedSTODEFunc(stode)
    flat.set_adj_matrix(simple_adj)
    assert flat._num_nodes == num_nodes
    assert torch.equal(flat._adj, simple_adj)

def test_flattened_forward(device, hidden_dim, num_nodes, simple_adj):
    odefunc = ODEFunc(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, None).to(device)
    flat = FlattenedSTODEFunc(stode).to(device)
    flat.set_adj_matrix(simple_adj)
    H = torch.randn(num_nodes, hidden_dim, device=device)
    flat_state = H.view(-1)
    t = torch.tensor(1.0, device=device)
    dh_flat = flat(t, flat_state)
    assert dh_flat.shape == (num_nodes * hidden_dim,)
    dh = stode(t, H)
    assert torch.allclose(dh_flat.view_as(dh), dh)

# ----------------------------------------------------------------------
# STODEIntegrator tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ['euler', 'rk4', 'dopri5', 'dopri8'])
def test_integrator_methods(device, hidden_dim, num_nodes, simple_adj, method):
    odefunc = ODEFunc(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, None).to(device)
    flat = FlattenedSTODEFunc(stode).to(device)
    integrator = STODEIntegrator(flat, method=method, step_size=0.1 if method in STODEIntegrator.FIXED_STEP else None)
    H0 = torch.randn(num_nodes, hidden_dim, device=device)
    t0 = torch.tensor(0.0, device=device)
    t1 = torch.tensor(1.0, device=device)
    H1 = integrator(H0, t0, t1, simple_adj)
    assert H1.shape == (num_nodes, hidden_dim)
    assert torch.isfinite(H1).all()

def test_integrator_adjoint(device, hidden_dim, num_nodes, simple_adj):
    odefunc = ODEFunc(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, None).to(device)
    flat = FlattenedSTODEFunc(stode).to(device)
    integrator = STODEIntegrator(flat, method='dopri5', adjoint=True)
    H0 = torch.randn(num_nodes, hidden_dim, device=device, requires_grad=True)
    t0 = torch.tensor(0.0, device=device)
    t1 = torch.tensor(1.0, device=device)
    H1 = integrator(H0, t0, t1, simple_adj)
    loss = H1.sum()
    loss.backward()
    assert H0.grad is not None

def test_integrator_step_size_validation(device, hidden_dim, num_nodes, simple_adj):
    odefunc = ODEFunc(hidden_dim).to(device)
    stode = STODEFunc(hidden_dim, odefunc, None).to(device)
    flat = FlattenedSTODEFunc(stode).to(device)
    # For fixed-step, step_size must be provided (or defaults)
    integrator = STODEIntegrator(flat, method='euler')  # default step_size 0.1
    H0 = torch.randn(num_nodes, hidden_dim, device=device)
    t0, t1 = torch.tensor(0.0), torch.tensor(1.0)
    H1 = integrator(H0, t0, t1, simple_adj)
    assert H1.shape == (num_nodes, hidden_dim)

# ----------------------------------------------------------------------
# STODELayer tests
# ----------------------------------------------------------------------

def test_stodelayer_init(device, hidden_dim, num_nodes):
    layer = STODELayer(hidden_dim, num_nodes).to(device)
    assert layer.hidden_dim == hidden_dim
    assert layer.num_nodes == num_nodes

def test_stodelayer_forward_simple(device, hidden_dim, num_nodes, base_data):
    layer = STODELayer(hidden_dim, num_nodes, update_type='gru').to(device)
    H_init = base_data['node_states']
    t_init = base_data['t_init']
    obs = []
    for i in range(len(base_data['adj_matrices'])):
        # Create a dummy observation at that time (e.g., random)
        H_obs = torch.randn(num_nodes, hidden_dim, device=device)
        t_obs = torch.tensor(base_data['walk_times'][0,0,i].item(), device=device)
        obs.append((H_obs, t_obs))
    adj = base_data['adj_matrices']
    output = layer(H_init, t_init, obs, adj)
    assert isinstance(output, STODEOutput)
    assert output.final_state.shape == (num_nodes, hidden_dim)
    assert output.num_observations == len(obs)

def test_stodelayer_update_types(device, hidden_dim, num_nodes):
    H_init = torch.randn(num_nodes, hidden_dim, device=device)
    t_init = torch.tensor(0.0, device=device)
    # Dummy observation at t=1.0
    H_obs = torch.randn(num_nodes, hidden_dim, device=device)
    t_obs = torch.tensor(1.0, device=device)
    obs = [(H_obs, t_obs)]
    adj = [torch.eye(num_nodes, device=device)]
    for update_type in ['gru', 'mlp', 'residual']:
        layer = STODELayer(hidden_dim, num_nodes, update_type=update_type).to(device)
        out = layer(H_init, t_init, obs, adj)
        assert out.final_state.shape == (num_nodes, hidden_dim)

def test_stodelayer_no_observations(device, hidden_dim, num_nodes):
    layer = STODELayer(hidden_dim, num_nodes).to(device)
    H_init = torch.randn(num_nodes, hidden_dim, device=device)
    t_init = torch.tensor(0.0, device=device)
    out = layer(H_init, t_init, [], [])
    assert torch.equal(out.final_state, H_init)
    assert out.num_observations == 0

def test_stodelayer_observation_ordering(device, hidden_dim, num_nodes):
    layer = STODELayer(hidden_dim, num_nodes).to(device)
    H_init = torch.randn(num_nodes, hidden_dim, device=device)
    t_init = torch.tensor(0.0, device=device)
    # Observations out of order
    obs = [
        (torch.randn(num_nodes, hidden_dim), torch.tensor(2.0)),
        (torch.randn(num_nodes, hidden_dim), torch.tensor(1.0))
    ]
    adj = [torch.eye(num_nodes, device=device)] * 2
    with pytest.raises(ValueError, match="Non-increasing times"):
        layer(H_init, t_init, obs, adj)

# ----------------------------------------------------------------------
# SpectralTemporalODE tests
# ----------------------------------------------------------------------

def test_stode_full_forward(device, base_data):
    model = SpectralTemporalODE(
        hidden_dim=base_data['hidden_dim'],
        num_nodes=base_data['num_nodes'],
        num_layers=2,
        ode_method='euler',
        ode_step_size=0.1
    ).to(device)
    output = model(
        node_states=base_data['node_states'],
        walk_encodings=base_data['walk_encodings'],
        walk_times=base_data['walk_times'],
        walk_masks=base_data['walk_masks'],
        adj_matrices=base_data['adj_matrices'],
        t_init=base_data['t_init'],
        return_all=True
    )
    assert isinstance(output, STODEOutput)
    assert output.final_state.shape == (base_data['num_nodes'], base_data['hidden_dim'])
    assert output.trajectory is not None
    assert len(output.trajectory) == 2
    assert output.num_observations == base_data['num_obs_times']

def test_stode_prepare_observations(device, base_data):
    model = SpectralTemporalODE(
        hidden_dim=base_data['hidden_dim'],
        num_nodes=base_data['num_nodes']
    ).to(device)
    obs = model._vectorized_prepare_observations(
        base_data['walk_encodings'],
        base_data['walk_times'],
        base_data['walk_masks']
    )
    # Should produce num_obs_times observations
    assert len(obs) == base_data['num_obs_times']
    for H, t in obs:
        assert H.shape == (base_data['num_nodes'], base_data['hidden_dim'])
        assert t.dim() == 0
    # Times should be sorted
    times = [t.item() for _, t in obs]
    assert times == sorted(times)

def test_stode_empty_observations(device, base_data):
    model = SpectralTemporalODE(
        hidden_dim=base_data['hidden_dim'],
        num_nodes=base_data['num_nodes']
    ).to(device)
    zero_masks = torch.zeros_like(base_data['walk_masks'])
    output = model(
        node_states=base_data['node_states'],
        walk_encodings=base_data['walk_encodings'],
        walk_times=base_data['walk_times'],
        walk_masks=zero_masks,
        adj_matrices=[],  # empty list
        t_init=base_data['t_init']
    )
    assert output.num_observations == 0
    # Should still produce final state via input projection
    assert output.final_state.shape == (base_data['num_nodes'], base_data['hidden_dim'])

def test_stode_mismatched_adj_length(device, base_data):
    model = SpectralTemporalODE(
        hidden_dim=base_data['hidden_dim'],
        num_nodes=base_data['num_nodes']
    ).to(device)
    with pytest.raises(ValueError, match="adj_matrices"):
        model(
            node_states=base_data['node_states'],
            walk_encodings=base_data['walk_encodings'],
            walk_times=base_data['walk_times'],
            walk_masks=base_data['walk_masks'],
            adj_matrices=base_data['adj_matrices'][:-1],  # one short
            t_init=base_data['t_init']
        )

def test_stode_checkpointing(device, base_data):
    model = SpectralTemporalODE(
        hidden_dim=base_data['hidden_dim'],
        num_nodes=base_data['num_nodes'],
        num_layers=2,
        use_checkpoint=True
    ).to(device)
    # Just ensure it runs without error
    output = model(
        node_states=base_data['node_states'],
        walk_encodings=base_data['walk_encodings'],
        walk_times=base_data['walk_times'],
        walk_masks=base_data['walk_masks'],
        adj_matrices=base_data['adj_matrices'],
        t_init=base_data['t_init']
    )
    assert output.final_state.shape == (base_data['num_nodes'], base_data['hidden_dim'])

def test_stode_gradient_flow(device, base_data):
    model = SpectralTemporalODE(
        hidden_dim=base_data['hidden_dim'],
        num_nodes=base_data['num_nodes'],
        num_layers=1  # single layer for speed
    ).to(device)
    # Enable grad on inputs
    node_states = base_data['node_states'].clone().detach().requires_grad_(True)
    walk_enc = base_data['walk_encodings'].clone().detach().requires_grad_(True)
    output = model(
        node_states=node_states,
        walk_encodings=walk_enc,
        walk_times=base_data['walk_times'],
        walk_masks=base_data['walk_masks'],
        adj_matrices=base_data['adj_matrices'],
        t_init=base_data['t_init']
    )
    loss = output.final_state.sum()
    loss.backward()
    assert node_states.grad is not None
    assert walk_enc.grad is not None
    # Check some parameters
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_grad = True
                break
    assert has_grad

# ----------------------------------------------------------------------
# Edge Cases and Integration
# ----------------------------------------------------------------------

def test_very_large_times(device, hidden_dim, num_nodes):
    model = SpectralTemporalODE(
        hidden_dim, num_nodes,
        ode_method='dopri5',      # adaptive method
        ode_step_size=None        # no fixed step
    ).to(device)

    H0 = torch.randn(num_nodes, hidden_dim, device=device)
    t0 = torch.tensor(0.0, device=device)
    t1 = torch.tensor(1e9, device=device)
    adj = torch.eye(num_nodes, device=device)

    # Should not crash
    H1 = model.layers[0].integrator(H0, t0, t1, adj)
    assert torch.isfinite(H1).all()

def test_time_precision_collision(device, hidden_dim, num_nodes):
    """Test that close times are not incorrectly merged."""
    model = SpectralTemporalODE(
        hidden_dim, num_nodes, time_precision=2  # low precision to force merging
    ).to(device)
    N, H = num_nodes, hidden_dim
    walk_enc = torch.randn(N, 1, 2, H, device=device)
    # Times: 0.1001 and 0.1002, with precision 2 they round to 0.10
    walk_times = torch.tensor([[[0.1001, 0.1002]]], device=device).expand(N, -1, -1)
    masks = torch.ones_like(walk_times)
    obs = model._vectorized_prepare_observations(walk_enc, walk_times, masks)
    # Should produce only 1 observation due to rounding
    assert len(obs) == 1
    # Higher precision should keep them separate
    model.time_precision = 4
    obs = model._vectorized_prepare_observations(walk_enc, walk_times, masks)
    assert len(obs) == 2

def test_zero_length_walks(device, hidden_dim, num_nodes):
    """Test when walk_length = 0 (no valid positions)."""
    model = SpectralTemporalODE(hidden_dim, num_nodes).to(device)
    N = num_nodes
    walk_enc = torch.randn(N, 1, 0, hidden_dim, device=device)  # L=0
    walk_times = torch.zeros(N, 1, 0, device=device)
    masks = torch.ones(N, 1, 0, device=device).bool()
    obs = model._vectorized_prepare_observations(walk_enc, walk_times, masks)
    assert len(obs) == 0

def test_adjacency_with_isolated_nodes(device, hidden_dim, num_nodes):
    """Test with adjacency that has isolated nodes."""
    adj = torch.eye(num_nodes, device=device)  # no edges
    model = SpectralTemporalODE(hidden_dim, num_nodes, num_eigenvectors=2).to(device)
    reg = SpectralRegularizer(hidden_dim, num_eigenvectors=2).to(device)
    H = torch.randn(num_nodes, hidden_dim, device=device)
    # Should not crash
    force = reg(H, adj)
    assert force.shape == (num_nodes, hidden_dim)

# ----------------------------------------------------------------------
# Run with pytest
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__])
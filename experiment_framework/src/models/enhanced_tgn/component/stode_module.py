import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Callable, NamedTuple, Protocol
from torchdiffeq import odeint, odeint_adjoint
from loguru import logger

from torch.utils.checkpoint import checkpoint
from .time_encoder import TimeEncoder

import time

class GRUODECell(nn.Module):
    """
    GRU-inspired ODE function: dh/dt = (1-z) * (h̃ - h)
    where z = σ(W_z h), r = σ(W_r h), h̃ = tanh(W_h(r⊙h) + t_mod)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Linear layers only (activations applied in forward for efficiency)
        self.W_update = nn.Linear(hidden_dim, hidden_dim)
        self.W_reset = nn.Linear(hidden_dim, hidden_dim)  
        self.W_candidate = nn.Linear(hidden_dim, hidden_dim)
        self.W_time = nn.Linear(1, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [scalar] or [batch_size, 1] or [1] - current time
            h: [batch_size, hidden_dim] - hidden state
        Returns:
            dh/dt: [batch_size, hidden_dim]
        """
        batch_size = h.size(0)
        
        # Check inputs
        if torch.isnan(h).any():
            print(f"GRUODECell input h has NaN!")
        if torch.isnan(t).any():
            print(f"GRUODECell input t has NaN!")
        
        # Normalize time to [batch_size, 1]
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1) if t.size(0) != 1 else t.view(1, 1)
        
        # Broadcast time if needed
        if t.size(0) != batch_size:
            t = t.expand(batch_size, -1)
        
        # Compute gates and derivative (fused, minimal allocations)
        z = torch.sigmoid(self.W_update(h))
        r = torch.sigmoid(self.W_reset(h))
        t_mod = torch.tanh(self.W_time(t))


        h_tilde = torch.tanh(self.W_candidate(r * h)) + t_mod
        dh_dt = (1 - z) * (h_tilde - h)
        # Check for NaN
        if torch.isnan(dh_dt).any():
            print(f"NaN in dh_dt! z: {z.min():.4f}/{z.max():.4f}, h_tilde: {h_tilde.min():.4f}/{h_tilde.max():.4f}, h: {h.min():.4f}/{h.max():.4f}")
        
        # Check output
        if torch.isnan(dh_dt).any():
            print(f"GRUODECell output dh_dt has NaN!")
            print(f"  z: {z.min():.4f}/{z.max():.4f}")
            print(f"  h_tilde: {h_tilde.min():.4f}/{h_tilde.max():.4f}")
            print(f"  h: {h.min():.4f}/{h.max():.4f}")
            print(f"  t: {t}")
        
        return dh_dt

class ODEFunc(nn.Module):
    """
    General ODE function: dh/dt = f(h, t)
    Supports GRU-inspired or MLP-based dynamics.
    """
    def __init__(
        self,
        hidden_dim: int,
        use_gru_ode: bool = True,
        hidden_layers: int = 2,
        activation: str = 'tanh',
        time_invariant: bool = False  # New: explicit control
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gru_ode = use_gru_ode
        self.time_invariant = time_invariant
        
        if use_gru_ode:
            self.dynamics = GRUODECell(hidden_dim)
        else:
            self.dynamics = self._build_mlp(
                hidden_dim, hidden_layers, activation, time_invariant
            )
        
        self._init_weights()
    
    def _build_mlp(self, dim: int, num_layers: int, activation: str, time_invariant: bool):
        """Build MLP with optional time concatenation."""
        act_map = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'softplus': nn.Softplus}
        Act = act_map.get(activation, nn.Tanh)
        
        layers = []
        # Input: hidden + time (if not time-invariant)
        in_dim = dim if time_invariant else dim + 1
        
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_dim = dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                layers.append(Act())
            
            in_dim = out_dim  # After first layer, always dim -> dim
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init)
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt.
        
        Args:
            t: [scalar] or [batch, 1] - current time
            h: [batch, hidden_dim] - hidden state
        """
        if self.use_gru_ode:
            return self.dynamics(t, h)
        
        # MLP branch: handle time
        if self.time_invariant:
            return self.dynamics(h)
        
        # Concatenate time to hidden state
        if t.dim() == 0:
            t = t.view(1, 1).expand(h.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Ensure batch alignment
        if t.size(0) != h.size(0):
            t = t.expand(h.size(0), -1)
        
        h_input = torch.cat([h, t], dim=-1)
        return self.dynamics(h_input)

class SpectralRegularizer(nn.Module):
    """
    Spectral regularization: F = -μ * (I - UU^T) H
    Memory-efficient via never materializing projection matrix.
    """
    def __init__(
        self,
        hidden_dim: int,
        mu: float = 0.1,
        num_eigenvectors: int = 10,
        adaptive_mu: bool = True,
        max_cache_size: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = num_eigenvectors
        self.adaptive_mu = adaptive_mu
        
        mu_tensor = torch.tensor(mu)
        self.mu = nn.Parameter(mu_tensor) if adaptive_mu else mu_tensor
        
        # LRU cache with size limit
        self._cache = {}
        self._cache_order = []
        self._max_cache = max_cache_size
    
    # def _get_cache_key(self, adj: torch.Tensor) -> int:
    #     """Stable hash for adjacency matrix."""
    #     return hash((adj.size(0), adj.detach().cpu().numpy().tobytes()))
    
    def _get_cache_key(self, adj):
        if adj.is_sparse:
            # Use indices and values for hashing
            indices = adj._indices().cpu().numpy().tobytes()
            values = adj._values().cpu().numpy().tobytes()
            return hash((adj.size(0), indices, values))
        else:
            return hash((adj.size(0), adj.cpu().numpy().tobytes()))
    
    def _update_cache(self, key: int, value: tuple):
        """LRU cache update."""
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self._max_cache:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache_order.append(key)
        self._cache[key] = value
    
    def _normalized_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        """L = I - D^{-1/2} A D^{-1/2}"""
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg_inv_sqrt = torch.rsqrt(adj.sum(dim=1) + 1e-8)
        # Efficient: D^{-1/2} A D^{-1/2} = (A * deg_inv_sqrt) * deg_inv_sqrt[:, None]
        norm_adj = adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        return torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype) - norm_adj
    
    def _eigen_decomposition(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k eigenvectors with caching."""
        key = self._get_cache_key(adj)
        
        if key in self._cache:
            return self._cache[key]
        
        L = self._normalized_laplacian(adj)
        
        try:
            vals, vecs = torch.linalg.eigh(L)
            # Skip first (zero eigenvalue), take next k
            vals, vecs = vals[1:self.k+1], vecs[:, 1:self.k+1]
        except RuntimeError as e:
            raise RuntimeError(f"Eigen-decomposition failed: {e}") from e
        
        self._update_cache(key, (vals, vecs))
        return vals, vecs
    
    def forward(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral regularization force.
        
        Args:
            H: [N, hidden_dim] node features
            adj: [N, N] adjacency matrix
            
        Returns:
            [N, hidden_dim] regularization force
        """
        if self.mu == 0.0:
            return torch.zeros_like(H) 
        _, U = self._eigen_decomposition(adj)  # U: [N, k]
        
        # F = -μ * (H - U @ U^T @ H) = -μ * (H - U @ (U^T @ H))
        # Memory: O(N*k) instead of O(N^2)
        UTH = U.T @ H  # [k, hidden_dim]
        H_smooth = U @ UTH  # [N, hidden_dim]
        
        return -self.mu * (H - H_smooth)
    
    def dirichlet_energy(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """E = tr(H^T L H) = sum_{i,j} A_{ij} ||H_i - H_j||^2"""
        # More efficient: don't build L explicitly
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg = adj.sum(dim=1)
        
        # E = sum_i deg_i ||h_i||^2 - sum_{i,j} A_{ij} h_i^T h_j
        term1 = (deg.unsqueeze(1) * H).sum()  # sum_i deg_i ||h_i||^2
        term2 = (H @ H.T * adj).sum()  # sum_{i,j} A_{ij} h_i^T h_j
        
        return term1 - term2



class STODEFunc(nn.Module):
    """
    Augmented ODE: dh/dt = f_ode(h, t) + F_spectral(h, adj)
    
    Maintains [num_nodes, hidden_dim] shape internally; 
    handles flattening via wrapper if needed by solver.
    """
    def __init__(
        self,
        hidden_dim: int,
        odefunc: nn.Module,
        spectral_reg: Optional[SpectralRegularizer] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.odefunc = odefunc
        self.spectral_reg = spectral_reg
        
        # Infer dimensions
        self.hidden_dim = getattr(odefunc, 'hidden_dim', None)
        if self.hidden_dim is None:
            raise ValueError("odefunc must have hidden_dim attribute")
    
    def forward(
        self, 
        t: torch.Tensor, 
        H: torch.Tensor, 
        adj_matrix: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute dynamics. Maintains 2D tensor shape [num_nodes, hidden_dim].
        
        Args:
            t: Scalar or [1] time
            H: [num_nodes, hidden_dim] state
            adj_matrix: Optional [num_nodes, num_nodes] for spectral reg
            return_diagnostics: Whether to return force magnitudes etc.
            
        Returns:
            dh/dt: [num_nodes, hidden_dim]
            or (dh/dt, info_dict) if return_diagnostics=True
        """
        if H.dim() != 2:
            raise ValueError(f"Expected H with 2 dims, got {H.dim()}")
        
        num_nodes, hid = H.shape
        if hid != self.hidden_dim:
            raise ValueError(f"H hidden dim {hid} != {self.hidden_dim}")
        
        # Base dynamics
        dh_dt = self.odefunc(t, H)
        
        info = {}
        
        # Spectral regularization
        if self.spectral_reg is not None and adj_matrix is not None:
            if adj_matrix.shape != (num_nodes, num_nodes):
                raise ValueError(f"Adj shape {adj_matrix.shape} != ({num_nodes}, {num_nodes})")
            
            force = self.spectral_reg(H, adj_matrix)
            dh_dt = dh_dt + force
            
            if return_diagnostics:
                info['spectral_force_norm'] = force.norm().item()
                info['spectral_force_mean'] = force.abs().mean().item()
        
        if return_diagnostics:
            info['total_dhdt_norm'] = dh_dt.norm().item()
            return dh_dt, info
        
        return dh_dt


class FlattenedSTODEFunc(nn.Module):
    """
    Wrapper for ODE solvers requiring flattened [batch] state.
    Buffers adjacency matrix (thread-unsafe but solver-compatible).
    """
    def __init__(self, stode_func: STODEFunc):
        super().__init__()
        self.core = stode_func
        self.register_buffer('_adj', torch.zeros(0, 0), persistent=False)
        self._num_nodes = 0
    
    def set_adj_matrix(self, adj: torch.Tensor):
        """Buffer adjacency for solver callbacks."""
        self._adj = adj
        self._num_nodes = adj.size(0)
    
    def forward(self, t: torch.Tensor, flat_state: torch.Tensor) -> torch.Tensor:
        """
        Flat state [num_nodes * hidden_dim] -> dynamics -> flat output.
        """
        expected = self._num_nodes * self.core.hidden_dim
        if flat_state.numel() != expected:
            raise ValueError(f"State size {flat_state.numel()} != expected {expected}")
        
        H = flat_state.view(self._num_nodes, self.core.hidden_dim)
        dh_dt = self.core(t, H, self._adj if self._num_nodes > 0 else None)
        return dh_dt.view(-1)

 

class ODEFuncProtocol(Protocol):
    """Protocol for ODE functions compatible with integrators."""
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
    def set_adj_matrix(self, adj: torch.Tensor) -> None: ...


class STODEIntegrator(nn.Module):
    """
    Integrates ST-ODE with piecewise-constant graph structure.
    
    Wrapper around torchdiffeq with proper shape handling and
    explicit adjacency management.
    """
    
    # Method categories
    FIXED_STEP = {'euler', 'rk4', 'midpoint'}
    ADAPTIVE = {'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun'}
    
    def __init__(
        self,
        odefunc: ODEFuncProtocol,
        method: str = 'dopri5',
        step_size: Optional[float] = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        adjoint: bool = False
    ):
        super().__init__()
        
        if not hasattr(odefunc, 'set_adj_matrix'):
            raise TypeError("odefunc must implement set_adj_matrix()")
        
        self.odefunc = odefunc
        self.method = method
        self.adjoint = adjoint
        
        # Solver-specific options
        if method in self.FIXED_STEP:
            self.step_size = step_size or 0.1
            self.rtol = None
            self.atol = None
        elif method in self.ADAPTIVE:
            self.rtol = rtol
            self.atol = atol
            self.step_size = None
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _prepare_time(self, t0: torch.Tensor, t1: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Construct time tensor without forcing device sync."""
        # Handle scalar tensors without .item()
        if t0.numel() == 1 and t1.numel() == 1:
            # Detach if not needed for gradients, keep on device
            t0_val = (t0.view(1) if t0.requires_grad else t0.detach().view(1)).to(device)
            t1_val = (t1.view(1) if t1.requires_grad else t1.detach().view(1)).to(device)
            return torch.cat([t0_val, t1_val])
        raise ValueError("t0 and t1 must be scalar tensors")
    
    def forward(
        self,
        h0: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate from t0 to t1 with fixed graph structure.
        
        Args:
            h0: [num_nodes, hidden_dim] initial state
            t0: scalar start time (tensor)
            t1: scalar end time (tensor)
            adj_matrix: [num_nodes, num_nodes] graph structure
            
        Returns:
            h1: [num_nodes, hidden_dim] final state
        """
        if h0.dim() != 2:
            raise ValueError(f"h0 must be 2D, got {h0.dim()}D")
        
        num_nodes, hidden_dim = h0.shape
        
        if adj_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"adj_matrix shape {adj_matrix.shape} != ({num_nodes}, {num_nodes})"
            )
        
        # Set graph context (restored in finally block)
        prev_adj = getattr(self.odefunc, '_adj_matrix', None)
        self.odefunc.set_adj_matrix(adj_matrix)
        
        try:
            # Flatten for ODE solver
            h0_flat = h0.view(-1)
            
            # Prepare time points
            t = self._prepare_time(t0, t1, h0.device)
            
            # Build options dict
            options = {}
            if self.step_size is not None:
                options['step_size'] = self.step_size
            
            # Select solver
            solver = odeint_adjoint if self.adjoint else odeint
            
            # Solve
            trajectory = solver(
                func=self.odefunc,
                y0=h0_flat,
                t=t,
                method=self.method,
                options=options if options else None,
                rtol=self.rtol,
                atol=self.atol,
            )
            
            # Extract final state [num_nodes * hidden_dim]
            final_flat = trajectory[-1]
            
            # Reshape back
            h1 = final_flat.view(num_nodes, hidden_dim)
            
        finally:
            # Restore previous graph state (or clear)
            if prev_adj is not None:
                self.odefunc.set_adj_matrix(prev_adj)            
        
        return h1


class STODETrajectory(nn.Module):
    """
    Returns full trajectory for visualization/analysis.
    Inherits core integration logic but returns all timesteps.
    """
    def __init__(self, integrator: STODEIntegrator, num_steps: int = 10):
        super().__init__()
        self.integrator = integrator
        self.num_steps = num_steps
    
    def forward(
        self,
        h0: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Return trajectory at num_steps intermediate points."""
        # Create intermediate time points
        times = torch.linspace(t0.item(), t1.item(), self.num_steps, device=h0.device)
        
        # Temporarily override method to ensure we get dense output
        # (adaptive methods may skip points)
        original_method = self.integrator.method
        
        if self.integrator.method in STODEIntegrator.ADAPTIVE:
            # Use fixed step for predictable output, or interpolate
            pass  # Keep adaptive, trajectory will have variable steps
        
        # Use internal solver but with full time grid
        # ... (implementation depends on specific needs)
        
        # Simplified: just call integrator multiple times (inefficient but clear)
        states = [h0]
        dt = (t1 - t0) / (self.num_steps - 1)
        
        for i in range(self.num_steps - 1):
            h_next = self.integrator(
                states[-1],
                t0 + i * dt,
                t0 + (i + 1) * dt,
                adj_matrix
            )
            states.append(h_next)
        
        return torch.stack(states)  # [num_steps, num_nodes, hidden_dim]



class STODEOutput(NamedTuple):
    """Structured output."""
    final_state: torch.Tensor
    trajectory: Optional[List[torch.Tensor]] = None  # Per-step or per-layer states
    num_observations: int = 0


class STODELayer(nn.Module):
    """
    ST-ODE layer: integrates between observations with spectral regularization.
    
    Flow: H_{i-1} --[ODE t_{i-1}->t_i]--> H̃_i --[Update]--> H_i
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        odefunc: Optional[nn.Module] = None,  # Dependency injection
        spectral_reg: Optional[nn.Module] = None,
        ode_method: str = 'dopri5',
        ode_step_size: Optional[float] = None,
        adjoint: bool = False,
        dropout: float = 0.1,
        update_type: str = 'gru',  # 'gru', 'mlp', 'residual'
        use_time_encoding: bool = False  # Enable if TimeEncoder needed
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Use injected components or create defaults
        self.odefunc = odefunc or ODEFunc(hidden_dim, use_gru_ode=True)
        self.spectral_reg = spectral_reg or SpectralRegularizer(hidden_dim, mu=0.1)
        
        # Build ST-ODE function
        self.st_odefunc = STODEFunc(
            hidden_dim=hidden_dim,
            odefunc=self.odefunc,
            spectral_reg=self.spectral_reg
        )
        
        # Wrap for integrator compatibility (adds set_adj_matrix)
        self.wrapped_odefunc = FlattenedSTODEFunc(self.st_odefunc)

        # Integrator operates on wrapped function
        self.integrator = STODEIntegrator(
            odefunc=self.wrapped_odefunc,  # Now has set_adj_matrix!
            method=ode_method,
            step_size=ode_step_size,
            adjoint=adjoint
        )
        
        # Update function
        if update_type == 'gru':
            self.update_fn = nn.GRUCell(hidden_dim, hidden_dim)
            self._gru_order = 'input_first'  # Document the convention
        elif update_type == 'mlp':
            self.update_fn = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.update_fn = lambda obs, hidden: obs + hidden
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Optional time encoding
        self.time_encoder = TimeEncoder(hidden_dim) if use_time_encoding else None
        
    
    def _validate_sequence(
        self,
        observations: List[Tuple[torch.Tensor, torch.Tensor]],
        adj_matrices: List[torch.Tensor],
        t_init: torch.Tensor
    ) -> None:
        """Validate input dimensions and temporal ordering."""
        if len(observations) == 0:
            return [], []
        
        t_init_val = t_init.item()
        valid_obs = []
        valid_adjs = []
        
        for i, ((H_obs, t_obs), adj) in enumerate(zip(observations, adj_matrices)):
            # Shape checks
            if H_obs.shape != (self.num_nodes, self.hidden_dim):
                raise ValueError(f"obs[{i}] shape {H_obs.shape} != ({self.num_nodes}, {self.hidden_dim})")
            if adj.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(f"adj[{i}] shape {adj.shape} != ({self.num_nodes}, {self.num_nodes})")
            
            t_curr = t_obs.item() if t_obs.numel() == 1 else float('inf')
            
            # Skip observations at or before t_init (they can't be evolved to)
            if t_curr <= t_init_val + 1e-6:
                continue
                
            valid_obs.append((H_obs, t_obs))
            valid_adjs.append(adj)
        
        return valid_obs, valid_adjs
    
    
    
    def forward(
        self,
        H_init: torch.Tensor,              # [num_nodes, hidden_dim]
        t_init: torch.Tensor,              # scalar, time of H_init
        observations: List[Tuple[torch.Tensor, torch.Tensor]],  # (H_obs, t)
        adj_matrices: List[torch.Tensor],  # [num_nodes, num_nodes] per interval
        return_trajectory: bool = False
    ) -> STODEOutput:
        """
        Process observation sequence through ST-ODE.
        
        Args:
            H_init: Initial state at time t_init
            t_init: Scalar tensor, initial time
            observations: List of (observation_features, timestamp)
                         timestamps must be strictly increasing and > t_init
            adj_matrices: Graph structure for interval [t_{i-1}, t_i]
            return_trajectory: If True, return all intermediate states
            
        Returns:
            STODEOutput with final_state and optional trajectory
        """
        observations, adj_matrices = self._validate_sequence(observations, adj_matrices, t_init)
        if len(observations) == 0:
            # No valid observations, return input unchanged
            return STODEOutput(final_state=H_init, num_observations=0)
        if isinstance(H_init, STODEOutput):
            raise TypeError(
                f"H_init must be a tensor, got STODEOutput. "
                f"Did you forget to extract .final? Use H_init.final to get the tensor."
            )
        # self._validate_sequence(observations, adj_matrices, t_init)
        
        H_current = H_init
        t_current = t_init
        trajectory = [H_init] if return_trajectory else None
        
        for (H_obs, t_next), adj_matrix in zip(observations, adj_matrices):
            # 1. Integrate ODE from current time to observation time
            H_ode = self.integrator(
                H_current,
                t_current,
                t_next,
                adj_matrix
            )
            
            # 2. Update with observation
            if isinstance(self.update_fn, nn.GRUCell):
                # GRUCell(input, hidden) -> we treat observation as input, ODE as hidden prior
                H_new = self.update_fn(
                    H_obs.reshape(-1, self.hidden_dim),
                    H_ode.reshape(-1, self.hidden_dim)
                ).reshape(self.num_nodes, self.hidden_dim)
            elif isinstance(self.update_fn, nn.Sequential):
                # MLP: concatenate and process
                combined = torch.cat([H_obs, H_ode], dim=-1)
                H_new = self.update_fn(combined)
            else:
                # Residual or other
                H_new = self.update_fn(H_obs, H_ode)
            
            # Post-processing
            H_new = self.norm(H_new)
            H_new = self.dropout(H_new)
            
            # Update state
            H_current = H_new
            t_current = t_next
            
            if return_trajectory:
                trajectory.append(H_current)
        
        return STODEOutput(
            final_state=H_current, 
            trajectory=trajectory,
            num_observations=len(observations)
        )
    
    def integrate_beyond(
        self,
        H_last: torch.Tensor,
        t_last: torch.Tensor,
        t_target: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate from last observation to future time (no update).
        
        Useful for forecasting beyond observed data.
        """
        return self.integrator(H_last, t_last, t_target, adj_matrix)






class SpectralTemporalODE(nn.Module):
    """
    Spectral-Temporal ODE for continuous-time node representation learning.
    adj_matrices as:
      - a single [N, N] Tensor  -> replicated to match observation count
      - List[[N, N] Tensor]     -> must match observation count  (
        FIXED: now matched against internally-computed obs 
        count rather than the caller's T)
        
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_eigenvectors: int = 10,
        mu: float = 0.1,
        adaptive_mu: bool = True,
        use_gru_ode: bool = True,
        ode_method: str = 'dopri5',
        ode_step_size: Optional[float] = None,
        num_layers: int = 1,
        adjoint: bool = False,
        dropout: float = 0.1,
        aggregation: str = 'mean',  # 'mean', 'sum', 'max'
        time_precision: int = 6,  # Decimal places for time hashing
        use_checkpoint: bool = False,        
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.time_precision = time_precision
        self.use_checkpoint = use_checkpoint
        
        if use_checkpoint:
            logger.warning(
                "use_checkpoint=True is incompatible with STODELayer (non-tensor args). "
                "Checkpoint is disabled."
            )
        
        # Aggregation function
        agg_map = {
            "mean": lambda x: x.mean(dim=0),
            "sum": lambda x: x.sum(dim=0),
            "max": lambda x: x.max(dim=0)[0],
        }
        
        self.agg_fn = agg_map.get(aggregation, agg_map["mean"])

        
        # Build layers - construct components then inject into STODELayer
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            # Create ODE function
            odefunc = ODEFunc(
                hidden_dim=hidden_dim,
                use_gru_ode=use_gru_ode
            )
            
            # Create spectral regularizer
            spectral_reg = SpectralRegularizer(
                hidden_dim=hidden_dim,
                mu=mu,
                num_eigenvectors=num_eigenvectors,
                adaptive_mu=adaptive_mu
            )
            # Create STODE layer with injected components
            layer = STODELayer(
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                odefunc=odefunc,           # Injected
                spectral_reg=spectral_reg,  # Injected
                ode_method=ode_method,
                ode_step_size=ode_step_size,
                adjoint=adjoint,
                dropout=dropout
            )
            self.layers.append(layer)
            
        
        
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
     # observation preparation 
    def _vectorized_prepare_observations(
        self,
        walk_encodings: torch.Tensor,  # [N, W, L, H]
        walk_times: torch.Tensor,       # [N, W, L]
        walk_masks: torch.Tensor        # [N, W, L]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Vectorized preparation of observations.
        O(N*W*L) memory, O(1) Python loops.
        """
        N, W, L, H = walk_encodings.shape
        device = walk_encodings.device
        dtype = walk_times.dtype
        
        # Flatten everything
        # Valid entries only
        valid_mask = walk_masks > 0  # [N, W, L]
        num_valid = valid_mask.sum()
        
        if num_valid == 0:
            return []  # Caller must handle
        
        # Gather valid encodings and times
        valid_encodings = walk_encodings[valid_mask]  # [num_valid, H]
        valid_times = walk_times[valid_mask]          # [num_valid]
        valid_nodes = torch.arange(N, device=device).view(-1, 1, 1).expand(N, W, L)[valid_mask]
        
        # Round times for stability
        time_mult = 10 ** self.time_precision
        time_ints = (valid_times * time_mult).long()  # [num_valid]
        
        # Sort by time for chronological processing
        sorted_indices = time_ints.argsort()
        time_ints = time_ints[sorted_indices]
        valid_nodes = valid_nodes[sorted_indices]
        valid_encodings = valid_encodings[sorted_indices]
        
        # Find unique time boundaries
        unique_times, inverse_indices, counts = torch.unique(
            time_ints, sorted=True, return_inverse=True, return_counts=True
        )
        num_unique = unique_times.size(0)
        
        # Scatter encodings into per-time-node aggregates
        # Use segment sum/coo operations
        # Create output tensor: [num_unique, N, H]
        H_obs_per_time = torch.zeros(num_unique, N, H, device=device, dtype=walk_encodings.dtype)
        counts_per_time = torch.zeros(num_unique, N, device=device, dtype=torch.long)
        
        # Scatter-add encodings
        # For each valid entry, add to (time_idx, node_idx)
        time_indices = torch.arange(num_unique, device=device)[inverse_indices]
        H_obs_per_time.index_put_(
            (time_indices, valid_nodes),
            valid_encodings,
            accumulate=True
        )
        counts_per_time.index_put_(
            (time_indices, valid_nodes),
            torch.ones_like(valid_nodes),
            accumulate=True
        )
        
        # Normalize by counts (mean aggregation)
        # Avoid div by zero: where count==0, keep zero
        counts_expanded = counts_per_time.unsqueeze(-1).clamp(min=1)
        H_obs_per_time = H_obs_per_time / counts_expanded.float()
        
        # Create list of (H_obs, t)
        observations = []
        for i, t_int in enumerate(unique_times):
            t = t_int.float() / time_mult
            H_obs = H_obs_per_time[i]  # [N, H]
            observations.append((H_obs, torch.as_tensor(t, dtype=dtype, device=device)))
        
        return observations
    
    def forward(
        self,
        node_states: torch.Tensor,           # [N, H]
        walk_encodings: torch.Tensor,         # [N, W, L, H]
        walk_times: torch.Tensor,              # [N, W, L]
        walk_masks: torch.Tensor,              # [N, W, L]
        adj_matrices: List[torch.Tensor],      # List of [N, N]
        t_init: torch.Tensor,                  # Scalar time for node_states
        return_all: bool = False
    ) -> STODEOutput:
        """
        Process through ST-ODE.
        
        Args:
            node_states: [N, H] initial states
            walk_encodings: [N, W, L, H] walk features
            walk_times: [N, W, L] timestamps
            walk_masks: [N, W, L] validity mask
            adj_matrices: List of [N, N] per observation time
            t_init: Scalar tensor, time of node_states
            return_all: Return intermediate states
            
        Returns:
            STODEOutput with final states and optional metadata
        """
        # Prepare observations (vectorized)
        observations = self._vectorized_prepare_observations(
            walk_encodings, walk_times, walk_masks
        )
        
        if not observations:
            # No observations: just project and return
            H_final = self.output_proj(self.input_proj(node_states))
            return STODEOutput(final_state=H_final, num_observations=0)
        
        # Validate adj_matrices length
        if len(adj_matrices) != len(observations):
            raise ValueError(
                f"adj_matrices ({len(adj_matrices)}) != observations ({len(observations)})"
            )
        
        # Initial projection
        H = self.input_proj(node_states)
        
        # Track layer outputs if requested
        layer_trajectory = [] if return_all else None
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                H = checkpoint(layer, H, t_init, observations, adj_matrices, use_reentrant=False).final_state
            else:
                layer_result = layer(H, t_init, observations, adj_matrices)
                H = layer_result.final_state
            
            if return_all:
                layer_trajectory.append(H)
        
        # Final projection
        H_final = self.output_proj(H)
        
        return STODEOutput(
            final_state=H_final,
            trajectory= layer_trajectory if return_all else None,
            num_observations= len(observations)
        )
    

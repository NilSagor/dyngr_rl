import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Union
from torchdiffeq import odeint, odeint_adjoint
from loguru import logger
import time
from collections import OrderedDict


_DEBUG = bool(os.environ.get("DEBUG_GRADIENTS", ""))

# GRU-inspired ODE function: dh/dt = (1-z) * (\tilde_h - h)
#      where z = \sigma(W_z h), r = \sigma(W_r h), \tilde_h = tanh(W_h(r⊙h) + t_mod)
# GRU-ODE cell

class GRUODECell(nn.Module):
    """
     ODE function
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Fused linear layers for speed
        self.W_gates = nn.Linear(hidden_dim, hidden_dim * 2)  # update + reset
        # self.W_reset = nn.Linear(hidden_dim, hidden_dim)
        self.W_candidate = nn.Linear(hidden_dim, hidden_dim)
        self.W_time = nn.Linear(1, hidden_dim, bias=False)

        # Pre-allocate buffers
        self.register_buffer('one', torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_gates.weight)
        nn.init.zeros_(self.W_gates.bias)
        nn.init.xavier_uniform_(self.W_candidate.weight)
        nn.init.zeros_(self.W_candidate.bias)
        nn.init.normal_(self.W_time.weight, std=0.01)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : scalar or [1] or [B, 1]
            h : [B, hidden_dim]
        Returns:
            dh/dt : [B, hidden_dim]
        """
        batch_size = h.size(0)

        # FIX LOW-1: replace print with conditional logger
        if _DEBUG and torch.isnan(h).any():
            logger.warning("GRUODECell: NaN in input h")

        # Normalise t to [B, 1]
        # t = t.reshape(-1) if t.dim() > 0 else t.unsqueeze(0)
        # t = t[:1].view(1, 1).expand(batch_size, 1)
        # Vectorized time handling - no branches
        # Robust time handling - ensure [B, 1] shape
        if t.dim() == 0:
            t_exp = t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            t_exp = t[:1].unsqueeze(0).expand(batch_size, 1) if t.size(0) > 0 else torch.zeros(batch_size, 1, device=t.device)
        else:
            t_exp = t.view(-1, 1)[:, :1].expand(-1, 1) if t.size(0) > 0 else torch.zeros(batch_size, 1, device=t.device)

        # z = torch.sigmoid(self.W_update(h))
        # r = torch.sigmoid(self.W_reset(h))
        # Fused gate computation
        gates = self.W_gates(h)
        z, r = torch.sigmoid(gates).chunk(2, dim=-1)

        t_mod = torch.tanh(self.W_time(t_exp))
        h_tilde = torch.tanh(self.W_candidate(r * h)) + t_mod

        dh_dt=(self.one - z) * (h_tilde - h)

        if _DEBUG and torch.isnan(dh_dt).any():
            logger.warning(
                f"GRUODECell: NaN in dh_dt | "
                f"z=[{z.min():.4f},{z.max():.4f}] "
                f"h_tilde=[{h_tilde.min():.4f},{h_tilde.max():.4f}]"
            )

        return dh_dt


# Generic ODE function
class ODEFunc(nn.Module):
    """dh/dt = f(h, t)  —  GRU-ODE or MLP dynamics."""

    def __init__(
        self,
        hidden_dim: int,
        use_gru_ode: bool = True,        
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gru_ode = use_gru_ode
        
        # self.dynamics = GRUODECell(hidden_dim) if use_gru_ode else self._build_mlp()
        if use_gru_ode:
            self.dynamics = GRUODECell(hidden_dim)
        else:
            self.dynamics = self._build_mlp()

        self._init_weights()        

    def _build_mlp(self):
        # return nn.Sequential(
        #     nn.Linear(self.hidden_dim + 1, self.hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim)
        # )
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Pass t and h separately to dynamics
        return self.dynamics(t, h)



# ============================================================================
# Spectral Regularizer 
# ============================================================================
class SpectralRegularizer(nn.Module):
    """
    Spectral regularizer with fast hash-based cache (no GPU sync).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        mu: float = 0.1,
        num_eigenvectors: int = 10,
        adaptive_mu: bool = True,
        max_cache_size: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = num_eigenvectors
        self.max_cache_size = max_cache_size
        
        mu_t = torch.tensor(float(mu))
        self.mu = nn.Parameter(mu_t) if adaptive_mu else mu_t
        
        # # Cache stored as buffer for persistence
        # self.register_buffer('_cached_U', torch.zeros(0, num_eigenvectors))
        
        # LRU cache: OrderedDict maintains insertion order
        # key -> (U, timestamp or access count)
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._access_count = 0
    
    def _get_cached_U(self, key: int) -> Optional[torch.Tensor]:
        """LRU cache lookup with promotion."""
        if key not in self._cache:
            return None
        
        # LRU: move to end (most recently used)
        U = self._cache.pop(key)
        self._cache[key] = U
        return U
    
    def _update_cache(self, key: int, U: torch.Tensor):
        """LRU cache update with eviction."""
        if key in self._cache:
            # Update existing
            self._cache.pop(key)
        elif len(self._cache) >= self.max_cache_size:
            # Evict least recently used (first item)
            self._cache.popitem(last=False)
        
        # Insert as most recently used
        self._cache[key] = U.detach()


    def _compute_adj_hash(self, adj: torch.Tensor) -> int:
        """Fast hash using deterministic sampling (no .item() calls until necessary)."""
        # Use sum of diagonal and corners - fast, no sync
        n = adj.size(0)
        diag_sum = adj.diagonal().sum().to(dtype=torch.float32)
        corner = (adj[0,0] + adj[0,-1] + adj[-1,0] + adj[-1,-1]).to(dtype=torch.float32)
        total = adj.sum().to(dtype=torch.float32)
        
        # Defer .item() until we have to compare
        hash_tensor = torch.cat([
            torch.tensor([n], device=adj.device),
            diag_sum.unsqueeze(0),
            corner.unsqueeze(0),
            total.unsqueeze(0)
        ])

        hash_val = hash(tuple(hash_tensor.cpu().numpy()))
        return hash_val    
    
    def _normalized_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg_inv_sqrt = torch.rsqrt(adj.sum(dim=1).clamp(min=1e-8))
        return torch.eye(adj.size(0), device=adj.device) - deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    
    def forward(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        mu_val = self.mu if isinstance(self.mu, float) else self.mu.item()
        if abs(mu_val) < 1e-8:
            return torch.zeros_like(H)
        
        # Compute hash 
        key = self._compute_adj_hash(adj)
        
        # Try LRU cache
        U = self._get_cached_U(key)
        
        if U is None:
            # Cache miss: compute eigenvectors
            L = self._normalized_laplacian(adj)
            try:
                _, U_full = torch.linalg.eigh(L)
                U = U_full[:, 1:self.k+1]  # Skip first eigenvector
            except RuntimeError:
                # Fallback: random orthonormal basis
                U = torch.randn(adj.size(0), self.k, device=adj.device)
                U = torch.linalg.qr(U)[0]
                                    
            # Update LRU cache
            self._update_cache(key, U)
        
        # Projection
        UTH = torch.matmul(U.T, H)
        H_smooth = torch.matmul(U, UTH)
        return -mu_val * (H - H_smooth)



# ============================================================================
# ST-ODE Function with Spectral
# ============================================================================

class STODEFuncWithSpectral(nn.Module):
    """Dual-path ODE function: temporal + spatial paths, fused into derivative."""
    
    def __init__(
        self,
        hidden_dim: int,
        use_gru_ode: bool = True,
        spectral_reg: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Path 1: Temporal dynamics (unchanged)
        self.temporal_path = ODEFunc(hidden_dim, use_gru_ode)
        
        # Path 2: Spatial evolution (graph-aware, time-invariant)
        # Uses simple MLP, not GRU-ODE, for pure graph structure evolution
        self.spatial_path = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        ) 

        # Spectral regularization for spatial path 
        self.spectral_reg = spectral_reg
        
        # Fusion: combine temporal and spatial derivatives
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Batch support: store adj as list 
        self._adj: Optional[torch.Tensor] = None  # [B, N, N] or [N, N]
        
    
    def set_adj_matrix(self, adj: torch.Tensor):
        """
        Set adjacency matrix for a specific time interval (batch support).
        Args:
            adj: [N, N] (static) or [B, N, N] (dynamic batch)            
        """
        self._adj = adj
        
    
    def forward(self, t: torch.Tensor, h_flat: torch.Tensor) -> torch.Tensor:
        # Reshape for batch processing (if needed)
        # if H.dim() == 1:
        #     # Flattened from [N, H] to [N*H]
        #     H = H.view(self.num_nodes, self.hidden_dim)
        # elif H.dim() == 3:
        #     # Batch mode [B, N, H] - flatten batch into nodes
        #     B, N, H_dim = H.shape
        #     H = H.view(B * N, H_dim)

        H = h_flat.view(-1, self.hidden_dim)
        N = H.size(0)
        
        # Temporal dynamics (same for all batches)
        # Path 1: Temporal evolution (no graph)
        dh_dt_temp = self.temporal_path(t, H)

        # Path 2: Spatial evolution (graph-only, no time)
        dh_dt_spat = self.spatial_path(H)  # Time-invariant

        # Add spectral regularization if available
        if self.spectral_reg is not None and self._adj is not None:
            spectral_force = self.spectral_reg(H, self._adj)
            dh_dt_spat = dh_dt_spat + spectral_force
        
         # Fuse the two paths
        # dh_dt_fused = self.fusion(torch.cat([dh_dt_temp, dh_dt_spat], dim=-1))
        
        # Fuse the two paths - concat on last dim
        combined = torch.cat([dh_dt_temp, dh_dt_spat], dim=-1)  # [N, 2H]
        dh_dt_fused = self.fusion(combined)  # [N, H]
        
        # Flatten back for odeint if needed
        if dh_dt_fused.dim() == 2:
            dh_dt_fused = dh_dt_fused.view(-1)  # [N*H] 

        return dh_dt_fused


# ============================================================================
# ST-ODE Integrator
# ============================================================================
class STODEIntegrator(nn.Module):
    """Batched ODE integrator."""
    
    FIXED_STEP = {"euler", "rk4", "midpoint"}
    ADAPTIVE = {"dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"}
    
    def __init__(
        self,
        odefunc: STODEFuncWithSpectral,
        method: str = "dopri5",
        step_size: Optional[float] = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        adjoint: bool = False,
        max_steps: int = 1000,
    ):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint = adjoint
        self.max_steps = max_steps
        
        if isinstance(step_size, str):
            logger.warning(f"step_size received as string '{step_size}', converting to None")
            step_size = None
        
        if method in self.FIXED_STEP:
            self.step_size = step_size if step_size is not None else 0.1
            self.rtol = self.atol = None
        elif method in self.ADAPTIVE:
            self.step_size = None
            self.rtol = rtol
            self.atol = atol
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.solver = odeint_adjoint if adjoint else odeint
    
    def integrate(
        self,
        h0: torch.Tensor,
        t_span: Tuple[float, float],
        adj_matrix: torch.Tensor,
        interval_idx: int = 0,
    ) -> torch.Tensor:
        """Single integration step."""
        self.odefunc.set_adj_matrix(adj_matrix)

        # Store original shape for reshaping
        original_shape = h0.shape  # [N, H]
        N, H = original_shape
        
        # Flatten for odeint: [N, H] -> [N*H]
        h0_flat = h0.reshape(-1)

        # Create time tensor
        t = torch.tensor([t_span[0], t_span[1]], device=h0.device, dtype=h0.dtype)

        options = {}
        if self.step_size is not None:
            options['step_size'] = self.step_size
        # options['max_num_steps'] = 100 # limit steps per integration
        start = time.time()       
        
        traj = self.solver(
            func=self.odefunc,
            y0=h0_flat,
            t=t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options if options else None,
        )
        
               
        elapsed = time.time() - start
        # logger.info(f"ODE integration took {elapsed:.2f}s, {len(traj)} steps")

        return traj[-1].view(original_shape)

    def batch_integrate(
        self,
        h0: torch.Tensor,           # [N, H] (initial state)
        t_spans: torch.Tensor,      # [B, 2] (batch of time spans: [t_start, t_end])
        adj_matrices: torch.Tensor, # [B, N, N] (dynamic adj for each interval)
    ) -> torch.Tensor:
        """
        Batch integrate over multiple time intervals (dynamic graphs).
        Args:
            h0: Initial state [N, H]
            t_spans: Batch of time spans [B, 2] (each row = [t_start, t_end])
            adj_matrices: Batch of adjacency matrices [B, N, N] (one per interval)
        Returns:
            H_ode_batch: [B, N, H] (ODE output for each interval)
        """
        B = t_spans.size(0)
        N, H = h0.shape
        
        # Initialize batch output
        H_ode_batch = torch.zeros(B, N, H, device=h0.device, dtype=h0.dtype)
        
        H_current = h0
        # Process each interval in batch (vectorized where possible)
        for b in range(B):
            t_start = float(t_spans[b, 0])
            t_end = float(t_spans[b, 1])
            
            # Skip if no time elapsed
            if abs(t_end - t_start) < 1e-6:
                H_ode_batch[b] = H_current
                continue
            
            # Integrate for this interval (dynamic adj)
            H_ode_batch[b] = self.integrate(
                h0=H_current,
                t_span=(t_start, t_end),
                adj_matrix=adj_matrices[b]                
            )
            H_current = H_ode_batch[b]
        
        return H_ode_batch


# ============================================================================
# ST-ODE Layer
# ============================================================================
class STODELayer(nn.Module):
    """
    Optimized ST-ODE layer with:
    - Shared spectral regularizer (no recomputation)
    - Batched observation processing where possible
    """    
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        spectral_reg: Optional[nn.Module] = None,
        ode_method: str = "dopri5",
        ode_step_size: Optional[float] = None,
        use_gru_ode: bool = True,
        adjoint: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Create ODE function with shared spectral regularizer
        self.odefunc = STODEFuncWithSpectral(
            hidden_dim=hidden_dim,
            use_gru_ode=use_gru_ode,
            spectral_reg=spectral_reg,
        )
        
        self.integrator = STODEIntegrator(
            odefunc=self.odefunc,
            method=ode_method,
            step_size=ode_step_size,
            adjoint=adjoint,
        )
        
        self.update_fn = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        H_init: torch.Tensor,           # [N, H]
        observations: torch.Tensor,      # [T, N, H]
        obs_times: torch.Tensor,         # [T]
        adj_matrix: torch.Tensor,        # [N, N] (static) or [T, N, N] (dynamic)
    ) -> torch.Tensor:
        """
        Process observations sequentially (required for ODE).
        """
        T = observations.size(0)
                
        H_current = H_init

        
        if isinstance(adj_matrix, list):
            # Convert list of [N,N] matrices to tensor [T, N, N]
            if len(adj_matrix) > 0:
                adj_matrix = torch.stack(adj_matrix, dim=0)
            else:
                # Fallback: create identity matrices
                adj_matrix = torch.eye(self.num_nodes, device=H_init.device).unsqueeze(0).expand(T, -1, -1)
        
        # Now safe to check shape
        is_dynamic = len(adj_matrix.shape) == 3  # [T, N, N]

        # Pre-validate/sort times
        if T > 1:
            sorted_times, sort_idx = torch.sort(obs_times)
            # Check if actually needs sorting
            if not torch.equal(obs_times, sorted_times):
                observations = observations[sort_idx]
                obs_times = sorted_times
                if is_dynamic:
                    adj_matrix = adj_matrix[sort_idx]
              
        
        # Create time spans for all intervals [t_prev, t_curr]   
        if T == 0:
            return H_current  

        # First interval: t_start = obs_times[0] - 1.0 (unit time before first obs)
        t_starts = torch.cat([
            torch.tensor([obs_times[0] - 1.0], device=obs_times.device),
            obs_times[:-1]
        ])

        t_ends = obs_times
        t_spans = torch.stack([t_starts, t_ends], dim=1)  # [T, 2]
        
        if not is_dynamic:  # Static graph [N,N]
            adj_batch = adj_matrix.unsqueeze(0).expand(T, -1, -1)  # [T, N, N]
        else:  # Dynamic graph [T, N, N]
            adj_batch = adj_matrix
        
        # Batch integrate all intervals (single call)
        H_ode_batch = self.integrator.batch_integrate(
            h0=H_init,
            t_spans=t_spans,
            adj_matrices=adj_batch,
        )  # [T, N, H]
        # GRU update with observation
        # GRU update is sequential (depends on previous state)
        for i in range(T):
            H_ode = H_ode_batch[i]
            H_obs = observations[i]
            
            # GRU update with observation
            H_new = self.update_fn(
                H_obs.reshape(-1, self.hidden_dim),
                H_ode.reshape(-1, self.hidden_dim)
            ).reshape(self.num_nodes, self.hidden_dim)
            
            H_current = self.dropout(self.norm(H_new))
        
        return H_current


# ============================================================================
# Main ST-ODE Module 
# ============================================================================
class SpectralTemporalODE(nn.Module):
    """
    Optimized Spectral-Temporal ODE with shared components.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_eigenvectors: int = 10,
        mu: float = 0.1,
        adaptive_mu: bool = True,
        use_gru_ode: bool = True,
        ode_method: str = "dopri5", #fixed step rk4, euler, midpoint
        ode_step_size: Optional[float] = None,
        num_layers: int = 1,
        adjoint: bool = False,
        dropout: float = 0.1,
        max_cache_size: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        if isinstance(ode_step_size, str) and ode_step_size.lower() in ("none", "null", ""):
            logger.warning(f"ode_step_size is string '{ode_step_size}', converting to None")
            ode_step_size = None
        
        # Must use fixed-step methods for batching
        # if ode_method not in STODEIntegrator.FIXED_STEP:
        #     logger.warning(f"Switching {ode_method} to rk4 for batching support")
        #     ode_method = "rk4"
        if ode_method not in STODEIntegrator.FIXED_STEP and ode_method not in STODEIntegrator.ADAPTIVE:
            raise ValueError(f"Unknown ODE method: {ode_method}")
        
        # Shared spectral regularizer across all layers!
        self.spectral_reg = SpectralRegularizer(
            hidden_dim=hidden_dim,
            mu=mu,
            num_eigenvectors=num_eigenvectors,
            adaptive_mu=adaptive_mu,
            max_cache_size=max_cache_size,
        )
        
        self.layers = nn.ModuleList([
            STODELayer(
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                spectral_reg=self.spectral_reg,  # Shared
                ode_method=ode_method,
                ode_step_size=ode_step_size,
                adjoint=adjoint,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        logger.debug(f"[DEBUG] ode_step_size value: {ode_step_size!r}, type: {type(ode_step_size)}")
        # self.amp_autocast = torch.amp.autocast(enabled=True, dtype=torch.float16)
    
    def _prepare_observations(
        self,
        walk_encodings: torch.Tensor,   # [N, W, L, H]
        walk_times: torch.Tensor,        # [N, W, L]
        walk_masks: torch.Tensor,        # [N, W, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized observation preparation."""
        N, W, L, H = walk_encodings.shape
        device = walk_encodings.device
        
        valid_mask = walk_masks > 0
        if not valid_mask.any():
            return torch.empty(0, N, H, device=device), torch.empty(0, device=device)
        
        # Flatten
        valid_enc = walk_encodings[valid_mask]   # [V, H]
        valid_t = walk_times[valid_mask]          # [V]
        valid_n = torch.arange(N, device=device).view(-1, 1, 1).expand(N, W, L)[valid_mask]
        
        # Sort
        sort_idx = valid_t.argsort()
        valid_t = valid_t[sort_idx]
        valid_n = valid_n[sort_idx]
        valid_enc = valid_enc[sort_idx]
        
        # Unique times
        unique_times, inverse = torch.unique(valid_t, sorted=True, return_inverse=True)
        T = unique_times.size(0)
        
        # Aggregate
        obs_tensor = torch.zeros(T, N, H, device=device, dtype=walk_encodings.dtype)
        counts = torch.zeros(T, N, device=device)
        
        flat_idx = inverse * N + valid_n
        obs_tensor.view(-1, H).index_add_(0, flat_idx, valid_enc)
        counts.view(-1).index_add_(0, flat_idx, torch.ones_like(valid_n, dtype=torch.float))
        
        obs_tensor = obs_tensor / counts.clamp(min=1).unsqueeze(-1)
        
        return obs_tensor, unique_times
    
    def reset_temporal_state(self):
        """Reset ODE solver state for new epoch."""
        self.last_update_time = torch.tensor(0.0)
        # Clear any internal solver buffers
        if hasattr(self, 'solver_state'):
            self.solver_state = None
    
    def forward(
        self,
        node_states: torch.Tensor,      # [N, H]
        walk_encodings: torch.Tensor,   # [N, W, L, H]
        walk_times: torch.Tensor,       # [N, W, L]
        walk_masks: torch.Tensor,       # [N, W, L]
        adj_matrix: torch.Tensor,       # [N, N] 
        # return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Returns evolved node states.
        """
        observations, obs_times = self._prepare_observations(
            walk_encodings, walk_times, walk_masks
        )
        
        if observations.size(0) == 0:
            return self.output_proj(self.input_proj(node_states))
        
        # with self.amp_autocast:
        H = self.input_proj(node_states)
                
        # Process through layers
        for layer in self.layers:
            H = layer(H, observations, obs_times, adj_matrix)
        
        return self.output_proj(H)


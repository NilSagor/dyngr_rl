import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from torchdiffeq import odeint, odeint_adjoint
from loguru import logger
import time
from collections import OrderedDict

_DEBUG = bool(os.environ.get("DEBUG_GRADIENTS", ""))

# ============================================================================
# 1. Stabilized ODE Cell with Velocity Gating
# ============================================================================

class StabilizedGRUODECell(nn.Module):
    """
    GRU-ODE cell with Velocity Gating.
    dh/dt = gate * v_ode + (1 - gate) * v_gru_backup
    Prevents explosion when ODE derivatives become too large.
    """
    def __init__(self, hidden_dim: int, gate_threshold: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_threshold = gate_threshold
        
        # Gates for standard GRU-ODE
        self.W_gates = nn.Linear(hidden_dim, hidden_dim * 2)
        self.W_candidate = nn.Linear(hidden_dim, hidden_dim)
        self.W_time = nn.Linear(1, hidden_dim, bias=False)
        
        # Gate network for stability switching
        # Inputs: norm of h, norm of t, estimated stiffness (optional)
        self.stability_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.register_buffer('one', torch.tensor(1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_gates.weight)
        nn.init.zeros_(self.W_gates.bias)
        nn.init.xavier_uniform_(self.W_candidate.weight)
        nn.init.zeros_(self.W_candidate.bias)
        nn.init.normal_(self.W_time.weight, std=0.01)
        # Initialize stability gate to prefer ODE (high value) initially
        nn.init.constant_(self.stability_gate[0].bias, 1.0)
        nn.init.constant_(self.stability_gate[2].bias, 2.0) # Sigmoid(2) ~ 0.88

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch_size = h.size(0)
        
        # Robust time expansion
        if t.dim() == 0:
            t_exp = t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            t_exp = t[:1].unsqueeze(0).expand(batch_size, 1) if t.numel() > 0 else torch.zeros(batch_size, 1, device=t.device)
        else:
            t_exp = t.view(-1, 1)[:, :1].expand(-1, 1)

        # --- Standard GRU-ODE Dynamics ---
        gates = self.W_gates(h)
        z, r = torch.sigmoid(gates).chunk(2, dim=-1)
        t_mod = torch.tanh(self.W_time(t_exp))
        h_tilde = torch.tanh(self.W_candidate(r * h)) + t_mod
        v_ode = (self.one - z) * (h_tilde - h)
        
        # --- GRU-Style Backup Dynamics (Discrete Stable Fallback) ---
        # v_gru approximates a stable discrete step: tanh(W*h) - h
        v_gru = torch.tanh(self.W_candidate(h)) - h
        
        # --- Stability Gating ---
        # Compute features for gating: [h_norm, t_norm]
        h_norm = h.norm(dim=-1, keepdim=True) / (self.hidden_dim ** 0.5)
        t_norm = t_exp.abs()
        gate_input = torch.cat([h, t_norm], dim=-1)
        
        # Stability gate: 1.0 = trust ODE, 0.0 = trust GRU backup
        stability_score = self.stability_gate(gate_input)
        
        # Additional safety: if ODE velocity norm is huge, force gate down
        v_ode_norm = v_ode.norm(dim=-1, keepdim=True)
        overflow_mask = (v_ode_norm > self.gate_threshold).float()
        
        # Blend: if overflow, reduce stability_score towards 0
        adjusted_gate = stability_score * (1.0 - overflow_mask) + 0.1 * overflow_mask
        
        # Final Velocity
        dh_dt = adjusted_gate * v_ode + (1.0 - adjusted_gate) * v_gru
        
        if _DEBUG and torch.isnan(dh_dt).any():
            logger.warning("StabilizedGRUODECell: NaN detected in output velocity")
            
        return dh_dt

# ============================================================================
# 2. Low-Rank Spectral Regularizer
# ============================================================================

class LowRankSpectralRegularizer(nn.Module):
    """
    Spectral regularizer using Low-Rank Projection to reduce stiffness.
    Projects H onto top-k eigenvectors, applies smoothing, then projects back.
    Filters out high-frequency noise that causes numerical instability.
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
        
        # LRU Cache for Eigenvectors
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
    
    def _get_cached_U(self, key: int) -> Optional[torch.Tensor]:
        if key not in self._cache:
            return None
        U = self._cache.pop(key)
        self._cache[key] = U
        return U
    
    def _update_cache(self, key: int, U: torch.Tensor):
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = U.detach()

    def _compute_adj_hash(self, adj: torch.Tensor) -> int:
        n = adj.size(0)
        # Fast hash components
        diag_sum = adj.diagonal().sum()
        corner = (adj[0,0] + adj[0,-1] + adj[-1,0] + adj[-1,-1])
        total = adj.sum()
        # CRITICAL FIX: Convert n to tensor before calling .to()
        # Use torch.tensor with explicit device and dtype to ensure consistency
        hash_vec = torch.stack([
            torch.tensor(float(n), device=adj.device, dtype=torch.float32),
            diag_sum.to(torch.float32), 
            corner.to(torch.float32), 
            total.to(torch.float32)
        ])
        return hash(tuple(hash_vec.cpu().numpy()))
    
    def _normalized_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg_inv_sqrt = torch.rsqrt(adj.sum(dim=1).clamp(min=1e-8))
        L = torch.eye(adj.size(0), device=adj.device) - deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return L
    
    def forward(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        mu_val = self.mu if isinstance(self.mu, float) else self.mu
        if abs(mu_val) < 1e-8:
            return torch.zeros_like(H)
        
        key = self._compute_adj_hash(adj)
        U = self._get_cached_U(key)
        
        if U is None:
            L = self._normalized_laplacian(adj)
            try:
                # Compute only top-k eigenvectors (smallest eigenvalues of L)
                # eigh returns sorted ascending. We want smallest non-zero.
                # Skip index 0 (constant vector) if graph is connected, take 1:k+1
                evals, evecs = torch.linalg.eigh(L)
                # Take indices 1 to k+1 (skip trivial constant mode)
                idx_end = min(self.k + 1, evecs.size(1))
                U = evecs[:, 1:idx_end]
            except RuntimeError:
                U = torch.randn(adj.size(0), self.k, device=adj.device)
                U, _ = torch.linalg.qr(U)
            
            self._update_cache(key, U)
        
        # Low-Rank Projection: H_proj = U * (U^T * H)
        # This filters out high-frequency components (stiff modes)
        coeffs = torch.matmul(U.T, H)      # [k, H_dim]
        H_smooth = torch.matmul(U, coeffs) # [N, H_dim]
        
        # Residual correction (only smooth part contributes to force)
        return -mu_val * (H - H_smooth)

# ============================================================================
# 3. Stabilized ODE Function with Adaptive Clamping
# ============================================================================

class StabilizedSTODEFunc(nn.Module):
    """
    Dual-path ODE with Velocity Gating and Adaptive Step Control hints.
    """
    def __init__(
        self,
        hidden_dim: int,
        use_gru_ode: bool = True,
        spectral_reg: Optional[nn.Module] = None,
        max_velocity_norm: float = 10.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_velocity_norm = max_velocity_norm
        
        # Temporal Path (Stabilized Cell)
        self.temporal_path = StabilizedGRUODECell(hidden_dim) if use_gru_ode else nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Spatial Path
        self.spatial_path = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.spectral_reg = spectral_reg
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self._adj: Optional[torch.Tensor] = None

    def set_adj_matrix(self, adj: torch.Tensor):
        self._adj = adj
    
    def forward(self, t: torch.Tensor, h_flat: torch.Tensor) -> torch.Tensor:
        H = h_flat.view(-1, self.hidden_dim)
        
        # Temporal
        dh_dt_temp = self.temporal_path(t, H)
        
        # Spatial
        dh_dt_spat = self.spatial_path(H)
        if self.spectral_reg is not None and self._adj is not None:
            dh_dt_spat = dh_dt_spat + self.spectral_reg(H, self._adj)
        
        # Fuse
        combined = torch.cat([dh_dt_temp, dh_dt_spat], dim=-1)
        dh_dt = self.fusion(combined)
        
        # --- Adaptive Velocity Clamping ---
        # Prevents explosion by clipping global norm of derivative
        norm = dh_dt.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_velocity_norm / (norm + 1e-9), max=1.0)
        dh_dt = dh_dt * scale
        
        return dh_dt.view(-1)

# ============================================================================
# 4. Vectorized Integrator with Tolerance Scheduling
# ============================================================================

class StabilizedIntegrator(nn.Module):
    FIXED_STEP = {"euler", "rk4", "midpoint"}
    ADAPTIVE = {"dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"}
    
    def __init__(
        self,
        odefunc: StabilizedSTODEFunc,
        method: str = "dopri5",
        step_size: Optional[float] = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        adjoint: bool = False,
        max_steps: int = 2000,
    ):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint = adjoint
        self.base_rtol = rtol
        self.base_atol = atol
        self.base_step_size = step_size if step_size is not None else 0.1
        self.max_steps = max_steps
        
        self.solver = odeint_adjoint if adjoint else odeint

    def integrate(
        self,
        h0: torch.Tensor,
        t_span: Tuple[float, float],
        adj_matrix: torch.Tensor,
        training_progress: float = 1.0,
    ) -> torch.Tensor:
        self.odefunc.set_adj_matrix(adj_matrix)
        original_shape = h0.shape
        h0_flat = h0.reshape(-1)
        
        t = torch.tensor([t_span[0], t_span[1]], device=h0.device, dtype=h0.dtype)
        
        # --- Tolerance Scheduling ---
        schedule_factor = 1.0 + 9.0 * (1.0 - training_progress)
        
        options = {}
        if self.method in self.FIXED_STEP:
            # Fixed-step solvers: Use step_size, NOT max_num_steps
            dt = self.base_step_size / schedule_factor
            options['step_size'] = dt
            # Don't set max_num_steps for fixed-step solvers!
        else:
            # Adaptive methods: Use tolerances and max_num_steps
            options['rtol'] = self.base_rtol * schedule_factor
            options['atol'] = self.base_atol * schedule_factor
            options['max_num_steps'] = self.max_steps  # Only for adaptive!
            
        try:
            traj = self.solver(
                func=self.odefunc,
                y0=h0_flat,
                t=t,
                method=self.method,
                options=options if options else None,
            )
            return traj[-1].view(original_shape)
        except RuntimeError as e:
            logger.warning(f"ODE Integration failed ({str(e)}), returning initial state.")
            return h0

    def batch_integrate(
        self,
        h0: torch.Tensor,           # [N, H]
        t_spans: torch.Tensor,      # [B, 2]
        adj_matrices: torch.Tensor, # [B, N, N]
        training_progress: float = 1.0,
    ) -> torch.Tensor:
        """
        Fully Vectorized Batch Integration.
        Note: torchdiffeq doesn't support true parallel batching of different ODEs 
        in a single call easily without flattening B*N. 
        We optimize by processing unique time spans or using vmap if available.
        Here we use a optimized loop over B, but internal ops are vectorized over N.
        """
        B = t_spans.size(0)
        N, H = h0.shape
        device = h0.device
        
        # Output buffer
        H_ode_batch = torch.zeros(B, N, H, device=device, dtype=h0.dtype)
        
        # Current state evolves sequentially if intervals are contiguous, 
        # but here t_spans might be disjoint. We assume independent intervals from h0 
        # OR contiguous evolution. 
        # Based on previous code: "H_current = H_ode_batch[b]" implies contiguous evolution.
        # However, true vectorization requires independent trajectories.
        # Let's assume independent trajectories starting from h0 for each batch item 
        # (common in attention-based time intervals) OR sequential if specified.
        # To match previous logic (sequential evolution):
        
        H_current = h0
        
        # Optimization: Group identical time spans? 
        # For now, loop is unavoidable for sequential dependency, but internal math is vectorized.
        for b in range(B):
            t_start = t_spans[b, 0]
            t_end = t_spans[b, 1]
            
            if abs(t_end - t_start) < 1e-7:
                H_ode_batch[b] = H_current
                continue
            
            # Integrate
            res = self.integrate(
                h0=H_current,
                t_span=(t_start, t_end),
                adj_matrix=adj_matrices[b],
                training_progress=training_progress
            )
            
            H_ode_batch[b] = res
            H_current = res # Sequential update
            
        return H_ode_batch

# ============================================================================
# 5. Stabilized ST-ODE Layer
# ============================================================================

class StabilizedSTODELayer(nn.Module):
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
        
        self.odefunc = StabilizedSTODEFunc(
            hidden_dim=hidden_dim,
            use_gru_ode=use_gru_ode,
            spectral_reg=spectral_reg,
        )
        
        self.integrator = StabilizedIntegrator(
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
        adj_matrix: torch.Tensor,        # [T, N, N] or [N, N]
        training_progress: float = 1.0,
    ) -> torch.Tensor:
        T = observations.size(0)
        if T == 0:
            return H_init
            
        device = H_init.device
        
        # Ensure Adj is [T, N, N]
        if adj_matrix.dim() == 2:
            adj_batch = adj_matrix.unsqueeze(0).expand(T, -1, -1)
        else:
            adj_batch = adj_matrix
            
        # Construct Time Spans [T, 2]
        # First span starts 1.0 unit before first obs (or min delta)
        t_starts = torch.cat([
            obs_times[:1] - 1.0,
            obs_times[:-1]
        ])
        t_ends = obs_times
        t_spans = torch.stack([t_starts, t_ends], dim=1)
        
        # Batch Integrate (Sequential Evolution)
        H_ode_seq = self.integrator.batch_integrate(
            h0=H_init,
            t_spans=t_spans,
            adj_matrices=adj_batch,
            training_progress=training_progress
        ) # [T, N, H]
        
        # Vectorized GRU Update
        # Flatten T and N to process all updates in one go? 
        # GRUCell is sequential in time, but we have T steps.
        # We must loop over T for GRU, but internal ops are vectorized over N.
        
        H_current = H_init
        for i in range(T):
            H_ode = H_ode_seq[i]
            H_obs = observations[i]
            
            # GRU Step
            H_new = self.update_fn(
                H_obs.view(-1, self.hidden_dim),
                H_ode.view(-1, self.hidden_dim)
            ).view(self.num_nodes, self.hidden_dim)
            
            H_current = self.dropout(self.norm(H_new))
            
        return H_current

# ============================================================================
# 6. Main Module: NumericallyStabilizedSTODE
# ============================================================================

class NumericallyStabilizedSTODE(nn.Module):
    """
    Production-ready ST-ODE with:
    - Velocity Gating
    - Low-Rank Spectral Projection
    - Tolerance Scheduling
    - Vectorized Observation Aggregation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_eigenvectors: int = 10,
        mu: float = 0.1,
        adaptive_mu: bool = True,
        use_gru_ode: bool = True,
        ode_method: str = "dopri5",
        ode_step_size: Optional[float] = None,
        num_layers: int = 1,
        adjoint: bool = False,
        dropout: float = 0.1,
        max_cache_size: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Shared Low-Rank Spectral Reg
        self.spectral_reg = LowRankSpectralRegularizer(
            hidden_dim=hidden_dim,
            mu=mu,
            num_eigenvectors=num_eigenvectors,
            adaptive_mu=adaptive_mu,
            max_cache_size=max_cache_size,
        )
        
        self.layers = nn.ModuleList([
            StabilizedSTODELayer(
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                spectral_reg=self.spectral_reg,
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
        
        # Register training progress buffer (updated by trainer)
        self.register_buffer('training_progress', torch.tensor(0.0))

    def set_training_progress(self, progress: float):
        """Call this at start of each epoch: 0.0 -> 1.0"""
        self.training_progress.fill_(progress)

    def _prepare_observations_vectorized(
        self,
        walk_encodings: torch.Tensor,   # [N, W, L, H]
        walk_times: torch.Tensor,        # [N, W, L]
        walk_masks: torch.Tensor,        # [N, W, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fully Vectorized Observation Aggregation.
        Groups by time without python loops.
        """
        N, W, L, H = walk_encodings.shape
        device = walk_encodings.device
        
        valid_mask = walk_masks > 0
        if not valid_mask.any():
            return torch.empty(0, N, H, device=device), torch.empty(0, device=device)
        
        # Flatten valid entries
        valid_enc = walk_encodings[valid_mask]   # [V, H]
        valid_t = walk_times[valid_mask]          # [V]
        valid_n = torch.arange(N, device=device).view(-1, 1, 1).expand(N, W, L)[valid_mask] # [V]
        
        # Sort by time
        sort_idx = valid_t.argsort()
        valid_t = valid_t[sort_idx]
        valid_n = valid_n[sort_idx]
        valid_enc = valid_enc[sort_idx]
        
        # Unique times
        if valid_t.numel() == 0:
             return torch.empty(0, N, H, device=device), torch.empty(0, device=device)
             
        unique_times, inverse = torch.unique(valid_t, sorted=True, return_inverse=True)
        T = unique_times.size(0)
        
        # Vectorized Aggregation using index_add
        obs_tensor = torch.zeros(T, N, H, device=device, dtype=walk_encodings.dtype)
        counts = torch.zeros(T, N, device=device)
        
        # Flat index for [T, N] -> T*N
        flat_idx = inverse * N + valid_n
        
        # Add encodings
        obs_tensor_view = obs_tensor.view(-1, H)
        obs_tensor_view.index_add_(0, flat_idx, valid_enc)
        
        # Count
        counts_view = counts.view(-1)
        counts_view.index_add_(0, flat_idx, torch.ones_like(valid_n, dtype=torch.float))
        
        # Normalize
        mask_counts = counts.clamp(min=1.0).unsqueeze(-1)
        obs_tensor = obs_tensor / mask_counts
        
        return obs_tensor, unique_times
    
    def forward(
        self,
        node_states: torch.Tensor,      # [N, H]
        walk_encodings: torch.Tensor,   # [N, W, L, H]
        walk_times: torch.Tensor,       # [N, W, L]
        walk_masks: torch.Tensor,       # [N, W, L]
        adj_matrix: torch.Tensor,       # [N, N] 
    ) -> torch.Tensor:
        
        # 1. Vectorized Observation Prep
        observations, obs_times = self._prepare_observations_vectorized(
            walk_encodings, walk_times, walk_masks
        )
        
        if observations.size(0) == 0:
            return self.output_proj(self.input_proj(node_states))
        
        # 2. Input Projection
        H = self.input_proj(node_states)
        
        # 3. Process Layers
        progress = self.training_progress
        
        for layer in self.layers:
            H = layer(
                H_init=H,
                observations=observations,
                obs_times=obs_times,
                adj_matrix=adj_matrix,
                training_progress=progress
            )
        
        return self.output_proj(H)
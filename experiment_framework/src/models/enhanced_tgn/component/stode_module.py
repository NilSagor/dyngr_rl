import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Callable
from torchdiffeq import odeint, odeint_adjoint

from .time_encoder import TimeEncoder


class GRUODECell(nn.Module):
    """
    GRU-inspired ODE function for continuous-time dynamics.
    
    This defines the dynamics f_ode(h, t) for the ODE:
    dh/dt = f_ode(h, t)
    
    Based on GRU structure but without input features (pure time evolution).
    """
    def __init__(self, hidden_dim: int):
        super(GRUODECell, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # GRU-like gates for ODE dynamics
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.candidate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Time modulation (allows dynamics to change with time)
        self.time_modulation = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        for module in [self.update_gate, self.reset_gate, self.candidate_proj, self.time_modulation]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt at time t.
        
        Args:
            t: Current time [batch_size, 1] or scalar
            h: Hidden state [batch_size, hidden_dim]
            
        Returns:
            dh/dt [batch_size, hidden_dim]
        """
        # Handle scalar t
        if t.dim() == 0:
            t = t.view(1, 1).expand(h.size(0), -1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Get time modulation
        t_mod = self.time_modulation(t)  # [batch_size, hidden_dim]
        
        # GRU-like dynamics
        z = self.update_gate(h)  # update gate
        r = self.reset_gate(h)   # reset gate
        
        # Candidate hidden state with reset
        h_tilde = self.candidate_proj(r * h)
        
        # Modulate with time
        h_tilde = h_tilde + t_mod
        
        # Compute derivative (GRU-inspired)
        dh_dt = (1 - z) * (h_tilde - h)
        
        return dh_dt


class ODEFunc(nn.Module):
    """
    General ODE function with optional spectral regularization.
    """
    def __init__(
        self,
        hidden_dim: int,
        use_gru_ode: bool = True,
        hidden_layers: int = 2,
        activation: str = 'tanh'
    ):
        super(ODEFunc, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.use_gru_ode = use_gru_ode
        
        if use_gru_ode:
            # Use GRU-inspired dynamics
            self.gru_ode = GRUODECell(hidden_dim)
        else:
            # Use MLP-based dynamics
            layers = []
            in_dim = hidden_dim
            for i in range(hidden_layers):
                out_dim = hidden_dim if i < hidden_layers - 1 else hidden_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'softplus':
                    layers.append(nn.Softplus())
                in_dim = out_dim
            self.mlp = nn.Sequential(*layers)
            
            # Initialize
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt.
        """
        if self.use_gru_ode:
            return self.gru_ode(t, h)
        else:
            return self.mlp(h)


class SpectralRegularizer(nn.Module):
    """
    Spectral regularization for graph-structured representations.
    
    Computes the projection onto low-frequency eigenvectors and
    applies a restoring force toward the smooth manifold.
    """
    def __init__(
        self,
        hidden_dim: int,
        mu: float = 0.1,
        num_eigenvectors: int = 10,
        adaptive_mu: bool = True
    ):
        super(SpectralRegularizer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_eigenvectors = num_eigenvectors
        self.mu = mu
        self.adaptive_mu = adaptive_mu
        
        if adaptive_mu:
            # Learnable mu parameter
            self.mu_param = nn.Parameter(torch.tensor(mu))
        else:
            self.register_buffer('mu_param', torch.tensor(mu))
        
        # Cache for eigen decomposition
        self.eigen_cache = {}
        self.cache_time = None
        self.cache_graph = None
    
    def compute_laplacian(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized graph Laplacian.
        
        L = I - D^{-1/2} A D^{-1/2}
        
        Args:
            adj_matrix: [num_nodes, num_nodes] adjacency matrix
            
        Returns:
            [num_nodes, num_nodes] normalized Laplacian
        """
        # Add self-loops for numerical stability
        adj = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        
        # Degree matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        laplacian = torch.eye(adj.size(0), device=adj.device) - deg_inv_sqrt @ adj @ deg_inv_sqrt
        
        return laplacian
    
    def compute_eigen_decomposition(
        self,
        adj_matrix: torch.Tensor,
        force_recompute: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top k eigenvectors of the Laplacian.
        
        Returns:
            eigenvalues: [num_eigenvectors]
            eigenvectors: [num_nodes, num_eigenvectors]
        """
        # Check cache
        graph_key = adj_matrix.sum().item()  # Simple hash - improve for production
        current_time = id(adj_matrix)  # Not ideal, but for demonstration
        
        if not force_recompute and graph_key in self.eigen_cache:
            return self.eigen_cache[graph_key]
        
        # Compute Laplacian
        L = self.compute_laplacian(adj_matrix)
        
        # Compute eigenvectors (top k smallest eigenvalues)
        # Using torch.linalg.eigh for symmetric matrices
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            
            # Sort by eigenvalue (ascending)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Take top k (excluding the first one which is 0)
            k = min(self.num_eigenvectors, eigenvalues.size(0) - 1)
            eigenvalues = eigenvalues[1:k+1]
            eigenvectors = eigenvectors[:, 1:k+1]
            
        except Exception as e:
            # Fallback for when eigen decomposition fails
            print(f"Eigen decomposition failed: {e}")
            eigenvalues = torch.ones(self.num_eigenvectors, device=adj_matrix.device)
            eigenvectors = torch.eye(adj_matrix.size(0), self.num_eigenvectors, device=adj_matrix.device)
        
        # Cache
        self.eigen_cache[graph_key] = (eigenvalues, eigenvectors)
        
        return eigenvalues, eigenvectors
    
    def compute_projection_matrix(
        self,
        eigenvectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute projection matrix P = U U^T.
        
        Args:
            eigenvectors: [num_nodes, num_eigenvectors]
            
        Returns:
            [num_nodes, num_nodes] projection matrix
        """
        # P = U U^T
        P = eigenvectors @ eigenvectors.T
        return P
    
    def forward(
        self,
        H: torch.Tensor,           # [num_nodes, hidden_dim] node states
        adj_matrix: torch.Tensor,   # [num_nodes, num_nodes] adjacency
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply spectral regularization force.
        
        force = -μ * (I - P) H
        
        Args:
            H: Node state matrix
            adj_matrix: Current graph adjacency
            return_components: Whether to return additional info
            
        Returns:
            Regularization force [num_nodes, hidden_dim]
            or (force, info_dict)
        """
        # Get eigenvectors
        eigenvalues, eigenvectors = self.compute_eigen_decomposition(adj_matrix)
        
        # Compute projection matrix
        P = self.compute_projection_matrix(eigenvectors)  # [num_nodes, num_nodes]
        
        # Project nodes onto smooth subspace
        H_smooth = P @ H  # [num_nodes, hidden_dim]
        
        # Compute high-frequency components
        H_high = H - H_smooth
        
        # Regularization force: -μ * H_high
        force = -self.mu_param * H_high
        
        if return_components:
            info = {
                'H_smooth': H_smooth,
                'H_high': H_high,
                'eigenvalues': eigenvalues,
                'projection_matrix': P,
                'force_magnitude': force.norm(dim=-1).mean().item()
            }
            return force, info
        
        return force
    
    def dirichlet_energy(self, H: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute Dirichlet energy: trace(H^T L H)
        
        This measures how much node states vary across edges.
        """
        L = self.compute_laplacian(adj_matrix)
        energy = torch.trace(H.T @ L @ H)
        return energy



class STODEFunc(nn.Module):
    """
    Augmented ODE function with spectral regularization.
    
    dh/dt = f_ode(h, t) - μ * (h - [P H]_u)
    """
    def __init__(
        self,
        hidden_dim: int,
        odefunc: nn.Module,
        spectral_reg: SpectralRegularizer,
        return_components: bool = False
    ):
        super(STODEFunc, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.odefunc = odefunc
        self.spectral_reg = spectral_reg
        self.return_components = return_components
        
        self.adj_matrix = None  # Will be set externally
    
    def set_adj_matrix(self, adj_matrix: torch.Tensor):
        """Set the current graph adjacency matrix."""
        self.adj_matrix = adj_matrix
    
    def forward(self, t: torch.Tensor, state: Union[torch.Tensor, Tuple]) -> Union[torch.Tensor, Tuple]:
        """
        Compute augmented ODE dynamics.
        
        Args:
            t: Current time
            state: Either [num_nodes * hidden_dim] flattened vector
                   or tuple (H, aux) for auxiliary outputs
        """
        if isinstance(state, tuple):
            H_flat, aux = state
        else:
            H_flat = state
            aux = None
        
        # Reshape to [num_nodes, hidden_dim]
        num_nodes = H_flat.size(0) // self.hidden_dim
        H = H_flat.view(num_nodes, self.hidden_dim)
        
        # Base ODE dynamics
        dh_dt_base = self.odefunc(t, H)
        
        # Spectral regularization force
        if self.adj_matrix is not None:
            force_reg, info = self.spectral_reg(H, self.adj_matrix, return_components=True)
            dh_dt = dh_dt_base + force_reg
        else:
            dh_dt = dh_dt_base
            info = {}
        
        # Flatten for ODE solver
        dh_dt_flat = dh_dt.view(-1)
        
        if self.return_components and isinstance(state, tuple):
            # Update aux with info
            aux.update(info)
            return (dh_dt_flat, aux)
        
        return dh_dt_flat



class STODEIntegrator(nn.Module):
    """
    Integrates the ST-ODE over time intervals.
    
    Handles the piecewise constant approximation of graph structure
    and integrates from t0 to t1.
    """
    def __init__(
        self,
        odefunc: nn.Module,
        method: str = 'rk4',
        step_size: float = 0.1,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        adjoint: bool = False
    ):
        super(STODEIntegrator, self).__init__()
        
        self.odefunc = odefunc
        self.method = method
        self.step_size = step_size
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        
    def forward(
        self,
        h0: torch.Tensor,           # [num_nodes, hidden_dim] initial state
        t0: torch.Tensor,            # [1] or scalar start time
        t1: torch.Tensor,            # [1] or scalar end time
        adj_matrix: torch.Tensor,    # [num_nodes, num_nodes] graph at t0
        return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Integrate ST-ODE from t0 to t1.
        
        Uses piecewise constant approximation: graph structure fixed at adj_matrix.
        """
        # Set current graph
        self.odefunc.set_adj_matrix(adj_matrix)
        
        # Prepare state
        num_nodes, hidden_dim = h0.shape
        h0_flat = h0.view(-1)
        
        # Time points
        t = torch.tensor([t0.item(), t1.item()], device=h0.device)
        
        # Integration options
        options = {}
        if self.method in ['euler', 'rk4']:
            options['step_size'] = self.step_size
        
        # Solve ODE
        ode_solve = odeint_adjoint if self.adjoint else odeint
        
        if return_aux:
            # Track auxiliary info
            aux = {}
            state = (h0_flat, aux)
            
            solution = ode_solve(
                self.odefunc,
                state,
                t,
                method=self.method,
                options=options,
                rtol=self.rtol,
                atol=self.atol
            )
            
            # solution is tuple (final_state, aux_history)
            final_state = solution[0]
            aux_history = solution[1]
            
        else:
            solution = ode_solve(
                self.odefunc,
                h0_flat,
                t,
                method=self.method,
                options=options,
                rtol=self.rtol,
                atol=self.atol
            )
            final_state = solution[-1]
            aux_history = {}
        
        # Reshape final state
        h1 = final_state.view(num_nodes, hidden_dim)
        
        if return_aux:
            return h1, aux_history
        return h1






class STODELayer(nn.Module):
    """
    Single ST-ODE layer that processes a sequence of observations.
    
    For a sequence [(h0, t0), (h1, t1), ..., (hL, tL)], this layer:
    1. Integrates from t_{i-1} to t_i using ST-ODE
    2. Updates with new observation (if provided)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        use_gru_ode: bool = True,
        mu: float = 0.1,
        num_eigenvectors: int = 10,
        adaptive_mu: bool = True,
        ode_method: str = 'rk4',
        ode_step_size: float = 0.125,
        adjoint: bool = False,
        dropout: float = 0.1
    ):
        super(STODELayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Base ODE function
        self.odefunc = ODEFunc(
            hidden_dim=hidden_dim,
            use_gru_ode=use_gru_ode
        )
        
        # Spectral regularizer
        self.spectral_reg = SpectralRegularizer(
            hidden_dim=hidden_dim,
            mu=mu,
            num_eigenvectors=num_eigenvectors,
            adaptive_mu=adaptive_mu
        )
        
        # Augmented ODE function
        self.st_odefunc = STODEFunc(
            hidden_dim=hidden_dim,
            odefunc=self.odefunc,
            spectral_reg=self.spectral_reg,
            return_components=True
        )
        
        # ODE integrator
        self.integrator = STODEIntegrator(
            odefunc=self.st_odefunc,
            method=ode_method,
            step_size=ode_step_size,
            adjoint=adjoint
        )
        
        # Observation update (GRU cell for new observations)
        self.observation_update = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Time encoding for observation times
        self.time_encoder = TimeEncoder(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        H_init: torch.Tensor,           # [num_nodes, hidden_dim] initial state
        observations: List[Tuple[torch.Tensor, torch.Tensor]],  # List of (H_obs, t)
        adj_matrices: List[torch.Tensor],  # Graph at each observation time
        return_all: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Process sequence of observations.
        
        Args:
            H_init: Initial hidden state
            observations: List of (H_obs, t) where H_obs is [num_nodes, hidden_dim]
                        new observation features, t is timestamp
            adj_matrices: Graph adjacency at each observation time
            return_all: If True, return all intermediate states
            
        Returns:
            Final hidden state or list of all states
        """
        H_current = H_init
        states = [H_current]
        
        t_prev = observations[0][1] if observations else None
        
        for i, (H_obs, t_curr) in enumerate(observations):
            if i > 0:
                # Integrate from t_prev to t_curr
                adj_matrix = adj_matrices[i-1]  # Use graph at start of interval
                H_integrated = self.integrator(
                    H_current,
                    t_prev,
                    t_curr,
                    adj_matrix
                )
                
                # Update with new observation
                H_updated = self.observation_update(H_obs, H_integrated)
                H_updated = self.norm(H_updated)
                H_updated = self.dropout(H_updated)
                
                H_current = H_updated
                states.append(H_current)
            
            t_prev = t_curr
        
        if return_all:
            return states
        return H_current



class SpectralTemporalODE(nn.Module):
    """
    Complete Spectral-Temporal ODE (ST-ODE) module for HYDRA.
    
    This module:
    1. Takes node states and temporal walk information
    2. Evolves states continuously using ODE with spectral regularization
    3. Updates at observation times with new walk encodings
    """
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_eigenvectors: int = 10,
        mu: float = 0.1,
        adaptive_mu: bool = True,
        use_gru_ode: bool = True,
        ode_method: str = 'rk4',
        ode_step_size: float = 0.125,
        num_layers: int = 1,
        adjoint: bool = False,
        dropout: float = 0.1
    ):
        super(SpectralTemporalODE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # ST-ODE layers
        self.layers = nn.ModuleList([
            STODELayer(
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                use_gru_ode=use_gru_ode,
                mu=mu,
                num_eigenvectors=num_eigenvectors,
                adaptive_mu=adaptive_mu,
                ode_method=ode_method,
                ode_step_size=ode_step_size,
                adjoint=adjoint,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Input projection (for walk embeddings)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def prepare_observations(
        self,
        walk_encodings: torch.Tensor,    # [num_nodes, num_walks, walk_len, hidden_dim]
        walk_times: torch.Tensor,         # [num_nodes, num_walks, walk_len] timestamps
        walk_masks: torch.Tensor          # [num_nodes, num_walks, walk_len] valid masks
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare observation sequence from walk encodings.
        
        For each node, we have multiple walks with multiple steps.
        This function aggregates them into a sequence of observations
        at unique timestamps.
        
        Returns:
            List of (H_obs, t) where H_obs is [num_nodes, hidden_dim]
        """
        num_nodes, num_walks, walk_len, _ = walk_encodings.shape
        
        # Collect all timestamps and corresponding encodings
        observations_dict = {}  # t -> list of encodings per node
        
        for n in range(num_nodes):
            for w in range(num_walks):
                for step in range(walk_len):
                    if walk_masks[n, w, step] > 0:
                        t = walk_times[n, w, step].item()
                        encoding = walk_encodings[n, w, step]  # [hidden_dim]
                        
                        if t not in observations_dict:
                            observations_dict[t] = [[] for _ in range(num_nodes)]
                        
                        observations_dict[t][n].append(encoding)
        
        # Sort by time
        sorted_times = sorted(observations_dict.keys())
        
        # Aggregate encodings at each time (mean over walks for each node)
        observations = []
        for t in sorted_times:
            H_obs_list = []
            for n in range(num_nodes):
                encodings = observations_dict[t][n]
                if encodings:
                    # Mean over walks at this time
                    H_n = torch.stack(encodings).mean(dim=0)
                else:
                    # No observation for this node at this time
                    H_n = torch.zeros(self.hidden_dim, device=walk_encodings.device)
                H_obs_list.append(H_n)
            
            H_obs = torch.stack(H_obs_list)  # [num_nodes, hidden_dim]
            t_tensor = torch.tensor(t, device=walk_encodings.device)
            observations.append((H_obs, t_tensor))
        
        return observations
    
    def forward(
        self,
        node_states: torch.Tensor,           # [num_nodes, hidden_dim] initial states
        walk_encodings: torch.Tensor,         # [num_nodes, num_walks, walk_len, hidden_dim]
        walk_times: torch.Tensor,              # [num_nodes, num_walks, walk_len]
        walk_masks: torch.Tensor,              # [num_nodes, num_walks, walk_len]
        adj_matrices: List[torch.Tensor],      # Graph at each observation time
        return_all: bool = False
    ) -> Union[torch.Tensor, Dict]:
        """
        Process through ST-ODE.
        
        Args:
            node_states: Initial node states (from SAM or previous layer)
            walk_encodings: Encoded walks from HCT
            walk_times: Timestamps for each walk step
            walk_masks: Masks for valid positions
            adj_matrices: Graph adjacency at each observation time
            return_all: Whether to return intermediate states
            
        Returns:
            Final node states or dictionary with all info
        """
        # Prepare observation sequence
        observations = self.prepare_observations(
            walk_encodings, walk_times, walk_masks
        )
        
        # Project input states
        H = self.input_proj(node_states)
        H = self.norm(H)
        
        # Process through layers
        layer_outputs = []
        
        for layer_idx, layer in enumerate(self.layers):
            H = layer(
                H,
                observations,
                adj_matrices,
                return_all=False
            )
            layer_outputs.append(H)
        
        # Final projection
        H_final = self.output_proj(H)
        H_final = self.norm(H_final)
        
        if return_all:
            return {
                'final': H_final,
                'layer_outputs': layer_outputs,
                'num_observations': len(observations)
            }
        
        return H_final
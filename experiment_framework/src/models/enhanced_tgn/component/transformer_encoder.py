import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import torch
import torch.nn as nn

# from utils.utils import MergeLayer

def _sanitize_tensor(
    x: torch.Tensor, 
    name: str = "tensor", 
    nan_val: float = 0.0, 
    inf_val: float = 10.0,
    neg_inf_val: float = -10.0
) -> torch.Tensor:
    """Helper: sanitize tensor for NaN/Inf with consistent logging."""
    if not torch.isfinite(x).all():
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        logger.warning(f"NaN/Inf in {name}: shape={x.shape}, has_nan={has_nan}, has_inf={has_inf}")
        return torch.nan_to_num(x, nan=nan_val, posinf=inf_val, neginf=neg_inf_val)
    return x



# new MergeLayer with temperature scaling
class MergeLayer(nn.Module):
    """
    Link prediction layer with optional temperature scaling for calibration.
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim, dropout=0.1, use_temperature=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Temperature scaling parameter (learnable)
        self.use_temperature = use_temperature
        if use_temperature:
            # Initialize near 1.0 for stable training
            self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        else:
            self.register_buffer('temperature', torch.ones(1))
        
    def forward(self, x1, x2, return_probs=False, threshold=None):
        x = torch.cat([x1, x2], dim=-1)
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        logits = self.fc2(h)
        
        if return_probs:
            # Convert to probabilities
            probs = torch.sigmoid(logits)
            
            # Apply temperature scaling to probabilities for calibration
            # This is the CORRECT way: scale probabilities, not logits
            if self.use_temperature and not self.training:
                temp = self.temperature.clamp(min=0.1, max=10.0)
                # Temperature scaling on probs: p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
                # Simplified approximation for small temperature changes:
                probs = torch.sigmoid(logits / temp)
            
            if threshold is not None:
                return (probs > threshold).float()
            return probs
        
        return logits
    
    def get_temperature(self):
        """Get current temperature value."""
        return self.temperature.item()
    
    def set_temperature(self, value):
        """Manually set temperature (for post-hoc calibration)."""
        if self.use_temperature:
            self.temperature.data.fill_(value)

# class MergeLayer(nn.Module):
#     """
#     Robust tensor merging layer for temporal attention mechanisms.
    
#     Industry standard usage (TGN ICML 2020):
#     - ONLY supports 2-input merging (x1, x2)
#     - Total input dim = dim(x1) + dim(x2)
#     - Hidden dim and output dim explicitly specified
    
#     Why no 4-arg support?
#     - Original TGN paper ONLY uses 2-input merging
#     - 4-arg pattern causes dimension mismatches (e.g., 204 vs 208 errors)
#     - Safer to enforce strict 2-input interface
#     """
    
#     def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
#         super().__init__()
        
#         # CRITICAL FIX 1: Explicit dimension validation
#         if input_dim1 <= 0 or input_dim2 <= 0:
#             raise ValueError(f"Input dimensions must be positive: {input_dim1}, {input_dim2}")
#         if hidden_dim <= 0 or output_dim <= 0:
#             raise ValueError(f"Hidden/output dimensions must be positive: {hidden_dim}, {output_dim}")
        
#         total_input_dim = input_dim1 + input_dim2
        
#         self.fc1 = nn.Linear(total_input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.act = nn.ReLU()
        
#         # Proper weight initialization (with bias handling)
#         nn.init.xavier_normal_(self.fc1.weight)
#         nn.init.zeros_(self.fc1.bias)  # Explicit bias initialization
#         nn.init.xavier_normal_(self.fc2.weight)
#         nn.init.zeros_(self.fc2.bias)
        
#         # Store dimensions for debugging/validation
#         self.input_dim1 = input_dim1
#         self.input_dim2 = input_dim2
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
    
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         """
#         Merge two tensors via concatenation + MLP.
        
#         Args:
#             x1: Tensor of shape [..., input_dim1]
#             x2: Tensor of shape [..., input_dim2]
        
#         Returns:
#             Tensor of shape [..., output_dim]
        
#         Raises:
#             ValueError: If input dimensions don't match expected sizes
#         """
#         # Runtime dimension validation
#         if x1.size(-1) != self.input_dim1:
#             raise ValueError(
#                 f"x1 dimension mismatch: expected {self.input_dim1}, got {x1.size(-1)}. "
#                 f"Check attention layer configuration."
#             )
#         if x2.size(-1) != self.input_dim2:
#             raise ValueError(
#                 f"x2 dimension mismatch: expected {self.input_dim2}, got {x2.size(-1)}. "
#                 f"This causes attention shape errors (e.g., 204 vs 208)."
#             )
        
#         # Safe concatenation
#         x = torch.cat([x1, x2], dim=-1)
#         h = self.act(self.fc1(x))
#         return self.fc2(h)



class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for walk positions.
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model] with positional encoding added
        """
        # Sanitize input before adding positional encoding
        x = _sanitize_tensor(x, "pos_enc_input")

        # Validate shape compatibility
        if x.size(-1) != self.pe.size(-1):
            logger.error(f"Dimension mismatch in PositionalEncoding: x={x.shape}, pe={self.pe.shape}")
            # Project x to match pe dimension
            projection = nn.Linear(x.size(-1), self.pe.size(-1), device=x.device)
            x = projection(x)

        # Clamp positional encoding to prevent extreme values
        pe_slice = self.pe[:, :x.size(1), :]
        pe_slice = torch.clamp(pe_slice, -10.0, 10.0)
        
        result = x + pe_slice
        
        # Sanitize output
        return _sanitize_tensor(result, "pos_enc_output")
    

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional bias.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Initialize with smaller weights for stability
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [batch_size, tgt_len, d_model]
            key: [batch_size, src_len, d_model]
            value: [batch_size, src_len, d_model]
            attn_bias: Optional [batch_size, nhead, tgt_len, src_len] bias to add to attention scores
            key_padding_mask: [batch_size, src_len] with True for padded positions
        Returns:
            output: [batch_size, tgt_len, d_model]
            attention: Optional attention weights
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.size(1)
        
         # Sanitize inputs before projection
        query = _sanitize_tensor(query, "attn_query_input")
        key = _sanitize_tensor(key, "attn_key_input")
        value = _sanitize_tensor(value, "attn_value_input")
        
        # Reshape for multi-head
        Q = query.view(batch_size, tgt_len, self.nhead, self.head_dim).transpose(1, 2)
        K = key.view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        V = value.view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Validate reshaped tensors
        if not torch.isfinite(Q).all() or not torch.isfinite(K).all() or not torch.isfinite(V).all():
            logger.error("NaN after view/transpose in attention!")
            Q = torch.nan_to_num(Q, nan=0.0)
            K = torch.nan_to_num(K, nan=0.0)
            V = torch.nan_to_num(V, nan=0.0)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Clamp scores before adding bias to prevent explosion
        attn_scores = torch.clamp(attn_scores, -50.0, 50.0)
        
        # Add attention bias if provided
        if attn_bias is not None:
            # Sanitize bias before adding
            attn_bias = _sanitize_tensor(attn_bias, "attn_bias")
            attn_bias = torch.clamp(attn_bias, -10.0, 10.0)
            attn_scores = attn_scores + attn_bias
        
        # Re-clamp after bias addition
        attn_scores = torch.clamp(attn_scores, -100.0, 100.0)
        
        # Apply key padding mask
        if key_padding_mask is not None:
            # FIX 7: Use finite mask value instead of -inf to prevent NaN in softmax
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -1e4  # Large negative but not -inf
            )
        
        # Softmax with numerical stability
        # Subtract max for numerical stability before softmax
        attn_scores_max = attn_scores.max(dim=-1, keepdim=True)[0]
        attn_scores = attn_scores - attn_scores_max
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Validate weights (should sum to 1)
        if not torch.isfinite(attn_weights).all():
            logger.error("NaN in attention weights after softmax!")
            attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / attn_weights.size(-1))
            # Renormalize
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Sanitize output of attention
        output = _sanitize_tensor(output, "attn_output_pre_reshape")
        
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        output = self.out_proj(output)
        
        # Final sanitization
        output = _sanitize_tensor(output, "attn_output_final")
        
        if need_weights:
            return output, attn_weights
        return output, None


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with pre-norm architecture.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu

        # Initialize FFN layers with smaller weights
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attn_bias: Optional attention bias
            key_padding_mask: [batch_size, seq_len] mask for padded positions
        """
        # Sanitize input
        x = _sanitize_tensor(x, "transformer_layer_input")
        
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_bias, key_padding_mask)
        x = self.dropout1(x)
        
        # Sanitize before residual addition
        x = _sanitize_tensor(x, "transformer_post_attn")
        x = residual + x
        
        # Sanitize after first residual
        x = _sanitize_tensor(x, "transformer_post_residual1")
        
        # FFN
        residual = x
        x = self.norm2(x)
        
        # Clamp FFN activations to prevent ReLU explosion
        x = self.linear1(x)
        x = _sanitize_tensor(x, "transformer_ffn_hidden")
        x = self.activation(x)
        x = torch.clamp(x, 0.0, 20.0)  # ReLU is already >= 0, clamp upper bound
        
        x = self.linear2(x)
        x = self.dropout2(x)
        
        # Sanitize before final residual
        x = _sanitize_tensor(x, "transformer_post_ffn")
        x = residual + x
        
        # Final sanitization
        return _sanitize_tensor(x, "transformer_layer_output")
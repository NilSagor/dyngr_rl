import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np




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
        return x + self.pe[:, :x.size(1), :]
    

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
        
        # Project and reshape for multi-head
        Q = self.q_proj(query).view(batch_size, tgt_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, nhead, tgt_len, src_len]
        
        # Add attention bias if provided
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [batch, nhead, tgt_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        output = self.out_proj(output)
        
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
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_bias, key_padding_mask)
        x = self.dropout1(x)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.dropout2(x)
        x = residual + x
        
        return x
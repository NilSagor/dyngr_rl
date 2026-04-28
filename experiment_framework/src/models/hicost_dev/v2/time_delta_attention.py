import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeDeltaAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_deltas, mask=None):
        # x: [batch, seq_len, d_model]
        # time_deltas: [batch, seq_len] (delta to current time)
        batch, seq_len, _ = x.shape

        # Encode time deltas into keys/values (additive bias)
        # Use sinusoidal encoding
        pe = torch.zeros_like(x)
        position = time_deltas.unsqueeze(-1)  # [batch, seq_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * -(math.log(10000.0) / self.d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        x = x + pe

        Q = self.q_proj(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1,2)
        K = self.k_proj(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) * self.scale

        # Causal mask: positions can only attend to earlier steps (i <= j)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        if mask is not None:
            # mask: [batch, seq_len] – True for padded positions
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(batch, seq_len, self.d_model)
        out = self.out_proj(out)
        return out

class TimeDeltaAttentionWalkEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = TimeDeltaAttention(hidden_dim, nhead, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)   # mean over sequence
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, time_deltas, masks):
        # features: [batch, num_walks, walk_len, input_dim]
        # time_deltas: [batch, num_walks, walk_len] (delta to current time)
        # masks: [batch, num_walks, walk_len] (1 for valid)
        batch, num_walks, walk_len, _ = features.shape
        # Flatten walks for parallel processing
        x = features.view(batch * num_walks, walk_len, -1)
        t = time_deltas.view(batch * num_walks, walk_len)
        m = masks.view(batch * num_walks, walk_len)

        x = self.input_proj(x)
        x = self.attention(x, t, mask=(m == 0))
        # Masked mean pooling
        x = (x * m.unsqueeze(-1)).sum(dim=1) / (m.sum(dim=1, keepdim=True) + 1e-8)
        x = self.dropout(x)
        x = self.output_proj(x)
        x = x.view(batch, num_walks, -1).mean(dim=1)   # average over walks
        return x
import torch 
import torch.nn as nn


class TimeEncoder(nn.Module):
    """ Fixed Time Encoder using trigeometric feature

        For 1D input (batch of timestamps) returns [batch_size, time_dim].
        For 2D input (batch of neighbor timestamps) returns [batch_size, n_neighbors, time_dim].
    """
    def __init__(self, dimension:int):
        super(TimeEncoder, self).__init__()
        self.dimension = dimension

        # pre-compute freq (non-learnable) log scale
        freqs = torch.exp(torch.linspace(0,8, dimension//2))
        self.register_buffer('freqs', freqs)

    def forward(self, times:torch.Tensor)->torch.Tensor:
        """
        Docstring for forward
        times [batch_size] or [batch_size, 1] Tensor of times
        return Times encoding [batch_size, dim]
        """
        original_shape = times.shape

        # Handle 1D input (batch of single timestamps)
        if times.dim() == 1:
            # [B] -> [B, 1]
            times = times.unsqueeze(-1)
            flattened = False
        # Handle 2D input (batch of neighbor timestamps)
        elif times.dim() == 2:
            # [B, K] -> [B*K, 1]  (flatten to process all timestamps at once)
            batch_size, n_neighbors = times.shape
            # Handle zero neighbors explicitly
            if n_neighbors == 0:
                return torch.zeros(
                    batch_size, 0, self.dimension,
                    device=times.device,
                    dtype=times.dtype
                )
            times = times.reshape(-1, 1)
            flattened = True
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got shape {original_shape}")
        
        # compute projection = times*freqs
        projection = times.unsqueeze(-1) * self.freqs # [B, dim//2]
        
        encodings = torch.cat([
            torch.cos(projection),
            torch.sin(projection)
        ], dim=-1) # [N, time_dim]

        
        
        if flattened:
            encodings = encodings.view(batch_size, n_neighbors, -1)
        else:
            encodings = encodings.squeeze(1)

        return encodings

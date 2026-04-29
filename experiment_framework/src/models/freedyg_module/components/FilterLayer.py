import torch 
import torch.nn as nn
import numpy as np


class FilterLayer(nn.Module):
    def __init__(self,max_input_length:int, hidden_dim: int):
        super(FilterLayer, self).__init__()

        self.max_input_length=max_input_length

        self.complex_weight = nn.Parameter(torch.randn(1,self.max_input_length//2+1, hidden_dim, 2,dtype=torch.float32))

        self.Dropout = nn.Dropout(0.1)

        # self.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, input_tensor: torch.Tensor):
      

        batch,seq_len, hidden = input_tensor.shape

        hidden_states=input_tensor
      
       
        x = torch.fft.rfft(hidden_states, n=self.max_input_length, dim=1, norm='forward')
        weight = torch.view_as_complex(self.complex_weight)
        x=x*weight
        sequence_emb_fft = torch.fft.irfft(x, n=self.max_input_length, dim=1, norm='forward')

        sequence_emb_fft = sequence_emb_fft[:,0:seq_len,:]
        hidden_states = self.Dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor

        return hidden_states
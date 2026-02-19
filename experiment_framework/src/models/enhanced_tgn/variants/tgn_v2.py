import torch 
import torch.nn as nn


from abc import ABC, abstractmethod

from ..base_enhance_tgn import BaseEnhancedTGN
from ..component.time_encoder import TimeEncoder

class TGNv2(BaseEnhancedTGN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_encoder = TimeEncoder(dimension=self.time_encoding_dim)
    
    # def forward(self, batch):
    #     pass

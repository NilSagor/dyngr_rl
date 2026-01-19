import torch
import torch.nn as nn
import lightning as L
from abc import ABC, abstractmethod
from torch_geometric.data import Data


class BaseDynamicGNN(L.LightningModule, ABC):
    def __init__(self, num_nodes, node_features, hidden_dim, time_encoding_dim = 32, num_layers = 2, dropout=0.1, learning_rate=1e-4, weight_decay=1e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_nodes = num_nodes
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.time_encoding_dim = time_encoding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.time_encoder = self._init_time_encoder()
        
        if node_features>0:
            self.node_embedding = nn.Embedding(num_nodes, node_features)
        else:
            self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        self.dropout_layer = nn.Dropout(dropout)

    def _init_time_encoder(self):
        return TimeEncode(self.time_encoding_dim)
    
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        metrics = self._compute_metrics(batch)
        for metric_name, value in metrics.items():
            self.log(f'test_{metric_name}', value, prog_bar=True)        
        return metrics
    

    
    def configure_optimizers(self):        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    @abstractmethod
    def _compute_loss(self, batch):
        pass

    @abstractmethod
    def _compute_metrics(self, batch):
        pass


class TimeEncode(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim

        freqs = torch.exp(
            torch.linspace(0, 1, time_dim//2)*torch.log(torch.tensor(10000.0))
        )
        self.register_buffer('freqs', freqs)

    def forward(self, timestamps):
        
        timestamps = timestamps.unsqueeze(-1) # [batch_size, 1]
        args = timestamps*self.freqs.unsqueeze(0) # [batch_size, time_dim//2]
        time_enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return time_enc #[batch_size, time_dim]

class MemoryModule(nn.Module):
    def __init__(self, num_nodes, memory_dim, message_dim, time_encoding_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim

        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))

        self.register_buffer('last_update', torch.zeros(num_nodes, dtype=torch.float32))

        self.memory_updater = nn.GRUCell(
            message_dim + time_encoding_dim, memory_dim
        )

    
    
    def get_memory(self, node_ids):
        return self.memory[node_ids]


    def update_memory(self, node_ids, messages, timestamps):
        
        current_memory = self.memory[node_ids]
        updated_memory = self.memory_updater(messages, current_memory)
        self.memory[node_ids] = updated_memory
        self.last_update[node_ids] = timestamps
    
    def reset_memory(self):
        self.memory.fill_(0)
        self.last_update.fill_(0)
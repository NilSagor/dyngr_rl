import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graphmixer_module.components.time_encoder import TimeEncoder
from src.models.graphmixer_module.components.neighbor_sampler import NeighborSampler

from src.models.graphmixer_module.components.MLPMixer import MLPMixer

from src.models.graphmixer_module.graphmixer_config import GraphMixerConfig

from loguru import logger

import lightning as L


from torchmetrics import AUROC, AveragePrecision, Accuracy


class GraphMixer(L.LightningModule):

    # def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
    #              time_feat_dim: int, num_tokens: int, num_layers: int = 2, token_dim_expansion_factor: float = 0.5,
    #              channel_dim_expansion_factor: float = 4.0, dropout: float = 0.1, device: str = 'cpu'):
    def __init__(self, config:GraphMixerConfig):
        """
        TCL model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_tokens: int, number of tokens
        :param num_layers: int, number of transformer layers
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(GraphMixer, self).__init__()
        self.save_hyperparameters()
        self.cfg = config
        
        self.device = torch.device(
            config.device if isinstance(config.device, str) else 'cpu'
        )
        
        # self.node_raw_features = torch.from_numpy(config.node_raw_features.astype(np.float32)).to(device)
        # self.edge_raw_features = torch.from_numpy(config.edge_raw_features.astype(np.float32)).to(device)

        if isinstance(config.node_raw_features, np.ndarray):
            self.node_raw_features = torch.from_numpy(config.node_raw_features.astype(np.float32)).to(self.device)
        else:
            self.node_raw_features = config.node_raw_features.to(self.device)
            
        if isinstance(config.edge_raw_features, np.ndarray):
            self.edge_raw_features = torch.from_numpy(config.edge_raw_features.astype(np.float32)).to(self.device)
        else:
            self.edge_raw_features = config.edge_raw_features.to(self.device)
        
        self.neighbor_sampler = config.neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = config.time_feat_dim
        self.num_tokens = config.num_tokens
        self.num_layers = config.num_layers
        self.token_dim_expansion_factor = config.token_dim_expansion_factor
        self.channel_dim_expansion_factor = config.channel_dim_expansion_factor
        self.dropout = config.dropout
        # self.device = device

        self.num_channels = self.edge_feat_dim
        
        # in GraphMixer, the time encoding function is not trainable
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(
            self.edge_feat_dim + self.time_feat_dim, 
            self.num_channels
        )

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(
            in_features=self.num_channels + self.node_feat_dim, 
            out_features=self.node_feat_dim, 
            bias=True
        )

        self.val_auroc = AUROC(task='binary')
        self.val_ap = AveragePrecision(task='binary')
        self.test_auroc = AUROC(task='binary')
        self.test_ap = AveragePrecision(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')
    
    
    
    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                node_interact_times: np.ndarray, num_neighbors: int = 20, time_gap: int = 2000):
        """
        Forward pass: compute edge probabilities for link prediction.
        """
                
        src_embeddings, dst_embeddings = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times, num_neighbors, time_gap
        )
        # Compute affinity score (dot product + sigmoid)
        scores = (src_embeddings * dst_embeddings).sum(dim=1)
        return torch.sigmoid(scores)
    
    def training_step(self, batch, batch_idx):
        sources = batch['sources'].cpu().numpy()
        destinations = batch['destinations'].cpu().numpy()
        timestamps = batch['timestamps'].cpu().numpy()
        neg_sources = batch['negative_sources'].cpu().numpy()
        neg_destinations = batch['negative_destinations'].cpu().numpy()
        
        num_neighbors = getattr(self.cfg, 'num_tokens', 20)
        time_gap = getattr(self.cfg, 'time_gap', 2000)
        
       
        pos_scores = self.forward(sources, destinations, timestamps, num_neighbors, time_gap)
        neg_scores = self.forward(neg_sources, neg_destinations, timestamps, num_neighbors, time_gap)
        
        # Binary cross entropy loss
        pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(sources))
        return loss
    
    def _shared_eval_step(self, batch, batch_idx, prefix: str):
        sources = batch['sources'].cpu().numpy()
        destinations = batch['destinations'].cpu().numpy()
        timestamps = batch['timestamps'].cpu().numpy()
        neg_sources = batch['negative_sources'].cpu().numpy()
        neg_destinations = batch['negative_destinations'].cpu().numpy()
        
        num_neighbors = getattr(self.cfg, 'num_tokens', 20)
        time_gap = getattr(self.cfg, 'time_gap', 2000)
        
       
        pos_scores = self.forward(sources, destinations, timestamps, num_neighbors, time_gap)
        neg_scores = self.forward(neg_sources, neg_destinations, timestamps, num_neighbors, time_gap)
        
        
        pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        preds = torch.cat([pos_scores, neg_scores])
        targets = torch.cat([
            torch.ones_like(pos_scores), 
            torch.zeros_like(neg_scores)
        ])
        
        # Select appropriate metrics
        if prefix == 'val':
            auroc = self.val_auroc(preds, targets)
            ap = self.val_ap(preds, targets)
            acc = self.val_accuracy(preds, targets > 0.5)
        else:
            auroc = self.test_auroc(preds, targets)
            ap = self.test_ap(preds, targets)
            acc = self.test_accuracy(preds, targets > 0.5)
        
        self.log(f'{prefix}_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(sources))
        self.log(f'{prefix}_auc', auroc, on_epoch=True, batch_size=len(sources))
        self.log(f'{prefix}_ap', ap, on_epoch=True, batch_size=len(sources))
        self.log(f'{prefix}_accuracy', acc, on_epoch=True, batch_size=len(sources))
        
        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'test')
    
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20, time_gap: int = 2000):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)

        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         num_neighbors: int = 20, time_gap: int = 2000):
        """
        given node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings of nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # link encoder
        # get temporal neighbors, including neighbor ids, edge ids and time information
        # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_times, ndarray, shape (batch_size, num_neighbors)
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        nodes_neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim + time_feat_dim)
        combined_features = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        # Tensor, shape (batch_size, num_neighbors, num_channels)
        combined_features = self.projection_layer(combined_features)

        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            combined_features = mlp_mixer(input_tensor=combined_features)

        # Tensor, shape (batch_size, num_channels)
        combined_features = torch.mean(combined_features, dim=1)

        # node encoder
        # get temporal neighbors of nodes, including neighbor ids
        # time_gap_neighbor_node_ids, ndarray, shape (batch_size, time_gap)
        time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                                                          node_interact_times=node_interact_times,
                                                                                          num_neighbors=time_gap)

        # Tensor, shape (batch_size, time_gap, node_feat_dim)
        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(time_gap_neighbor_node_ids)]

        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask = torch.from_numpy((time_gap_neighbor_node_ids > 0).astype(np.float32))
        # note that if a node has no valid neighbor (whose valid_time_gap_neighbor_node_ids_mask are all zero), directly set the mask to -np.inf will make the
        # scores after softmax be nan. Therefore, we choose a very large negative number (-1e10) instead of -np.inf to tackle this case
        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = -1e10
        # Tensor, shape (batch_size, time_gap)
        scores = torch.softmax(valid_time_gap_neighbor_node_ids_mask, dim=1).to(self.device)

        # Tensor, shape (batch_size, node_feat_dim), average over the time_gap neighbors
        nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)

        # Tensor, shape (batch_size, node_feat_dim), add features of nodes in node_ids
        output_node_features = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[torch.from_numpy(node_ids)]

        # Tensor, shape (batch_size, node_feat_dim)
        node_embeddings = self.output_layer(torch.cat([combined_features, output_node_features], dim=1))

        return node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )



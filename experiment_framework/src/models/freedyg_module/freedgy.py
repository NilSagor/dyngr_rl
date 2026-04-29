# src/models/freedyg_module/freedyg.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import lightning as L

from torchmetrics import AUROC, AveragePrecision, Accuracy

from src.models.freedyg_module.freedyg_config import FreeDyGConfig
from src.models.freedyg_module.components.TimeEncoder import TimeEncoder
# from src.models.freedyg_module.components.NeighborSampler import NeighborSampler
from src.models.freedyg_module.components.neighbor_sampler import NeighborSampler
from src.models.freedyg_module.components.MLPMixer import MLPMixer
from src.models.freedyg_module.components.NIFEncoder import NIFEncoder


class FreeDyG(L.LightningModule):
    """
    FreeDyG: Feature-Rich Dynamic Graph Learning with MLP-Mixer.
    
    Key differences from GraphMixer:
    - Single encoder (no separate link/node encoders)
    - NIF (Neighbor Interaction Feature) encoding
    - Learned weighted aggregation instead of mean pooling
    - Unified projection for node/edge/time/nif features
    """

    def __init__(self, config: FreeDyGConfig):
        super(FreeDyG, self).__init__()
        self.save_hyperparameters()
        self.cfg = config
        
        # Device handling
        self.device = torch.device(
            config.device if isinstance(config.device, str) else 'cpu'
        )
        
        # Feature tensors
        if isinstance(config.node_raw_features, np.ndarray):
            self.node_raw_features = torch.from_numpy(
                config.node_raw_features.astype(np.float32)
            ).to(self.device)
        else:
            self.node_raw_features = config.node_raw_features.to(self.device)
            
        if isinstance(config.edge_raw_features, np.ndarray):
            self.edge_raw_features = torch.from_numpy(
                config.edge_raw_features.astype(np.float32)
            ).to(self.device)
        else:
            self.edge_raw_features = config.edge_raw_features.to(self.device)
        
        self.neighbor_sampler = config.neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = config.time_feat_dim
        self.channel_embedding_dim = config.channel_embedding_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.max_input_sequence_length = config.max_input_sequence_length
        self.nif_feat_dim = config.nif_feat_dim
        
        # Time encoder (frozen)
        self.time_encoder = TimeEncoder(
            time_dim=self.time_feat_dim, 
            parameter_requires_grad=False
        )
        
        # NIF Encoder
        self.nif_encoder = NIFEncoder(
            nif_feat_dim=self.nif_feat_dim, 
            device=self.device
        )
        
        # Projection layers for each feature type → channel_embedding_dim
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(self.node_feat_dim, self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(self.edge_feat_dim, self.channel_embedding_dim, bias=True),
            'time': nn.Linear(self.time_feat_dim, self.channel_embedding_dim, bias=True),
            'nif': nn.Linear(self.nif_feat_dim, self.channel_embedding_dim, bias=True)
        })
        
        # Reduce concatenated features back to channel_embedding_dim
        self.reduce_layer = nn.Linear(
            4 * self.channel_embedding_dim, 
            self.channel_embedding_dim
        )
        
        # MLP-Mixer blocks
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(
                num_tokens=self.max_input_sequence_length,
                num_channels=self.channel_embedding_dim,
                token_dim_expansion_factor=0.5,
                channel_dim_expansion_factor=4.0,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Learned weighted aggregation (replaces mean pooling)
        self.weightagg = nn.Linear(self.channel_embedding_dim, 1)
        
        # Output projection for link prediction
        self.output_layer = nn.Linear(
            self.channel_embedding_dim, 
            self.channel_embedding_dim
        )
        
        # Metrics
        self.val_auroc = AUROC(task='binary')
        self.val_ap = AveragePrecision(task='binary')
        self.test_auroc = AUROC(task='binary')
        self.test_ap = AveragePrecision(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                node_interact_times: np.ndarray):
        """
        Compute edge probabilities for link prediction.
        """
        src_emb, dst_emb = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times
        )
        # Affinity score: dot product + sigmoid
        scores = (src_emb * dst_emb).sum(dim=1)
        return torch.sigmoid(scores)

    def compute_src_dst_node_temporal_embeddings(
        self, 
        src_node_ids: np.ndarray, 
        dst_node_ids: np.ndarray, 
        node_interact_times: np.ndarray
    ):
        """Compute temporal embeddings for source and destination nodes."""
        
        # Sample neighbors for both src and dst
        src_n_ids, src_e_ids, src_t_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=self.max_input_sequence_length
        )
        dst_n_ids, dst_e_ids, dst_t_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=self.max_input_sequence_length
        )
        
        # NIF features
        src_nif, dst_nif = self.nif_encoder(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            src_nodes_neighbor_ids=src_n_ids,
            dst_nodes_neighbor_ids=dst_n_ids
        )
        
        # Raw features + time encoding
        src_node_feat, src_edge_feat, src_time_feat = self._get_features(
            node_interact_times, src_n_ids, src_e_ids, src_t_times
        )
        dst_node_feat, dst_edge_feat, dst_time_feat = self._get_features(
            node_interact_times, dst_n_ids, dst_e_ids, dst_t_times
        )
        
        # Project all features to channel_embedding_dim
        src_node_feat = self.projection_layer['node'](src_node_feat)
        src_edge_feat = self.projection_layer['edge'](src_edge_feat)
        src_time_feat = self.projection_layer['time'](src_time_feat)
        src_nif = self.projection_layer['nif'](src_nif)
        
        dst_node_feat = self.projection_layer['node'](dst_node_feat)
        dst_edge_feat = self.projection_layer['edge'](dst_edge_feat)
        dst_time_feat = self.projection_layer['time'](dst_time_feat)
        dst_nif = self.projection_layer['nif'](dst_nif)
        
        # Concatenate and reduce
        src_combined = torch.cat([src_node_feat, src_edge_feat, src_time_feat, src_nif], dim=-1)
        dst_combined = torch.cat([dst_node_feat, dst_edge_feat, dst_time_feat, dst_nif], dim=-1)
        
        src_combined = self.reduce_layer(src_combined)
        dst_combined = self.reduce_layer(dst_combined)
        
        # Apply MLP-Mixer layers
        for mixer in self.mlp_mixers:
            src_combined = mixer(src_combined)
            dst_combined = mixer(dst_combined)
        
        # Learned weighted aggregation (FreeDyG's key innovation)
        src_weights = F.softmax(self.weightagg(src_combined).squeeze(-1), dim=1)  # (B, seq_len)
        dst_weights = F.softmax(self.weightagg(dst_combined).squeeze(-1), dim=1)
        
        src_emb = (src_weights.unsqueeze(-1) * src_combined).sum(dim=1)  # (B, channel_dim)
        dst_emb = (dst_weights.unsqueeze(-1) * dst_combined).sum(dim=1)
        
        # Final projection
        src_emb = self.output_layer(src_emb)
        dst_emb = self.output_layer(dst_emb)
        
        return src_emb, dst_emb

    def _get_features(self, node_interact_times: np.ndarray, 
                      nodes_neighbor_ids: np.ndarray,
                      nodes_edge_ids: np.ndarray,
                      nodes_neighbor_times: np.ndarray):
        """Extract node, edge, and time features for a sequence of neighbors."""
        # Node features
        node_feat = self.node_raw_features[
            torch.from_numpy(nodes_neighbor_ids).to(self.device)
        ]  # (B, seq_len, node_feat_dim)
        
        # Edge features
        edge_feat = self.edge_raw_features[
            torch.from_numpy(nodes_edge_ids).to(self.device)
        ]  # (B, seq_len, edge_feat_dim)
        
        # Time features (relative encoding)
        time_deltas = node_interact_times[:, np.newaxis] - nodes_neighbor_times
        time_feat = self.time_encoder(
            torch.from_numpy(time_deltas).float().to(self.device)
        )  # (B, seq_len, time_feat_dim)
        
        # Mask padded neighbors (id == 0)
        padding_mask = torch.from_numpy(nodes_neighbor_ids == 0).to(self.device)
        time_feat[padding_mask] = 0.0
        
        return node_feat, edge_feat, time_feat

    def training_step(self, batch, batch_idx):
        src = batch['sources'].cpu().numpy()
        dst = batch['destinations'].cpu().numpy()
        ts = batch['timestamps'].cpu().numpy()
        neg_src = batch['negative_sources'].cpu().numpy()
        neg_dst = batch['negative_destinations'].cpu().numpy()
        
        pos_scores = self.forward(src, dst, ts)
        neg_scores = self.forward(neg_src, neg_dst, ts)
        
        pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, 
                 prog_bar=True, batch_size=len(src))
        return loss

    def _shared_eval_step(self, batch, batch_idx, prefix: str):
        src = batch['sources'].cpu().numpy()
        dst = batch['destinations'].cpu().numpy()
        ts = batch['timestamps'].cpu().numpy()
        neg_src = batch['negative_sources'].cpu().numpy()
        neg_dst = batch['negative_destinations'].cpu().numpy()
        
        pos_scores = self.forward(src, dst, ts)
        neg_scores = self.forward(neg_src, neg_dst, ts)
        
        pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        preds = torch.cat([pos_scores, neg_scores])
        targets = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])
        
        # Select metrics
        metrics = {
            'val': (self.val_auroc, self.val_ap, self.val_accuracy),
            'test': (self.test_auroc, self.test_ap, self.test_accuracy)
        }
        auroc_fn, ap_fn, acc_fn = metrics[prefix]
        
        auroc = auroc_fn(preds, targets)
        ap = ap_fn(preds, targets)
        acc = acc_fn(preds, targets > 0.5)
        
        self.log(f'{prefix}_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(src))
        self.log(f'{prefix}_auc', auroc, on_epoch=True, batch_size=len(src))
        self.log(f'{prefix}_ap', ap, on_epoch=True, batch_size=len(src))
        self.log(f'{prefix}_accuracy', acc, on_epoch=True, batch_size=len(src))
        
        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """Inject neighbor sampler (called by runner)."""
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def _log_model_status(self) -> None:
        """Debug helper: log architecture summary."""
        logger.info(f"=== FreeDyG Architecture ===")
        logger.info(f"Node feat dim: {self.node_feat_dim}")
        logger.info(f"Edge feat dim: {self.edge_feat_dim}")
        logger.info(f"Channel embedding dim: {self.channel_embedding_dim}")
        logger.info(f"Max sequence length: {self.max_input_sequence_length}")
        logger.info(f"MLP-Mixer layers: {self.num_layers}")
        logger.info(f"NIF feature dim: {self.nif_feat_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        logger.info("=" * 40)
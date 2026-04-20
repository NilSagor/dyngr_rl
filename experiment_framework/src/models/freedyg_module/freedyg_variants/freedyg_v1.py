import torch
import torch.nn as nn
import lightning.pytorch as pl

from torchmetrics import Accuracy, AUROC, AveragePrecision
from src.models.freedyg_module.Freedyg import FreeDyG
from src.models.freedyg_module.MergeLayer import MergeLayer

class FreeDyGLightningModule(pl.LightningModule):
    def __init__(self, args, node_raw_features, edge_raw_features, neighbor_sampler, compile_model=False):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler

        # Build FreeDyG backbone + link predictor
        self.dynamic_backbone = FreeDyG(
            node_raw_features=node_raw_features.cpu().numpy(),
            edge_raw_features=edge_raw_features.cpu().numpy(),
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            channel_embedding_dim=args.channel_embedding_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_input_sequence_length=args.max_input_sequence_length,
            device=self.device
        )
        self.link_predictor = MergeLayer(
            input_dim1=node_raw_features.shape[1],
            input_dim2=node_raw_features.shape[1],
            hidden_dim=node_raw_features.shape[1],
            output_dim=1
        )
        self.model = nn.Sequential(self.dynamic_backbone, self.link_predictor)

        if compile_model:
            self.model = torch.compile(self.model)

        self.loss_fn = nn.BCELoss()

        # Metrics
        self.train_ap = AveragePrecision(task="binary")
        self.train_auc = AUROC(task="binary")
        self.train_acc = Accuracy(task="binary")

        self.val_ap = AveragePrecision(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_acc = Accuracy(task="binary")

        self.test_ap = AveragePrecision(task="binary")
        self.test_auc = AUROC(task="binary")
        self.test_acc = Accuracy(task="binary")

    def forward(self, src_ids, dst_ids, timestamps):
        src_emb, dst_emb = self.dynamic_backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_ids.cpu().numpy(),
            dst_node_ids=dst_ids.cpu().numpy(),
            node_interact_times=timestamps.cpu().numpy()
        )
        prob = self.link_predictor(src_emb, dst_emb).squeeze(-1).sigmoid()
        return prob

    def shared_step(self, batch, batch_idx, stage="train"):
        src_ids, dst_ids, timestamps, edge_ids = batch

        # Negative sampling
        if stage == "train":
            neg_sampler = self.trainer.datamodule.train_neg_sampler
        elif stage == "val":
            neg_sampler = self.trainer.datamodule.val_neg_sampler
        else:  # test
            neg_sampler = self.trainer.datamodule.test_neg_sampler

        neg_dst = neg_sampler.sample(len(src_ids))[1]
        neg_src = src_ids  # keep source same

        # Compute embeddings
        pos_src_emb, pos_dst_emb = self.dynamic_backbone.compute_src_dst_node_temporal_embeddings(
            src_ids.cpu().numpy(), dst_ids.cpu().numpy(), timestamps.cpu().numpy()
        )
        neg_src_emb, neg_dst_emb = self.dynamic_backbone.compute_src_dst_node_temporal_embeddings(
            neg_src.cpu().numpy(), neg_dst.cpu().numpy(), timestamps.cpu().numpy()
        )

        pos_prob = self.link_predictor(pos_src_emb, pos_dst_emb).squeeze(-1).sigmoid()
        neg_prob = self.link_predictor(neg_src_emb, neg_dst_emb).squeeze(-1).sigmoid()

        preds = torch.cat([pos_prob, neg_prob], dim=0)
        labels = torch.cat([torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0)

        loss = self.loss_fn(preds, labels)

        # Update metrics
        if stage == "train":
            self.train_ap.update(preds, labels.int())
            self.train_auc.update(preds, labels.int())
            self.train_acc.update(preds, labels.int())
        elif stage == "val":
            self.val_ap.update(preds, labels.int())
            self.val_auc.update(preds, labels.int())
            self.val_acc.update(preds, labels.int())
        else:
            self.test_ap.update(preds, labels.int())
            self.test_auc.update(preds, labels.int())
            self.test_acc.update(preds, labels.int())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch[0]))
        return loss

    def on_train_epoch_end(self):
        self.log("train_ap", self.train_ap.compute(), prog_bar=True)
        self.log("train_auc", self.train_auc.compute(), prog_bar=True)
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_ap.reset()
        self.train_auc.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "val")
        self.log("val_loss", loss, sync_dist=True, batch_size=len(batch[0]))
        return loss

    def on_validation_epoch_end(self):        
        self.log("val_ap", self.val_ap.compute(), sync_dist=True, prog_bar=True)
        self.log("val_auc", self.val_auc.compute(), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), sync_dist=True, prog_bar=True)        
        self.val_ap.reset()
        self.val_auc.reset()
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "test")
        self.log("test_loss", loss, sync_dist=True, batch_size=len(batch[0]))
        return loss

    def on_test_epoch_end(self):
        self.log("test_ap", self.test_ap.compute(), sync_dist=True)
        self.log("test_auc", self.test_auc.compute(), sync_dist=True)
        self.log("test_acc", self.test_acc.compute(), sync_dist=True)
        self.test_ap.reset()
        self.test_auc.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        return optimizer
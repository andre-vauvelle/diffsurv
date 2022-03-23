from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as f
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
from torch import nn
from torchmetrics import MetricCollection, AveragePrecision, Precision, Accuracy, AUROC
import pandas as pd

from data.preprocess.utils import vocab_omop_embedding, SYMBOL_IDX
from definitions import TENSORBOARD_DIR
from models.heads import PredictionHead
from modules.loss import CoxPHLoss


class MultilayerBase(pl.LightningModule):
    def __init__(self,
                 # input_dim=4846,
                 input_dim=1390,
                 output_dim=1390, embedding_dim=128, hidden_dropout_prob=0.2, lr=1e-4,
                 pretrained_embedding_path=None, freeze_pretrained=False, single_multihot_training=True, count=True,
                 used_covs=('age_ass', 'sex'), only_covs=False):
        super().__init__()
        self.lr = lr

        # self.loss_func = nn.BCEWithLogitsLoss()  # Required for multihot training
        self.loss_func = CoxPHLoss()
        self.single_multihot_training = single_multihot_training
        self.count = count
        self.used_covs = used_covs

        if pretrained_embedding_path is None:
            self.embed = nn.EmbeddingBag(num_embeddings=input_dim, embedding_dim=embedding_dim,
                                         padding_idx=SYMBOL_IDX['PAD'],
                                         mode="mean", sparse=True)
        else:
            # Use preprocess_ukb_omop.py to preprocess
            pretrained_embedding = torch.load(pretrained_embedding_path).float()
            # pretrained_embedding = pd.read_feather(pretrained_embedding_path)
            self.embed = nn.EmbeddingBag.from_pretrained(pretrained_embedding, freeze=freeze_pretrained, sparse=True)

        if self.used_covs is None:
            self.head = PredictionHead(in_features=embedding_dim, out_features=output_dim)
        elif self.used_covs is not None and only_covs:
            self.head = PredictionHead(in_features=len(self.used_covs), out_features=output_dim)
        else:
            self.head = PredictionHead(in_features=embedding_dim + len(self.used_covs), out_features=output_dim)

        self.save_hyperparameters()

        self.input_dim = input_dim
        self.output_dim = output_dim

        metrics = MetricCollection(
            [
                AveragePrecision(num_classes=self.output_dim, compute_on_step=False, average='weighted'),
                Precision(compute_on_step=False, average='micro'),
                Accuracy(compute_on_step=False, average='micro'),
                AUROC(num_classes=self.output_dim, compute_on_step=False)
            ]
        )

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def forward(self, idx, covariates=None) -> torch.Tensor:
        if not self.only_covs:
            pooled = covariates
        else:
            pooled = self.embed(idx)
            if self.used_covs is not None:
                pooled = torch.cat((pooled, covariates), dim=1)

        logits = self.head(pooled)

        return logits

    @staticmethod
    def _row_unique(x):
        """get unique unique idx for each row"""
        # sorting the rows so that duplicate values appear together
        # e.g., first row: [1, 2, 3, 3, 3, 4, 4]
        y, indices = x.sort(dim=-1)

        # subtracting, so duplicate values will become 0
        # e.g., first row: [1, 2, 3, 0, 0, 4, 0]
        y[:, 1:] *= ((y[:, 1:] - y[:, :-1]) != 0).long()

        # retrieving the original indices of elements
        indices = indices.sort(dim=-1)[1]

        # re-organizing the rows following original order
        # e.g., first row: [1, 2, 3, 4, 0, 0, 0]
        result = torch.gather(y, 1, indices)
        return result

    def _shared_eval_step(self, batch, batch_idx):
        rank_zero_warn("`_shared_eval_step` must be implemented to be used with the Lightning Trainer")

    def training_step(self, batch, batch_idx, optimizer_idx):
        rank_zero_warn("`training_step` must be implemented to be used with the Lightning Trainer")

    def validation_step(self, batch, batch_idx):
        rank_zero_warn("`validation_step` must be implemented to be used with the Lightning Trainer")

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)
        self.log('hp_metric', output['val/AveragePrecision'])

    def test_step(self, batch, batch_idx):
        rank_zero_warn("`test_step` must be implemented to be used with the Lightning Trainer")

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(output, prog_bar=True)

    def configure_optimizers(self):

        sparse = [p for n, p in self.named_parameters() if 'embed' in n]
        not_sparse = [p for n, p in self.named_parameters() if 'embed' not in n]
        optimizer_sparse = torch.optim.SparseAdam(sparse, lr=self.lr)
        optimizer = torch.optim.Adam(not_sparse, lr=self.lr)
        return optimizer_sparse, optimizer


class MultilayerRisk(MultilayerBase):
    def __init__(self,
                 input_dim=1390,
                 output_dim=1390, embedding_dim=128, hidden_dropout_prob=0.2, lr=1e-4,
                 pretrained_embedding_path=None, freeze_pretrained=False, single_multihot_training=True, count=True,
                 ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, embedding_dim=embedding_dim,
                         hidden_dropout_prob=hidden_dropout_prob,
                         lr=lr, pretrained_embedding_path=pretrained_embedding_path,
                         freeze_pretrained=freeze_pretrained,
                         single_multihot_training=single_multihot_training, count=count
                         )

    def _shared_eval_step(self, batch, batch_idx):
        (token_idx, age_idx, position, segment, mask, covariates), (label_multihot, label_times) = batch
        if not self.count:
            token_idx = self._row_unique(token_idx)

        logits = self(token_idx, covariates=covariates)

        loss = self.loss_func(logits, label_multihot, label_times)
        return loss, logits, label_multihot

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss, _, _ = self._shared_eval_step(batch, batch_idx)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, label_multihot = self._shared_eval_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)

        predictions = torch.sigmoid(logits)
        self.valid_metrics.update(predictions, label_multihot.int())

    def test_step(self, batch, batch_idx):
        # TODO: implement
        rank_zero_warn("`test_step` must be implemented to be used with the Lightning Trainer testing")


class MultilayerMLM(MultilayerBase):
    def __init__(self):
        super().__init__()

    def _shared_eval_step(self, batch, batch_idx):
        token_idx, age_idx, position, segment, mask_labels, noise_labels, mask = batch
        if not self.count:
            token_idx = self._row_unique(token_idx)

        logits = self(token_idx)

        mask_labels_expanded = mask_labels.view(-1)
        logits_expanded = logits.repeat(mask_labels.shape[1], 1)

        loss = self.loss_func(logits_expanded, mask_labels_expanded)
        return loss, logits_expanded, mask_labels_expanded

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss, logits_expanded, mask_labels_expanded = self._shared_eval_step(batch, batch_idx)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits_expanded, mask_labels_expanded = self._shared_eval_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)

        predictions = f.softmax(logits_expanded.view(-1, self.output_dim), dim=1)
        keep = mask_labels_expanded.view(-1) != -1
        mask_labels_reduced = mask_labels_expanded.view(-1)[keep]
        predictions = predictions[keep]
        self.valid_metrics.update(predictions, mask_labels_reduced)

    def test_step(self, batch, batch_idx):
        token_idx, age_idx, position, segment, mask_labels, noise_labels = batch
        logits = self(token_idx)
        expanded_logits = logits.repeat(mask_labels.shape[1], 1)
        loss = self.loss_func(expanded_logits, mask_labels.view(-1))
        self.log('test/loss', loss, prog_bar=True)

        predictions = f.softmax(expanded_logits.view(-1, self.input_dim), dim=1)
        keep = mask_labels.view(-1) != -1
        mask_labels = mask_labels.view(-1)[keep]
        predictions = predictions[keep]
        self.test_metrics.update(predictions, mask_labels)

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
from models.metrics import CIndex
from modules.base import BaseModel
from modules.loss import CoxPHLoss
from modules.tasks import RiskMixin
from omni.common import safe_string


class MultilayerBase(BaseModel):
    def __init__(self,
                 # input_dim=4846,
                 input_dim=1390,
                 output_dim=1390, embedding_dim=128, hidden_dropout_prob=0.2, lr=1e-4,
                 pretrained_embedding_path=None, freeze_pretrained=False, count=True,
                 used_covs=('age_ass', 'sex'), only_covs=False, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr

        # self.loss_func = nn.BCEWithLogitsLoss()  # Required for multihot training
        self.count = count
        self.used_covs = used_covs
        self.only_covs = only_covs

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

        self.input_dim = input_dim
        self.output_dim = output_dim

        # TODO: consider moving to mixin
        metrics = MetricCollection(
            [
                AveragePrecision(num_classes=self.output_dim, compute_on_step=False, average='weighted'),
                Precision(compute_on_step=False, average='micro'),
                Accuracy(compute_on_step=False, average='micro'),
                AUROC(num_classes=self.output_dim, compute_on_step=False)
            ]
        )

        self.valid_metrics = metrics.clone(prefix='val/')
        self.save_hyperparameters()

    def forward(self, idx, covariates=None) -> torch.Tensor:
        if self.only_covs:
            pooled = covariates.float().requires_grad_()
        else:
            pooled = self.embed(idx)
            if self.used_covs is not None:
                pooled = torch.cat((pooled, covariates), dim=1)

        logits = self.head(pooled)

        return logits


class MultilayerRisk(RiskMixin, MultilayerBase):
    def __init__(self,
                 input_dim=1390,
                 output_dim=1390, embedding_dim=256, hidden_dropout_prob=0.1, lr=1e-4,
                 pretrained_embedding_path=None, freeze_pretrained=False, count=True,
                 only_covs=False, label_vocab=None, grouping_labels=None, used_covs=('age_ass', 'sex'), weightings=None,
                 use_weighted_loss=False, loss=None
                 ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, embedding_dim=embedding_dim,
                         hidden_dropout_prob=hidden_dropout_prob,
                         lr=lr, pretrained_embedding_path=pretrained_embedding_path,
                         freeze_pretrained=freeze_pretrained, used_covs=used_covs,
                         count=count,
                         only_covs=only_covs, label_vocab=label_vocab, grouping_labels=grouping_labels, loss=loss,
                         weightings=weightings,
                         use_weighted_loss=use_weighted_loss)
        self.save_hyperparameters()

    def _shared_eval_step(self, batch, batch_idx):
        (token_idx, age_idx, position, segment, mask, covariates), (label_multihot, label_times, censorings,
                                                                    exclusions) = batch
        if not self.count:
            token_idx = self._row_unique(token_idx)

        logits = self(token_idx, covariates=covariates)

        loss = self.loss_func(logits, label_multihot, label_times)
        return loss, logits, label_multihot, label_times


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

from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as f
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchmetrics import MetricCollection, AveragePrecision, Precision, Accuracy, AUROC
import pandas as pd

from data.datasets import PADDING_IDX
from data.preprocess.utils import vocab_omop_embedding
from definitions import TENSORBOARD_DIR


class MultilayerMLM(pl.LightningModule):
    def __init__(self, input_dim=4846, output_dim=1390, embedding_dim=50, hidden_dropout_prob=0.2, lr=1e-4,
                 pretrained_embedding_path=None, freeze_pretrained=False, single_multihot_training=True):
        super().__init__()
        self.lr = lr

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.single_multihot_training = single_multihot_training

        if pretrained_embedding_path is None:

            self.embed = nn.EmbeddingBag(num_embeddings=input_dim, embedding_dim=embedding_dim, padding_idx=PADDING_IDX,
                                         mode="mean")
        else:
            # Use preprocess_ukb_omop.py to preprocess
            # pretrained_embedding = torch.load(pretrained_embedding_path)
            pretrained_embedding = pd.read_feather(pretrained_embedding_path)
            self.embed = nn.EmbeddingBag.from_pretrained(pretrained_embedding, freeze=freeze_pretrained)

        # self.embeddings = nn.Sequential(
        #     self.embed,
        #     nn.LayerNorm(normalized_shape=embedding_dim),
        #     nn.Dropout(hidden_dropout_prob)
        # )
        self.head = nn.Linear(in_features=embedding_dim, out_features=output_dim)

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

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, idx) -> torch.Tensor:
        pooled = self.embed(idx)

        logits = self.head(pooled)

        return logits

    def training_step(self, batch, batch_idx):
        token_idx, age_idx, position, segment, mask_labels, noise_labels = batch
        logits = self(token_idx)

        mask_labels_expanded = mask_labels.view(-1)
        logits_expanded = logits.repeat(mask_labels.shape[1], 1)

        loss = self.loss_func(logits_expanded, mask_labels_expanded)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        token_idx, age_idx, position, segment, mask_labels, noise_labels = batch
        logits = self(token_idx)

        mask_labels_expanded = mask_labels.view(-1)
        logits_expanded = logits.repeat(mask_labels.shape[1], 1)

        loss = self.loss_func(logits_expanded, mask_labels_expanded)
        self.log('val_loss', loss, prog_bar=True)

        predictions = f.softmax(logits_expanded.view(-1, self.output_dim), dim=1)
        keep = mask_labels.view(-1) != -1
        mask_labels = mask_labels.view(-1)[keep]
        predictions = predictions[keep]
        self.valid_metrics.update(predictions, mask_labels)

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)

    def test_step(self, batch, batch_idx):
        token_idx, age_idx, position, segment, mask_labels, noise_labels = batch
        logits = self(token_idx)
        expanded_logits = logits.repeat(mask_labels.shape[1], 1)
        loss = self.loss_func(expanded_logits, mask_labels.view(-1))
        self.log('test_loss', loss, prog_bar=True)

        predictions = f.softmax(expanded_logits.view(-1, self.input_dim), dim=1)
        keep = mask_labels.view(-1) != -1
        mask_labels = mask_labels.view(-1)[keep]
        predictions = predictions[keep]
        self.test_metrics.update(predictions, mask_labels)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(output, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

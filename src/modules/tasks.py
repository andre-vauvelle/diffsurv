import importlib

import torch
from diffsort import diffsort
from pytorch_lightning.utilities import rank_zero_warn
from torchmetrics import MetricCollection

import pytorch_lightning as pl
from models.metrics import CIndex
from omni.common import safe_string, unsafe_string
from modules.loss import CoxPHLoss, SortingCrossEntropyLoss
from modules.sorter import CustomDiffSortNet
import pdb


class RiskMixin(pl.LightningModule):
    def __init__(self, grouping_labels, label_vocab, loss=None, weightings=None, use_weighted_loss=False, **kwargs):
        super().__init__(**kwargs)
        self.label_vocab = label_vocab
        self.grouping_labels = grouping_labels

        c_index_metric_names = list(self.label_vocab['token2idx'].keys())
        c_index_metrics = MetricCollection(
            {'c_index/' + safe_string(name): CIndex() for name in c_index_metric_names}
        )
        self.valid_cindex = c_index_metrics.clone(prefix='val/')

        if loss is None:
            self.loss_func = CoxPHLoss()
        else:
            loss = loss[0]
            loss_args = loss['args']
            self.sorter = CustomDiffSortNet(
                sorting_network_type=loss_args['sorting_network'],
                size=loss_args['num_compare'],
                steepness=loss_args['steepness'],
                art_lambda=loss_args['art_lambda'],
                distribution=loss_args['distribution'],
            )
            self.loss_func = SortingCrossEntropyLoss(self.sorter)

        if weightings is not None:
            self.loss_func_w = CoxPHLoss(weightings=weightings)
        else:
            self.loss_func_w = None

        self.use_weighted_loss = use_weighted_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss, logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)
        loss = self.loss_func(logits, label_multihot, label_times)
        self.log('train/CoxPH', loss)

        if self.loss_func_w is not None:
            loss_weighted = self.loss_func_w(logits, label_multihot, label_times)
            self.log('train/CoxPH_weighted', loss)
            if self.use_weighted_loss:
                loss = loss_weighted

        self.log('train/loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)
        self.log('val/CoxPH', loss, prog_bar=True)
        # calc and use weighted loss
        if self.loss_func_w is not None:
            loss_weighted = self.loss_func_w(logits, label_multihot, label_times)
            self.log('val/CoxPH_weighted', loss)
            if self.use_weighted_loss:
                loss = loss_weighted
        self.log('val/loss', loss, prog_bar=True)

        predictions = torch.sigmoid(logits)
        self.valid_metrics.update(predictions, label_multihot.int())
        label_times = batch[1][1]
        exclusions = batch[1][3]

        # c-index is applied per label
        for name, metric in self.valid_cindex.items():
            # idx = self._groping_idx[name]
            idx = self.label_vocab['token2idx'][unsafe_string(name.split('/')[-1])]
            e = exclusions[:, idx]  # exclude patients with prior history of event
            p, l, t = predictions[1 - e, idx], label_multihot[1 - e, idx], label_times[1 - e, idx]
            metric.update(p, l.int(), t)

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)
        output = self.valid_cindex.compute()
        self._group_cindex(output)
        self.valid_cindex.reset()
        self.log_dict(output, prog_bar=False)
        self.log('hp_metric', output['val/c_index/all'])

    def test_step(self, batch, batch_idx):
        # TODO: implement
        rank_zero_warn("`test_step` must be implemented to be used with the Lightning Trainer testing")

    def _group_cindex(self, output):
        """
        Group c-index by label
        :param output:
        :return:
        """

        for name, labels in self.grouping_labels.items():
            values = []
            for label in labels:
                try:
                    v = output['val/c_index/' + safe_string(label)]
                    if not torch.isnan(v):
                        values.append(v)
                except KeyError:
                    pass
            if len(values) > 0:
                average_value = sum(values) / len(values)
                output.update({'val/c_index/' + safe_string(name): average_value})

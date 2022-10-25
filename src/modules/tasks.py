import importlib
import pdb

import pytorch_lightning as pl
import torch
import torchvision.utils
from diffsort import diffsort
from pytorch_lightning.utilities import rank_zero_warn
from torchmetrics import MetricCollection

from models.metrics import CIndex
from modules.loss import CoxPHLoss, CustomBCEWithLogitsLoss, SortingCrossEntropyLoss
from modules.sorter import CustomDiffSortNet
from omni.common import safe_string, unsafe_string


class RiskMixin(pl.LightningModule):
    def __init__(
        self,
        grouping_labels,
        label_vocab,
        task="risk",
        weightings=None,
        use_weighted_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_vocab = label_vocab
        self.grouping_labels = grouping_labels

        c_index_metric_names = list(self.label_vocab["token2idx"].keys())
        c_index_metrics = MetricCollection(
            {"c_index/" + safe_string(name): CIndex() for name in c_index_metric_names}
        )
        self.valid_cindex = c_index_metrics.clone(prefix="val/")

        if task == "risk":
            self.loss_func = CoxPHLoss()
        elif task == "next":
            self.loss_func = CustomBCEWithLogitsLoss()

        if weightings is not None:
            self.loss_func_w = CoxPHLoss(weightings=weightings)
        else:
            self.loss_func_w = None

        self.use_weighted_loss = use_weighted_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)

        if self.loss_func_w is not None:
            loss = self.loss_func_w(logits, label_multihot, label_times)
        else:
            loss = self.loss_func(logits, label_multihot, label_times)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)
        # calc and use weighted loss
        if self.loss_func_w is not None:
            loss = self.loss_func_w(logits, label_multihot, label_times)
        else:
            loss = self.loss_func(logits, label_multihot, label_times)

        if not torch.isnan(loss):
            self.log("val/loss", loss, prog_bar=True)

        predictions = torch.sigmoid(logits)
        self.valid_metrics.update(predictions, label_multihot.int())
        label_times = batch["label_times"]
        exclusions = batch["exclusions"]

        # c-index is applied per label
        for name, metric in self.valid_cindex.items():
            # idx = self._groping_idx[name]
            idx = self.label_vocab["token2idx"][unsafe_string(name.split("/")[-1])]
            e = exclusions[:, idx]  # exclude patients with prior history of event
            e_idx = (1 - e).bool()
            p, l, t = logits[e_idx, idx], label_multihot[e_idx, idx], label_times[e_idx, idx]
            metric.update(p, l.int(), t)

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=False)
        output = self.valid_cindex.compute()
        self._group_cindex(output)
        self.valid_cindex.reset()
        self.log_dict(output, prog_bar=False)
        self.log("hp_metric", output["val/c_index/all"], prog_bar=True)

    def test_step(self, batch, batch_idx):
        # TODO: implement
        rank_zero_warn(
            "`test_step` must be implemented to be used with the Lightning Trainer testing"
        )

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
                    v = output["val/c_index/" + safe_string(label)]
                    if not torch.isnan(v):
                        values.append(v)
                except KeyError:
                    pass
            if len(values) > 0:
                average_value = sum(values) / len(values)
                output.update({"val/c_index/" + safe_string(name): average_value})


class SortingRiskMixin(RiskMixin):
    """Needs a seperate mixin due to loss function requiring permutation matrices
    #TODO: possible refactor to avoid this
    """

    def __init__(
        self,
        sorting_network="bitonic",
        steepness=30,
        art_lambda=0.2,
        distribution="cauchy",
        sorter_size=128,
        ignore_censoring=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sorter_size = sorter_size

        self.sorter = CustomDiffSortNet(
            sorting_network_type=sorting_network,
            size=self.sorter_size,
            steepness=steepness,
            art_lambda=art_lambda,
            distribution=distribution,
        )
        self.ignore_censoring = ignore_censoring

    def sorting_step(self, logits, label_multihot, label_times):
        losses = torch.zeros(logits.shape[1])
        predictions = torch.zeros_like(label_times)
        for i in range(logits.shape[1]):
            lh, d, e = logits[:, i], label_times[:, i], label_multihot[:, i]

            # TODO: could refactor to dataloader
            # Get the soft permutation matrix
            sort_out, perm_prediction = self.sorter(lh.unsqueeze(0))
            perm_ground_truth = _get_soft_perm(e, d)

            if self.ignore_censoring:
                weight = e
            else:
                weight = None
            loss = torch.nn.BCELoss(weight=weight)(perm_prediction, perm_ground_truth)
            predictions[:, i] = lh
            losses[i] = loss

        loss_idx = losses.gt(0)
        loss = losses[loss_idx].mean()
        return loss, predictions, perm_prediction, perm_ground_truth

    def training_step(self, batch, batch_idx, optimizer_idx):
        logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)
        loss, _, _, _ = self.sorting_step(logits, label_multihot, label_times)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)

        loss, predictions, perm_prediction, perm_ground_truth = self.sorting_step(
            logits, label_multihot, label_times
        )

        self.log("val/loss", loss, prog_bar=True)

        self.valid_metrics.update(predictions, label_multihot.int())
        # label_times = batch['label_times']
        exclusions = batch["exclusions"]

        # c-index is applied per label
        for name, metric in self.valid_cindex.items():
            # idx = self._groping_idx[name]
            idx = self.label_vocab["token2idx"][unsafe_string(name.split("/")[-1])]
            e = exclusions[:, idx]  # exclude patients with prior history of event
            e_idx = (1 - e).bool()
            p, l, t = predictions[e_idx, idx], label_multihot[e_idx, idx], label_times[e_idx, idx]
            metric.update(-1 * p, l.int(), t)  # -1 due to inverse risk/logit vs coxph

        return loss, predictions, perm_prediction, perm_ground_truth

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)
        output = self.valid_cindex.compute()
        self._group_cindex(output)
        self.valid_cindex.reset()
        self.log_dict(output, prog_bar=False)
        self.log("hp_metric", output["val/c_index/all"], prog_bar=True)

    def test_step(self, batch, batch_idx):
        # TODO: implement
        rank_zero_warn(
            "`test_step` must be implemented to be used with the Lightning Trainer testing"
        )


def _get_soft_perm(events, d):
    """
    Returns the soft permutation matrix label for the given events and durations.

    For a right-censored sample `i`, we only know that the risk must be lower than the risk of all other
    samples with an event time lower than the censoring time of `i`, i.e. they must be ranked after
    these events. We can thus assign p=0 of sample `i` being ranked before any prior events, and uniform
    probability that it has a higher ranking.

    For another sample `j` with an event at `t_j`, we know that the risk must be lower than the risk of
    other samples with an event time lower than `t_j`, and higher than the risk of other samples either
    with an event time higher than `t_j` or with a censoring time higher than `t_j`. We do not know how
    the risk compares to samples with censoring time lower than `t_j`, and thus have to assign uniform
    probability to their rankings.
    :param events: binary vector indicating if event happened or not
    :param d: time difference between observation start and event time
    :return:
    """
    # Initialize the soft permutation matrix
    perm_matrix = torch.zeros(events.shape[0], events.shape[0], device=events.device)

    idx = torch.argsort(d, descending=False)

    # Used to return to origonal order
    perm_un_ascending = torch.nn.functional.one_hot(idx).transpose(-2, -1).float()

    events = events[idx]
    event_counts = 0

    # TODO: refactor interms of comparable events
    for i, e in enumerate(events):
        # Right censored samples
        if not e:
            # assign 0 for all samples with event time lower than the censoring time
            perm_matrix[i, :i] = 0
            # assign uniform probability to all samples with event time higher than the censoring time
            # includes previous censored events that happened before the event time
            perm_matrix[i, event_counts:] = 1 / (perm_matrix[i, event_counts:].shape[0])
        # events
        else:
            # assign uniform probability to an event and all censored events with shorted time,
            perm_matrix[i, event_counts : i + 1] = 1 / (
                perm_matrix[i, event_counts : i + 1].shape[0]
            )
            event_counts += 1

    # permute to match the order of the input
    perm_matrix = perm_un_ascending @ perm_matrix

    # Unsqueeze for one batch
    return perm_matrix.unsqueeze(0)


def test_diff_sort_loss_get_soft_perm():
    """Test the soft permutation matrix label for the given events and durations."""
    test_events = torch.Tensor([0, 0, 1, 0, 1, 0, 0])
    test_durations = torch.Tensor([1, 3, 2, 4, 5, 6, 7])
    # logh = torch.Tensor([0, 2, 1, 3, 4, 5, 6])

    required_perm_matrix = torch.Tensor(
        [
            [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
            [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            [1 / 2, 1 / 2, 0, 0, 0, 0, 0],
            [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            [0, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 0, 0],
            [0, 0, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
            [0, 0, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
        ]
    )
    required_perm_matrix = required_perm_matrix.unsqueeze(0)

    test_events = test_events.unsqueeze(-1)
    test_durations = test_durations.unsqueeze(-1)

    true_perm_matrix = _get_soft_perm(test_events[:, 0], test_durations[:, 0])

    assert torch.allclose(required_perm_matrix, true_perm_matrix)

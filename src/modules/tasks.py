import math
from typing import Any, Dict, Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torchmetrics import MetricCollection

from models.metrics import CIndex, ExactMatch, TopK
from modules.loss import CoxPHLoss, CustomBCEWithLogitsLoss, RankingLoss
from modules.sorter import CustomDiffSortNet
from omni.common import safe_string, unsafe_string


class RiskMixin(pl.LightningModule):
    """
    :arg setting: If synthetic then we have access to true hazards from simulation model,
    then will enable logging of metrics using true risk
    """

    def __init__(
        self,
        grouping_labels,
        label_vocab,
        loss_str="cox",
        weightings=None,
        use_weighted_loss=False,
        setting: str = "realworld",
        sorter_size: int = 128,
        log_weights=False,
        cph_method: str = "efron",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.log_weights = log_weights
        self.label_vocab = label_vocab
        self.grouping_labels = grouping_labels
        self.setting = setting
        self.loss_str = loss_str
        self.sorter_size = sorter_size

        c_index_metric_names = list(self.label_vocab["token2idx"].keys())
        c_index_metrics = MetricCollection(
            {"c_index/" + safe_string(name): CIndex() for name in c_index_metric_names}
        )
        self.valid_cindex = c_index_metrics.clone(prefix="val/")
        self.test_cindex = c_index_metrics.clone(prefix="test/")

        metrics = MetricCollection([TopK()])
        self.valid_topk = metrics.clone(prefix="val/")
        self.test_topk = metrics.clone(prefix="test/")

        metrics = MetricCollection([ExactMatch(size=self.sorter_size)])
        self.valid_em = metrics.clone(prefix="val/")
        self.test_em = metrics.clone(prefix="test/")

        if self.setting == "synthetic":
            c_index_metrics = MetricCollection(
                {"c_index_risk/" + safe_string(name): CIndex() for name in c_index_metric_names}
            )
            self.valid_cindex_risk = c_index_metrics.clone(prefix="val/")
            self.test_cindex_risk = c_index_metrics.clone(prefix="test/")

            metrics = MetricCollection([ExactMatch(size=self.sorter_size)])
            self.valid_em_risk = metrics.clone(prefix="val/", postfix="_risk")
            self.test_em_risk = metrics.clone(prefix="test/", postfix="_risk")

            metrics = MetricCollection([TopK()])
            self.valid_topk_risk = metrics.clone(prefix="val/", postfix="_risk")
            self.test_topk_risk = metrics.clone(prefix="test/", postfix="_risk")

        if loss_str == "cox":
            self.loss_func = CoxPHLoss(method=cph_method)
        elif loss_str == "ranking":
            self.loss_func = RankingLoss()
        elif loss_str == "binary":
            self.loss_func = CustomBCEWithLogitsLoss()
        else:
            raise NotImplementedError(
                "loss_str must be one of the implmented {'cox', 'ranking', 'binary'}"
            )

        if weightings is not None:
            self.loss_func_w = CoxPHLoss(weightings=weightings)
        else:
            self.loss_func_w = None

        self.use_weighted_loss = use_weighted_loss

    def training_step(self, batch, batch_idx, optimizer_idx: Optional[int] = None):
        logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)

        if self.loss_func_w is not None:
            loss = self.loss_func_w(logits, label_multihot, label_times)
        else:
            loss = self.loss_func(logits, label_multihot, label_times)

        self.log("train/loss", loss, prog_bar=True)
        if self.log_weights:
            for i, p in enumerate(self.head.final.weight.flatten()):
                self.log(f"param_est_{i}", -p, prog_bar=True)

        return loss

    def log_cindex(self, cindex: MetricCollection, exclusions, logits, label_multihot, label_times):
        for name, metric in cindex.items():
            # idx = self._groping_idx[name]
            idx = self.label_vocab["token2idx"][unsafe_string(name.split("/")[-1])]
            e = exclusions[:, idx]  # exclude patients with prior history of event
            e_idx = (1 - e).bool()
            p, l, t = (
                logits[e_idx, idx],
                label_multihot[e_idx, idx],
                label_times[e_idx, idx],
            )
            metric.update(p, l.int(), t)

    def validation_step(self, batch, batch_idx):
        logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)

        self.valid_topk.update(
            -logits.squeeze(-1), label_multihot.squeeze(-1), label_times.squeeze(-1)
        )

        self.valid_em.update(-logits.squeeze(-1), label_times.squeeze(-1))
        label_times = batch["label_times"]
        exclusions = batch["exclusions"]

        # c-index is applied per label, collect inputs
        self.log_cindex(self.valid_cindex, exclusions, logits, label_multihot, label_times)
        if self.setting == "synthetic":
            all_observed = torch.ones_like(label_multihot)
            self.log_cindex(
                self.valid_cindex_risk,
                exclusions,
                -logits,
                all_observed,
                batch["risk"],  # *-1 since lower times is higher risk and vice versa
            )
            self.valid_em_risk.update(logits.squeeze(-1), batch["risk"].squeeze(-1).squeeze(-1))
            self.valid_topk_risk.update(
                logits.squeeze(-1),
                torch.ones_like(label_multihot.squeeze(-1)),
                batch["risk"].squeeze(-1).squeeze(-1),
            )

    def on_validation_epoch_end(self) -> None:
        output = self.valid_topk.compute()
        self.valid_topk.reset()
        self.log_dict(output, prog_bar=False, sync_dist=True)

        output = self.valid_em.compute()
        self.valid_em.reset()
        self.log_dict(output, prog_bar=False, sync_dist=True)

        # Get calc cindex metric with collected inputs
        output = self.valid_cindex.compute()
        self._group_cindex(output, key="val/c_index/")
        self.valid_cindex.reset()
        self.log_dict(output, prog_bar=False)
        self.log("hp_metric", output["val/c_index/all"], prog_bar=True, sync_dist=True)

        if self.setting == "synthetic":
            output = self.valid_cindex_risk.compute()
            self._group_cindex(output, key="val/c_index_risk/")
            self.valid_cindex_risk.reset()
            self.log_dict(output, prog_bar=False, sync_dist=True)

            output = self.valid_em_risk.compute()
            self.valid_em_risk.reset()
            self.log_dict(output, prog_bar=False, sync_dist=True)

            output = self.valid_topk_risk.compute()
            self.valid_topk_risk.reset()
            self.log_dict(output, prog_bar=False, sync_dist=True)

    def _group_cindex(self, output, key="val/c_index/"):
        """
        Group c-index by label
        :param output:
        :return:
        """

        for name, labels in self.grouping_labels.items():
            values = []
            for label in labels:
                try:
                    v = output[key + safe_string(label)]
                    if not torch.isnan(v):
                        values.append(v)
                except KeyError:
                    pass
            if len(values) > 0:
                average_value = sum(values) / len(values)
                output.update({key + safe_string(name): average_value})

    def test_step(self, batch, batch_idx):
        logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)

        self.test_topk.update(
            -logits.squeeze(-1), label_multihot.squeeze(-1), label_times.squeeze(-1)
        )

        self.test_em.update(-logits.squeeze(-1), label_times.squeeze(-1))
        label_times = batch["label_times"]
        exclusions = batch["exclusions"]

        # c-index is applied per label, collect inputs
        self.log_cindex(self.test_cindex, exclusions, logits, label_multihot, label_times)
        if self.setting == "synthetic":
            all_observed = torch.ones_like(label_multihot)
            self.log_cindex(
                self.test_cindex_risk,
                exclusions,
                -logits,
                all_observed,
                batch["risk"],  # *-1 since lower times is higher risk and vice versa
            )
            self.test_em_risk.update(logits.squeeze(-1), batch["risk"].squeeze(-1).squeeze(-1))
            self.test_topk_risk.update(
                logits.squeeze(-1),
                torch.ones_like(label_multihot.squeeze(-1)),
                batch["risk"].squeeze(-1).squeeze(-1),
            )

    def on_test_epoch_end(self) -> None:
        output = self.test_topk.compute()
        self.test_topk.reset()
        self.log_dict(output, prog_bar=False, sync_dist=True)

        output = self.test_em.compute()
        self.test_em.reset()
        self.log_dict(output, prog_bar=False, sync_dist=True)

        # Get calc cindex metric with collected inputs
        output = self.test_cindex.compute()
        self._group_cindex(output, key="test/c_index/")
        self.test_cindex.reset()
        self.log_dict(output, prog_bar=False)
        self.log("hp_metric", output["test/c_index/all"], prog_bar=True, sync_dist=True)

        if self.setting == "synthetic":
            output = self.test_cindex_risk.compute()
            self._group_cindex(output, key="test/c_index_risk/")
            self.test_cindex_risk.reset()
            self.log_dict(output, prog_bar=False, sync_dist=True)

            output = self.test_em_risk.compute()
            self.test_em_risk.reset()
            self.log_dict(output, prog_bar=False, sync_dist=True)

            output = self.test_topk_risk.compute()
            self.test_topk_risk.reset()
            self.log_dict(output, prog_bar=False, sync_dist=True)


class SortingRiskMixin(RiskMixin):
    """Needs a seperate mixin due to loss function requiring permutation matrices
    #TODO: possible refactor to avoid this
    """

    def __init__(
        self,
        sorting_network: Literal["odd_even", "bitonic"],
        steepness: Optional[float] = None,
        art_lambda: float = 0.2,
        distribution="cauchy",
        sorter_size: int = 128,
        ignore_censoring: bool = True,
        norm_risk: bool = True,
        optimize_topk: bool = True,
        optimize_combined: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(sorter_size=sorter_size, *args, **kwargs)
        self.sorter_size = sorter_size
        if steepness is None:
            if sorting_network == "odd_even":
                steepness = 2 * self.sorter_size
            else:
                steepness = math.log2(self.sorter_size) * (
                    1 + math.log2(self.sorter_size)
                )  # bitonic layers

        self.sorter = CustomDiffSortNet(
            sorting_network_type=sorting_network,
            size=self.sorter_size,
            steepness=steepness,
            art_lambda=art_lambda,
            distribution=distribution,
        )
        self.ignore_censoring = ignore_censoring
        self.norm_risk = norm_risk
        self.steepness = steepness
        self.optimize_topk = optimize_topk or optimize_combined
        self.optimize_combined = optimize_combined

    def sorting_step(self, logits, perm_ground_truth, events):
        lh = logits

        # Normalize within risk set...
        if self.norm_risk:
            if len(lh.shape) == 3:
                lh = (lh - lh.mean(dim=1, keepdim=True)) / lh.std(dim=1, keepdim=True)
            else:
                lh = (lh - lh.mean(dim=0, keepdim=True)) / lh.std(dim=0, keepdim=True)

        x_shape = lh.shape
        if len(x_shape) == 3:
            lh = lh.reshape(-1, lh.shape[1])

        sort_out, perm_prediction = self.sorter(lh)

        possible_predictions = (perm_ground_truth * perm_prediction).sum(dim=1)

        top_k_loss = 0.0
        if self.optimize_topk:
            risk_set_size = perm_ground_truth.shape[-1]

            losses = None
            for pgt, pp in zip(perm_ground_truth, perm_prediction):
                possible_top_k_idxs = torch.argwhere(
                    pgt[:, -max(risk_set_size // 10, 0) :].sum(axis=1) > 0
                ).flatten()

                top_k_loss = -pp[
                    possible_top_k_idxs,
                    -max(risk_set_size // 10, 0) :,
                ].sum()

                if losses is None:
                    losses = top_k_loss
                else:
                    losses = losses + top_k_loss

            top_k_loss = losses / len(perm_ground_truth)

            """
            possible_top_k_mask = perm_ground_truth[:, :, -risk_set_size // 10 :].sum(axis=-1) > 0
            top_k_loss = -perm_prediction[:, :, -risk_set_size // 10 :][possible_top_k_mask].mean()
            """

            if not self.optimize_combined:
                return top_k_loss, lh, perm_prediction, perm_ground_truth

        if self.ignore_censoring:
            possible_events_only = possible_predictions.flatten()[events.flatten() == 1]
            predictions = possible_events_only
        else:
            predictions = possible_predictions

        loss = torch.nn.BCELoss()(
            torch.clamp(predictions, 1e-8, 1 - 1e-8),
            torch.ones_like(predictions),
        )

        if self.optimize_combined:
            loss = loss + top_k_loss

        return loss, lh, perm_prediction, perm_ground_truth

    def training_step(self, batch, batch_idx, optimizer_idx: Optional[int] = None, *args, **kwargs):
        logits, label_multihot, label_times = self._shared_eval_step(batch, batch_idx)
        loss, _, _, _ = self.sorting_step(logits, batch["soft_perm_mat"], label_multihot)

        self.log("train/loss", loss, prog_bar=True)
        if self.log_weights:
            for i, p in enumerate(self.head.final.weight.flatten()):
                self.log(f"param_est_{i}", p, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        covariates = batch["covariates"]
        label_multihot = batch["labels"]
        label_times = batch["label_times"]
        logits = self(covariates)

        exclusions = batch["exclusions"]

        # c-index is applied per label, collect inputs
        self.log_cindex(self.valid_cindex, exclusions, -logits, label_multihot, label_times)
        self.valid_em.update(logits.squeeze(-1), label_times.squeeze(-1))
        self.valid_topk.update(
            logits.squeeze(-1), label_multihot.squeeze(-1), label_times.squeeze(-1)
        )

        if self.setting == "synthetic":
            all_observed = torch.ones_like(label_multihot)
            self.log_cindex(
                self.valid_cindex_risk,
                exclusions,
                logits,
                all_observed,
                batch["risk"],  # *-1 since lower times is higher risk and vice versa
            )
            self.valid_em_risk.update(logits.squeeze(-1), -batch["risk"].squeeze(-1).squeeze(-1))
            self.valid_topk_risk.update(
                logits.squeeze(-1),
                torch.ones_like(label_multihot.squeeze(-1)),
                batch["risk"].squeeze(-1).squeeze(-1),
            )

    def test_step(self, batch, batch_idx):
        covariates = batch["covariates"]
        label_multihot = batch["labels"]
        label_times = batch["label_times"]
        logits = self(covariates)

        exclusions = batch["exclusions"]

        # c-index is applied per label, collect inputs
        self.log_cindex(self.test_cindex, exclusions, -logits, label_multihot, label_times)
        self.test_em.update(logits.squeeze(-1), label_times.squeeze(-1))
        self.test_topk.update(
            logits.squeeze(-1), label_multihot.squeeze(-1), label_times.squeeze(-1)
        )

        if self.setting == "synthetic":
            all_observed = torch.ones_like(label_multihot)
            self.log_cindex(
                self.test_cindex_risk,
                exclusions,
                logits,
                all_observed,
                batch["risk"],  # *-1 since lower times is higher risk and vice versa
            )
            self.test_em_risk.update(logits.squeeze(-1), -batch["risk"].squeeze(-1).squeeze(-1))
            self.test_topk_risk.update(
                logits.squeeze(-1),
                torch.ones_like(label_multihot.squeeze(-1)),
                batch["risk"].squeeze(-1).squeeze(-1),
            )

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        logits = self(0, covariates=batch["covariates"])

        batch.update({"logits": logits})

        # Let's get things in a friendly format for pandas dataframe..
        numpy_batch = {}
        for k, v in batch.items():
            # Detach from graph and make sure on cpu
            new_v = v.detach().cpu()
            if k != "covariates":
                new_v = new_v.flatten()
                numpy_batch[k] = new_v.numpy().tolist()
            # Covariates have multiple dim so should not be flattened
            else:
                dim = new_v.shape[1]
                for i in range(dim):
                    numpy_batch[k + f"_{i}"] = new_v[:, i].numpy().tolist()

        return numpy_batch


def _get_possible_permutation_matrix(
    events: torch.Tensor,
    durations: torch.Tensor,
    inc_censored_in_ties=True,
    eps: float = 1e-6,
):
    """
    Returns the possible permutation matrix label for the given events and durations.

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
    :param durations: time difference between observation start and event time
    :param inc_censored_in_ties: if we want to include censored events as possible permutations in ties with events
    :return:
    """
    # Initialize the soft permutation matrix
    perm_matrix = torch.zeros(events.shape[0], events.shape[0], device=events.device)

    # eps here forces ties between censored and event to be ordred event first (ascending)
    idx = torch.argsort(durations - events * eps, descending=False)

    # Used to return to origonal order
    perm_un_ascending = torch.nn.functional.one_hot(idx).transpose(-2, -1).float()
    ordered_durations = durations[idx]
    ordered_events = events[idx]

    # events_ordered = events[idx]
    event_counts = 0

    idx_stack = list(range(idx.shape[0]))
    idx_stack.reverse()
    i_event = []
    i_censored = []
    while idx_stack:
        if ordered_events[idx_stack[-1]]:
            i_event.append(idx_stack.pop())
        else:
            i_censored.append(idx_stack.pop())

        # Handle Ties: Look ahead, if next has the same time, add next index!
        i_all = i_event + i_censored
        if (
            idx_stack
            and i_all
            and (ordered_durations[i_all[-1]] == ordered_durations[idx_stack[-1]])
        ):
            continue

        if inc_censored_in_ties and i_censored:
            # Right censored samples
            # assign 0 for all samples with event time lower than the censoring time
            # perm_matrix[i, : i[-1]] = 0
            # assign uniform probability to all samples with event time higher than the censoring time
            # includes previous censored events that happened before the event time
            perm_matrix[i_censored, event_counts:] = 1
            i_censored = []  # clear idx on  censored

        # Events
        # Assign uniform probability to an event and all censored events with shorter time,
        if i_event:
            if inc_censored_in_ties:
                perm_matrix[i_event, event_counts : max(i_all) + 1] = 1
            else:
                perm_matrix[i_event, event_counts : max(i_event) + 1] = 1
            event_counts += int(sum(ordered_events[i_event]))
            i_event = []  # reset indices no more ties

        if not inc_censored_in_ties and i_censored:
            perm_matrix[i_censored, event_counts:] = 1
            i_censored = []  # clear idx on  censored

    # Permute to match the order of the input
    perm_matrix = perm_un_ascending @ perm_matrix

    return perm_matrix


# TODO: add pytest and fixtures...
def test_get_possible_permutation_matrix():
    """Test the soft permutation matrix label for the given events and durations."""
    test_events = torch.Tensor([0, 0, 1, 0, 1, 0, 0])
    test_durations = torch.Tensor([1, 3, 2, 4, 5, 6, 7])
    # logh = torch.Tensor([0, 2, 1, 3, 4, 5, 6])

    required_perm_matrix = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
        ]
    )
    required_perm_matrix = required_perm_matrix.unsqueeze(0)

    test_events = test_events.unsqueeze(-1)
    test_durations = test_durations.unsqueeze(-1)

    true_perm_matrix = _get_possible_permutation_matrix(test_events[:, 0], test_durations[:, 0])

    assert torch.allclose(required_perm_matrix, true_perm_matrix)


def test_ties_all_events_get_possible_permutation_matrix():
    """Test the soft permutation matrix label for the given events and durations."""
    test_events = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
    test_durations = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
    # logh = torch.Tensor([0, 2, 1, 3, 4, 5, 6])

    required_perm_matrix = torch.ones((7, 7))
    required_perm_matrix = required_perm_matrix.unsqueeze(0)

    test_events = test_events.unsqueeze(-1)
    test_durations = test_durations.unsqueeze(-1)

    true_perm_matrix = _get_possible_permutation_matrix(test_events[:, 0], test_durations[:, 0])

    assert torch.allclose(required_perm_matrix, true_perm_matrix)


def test_ties_get_possible_permutation_matrix():
    """Test the soft permutation matrix label for the given events and durations."""
    test_events = torch.Tensor([0, 1, 0, 1, 1, 0, 0])
    test_durations = torch.Tensor([1, 2, 3, 4, 4, 4, 5])
    # logh = torch.Tensor([0, 2, 1, 3, 4, 5, 6])

    # -2 censored event same time so can permuate with events under independent censoring
    # -1 cannot permuate with events but can with censored event of the same time
    required_perm_matrix = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ]
    )
    required_perm_matrix = required_perm_matrix.unsqueeze(0)

    test_events = test_events.unsqueeze(-1)
    test_durations = test_durations.unsqueeze(-1)

    perm_matrix = _get_possible_permutation_matrix(
        test_events[:, 0], test_durations[:, 0], inc_censored_in_ties=False
    )

    assert torch.allclose(required_perm_matrix, perm_matrix)


def test_ties_inc_get_possible_permutation_matrix():
    """Test the soft permutation matrix label for the given events and durations."""
    test_events = torch.Tensor([0, 1, 0, 0, 1, 1, 0])
    test_durations = torch.Tensor([1, 2, 3, 4, 4, 4, 5])
    # logh = torch.Tensor([0, 2, 1, 3, 4, 5, 6])
    # censored event same time so can permuate with events under independent c
    # cannot permuate with events but can with censored event of the same time
    required_perm_matrix = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1],
        ]
    )
    required_perm_matrix = required_perm_matrix.unsqueeze(0)
    test_events = test_events.unsqueeze(-1)
    test_durations = test_durations.unsqueeze(-1)
    perm_matrix = _get_possible_permutation_matrix(
        test_events[:, 0], test_durations[:, 0], inc_censored_in_ties=True
    )
    assert torch.allclose(required_perm_matrix, perm_matrix)


if __name__ == "__main__":
    test_ties_get_possible_permutation_matrix()
    test_ties_inc_get_possible_permutation_matrix()
    test_ties_all_events_get_possible_permutation_matrix()
    test_get_possible_permutation_matrix()

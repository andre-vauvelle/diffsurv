import numba
import numpy as np
import torch
import torchmetrics
from sortedcontainers import SortedList


@numba.njit(parallel=False, nogil=True)
def loop_cindex(events, event_times, predictions):
    idxs = np.argsort(event_times)

    events = events[idxs]
    event_times = event_times[idxs]
    predictions = predictions[idxs]

    n_concordant = 0
    n_comparable = 0

    for i in numba.prange(len(events)):
        for j in range(i + 1, len(events)):
            if events[i] and events[j]:
                n_comparable += 1
                n_concordant += (event_times[i] > event_times[j]) == (
                    predictions[i] > predictions[j]
                )
            elif events[i]:
                n_comparable += 1
                n_concordant += predictions[i] < predictions[j]
    if n_comparable > 0:
        return n_concordant / n_comparable
    else:
        return np.nan


def sorted_list_concordance_index(events, time, predictions):
    """
    O(n log n) implementation of https://square.github.io/pysurvival/metrics/c_index.html from https://github.com/lasso-net/lassonet
    """
    assert len(predictions) == len(time) == len(events)
    predictions = predictions * -1  # ordered opposite from sorted_list implementation...
    n = len(predictions)
    order = sorted(range(n), key=time.__getitem__)
    past = SortedList()
    num = 0
    den = 0
    for i in order:
        num += len(past) - past.bisect_right(predictions[i])
        den += len(past)
        if events[i]:
            past.add(predictions[i])
    return num / den


class ExactMatch(torchmetrics.Metric):
    """
    :param size: size of the set on which to calc exact match and element wise match
    """

    def __init__(self, size: int):
        super().__init__(full_state_update=False)
        self.size = size
        self.add_state("exact_match", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("element_wise", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("exact_match5", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_elements", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_sets", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        acc = torch.argsort(target[: self.size], dim=-1) == torch.argsort(
            preds[: self.size], dim=-1
        )

        self.exact_match += acc.all(-1).sum()
        self.element_wise += acc.sum()

        preds5 = preds[:5]
        target5 = target[:5]
        acc5 = torch.argsort(target5, dim=-1) == torch.argsort(preds5, dim=-1)
        self.exact_match5 += acc5.all(-1).sum()

        self.total_elements += self.size
        self.total_sets += 1

    def compute(self):
        return dict(
            exact_match=self.exact_match / self.total_sets,
            element_wise=self.element_wise / self.total_elements,
            exact_match_5=self.exact_match5 / self.total_sets,
        )


class CIndex(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, method="sorted_list"):
        super().__init__(dist_sync_on_step=dist_sync_on_step, full_state_update=True)
        if method == "loop":
            self.cindex_fn = loop_cindex
        elif method == "sorted_list":
            self.cindex_fn = sorted_list_concordance_index

        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("events", default=[], dist_reduce_fx="cat")
        self.add_state("times", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, events: torch.Tensor, times: torch.Tensor):
        self.logits.append(logits)
        self.events.append(events)
        self.times.append(times.flatten())

    def compute(self):
        # this version is much faster, but doesn't handle ties correctly.
        # numba doesn't handle half precision correctly, so we use float32
        if isinstance(self.events, list):
            self.events = torch.cat(self.events)
        if isinstance(self.logits, list):
            self.logits = torch.cat(self.logits)
        if isinstance(self.times, list):
            self.times = torch.cat(self.times)
        return torch.Tensor(
            [
                self.cindex_fn(
                    self.events.cpu().float().numpy(),
                    self.times.cpu().float().numpy(),
                    1 - self.logits.cpu().float().numpy(),  # just - x  not 1 - x?
                )
            ]
        )


class TopK(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, full_state_update=True)

        self.add_state("scores", default=[], dist_reduce_fx="cat")

        self.add_state("events", default=[], dist_reduce_fx="cat")
        self.add_state("times", default=[], dist_reduce_fx="cat")
        self.add_state("logits", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, events: torch.Tensor, times: torch.Tensor):
        # TODO: prevent circular dependency
        from modules.tasks import _get_possible_permutation_matrix

        events = events.bool()

        self.logits.append(logits)
        self.events.append(events)
        self.times.append(times.flatten())

        possible_perm = _get_possible_permutation_matrix(events, times)

        possible_top_k_idxs = set(
            torch.argwhere(possible_perm[:, -max(len(possible_perm) // 10, 1) :].sum(axis=-1) > 0)
            .flatten()
            .detach()
            .cpu()
            .data.numpy()
        )
        pred_topk = set(torch.argsort(logits)[-max(len(logits) // 10, 1) :].tolist())

        score = len(pred_topk & possible_top_k_idxs) / len(pred_topk)

        self.scores.append(score)

    def compute(self):
        batched_score = np.mean(self.scores)

        # TODO: prevent circular dependency
        from modules.tasks import _get_possible_permutation_matrix

        if isinstance(self.events, list):
            events = torch.cat(self.events)
        if isinstance(self.logits, list):
            logits = torch.cat(self.logits)
        if isinstance(self.times, list):
            times = torch.cat(self.times)

        possible_perm = _get_possible_permutation_matrix(events, times)

        possible_top_k_idxs = set(
            torch.argwhere(possible_perm[:, -max(len(possible_perm) // 10, 1) :].sum(axis=-1) > 0)
            .flatten()
            .detach()
            .cpu()
            .data.numpy()
        )
        pred_topk = set(torch.argsort(logits)[-max(len(logits) // 10, 1) :].tolist())

        score = len(pred_topk & possible_top_k_idxs) / len(pred_topk)

        return dict(batch_topk=batched_score, topk=score)

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


class CIndex(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, method="sorted_list"):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
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

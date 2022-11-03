import numba
import numpy as np
import torch
import torchmetrics


@numba.njit(parallel=False, nogil=True)
def cindex(events, event_times, predictions):
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


class CIndex(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, method="loop"):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.method = method

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
        return torch.Tensor(
            [
                cindex(
                    torch.cat(self.events).cpu().float().numpy(),
                    torch.cat(self.times).cpu().float().numpy(),
                    1 - torch.cat(self.logits).cpu().float().numpy(),  # just - x  not 1 - x?
                )
            ]
        )

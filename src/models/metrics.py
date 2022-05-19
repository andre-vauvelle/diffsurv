import numba
import numpy as np
import pandas as pd
import torchmetrics
import torch


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
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("events", default=[], dist_reduce_fx="cat")
        self.add_state("times", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, events: torch.Tensor, times: torch.Tensor):
        self.logits.append(logits)
        self.events.append(events)
        self.times.append(times)

    def compute(self):
        # this version is much faster, but doesn't handle ties correctly.
        # numba doesn't handle half precision correctly, so we use float32
        return torch.Tensor(
            [
                cindex(
                    torch.cat(self.events).cpu().float().numpy(),
                    torch.cat(self.times).cpu().float().numpy(),
                    torch.cat(self.logits).cpu().float().numpy(),
                )
            ]
        )

        # if False:
        #     return torch.Tensor(
        #         [
        #             lifelines.utils.concordance_index(
        #                 torch.cat(self.times).cpu().numpy(),
        #                 1 - torch.cat(self.logits).cpu().numpy(),
        #                 event_observed=torch.cat(self.events).cpu().numpy(),
        #             )
        #         ]
        #     )

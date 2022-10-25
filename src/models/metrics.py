import numba
import numpy as np
import pandas as pd
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


def kendall_embedding_loop(vector):
    n = len(vector)
    rank = torch.argsort(vector)
    embedding = torch.zeros(n, n)
    for i, iv in enumerate(rank):
        for j, jv in enumerate(rank):
            embedding[i, j] = 1 if iv < jv else 0
    return embedding


# def kendall_cindex(logits, events, times):
#     """Vectorised version of c-index using kendall embeddings to find concordenet and discordent pairs"""
#     e


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
        self.times.append(times)

    def compute(self):
        # this version is much faster, but doesn't handle ties correctly.
        # numba doesn't handle half precision correctly, so we use float32
        if self.method == "kendall":
            NotImplemented
        else:
            return torch.Tensor(
                [
                    cindex(
                        torch.cat(self.events).cpu().float().numpy(),
                        torch.cat(self.times).cpu().float().numpy(),
                        1 - torch.cat(self.logits).cpu().float().numpy(),
                    )
                ]
            )


if __name__ == "__main__":
    test_vector = torch.Tensor([5, 6, 3, 9])
    kendall_embedding_loop(test_vector)

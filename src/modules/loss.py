import torch

import pdb
import diffsort

from modules.sorter import CustomDiffSortNet


class SortingCrossEntropyLoss(torch.nn.Module):
    """
    Sorting Cross Entropy loss. Uses a differentiable sorter to find the loss between a true permutation and an
    estimated permutation. Permutation are the set of events (patients) within a set ranked by risk (most likely event
    first).

    In the absence of full information with some labels missing due to censoring, soft-labels are used to determine the
    true permutation matrix. Censored soft-labels are determined by the censoring time, an equal probability is
    assigned to all labels with event times greater than the censoring time (could not have happened before censored
    event). Events without censoring are assigned equal probability for each censored event
    that happened before its event time, as the censored events, if continually observed could have manifested.
    Cross entropy loss is applied using between estimated and true permutation matrices.
    """

    def __init__(self, sorter,
                 eps=1e-6, weightings=None):
        super().__init__()
        self.eps = eps
        self.sorter = sorter
        self.weightings = weightings

    def forward(self, logits, events, durations):
        losses = []
        for i in range(logits.shape[1]):
            lh, d, e = logits[:, i], durations[:, i], events[:, i]

            # TODO: could refactor to dataloader
            # Get the soft permutation matrix
            _, perm_prediction = self.sorter(lh.unsqueeze(0))
            perm_ground_truth = self._get_soft_perm(e, d)

            loss = torch.nn.BCELoss()(perm_prediction, perm_ground_truth)
            losses.append(loss)

        # drop losses less than zero, ie no events in risk set
        loss_tensor = torch.stack(losses)
        loss_idx = loss_tensor.gt(0)

        if self.weightings is None:
            loss = loss_tensor[loss_idx].mean()
        else:
            # re-normalize weights
            weightings = self.weightings[loss_idx] / self.weightings[loss_idx].sum()
            loss = loss_tensor[loss_idx].mul(weightings).sum()

        return loss

    @staticmethod
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
                perm_matrix[i, event_counts:i + 1] = 1 / (perm_matrix[i, event_counts:i + 1].shape[0])
                event_counts += 1

        # permute to match the order of the input
        perm_matrix = perm_un_ascending @ perm_matrix

        # Unsqueeze for one batch
        return perm_matrix.unsqueeze(0)


def test_diff_sort_loss_get_soft_perm():
    """Test the soft permutation matrix label for the given events and durations."""
    test_events = torch.Tensor([0, 0, 1, 0, 1, 0, 0])
    test_durations = torch.Tensor([1, 3, 2, 4, 5, 6, 7])
    logh = torch.Tensor([0, 2, 1, 3, 4, 5, 6])

    required_perm_matrix = torch.Tensor([[1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
                                         [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                                         [1 / 2, 1 / 2, 0, 0, 0, 0, 0],
                                         [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                                         [0, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 0, 0],
                                         [0, 0, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                                         [0, 0, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]])
    required_perm_matrix = required_perm_matrix.unsqueeze(0)

    test_events = test_events.unsqueeze(-1)
    test_durations = test_durations.unsqueeze(-1)

    sorter = CustomDiffSortNet(sorting_network_type='bitonic', size=7)
    loss = SortingCrossEntropyLoss(sorter)

    true_perm_matrix = loss._get_soft_perm(test_events[:, 0], test_durations[:, 0])

    assert torch.allclose(required_perm_matrix, true_perm_matrix)


class CoxPHLoss(torch.nn.Module):
    def __init__(self, weightings=None):
        super().__init__()
        self.weightings = weightings

    def forward(self, logh, events, durations, eps=1e-7):
        """
        Simple approximation of the COX-ph. Log hazard is not computed on risk-sets, but on ranked list instead.
        This approximation is valid for datamodules w/ low percentage of ties.
        Credit to Haavard Kamme/PyCox
        :param logh: log hazard
        :param durations: (batch_size, n_risk_sets, n_events)
        :param events: 1 if event, 0 if censored
        :param eps: small number to avoid log(0)
        :param weightings: weighting of the loss function
        :return:
        """

        losses = []
        for i in range(logh.shape[1]):
            lh, d, e = logh[:, i], durations[:, i], events[:, i]

            # sort:
            idx = d.sort(descending=True, dim=0)[1]
            e = e[idx].squeeze(-1)
            lh = lh[idx].squeeze(-1)
            # calculate loss:
            gamma = lh.max()
            log_cumsum_h = lh.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
            if e.sum() > 0:
                loss = -lh.sub(log_cumsum_h).mul(e).sum().div(e.sum())
            else:
                loss = -lh.sub(log_cumsum_h).mul(e).sum()  # would this not always be zero?
            losses.append(loss)

        # drop losses less than zero, ie no events in risk set
        loss_tensor = torch.stack(losses)
        loss_idx = loss_tensor.gt(0)

        if self.weightings is None:
            loss = loss_tensor[loss_idx].mean()
        else:
            # re-normalize weights
            weightings = self.weightings[loss_idx] / self.weightings[loss_idx].sum()
            loss = loss_tensor[loss_idx].mul(weightings).sum()

        return loss


if __name__ == '__main__':
    test_diff_sort_loss_get_soft_perm()

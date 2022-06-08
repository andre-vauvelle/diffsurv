import numpy as np
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


class CustomBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self):
        super().__init__()

    def forward(self, logh, events, durations=None, eps=1e-7):
        return super().forward(logh, events)


class CoxPHLoss(torch.nn.Module):
    def __init__(self, weightings=None, method='ranked_list'):
        super().__init__()
        self.method = method
        if weightings is not None:
            self.register_buffer('weightings', weightings)
        else:
            self.weightings = weightings

    def forward(self, logh, events, durations, eps=1e-7):
        """
        Simple approximation of the COX-ph. Log hazard is not computed on risk-sets, but on ranked list instead.
        This approximation is valid for datamodules w/ low percentage of ties.
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
            if self.method == 'efron':
                loss = self._efron_loss(lh, d, e, eps)
            elif self.method == 'ranked_list':
                loss = self._loss_ranked_list(lh, d, e, eps)
            else:
                raise ValueError('Unknown method: {}, choose one of ["efron", "ranked_list"]'.format(self.method))
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
    def _loss_ranked_list(lh, d, e, eps=1e-7):
        """Ranked list method for COX-PH.
        Credit to Haavard Kamme/PyCox
        """

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
        return loss

    @staticmethod
    def _efron_loss(lh, d, e, eps=1e-7):
        """Efron method for COX-PH.
        Credit to https://bydmitry.github.io/efron-tensorflow.html
        """
        raise NotImplementedError


def tgt_equal_tgt(time):
    """
    Used for tied times. Returns a diagonal by block matrix.
    Diagonal blocks of 1 if same time.
    Sorted over time. A_ij = i if t_i == t_j.
    Parameters
    ----------
    time: ndarray
        Time sorted in ascending order.
    Returns
    -------
    tied_matrix: ndarray
        Diagonal by block matrix.
    """
    t_i = time.astype(np.float32).reshape(1, -1)
    t_j = time.astype(np.float32).reshape(-1, 1)
    tied_matrix = np.where(t_i == t_j, 1., 0.).astype(np.float32)

    assert (tied_matrix.ndim == 2)
    block_sizes = np.sum(tied_matrix, axis=1)
    block_index = np.sum(tied_matrix - np.triu(tied_matrix), axis=1)

    tied_matrix = tied_matrix * (block_index / block_sizes)[:, np.newaxis]
    return tied_matrix


def tgt_leq_tgt(time):
    """
    Lower triangular matrix where A_ij = 1 if t_i leq t_j.
    Parameters
    ----------
    time: ndarray
        Time sorted in ascending order.
    Returns
    -------
    tril: ndarray
        Lower triangular matrix.
    """
    t_i = time.astype(np.float32).reshape(1, -1)
    t_j = time.astype(np.float32).reshape(-1, 1)
    tril = np.where(t_i <= t_j, 1., 0.).astype(np.float32)
    return tril


def cox_loss_ties(pred, cens, tril, tied_matrix):
    """
    Compute the Efron version of the Cox loss. This version take into
    account the ties.
    t unique time
    H_t denote the set of indices i such that y^i = t and c^i =1.
    c^i = 1 event occured.
    m_t = |H_t| number of elements in H_t.
    l(theta) = sum_t (sum_{i in H_t} h_{theta}(x^i)
                     - sum_{l=0}^{m_t-1} log (
                        sum_{i: y^i >= t} exp(h_{theta}(x^i))
                        - l/m_t sum_{i in H_t} exp(h_{theta}(x^i)))
    Parameters
    ----------
    pred : torch tensor
        Model prediction.
    cens : torch tensor
        Event tensor.
    tril : torch tensor
        Lower triangular tensor.
    tied_matrix : torch tensor
        Diagonal by block tensor.
    Returns
    -------
    loss : float
        Efron version of the Cox loss.
    """

    # Note that the observed variable is not required as we are sorting the
    # inputs when generating the batch according to survival time.

    # exp(h_{theta}(x^i))
    exp_pred = torch.exp(pred)
    # Term corresponding to the sum over events in the risk pool
    # sum_{i: y^i >= t} exp(h_{theta}(x^i))
    future_theta = torch.mm(tril.transpose(1, 0), exp_pred)
    # sum_{i: y^i >= t} exp(h_{theta}(x^i))
    # - l/m_t sum_{i in H_t} exp(h_{theta}(x^i))
    tied_term = future_theta - torch.mm(tied_matrix, exp_pred)
    # log (sum_{i: y^i >= t} exp(h_{theta}(x^i))
    #      - l/m_t sum_{i in H_t} exp(h_{theta}(x^i))
    tied_term = torch.log(tied_term)
    # event row vector to column
    tied_term = tied_term.view((-1, 1))
    cens = cens.view((-1, 1))
    # sum_t (sum_{i in H_t} h_{theta}(x^i)
    #       - sum_{l=0}^{m_t-1} log (
    #          sum_{i: y^i >= t} exp(h_{theta}(x^i))
    #          - l/m_t sum_{i in H_t} exp(h_{theta}(x^i)))
    loss = (pred - tied_term) * cens
    # Negative loglikelihood
    loss = -torch.mean(loss)
    return loss


if __name__ == '__main__':
    test_diff_sort_loss_get_soft_perm()

import math
from typing import Literal

import numba
import numpy as np
import torch


class CauchyFunction(torch.nn.Module):
    def __init__(self, steepness=1):
        super().__init__()
        self.steepness = steepness

    def forward(self, d):
        v = self.steepness * d
        alpha = 1 / math.pi * torch.atan(v) + 0.5
        return alpha


class RankingLoss(torch.nn.Module):
    """
    Implments Ranking loss where
    \text{ranking-loss} = \frac{1}{\\lvert{\\mathcal{A}\rvert}}\\sum_{(x_i, x_j) \\in \\mathcal{A}} \\phi(f_\theta(x_i) - f_\theta(x_j))
    Keyword Arguments:
        ranking_function{nn.Module} -- Function to pass differences to, could also be C
    """

    def __init__(self, ranking_function: torch.nn.Module = torch.nn.LogSigmoid):
        super().__init__()
        self.ranking_function = ranking_function()

    def forward(self, logh: torch.Tensor, label_multihot, label_times):
        assert (
            logh.shape[1] == 2
        ), f"must have pair wise comparisons,max num_compare==2, currently is {logh.shape[1]}"
        # index 1 is always case... current implmentation relies on correct ordering of indicies
        return torch.sum(-self.ranking_function(logh[:, 1, :] - logh[:, 0, :])) / logh.shape[0]


@numba.njit
def _pair_rank_mat(mat, idx_durations, events, dtype="float32"):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat


def test_ranking_loss():
    test_events = torch.Tensor([0, 0, 1, 0, 1, 0, 0])
    test_durations = torch.Tensor([1, 3, 2, 4, 5, 6, 7])
    logh = torch.Tensor([0, 2, 1, 3, 4, 5, 6])

    loss_fn = RankingLoss()
    loss = loss_fn(logh, test_events, test_durations)
    assert not torch.isnan(loss)


def pair_rank_mat(idx_durations, events, dtype="float32"):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.

    Arguments:
        idx_durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.

    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})

    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat


class CustomBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self):
        super().__init__()

    def forward(self, logh, events, durations=None, eps=1e-7):
        return super().forward(logh, events)


class CoxPHLoss(torch.nn.Module):
    def __init__(
        self,
        weightings=None,
        method: Literal["breslow", "efron", "ranked_list"] = "efron",
    ):
        super().__init__()
        self.method = method
        if weightings is not None:
            self.register_buffer("weightings", weightings)
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
        if len(logh) == 2:
            logh = logh.unsqueeze(0)
            events = events.unsqueeze(0)
            durations = durations.unsqueeze(0)
        for b in range(logh.shape[0]):
            for i in range(logh.shape[2]):
                lh, d, e = logh[b, :, i], durations[b, :, i], events[b, :, i]
                if self.method == "efron":
                    loss = self._efron_loss(lh, d, e, eps)
                elif self.method == "breslow":
                    loss = self._breslow_loss(lh, d, e, eps)
                elif self.method == "ranked_list":
                    loss = self._loss_ranked_list(lh, d, e, eps)
                else:
                    raise ValueError(
                        f'Unknown method: {self.method}, choose one of ["efron", "ranked_list",'
                        ' "breslow"]'
                    )
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
    def _get_event_ties_lcse(log_h, durations, events):
        log_h = log_h.flatten()
        # sort input
        durations, idx = durations.sort(descending=True)
        log_h = log_h[idx]
        events = events[idx]

        event_ind = events.nonzero().flatten()

        # logcumsumexp of events
        event_lcse = torch.logcumsumexp(log_h, dim=0)[event_ind]

        # number of events for each unique risk set
        _, tie_inverses, tie_count = torch.unique_consecutive(
            durations[event_ind], return_counts=True, return_inverse=True
        )

        # position of last event (lowest duration) of each unique risk set
        tie_pos = tie_count.cumsum(axis=0) - 1

        # logcumsumexp by tie for each event
        event_tie_lcse = event_lcse[tie_pos][tie_inverses]
        return event_tie_lcse, tie_count, tie_inverses, tie_pos, event_ind

    def _efron_loss(self, log_h, durations, events, eps=1e-7):
        """Efron method for COX-PH.
        Credit to https://github.com/lasso-net/lassonet/blob/78bd5875cd43ae667690e5bda524818c05efe62d/lassonet/utils.py
        """
        (
            event_tie_lcse,
            tie_count,
            tie_inverses,
            tie_pos,
            event_ind,
        ) = self._get_event_ties_lcse(log_h, durations, events)
        # logsumexp of ties, duplicated within tie set
        tie_lse = scatter_logsumexp(log_h[event_ind], tie_inverses, dim=0)[tie_inverses]
        # multiply (add in log space) with corrective factor
        aux = torch.ones_like(tie_inverses)
        aux[tie_pos[:-1] + 1] -= tie_count[:-1]
        event_id_in_tie = torch.cumsum(aux, dim=0) - 1
        discounted_tie_lse = (
            tie_lse + torch.log(event_id_in_tie) - torch.log(tie_count[tie_inverses])
        )

        # denominator
        log_den = log_substract(event_tie_lcse, discounted_tie_lse).mean()

        # numerator
        log_num = log_h[event_ind].mean()
        # loss is negative log likelihood
        return log_den - log_num

    def _breslow_loss(self, log_h, durations, events, eps=1e-7):
        """Breslow method for COX-PH.
        Credit to https://github.com/lasso-net/lassonet/blob/78bd5875cd43ae667690e5bda524818c05efe62d/lassonet/utils.py

        """
        event_tie_lcse, _, _, _, event_ind = self._get_event_ties_lcse(log_h, durations, events)
        log_num = log_h[event_ind].mean()
        # loss is negative log likelihood
        log_den = event_tie_lcse.mean()
        return log_den - log_num


# from https://github.com/lasso-net/lassonet/blob/78bd5875cd43ae667690e5bda524818c05efe62d/lassonet/utils.py
def scatter_logsumexp(input, index, *, dim=-1, output_size=None):
    """Inspired by torch_scatter.logsumexp
    Uses torch.scatter_reduce for performance
    """
    max_value_per_index = scatter_reduce(
        input, dim=dim, index=index, output_size=output_size, reduce="amax"
    )
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = input - max_per_src_element
    sum_per_index = scatter_reduce(
        recentered_scores.exp(),
        dim=dim,
        index=index,
        output_size=output_size,
        reduce="sum",
    )
    return max_value_per_index + sum_per_index.log()


if hasattr(torch.Tensor, "scatter_reduce_"):
    # version >= 1.12
    def scatter_reduce(input, dim, index, reduce, *, output_size=None):
        src = input
        if output_size is None:
            output_size = index.max() + 1
        return torch.empty(output_size, device=input.device).scatter_reduce(
            dim=dim, index=index, src=src, reduce=reduce, include_self=False
        )

else:
    scatter_reduce = torch.scatter_reduce


def log_substract(x, y):
    """log(exp(x) - exp(y))"""
    return x + torch.log1p(-(y - x).exp())


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
    tied_matrix = np.where(t_i == t_j, 1.0, 0.0).astype(np.float32)

    assert tied_matrix.ndim == 2
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
    tril = np.where(t_i <= t_j, 1.0, 0.0).astype(np.float32)
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


if __name__ == "__main__":
    test_ranking_loss()

import torch


class CoxPHLoss(torch.nn.Module):
    def forward(self, logh, events, durations, eps=1e-7):
        """
        Simple approximation of the COX-ph. Log hazard is not computed on risk-sets, but on ranked list instead.
        This approximation is valid for datamodules w/ low percentage of ties.
        Credit to Haavard Kamme/PyCox
        :param logh:
        :param durations:
        :param events:
        :param eps:
        :return:
        """
        # sort:
        idx = durations.sort(descending=True, dim=0)[1]
        events = events[idx].squeeze(-1)
        logh = logh[idx].squeeze(-1)
        # calculate loss:
        gamma = logh.max()
        log_cumsum_h = logh.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        if events.sum() > 0:
            loss = -logh.sub(log_cumsum_h).mul(events).sum().div(events.sum())
        else:
            loss = -logh.sub(log_cumsum_h).mul(events).sum()
        return loss

import torch


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

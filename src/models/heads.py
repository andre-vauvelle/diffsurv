import torch
from torch import nn


class PredictionHead(nn.Module):
    """
    Prediction MLP head for the model.
    """

    def __init__(
        self,
        in_features,
        out_features,
        hidden_dim,
        n_layers,
        act_fn=nn.LeakyReLU,
        dropout=0.2,
        norm=nn.LayerNorm,
        batch_norm=True,
    ):
        super().__init__()
        if n_layers > 0:
            sequence = []
            if batch_norm:
                sequence.append(nn.BatchNorm1d(in_features))
            sequence.extend([nn.Linear(in_features, hidden_dim), act_fn()])
            if norm is not None:
                sequence.append(norm(hidden_dim))
            for _ in range(n_layers - 1):
                sequence.append(nn.Linear(hidden_dim, hidden_dim))
                sequence.append(act_fn())
                if norm is not None:
                    sequence.append(norm(hidden_dim))
            sequence.append(nn.Dropout(dropout))
            self.final = nn.Linear(in_features=hidden_dim, out_features=out_features, bias=False)
        else:
            sequence = [nn.Dropout(dropout)]
            if batch_norm:
                sequence.append(nn.BatchNorm1d(in_features))
            # sequence.append(nn.Linear(in_features=in_features, out_features=out_features))
            self.final = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.layers = nn.Sequential(*sequence)
        self.act_fn = act_fn

        self.apply(self.init_weights_default)

    def forward(self, hidden_states):
        hidden_states = self.layers(hidden_states)
        hidden_states = self.final(hidden_states)
        return hidden_states

    def init_weights_default(self, m):
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

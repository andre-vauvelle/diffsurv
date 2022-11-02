from torch import autocast, nn


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
    ):
        super().__init__()
        if n_layers > 0:
            sequence = [nn.Batchnorm1d(in_features), nn.Linear(in_features, hidden_dim), act_fn()]
            if norm is not None:
                sequence.append(norm(hidden_dim))
            for _ in range(n_layers - 1):
                sequence.append(nn.Linear(hidden_dim, hidden_dim))
                sequence.append(act_fn())
                if norm is not None:
                    sequence.append(norm(hidden_dim))
            sequence.append(nn.Dropout(dropout))
            sequence.append(nn.Linear(in_features=hidden_dim, out_features=out_features))
        else:
            sequence = [
                nn.Dropout(dropout),
                nn.Batchnorm1d(in_features),
                nn.Linear(in_features=in_features, out_features=out_features),
            ]
        self.layers = nn.Sequential(*sequence)
        self.act_fn = act_fn

    def forward(self, hidden_states):
        hidden_states = self.layers(hidden_states)
        return hidden_states

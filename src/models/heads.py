from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.dense = nn.Linear(in_features=in_features, out_features=out_features)
        self.act_fn = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

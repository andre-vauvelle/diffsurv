import torch
from pytorch_lightning.utilities import rank_zero_warn
from torchmetrics import MetricCollection, Precision

from data.preprocess.utils import SYMBOL_IDX
from modules.base import BaseModel
from models.heads import PredictionHead
from modules.tasks import RiskMixin


class BiRNNBase(BaseModel):
    def __init__(self,
                 input_dim=1390,
                 output_dim=1390,
                 embedding_dim=120,
                 num_hidden_layers=3,
                 hidden_dropout_prob=0.2, lr=1e-4,
                 intermediate_size=256, hidden_act="relu",
                 bidirectional=True,
                 used_covs=('age_ass', 'sex'),
                 ):
        super().__init__()

        self.output_dim = output_dim
        self.lr = lr
        self.used_covs = used_covs

        self.embed = torch.nn.Embedding(input_dim, embedding_dim, sparse=True, padding_idx=SYMBOL_IDX['PAD'])

        self.model = torch.nn.GRU(input_size=embedding_dim, hidden_size=intermediate_size, num_layers=num_hidden_layers,
                                  batch_first=True, bidirectional=bidirectional,
                                  dropout=hidden_dropout_prob)

        head_input_dim = embedding_dim
        head_input_dim = head_input_dim * num_hidden_layers * (2 if bidirectional else 1)
        if used_covs is not None:
            head_input_dim = head_input_dim + len(used_covs)

        self.head = PredictionHead(in_features=head_input_dim, out_features=output_dim)

        metrics = MetricCollection(
            [
                Precision(compute_on_step=False, average='micro'),
            ]
        )

        self.valid_metrics = metrics.clone(prefix='val/')
        # self.cls = CustomBertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)

    def _shared_eval_step(self, batch, batch_idx):
        rank_zero_warn("`_shared_eval_step` must be implemented to be used with the Lightning Trainer")

    def forward(self, input_tuple, covariates) -> torch.Tensor:
        token_idx = input_tuple
        embedding = self.embed(token_idx)
        _, layer_outputs = self.model(embedding)
        rnn_out = layer_outputs.view(layer_outputs.size(1), -1)

        if self.used_covs is not None:
            pooled = torch.cat((rnn_out, covariates), dim=1)
        else:
            pooled = rnn_out
        logits = self.head(pooled)
        return logits


class BiRNNRisk(RiskMixin, BiRNNBase):
    def __init__(self,
                 input_dim=1390,
                 output_dim=1390,
                 embedding_dim=120,
                 num_hidden_layers=8,
                 hidden_dropout_prob=0.2, lr=1e-4,
                 intermediate_size=256, hidden_act="relu",
                 bidirectional=True,
                 used_covs=('age_ass', 'sex'),
                 label_vocab=None,
                 weightings=None,
                 use_weighted_loss=False,
                 loss=None,
                 grouping_labels=None
                 ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, embedding_dim=embedding_dim,
                         num_hidden_layers=num_hidden_layers, hidden_dropout_prob=hidden_dropout_prob,
                         lr=lr, intermediate_size=intermediate_size, hidden_act=hidden_act,
                         bidirectional=bidirectional, used_covs=used_covs, label_vocab=label_vocab,
                         weightings=weightings, loss=loss, use_weighted_loss=use_weighted_loss,
                         grouping_labels=grouping_labels, )
        self.save_hyperparameters()

    def _shared_eval_step(self, batch, batch_idx):
        (token_idx, age_idx, position, segment, mask, covariates), (label_multihot, label_times, censorings,
                                                                    exclusions) = batch

        logits = self(token_idx, covariates=covariates)

        loss = self.loss_func(logits, label_multihot, label_times)
        return loss, logits, label_multihot, label_times

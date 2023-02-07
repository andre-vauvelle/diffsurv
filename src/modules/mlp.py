import torch
from torch import nn
from torchmetrics import MetricCollection, Precision

from data.preprocess.utils import SYMBOL_IDX
from models.heads import PredictionHead
from modules.base import BaseModel
from modules.tasks import RiskMixin, SortingRiskMixin


class MultilayerBase(BaseModel):
    def __init__(
        self,
        input_dim=1390,
        output_dim=1390,
        embedding_dim=128,
        lr=1e-4,
        head_hidden_dim=256,
        head_layers=1,
        hidden_dropout_prob=0.2,
        pretrained_embedding_path=None,
        freeze_pretrained=False,
        count=True,
        cov_size=2,
        only_covs=False,
        batch_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = lr

        self.count = count
        self.cov_size = cov_size
        self.only_covs = only_covs

        if not only_covs:
            if pretrained_embedding_path is None:
                self.embed = nn.EmbeddingBag(
                    num_embeddings=input_dim,
                    embedding_dim=embedding_dim,
                    padding_idx=SYMBOL_IDX["PAD"],
                    mode="mean",
                    sparse=True,
                )
            else:
                # Use preprocess_ukb_omop.py to preprocess
                pretrained_embedding = torch.load(pretrained_embedding_path).float()
                # pretrained_embedding = pd.read_feather(pretrained_embedding_path)
                self.embed = nn.EmbeddingBag.from_pretrained(
                    pretrained_embedding, freeze=freeze_pretrained, sparse=True
                )

        if only_covs:
            head_input_dim = self.cov_size
        else:
            head_input_dim = embedding_dim + self.cov_size

        # else:
        self.head = PredictionHead(
            in_features=head_input_dim,
            out_features=output_dim,
            hidden_dim=head_hidden_dim,
            n_layers=head_layers,
            dropout=hidden_dropout_prob,
            batch_norm=batch_norm,
            norm=nn.BatchNorm1d,
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

        # TODO: consider moving to mixin
        metrics = MetricCollection([Precision(compute_on_step=False, average="micro")])
        self.valid_metrics = metrics.clone(prefix="val/")
        self.save_hyperparameters()

    def forward(self, covariates) -> torch.Tensor:
        pooled = covariates.float().requires_grad_()
        logits = self.head(pooled)

        return logits

    def _shared_eval_step(self, batch, batch_idx):
        covariates = batch["covariates"]
        label_multihot = batch["labels"]
        label_times = batch["label_times"]

        x_shape = covariates.shape
        if len(x_shape) == 3:
            covariates = covariates.reshape(-1, *x_shape[2:])
        logits = self(covariates)
        if len(x_shape) == 3:
            logits = logits.reshape(*x_shape[:2], 1)

        return logits, label_multihot, label_times


class MultilayerRisk(RiskMixin, MultilayerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()


class MultilayerDiffsort(SortingRiskMixin, MultilayerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

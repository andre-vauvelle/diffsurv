import torch

from models.imaging import SVHNConvNet
from modules.base import BaseModel
from modules.tasks import RiskMixin, SortingRiskMixin
from torchmetrics import MetricCollection, Precision


class ConvModule(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_net = SVHNConvNet()
        metrics = MetricCollection([Precision(compute_on_step=False, average="micro")])
        self.valid_metrics = metrics.clone(prefix="val/")

    def forward(self, img) -> torch.Tensor:
        return self.conv_net(img)

    def _shared_eval_step(self, batch, batch_idx):
        covariates = batch["covariates"]
        label_multihot = batch["labels"]
        label_times = batch["label_times"]
        logits = self(covariates)
        return logits, label_multihot, label_times


class ConvRisk(RiskMixin, ConvModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()


class ConvDiffsort(SortingRiskMixin, ConvModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

import torch
from torch import nn
from torchvision.models import densenet121

from models.imaging import SVHNConvNet
from modules.base import BaseModel
from modules.tasks import RiskMixin, SortingRiskMixin


class ConvModule(BaseModel):
    def __init__(self, model="svnh", **kwargs):
        super().__init__(**kwargs)
        if model == "densenet":
            self.conv_net = densenet121(pretrained=True)
            self.conv_net.classifier = nn.Linear(self.conv_net.classifier.in_features, 1)
        else:
            self.conv_net = SVHNConvNet()

    def forward(self, img) -> torch.Tensor:
        x_shape = img.shape
        if len(x_shape) == 5:
            x = img.view(-1, *x_shape[-3:])
        return self.conv_net(x)

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
        steepness = self.steepness
        self.save_hyperparameters()

import torch
from torch import nn
from torchvision.models import convnext_small, convnext_tiny, densenet121

from models.imaging import SVHNConvNet
from modules.base import BaseModel
from modules.tasks import RiskMixin, SortingRiskMixin


class ConvModule(BaseModel):
    def __init__(self, model="svnh", img_size=48, **kwargs):
        super().__init__(**kwargs)
        if model == "densenet":
            self.conv_net = densenet121(pretrained=True)
            self.conv_net.classifier = nn.Linear(
                self.conv_net.classifier.in_features, 1, bias=False
            )
        elif model == "small":
            self.conv_net = SVHNConvNet(img_size=img_size)
        elif model == "convnext_small":
            self.conv_net = convnext_small(pretrained=True)
            self.conv_net.classifier = nn.Linear(
                self.conv_net.classifier[2].out_features, 1, bias=False
            )
        elif model == "convnext_tiny":
            self.conv_net = convnext_tiny(pretrained=True)
            self.conv_net.classifier = nn.Linear(
                self.conv_net.classifier[2].out_features, 1, bias=False
            )
        else:
            raise ValueError(
                f"Model {model} not recongized, must be either densenet, small, convnext_small, "
                "convnext_tiny"
            )

    def forward(self, img) -> torch.Tensor:
        x_shape = img.shape
        if len(x_shape) == 5:
            img = img.view(-1, *x_shape[-3:])
        logits = self.conv_net(img)
        logits = logits.view(*x_shape[:-3], 1)
        return logits

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

from functools import partial
from typing import Any, Callable, Optional, Union

import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.optim import Optimizer
from torchvision.models import convnext_small, convnext_tiny, densenet121, efficientnet_b0

from models.imaging import SVHNConvNet
from modules.base import BaseModel
from modules.tasks import RiskMixin, SortingRiskMixin


class ConvModule(BaseModel):
    def __init__(self, model="svnh", img_size=48, head_steps=200, weight_decay=0, **kwargs):
        super().__init__(**kwargs)
        self.head_steps = head_steps
        self.weight_decay = weight_decay
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        if model == "densenet":
            self.conv_net = densenet121(pretrained=True)
            self.conv_net.classifier = nn.Linear(
                self.conv_net.classifier.in_features, 1, bias=False
            )
        elif model == "small":
            self.conv_net = SVHNConvNet(img_size=img_size)
        elif model == "efficientnet":
            self.conv_net = efficientnet_b0(pretrained=True)
            self.conv_net.classifier = nn.Linear(
                self.conv_net.classifier[1].in_features, 1, bias=False
            )
        elif model == "convnext_small":
            self.conv_net = convnext_small(pretrained=True)
            self.conv_net.classifier = nn.Sequential(
                nn.Flatten(1), norm_layer(768), nn.Linear(768, 1, bias=False)
            )
        elif model == "convnext_tiny":
            self.conv_net = convnext_tiny(pretrained=True)
            self.conv_net.classifier = nn.Sequential(
                nn.Flatten(1), norm_layer(768), nn.Linear(768, 1, bias=False)
            )
        else:
            raise ValueError(
                f"Model {model} not recongized, must be either densenet, small, convnext_small, "
                "convnext_tiny"
            )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.conv_net.features.parameters()},
                {"params": self.conv_net.classifier.parameters()},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        lr_schedulers = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, mode="max", patience=2, verbose=True
            ),
            "monitor": "val/c_index/all",
        }
        return optimizer, lr_schedulers

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

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # skip training the features for the first head_steps
        if self.trainer.global_step < self.head_steps:
            optimizer.param_groups[0]["lr"] = 0
        else:
            optimizer.param_groups[0]["lr"] = self.lr

        optimizer.step(closure=optimizer_closure)


class ConvRisk(RiskMixin, ConvModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()


class ConvDiffsort(SortingRiskMixin, ConvModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        steepness = self.steepness
        self.save_hyperparameters()

import torch
import torch.nn.functional as F
from torch import nn


class SVHNConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * 3 * 256, 64),
            nn.ReLU(),
            nn.Linear(64, 1, bias=False),
        )

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 5:
            x = x.view(-1, *x_shape[-3:])
        x = self.features(x)
        x = x.view(*x_shape[:-3], -1)
        x = self.classifier(x)
        return x

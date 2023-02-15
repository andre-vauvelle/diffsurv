import torch
import torch.nn.functional as F
from torch import nn


class SVHNConvNet(nn.Module):
    def __init__(self, img_size=48):
        scale = img_size / 48
        fc1_size = int(3 * 3 * 256 * scale**2)
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(fc1_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 5:
            x = x.view(-1, *x_shape[-3:])
        x = torch.utils.checkpoint.checkpoint(self.convblock1, x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.utils.checkpoint.checkpoint(self.convblock2, x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.utils.checkpoint.checkpoint(self.convblock3, x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.utils.checkpoint.checkpoint(self.convblock4, x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(*x_shape[:-3], -1)
        x = torch.utils.checkpoint.checkpoint(self.fc1, x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

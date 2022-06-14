import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from typing import List


def torch_show(imgs, titles: List[str], suptitle: str):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    fig.suptitle(suptitle, fontsize=16)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title=title)

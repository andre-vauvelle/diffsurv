from typing import Literal, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_warn


class BaseModel(pl.LightningModule):
    def __init__(self, optimizer: Literal["adam", "sgd", "adamw"] = "adam", weight_decay=0, lr=0.1):
        super().__init__()
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr = lr

    def configure_optimizers(self):
        sparse = [p for n, p in self.named_parameters() if "embed" in n]
        not_sparse = [p for n, p in self.named_parameters() if "embed" not in n]
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(not_sparse, lr=self.lr)
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(not_sparse, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(not_sparse, lr=self.lr, weight_decay=self.weight_decay)

        if sparse:
            optimizer_sparse = torch.optim.SparseAdam(sparse, lr=self.lr)
            return optimizer_sparse, optimizer
        else:
            return optimizer

    @staticmethod
    def _row_unique(x):
        """get unique unique idx for each row"""
        # sorting the rows so that duplicate values appear together
        # e.g., first row: [1, 2, 3, 3, 3, 4, 4]
        y, indices = x.sort(dim=-1)

        # subtracting, so duplicate values will become 0
        # e.g., first row: [1, 2, 3, 0, 0, 4, 0]
        y[:, 1:] *= ((y[:, 1:] - y[:, :-1]) != 0).long()

        # retrieving the original indices of elements
        indices = indices.sort(dim=-1)[1]

        # re-organizing the rows following original order
        # e.g., first row: [1, 2, 3, 4, 0, 0, 0]
        result = torch.gather(y, 1, indices)
        return result

    def _shared_eval_step(self, batch, batch_idx):
        rank_zero_warn(
            "`_shared_eval_step` must be implemented to be used with the Lightning Trainer"
        )

    def training_step(self, batch, batch_idx, optimizer_idx: Optional[int] = None):
        rank_zero_warn("`training_step` must be implemented to be used with the Lightning Trainer")

    def validation_step(self, batch, batch_idx):
        rank_zero_warn(
            "`validation_step` must be implemented to be used with the Lightning Trainer"
        )

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)
        self.log("hp_metric", output["val/AveragePrecision"])

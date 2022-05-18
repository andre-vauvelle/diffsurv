import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule):
    def __init__(self,
                 input_dim=1390,
                 output_dim=1390,
                 used_covs=('age_ass', 'sex'),
                 ):
        super().__init__()

    def forward(self, input_tuple, covariates) -> torch.Tensor:
        token_idx, age_idx, position, segment, mask = input_tuple

        _, pooled = self.model(token_idx, age_idx, position, segment, mask)
        if self.used_covs is not None:
            pooled = torch.cat((pooled, covariates), dim=1)

        logits = self.head(pooled)
        return logits

    def configure_optimizers(self):

        sparse = [p for n, p in self.named_parameters() if 'embed' in n]
        not_sparse = [p for n, p in self.named_parameters() if 'embed' not in n]
        optimizer_sparse = torch.optim.SparseAdam(sparse, lr=self.lr)
        optimizer = torch.optim.Adam(not_sparse, lr=self.lr)
        return optimizer_sparse, optimizer

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
        rank_zero_warn("`_shared_eval_step` must be implemented to be used with the Lightning Trainer")

    def training_step(self, batch, batch_idx, optimizer_idx):
        rank_zero_warn("`training_step` must be implemented to be used with the Lightning Trainer")

    def validation_step(self, batch, batch_idx):
        rank_zero_warn("`validation_step` must be implemented to be used with the Lightning Trainer")

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)
        self.log('hp_metric', output['val/AveragePrecision'])

    def test_step(self, batch, batch_idx):
        rank_zero_warn("`test_step` must be implemented to be used with the Lightning Trainer")

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(output, prog_bar=True)

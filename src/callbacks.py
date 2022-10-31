import os
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

import wandb
from definitions import RESULTS_DIR
from models.loggers import CustomWandbLogger
from omni.common import _create_folder_if_not_exist


class OnTrainEndResults(Callback):
    """Get results on training end!"""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        dataloader = trainer.val_dataloaders[0]
        store = []
        for batch in tqdm(iter(dataloader)):
            batch: dict
            outputs = pl_module(0, covariates=batch["covariates"])

            batch.update({"logits": outputs})

            # Let's get things in a friendly format for pandas dataframe..
            numpy_batch = {}
            for k, v in batch.items():
                # Detach from graph and make sure on cpu
                new_v = v.detach().cpu()
                if k != "covariates":
                    new_v = new_v.flatten()
                    numpy_batch[k] = new_v.numpy().tolist()
                # Covariates have multiple dim so should not be flattened
                else:
                    dim = new_v.shape[1]
                    for i in range(dim):
                        numpy_batch[k + f"_{i}"] = new_v[:, i].numpy().tolist()

            store.append(numpy_batch)

        results = pd.DataFrame(store)
        results_df = results.explode(list(results.columns), ignore_index=True)

        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            exp: wandb.sdk.wandb_run.Run = logger.experiment
            path = os.path.join(RESULTS_DIR, self.save_dir, str(exp._run_id) + "_results.parquet")
            _create_folder_if_not_exist(path)
            results_df.to_parquet(path)
            exp.log_artifact(path, type="dataset")
        else:
            path = os.path.join(RESULTS_DIR, self.save_dir, "results.parquet")
            _create_folder_if_not_exist(path)
            results_df.to_parquet(path)


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if not batch_idx % 200:
            wandb_logger = trainer.logger

            # `outputs` comes from `LightningModule.validation_step`
            loss, predictions, perm_prediction, perm_ground_truth = outputs

            # Let's log 20 sample image predictions from first batch

            label_times = batch["label_times"]

            idx = torch.argsort(label_times.squeeze(), descending=False)
            perm_ascending = torch.nn.functional.one_hot(idx).transpose(-2, -1).float()
            perm_prediction_asc = perm_ascending.T @ perm_prediction
            perm_ground_truth_asc = perm_ascending.T @ perm_ground_truth

            captions = ["Soft Permutation", "Predicted Permutation"]

            wandb_logger.log_image(
                key=f"batch:{batch_idx}",
                images=[perm_ground_truth_asc, perm_prediction_asc],
                caption=captions,
            )

            # lt_dist = torch.nn.functional.normalize(label_times.squeeze()[idx], dim=0)
            # pred_dist = torch.nn.functional.normalize(predictions[idx], dim=0)
            #
            # data = list(zip(lt_dist, pred_dist))
            # table = wandb.Table(data=data, columns=["label", "value"])
            # wandb_logger.({"my_bar_chart_id": wandb.plot.bar(table, "label", "value", title="Custom Bar Chart")})
            #
            # torch_show([perm_ground_truth_asc, perm_prediction_asc])
            # plt.show()


# log_predictions_callback = LogPredictionsCallback()

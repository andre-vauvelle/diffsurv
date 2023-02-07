import os
import socket
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.profiler import tensorboard_trace_handler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from definitions import RESULTS_DIR
from omni.common import _create_folder_if_not_exist


class TorchTensorboardProfilerCallback(Callback):
    """Quick-and-dirty Callback for invoking TensorboardProfiler during training.

    For greater robustness, extend the pl.profiler.profilers.BaseProfiler. See
    https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html"""

    def __init__(self, save_dir: str = "wandb/latest-run/tbprofile"):
        super().__init__()
        self.save_dir = save_dir
        wait, warmup, active, repeat = 1, 1, 2, 1
        # total_steps = (wait + warmup + active) * (1 + repeat)
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=tensorboard_trace_handler(save_dir),
            with_stack=False,
        )
        self.profiler = profiler

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.profiler.step()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger = trainer.logger
        exp: wandb.sdk.wandb_run.Run = logger.experiment
        worker_name = f"{socket.gethostname()}_{str(os.getpid())}"
        filename = [f for f in os.listdir(self.save_dir) if worker_name in f][0]
        path = os.path.join(self.save_dir, filename)
        exp.log_artifact(path, type="profile")


class OnTrainEndResults(Callback):
    """Get results on training end!"""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir

    def _extract(
        self, trainer: "pl.Trainer", dataloader: DataLoader, pl_module: "pl.LightningModule"
    ) -> None:
        store = []
        for batch in tqdm(iter(dataloader)):
            batch: dict
            logits = pl_module(covariates=batch["covariates"])

            batch.update({"logits": logits})

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

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        dataloader = trainer.val_dataloaders[0]
        self._extract(trainer, dataloader, pl_module)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        dataloader = trainer.predict_dataloaders[0]
        self._extract(trainer, dataloader, pl_module)


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

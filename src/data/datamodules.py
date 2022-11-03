import os
from typing import Literal, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

import wandb
from data.datasets import DatasetRisk


class DataModuleRisk(pl.LightningDataModule):
    """
    Args:
        :param wandb_artifact: wandb artficact dataset to use.
        :param local_path: local path to data, only used if there is no wandb_artifact...
    """

    def __init__(
        self,
        wandb_artifact: Optional[str] = "qndre/diffsurv/pysurv_square_0.3.pt:v0",
        local_path: Optional[str] = None,
        setting: Optional[Literal["realworld", "synthetic"]] = None,
        val_split=0.2,
        batch_size=32,
        num_workers=1,
    ):
        super().__init__()
        self.wandb_artifact = wandb_artifact
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        if wandb_artifact is not None:
            api = wandb.Api()
            artifact = api.artifact(self.wandb_artifact)
            wandb_dir = artifact.download(root=f"../data/wandb/{wandb_artifact}")
            wandb_path = os.listdir(wandb_dir)[0]
            self.path = os.path.join(wandb_dir, wandb_path)
            self.setting = artifact.metadata["setting"]
        elif local_path is not None:
            self.path = local_path
            if setting is None:
                raise Exception(
                    "setting argument must be set to either 'realworld' or 'synthetic if using"
                    f" local path, currently: {setting}"
                )
        else:
            raise Exception("Needs either local_path or wandb_artifact... Both are None")
        data = torch.load(os.path.join(self.path))
        self.input_dim = data["x_covar"].shape[1]
        self.cov_size = data["x_covar"].shape[1]
        self.output_dim = data["y_times"].shape[1]
        self.label_vocab = {"token2idx": {"event0": 0}, "idx2token": {0: "event0"}}
        self.grouping_labels = {"all": ["event0"]}

    def get_dataloader(self, stage: Literal["train", "val"]):
        data = torch.load(self.path)
        x_covar, y_times, censored_events = (
            data["x_covar"],
            data["y_times"],
            data["censored_events"],
        )
        if self.setting == "synthetic":
            risk = data["risk"]
        else:
            risk = None

        n_patients = x_covar.shape[0]
        if stage == "train":
            n_training_patients = int(n_patients * (1 - self.val_split))
            dataset = DatasetRisk(
                x_covar[:n_training_patients],
                y_times[:n_training_patients],
                censored_events[:n_training_patients],
                risk[:n_training_patients] if risk is not None else None,
            )
        elif stage == "val":
            n_validation_patients = int(n_patients * self.val_split)
            dataset = DatasetRisk(
                x_covar[-n_validation_patients:],
                y_times[-n_validation_patients:],
                censored_events[-n_validation_patients:],
                risk[-n_validation_patients:] if risk is not None else None,
            )
        else:
            raise Exception("Stage must be either 'train' or 'val' ")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
        )

    def train_dataloader(self):
        train_dataloader = self.get_dataloader(stage="train")
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = self.get_dataloader(stage="val")
        return val_dataloader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    def test_dataloader(self):
        pass

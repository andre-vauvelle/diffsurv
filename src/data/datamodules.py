import os
from typing import List, Literal, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

import wandb
from data.datasets import CaseControlRiskDataset, DatasetRisk
from definitions import DATA_DIR


class DataModuleCXR(pl.LightningDataModule):
    def __init__(
        self,
        local_path: str = os.path.join(DATA_DIR, "mimic", "splits.csv"),
        batch_size: int = 32,
        val_batch_size: Optional[int] = 32,
        risk_set_size: Optional[int] = None,
        num_workers: int = 0,
        inc_censored_in_ties: bool = True,
        batch_risk_tuple: Optional[Union[tuple, str]] = None,
    ):
        super().__init__()
        if batch_risk_tuple is not None:  # wandb hack
            if isinstance(batch_risk_tuple, str):
                batch_risk_tuple: list[int] = [int(i) for i in batch_risk_tuple.split(",")]
                assert len(batch_risk_tuple) == 2
            batch_size = batch_risk_tuple[0]
            risk_set_size = batch_risk_tuple[1]
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.risk_set_size = risk_set_size
        self.controls_per_case = risk_set_size - 1
        self.num_workers = os.cpu_count() - 2 if num_workers == -1 else num_workers
        self.setting = "realworld"
        self.input_dim = 50
        self.cov_size = 50
        self.output_dim = 1
        self.inc_censored_in_ties = inc_censored_in_ties
        self.label_vocab = {"token2idx": {"event0": 0}, "idx2token": {0: "event0"}}
        self.grouping_labels = {"all": ["event0"]}
        self.local_path = local_path

        self.save_hyperparameters()

    def get_dataloader(self, stage: Literal["train", "val", "test"]):
        # Pre-split provided
        splits = pd.read_csv(self.local_path)

        idx = (splits.split == stage) & splits.exists

        y_times = torch.Tensor(splits.loc[idx, "tte"].values).float()
        censored_events = 1 - torch.LongTensor(splits.loc[idx, "event"].values)
        x_covar = splits.loc[idx, "path"].values

        if stage == "train":
            dataset = CaseControlRiskDataset(
                self.controls_per_case,
                x_covar,
                y_times,
                censored_events,
                inc_censored_in_ties=self.inc_censored_in_ties,
                transform=transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.Resize(230),
                        transforms.CenterCrop(224),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                ),
            )
        else:
            dataset = DatasetRisk(
                x_covar,
                y_times,
                censored_events,
                transform=transforms.Compose(
                    [
                        transforms.Resize(230),
                        transforms.CenterCrop(224),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                ),
            )
            # TODO: THe MOdel!

        # Validation must be not have casecontrol sampling (Otherwise not all patients included)
        # if self.controls_per_case is None or stage == "val":
        if self.risk_set_size >= 5:
            val_batch_size = self.risk_set_size
        else:
            val_batch_size = 5  # ensures that EM6 metric can be calculated
        return DataLoader(
            dataset,
            batch_size=self.batch_size if stage == "train" else val_batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True if stage == "train" else False,
            pin_memory=True,
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
        test_dataloader = self.get_dataloader(stage="test")
        return test_dataloader


class DataModuleRisk(pl.LightningDataModule):
    """
    Args:
        :param wandb_artifact: wandb artficact dataset to use.
        :param local_path: local path to data, only used if there is no wandb_artifact...
        :param k_fold: (kth fold, total_folds)
    """

    def __init__(
        self,
        wandb_artifact: Optional[str] = None,
        local_path: Optional[str] = None,
        setting: str = "realworld",
        val_split: Optional[float] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = 32,
        risk_set_size: Optional[int] = None,
        num_workers: int = 0,
        return_perm_mat: bool = True,
        inc_censored_in_ties: bool = True,
        k_fold: Optional[tuple] = (1, 5),
        use_risk: bool = False,
        random_sample: bool = False,
        batch_risk_tuple: Optional[Union[tuple, str]] = None,
    ):
        super().__init__()
        self.random_sample = random_sample
        self.val_batch_size = val_batch_size
        self.k_fold = k_fold
        self.inc_censored_in_ties = inc_censored_in_ties
        if batch_risk_tuple is not None:  # wandb hack
            if isinstance(batch_risk_tuple, str):
                batch_risk_tuple: list[int] = [int(i) for i in batch_risk_tuple.split(",")]
                assert len(batch_risk_tuple) == 2
            batch_size = batch_risk_tuple[0]
            risk_set_size = batch_risk_tuple[1]
        self.risk_set_size = risk_set_size
        self.batch_size = batch_size
        self.controls_per_case = risk_set_size - 1  # one is a case...
        self.wandb_artifact = wandb_artifact
        self.val_split = val_split
        self.use_risk = use_risk
        self.num_workers = os.cpu_count() - 2 if num_workers == -1 else num_workers
        self.return_perm_mat = return_perm_mat
        if wandb_artifact is not None:
            api = wandb.Api(overrides={"project": "diffsurv", "entity": "cardiors"})
            artifact = api.artifact(self.wandb_artifact)
            self.wandb_dir = artifact.download(root=f"../data/wandb/{self.wandb_artifact}")
            setting = artifact.metadata["setting"]
            self.setting = setting
            self.input_dim = artifact.metadata["input_dim"]
            self.cov_size = artifact.metadata["input_dim"]
            self.output_dim = artifact.metadata["output_dim"]
        elif local_path is not None:
            self.path = local_path
            if setting is None:
                raise Exception(
                    "setting argument must be set to either 'realworld' or 'synthetic if using"
                    f" local path, currently: {setting}"
                )
            data = torch.load(os.path.join(self.path))
            self.setting = setting
            self.input_dim = data["x_covar"].shape[1]
            self.cov_size = data["x_covar"].shape[1]
            self.output_dim = data["y_times"].shape[1]
        else:
            raise Exception("Needs either local_path or wandb_artifact... Both are None")
        self.label_vocab = {"token2idx": {"event0": 0}, "idx2token": {0: "event0"}}
        self.grouping_labels = {"all": ["event0"]}
        self.save_hyperparameters()

    def get_dataloader(self, stage: Literal["train", "val", "test"]):
        if self.wandb_artifact and (
            "kkbox_v1:" in self.wandb_artifact or "SVNH" in self.wandb_artifact
        ):
            # Pre-split provided
            if stage == "train":
                wandb_path = [p for p in os.listdir(self.wandb_dir) if "train" in p][0]
                self.path = os.path.join(self.wandb_dir, wandb_path)
                shuffle = True
            elif stage == "val":
                wandb_path = [p for p in os.listdir(self.wandb_dir) if "val" in p][0]
                self.path = os.path.join(self.wandb_dir, wandb_path)
                shuffle = False
            else:
                wandb_path = [p for p in os.listdir(self.wandb_dir) if "test" in p][0]
                self.path = os.path.join(self.wandb_dir, wandb_path)
                shuffle = False
            data = torch.load(self.path)
            x_covar, y_times, censored_events = (
                data["x_covar"],
                data["y_times"].flatten(),
                data["censored_events"].flatten(),
            )
            if stage == "train":
                dataset = CaseControlRiskDataset(
                    self.controls_per_case,
                    x_covar,
                    y_times if not self.use_risk else data["risk"],
                    censored_events,
                    risk=None if "kkbox_v1" in self.wandb_artifact else data["risk"],
                    inc_censored_in_ties=self.inc_censored_in_ties,
                )
            else:
                dataset = DatasetRisk(
                    x_covar,
                    y_times,
                    censored_events,
                    risk=None if "kkbox_v1" in self.wandb_artifact else data["risk"],
                )
        else:
            # Manually split data
            if self.wandb_artifact:
                wandb_path = os.listdir(self.wandb_dir)[0]
                self.path = os.path.join(self.wandb_dir, wandb_path)
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
                if not self.k_fold:
                    n_training_patients = (
                        int(n_patients * (1 - self.val_split)) if self.val_split else n_patients
                    )
                    dataset = CaseControlRiskDataset(
                        self.controls_per_case,
                        x_covar[:n_training_patients],
                        y_times[:n_training_patients],
                        censored_events[:n_training_patients],
                        risk[:n_training_patients] if risk is not None else None,
                        return_perm_mat=self.return_perm_mat,
                        inc_censored_in_ties=self.inc_censored_in_ties,
                        random_sample=self.random_sample,
                    )
                else:
                    idx = set(range(n_patients))
                    kth_fold, total_folds = self.k_fold
                    shift = int(kth_fold * (n_patients / total_folds))
                    remove_idx = set(range(shift, shift + int(n_patients / total_folds)))
                    fold_idx = list(idx - remove_idx)

                    dataset = CaseControlRiskDataset(
                        self.controls_per_case,
                        x_covar[fold_idx],
                        y_times[fold_idx],
                        censored_events[fold_idx],
                        risk[fold_idx] if risk is not None else None,
                        return_perm_mat=self.return_perm_mat,
                        inc_censored_in_ties=self.inc_censored_in_ties,
                        random_sample=self.random_sample,
                    )
                shuffle = True
            elif stage == "val" or stage == "test":
                if not self.k_fold:
                    n_validation_patients = (
                        int(n_patients * self.val_split) if self.val_split else n_patients
                    )
                    dataset = DatasetRisk(
                        x_covar[-n_validation_patients:],
                        y_times[-n_validation_patients:],
                        censored_events[-n_validation_patients:],
                        risk[-n_validation_patients:] if risk is not None else None,
                    )
                else:
                    kth_fold, total_folds = self.k_fold
                    shift = int(kth_fold * (n_patients / total_folds))
                    remove_idx = range(shift, shift + int(n_patients / total_folds))
                    fold_idx = list(remove_idx)
                    dataset = DatasetRisk(
                        x_covar[fold_idx],
                        y_times[fold_idx],
                        censored_events[fold_idx],
                        risk[fold_idx] if risk is not None else None,
                    )

                shuffle = False
            else:
                raise Exception("Stage must be either 'train' or 'val' or 'test' ")

        # Validation must be not have casecontrol sampling (Otherwise not all patients included)
        # if self.controls_per_case is None or stage == "val":
        if self.risk_set_size >= 5:
            val_batch_size = self.risk_set_size
        else:
            val_batch_size = 5  # ensures that EM6 metric can be calculated
        return DataLoader(
            dataset,
            batch_size=self.batch_size if stage == "train" else val_batch_size,
            num_workers=self.num_workers,
            drop_last=True,  # Need to drop for efrons
            shuffle=shuffle,
            pin_memory=True,
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
        test_dataloader = self.get_dataloader(stage="test")
        return test_dataloader

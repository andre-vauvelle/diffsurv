import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.cli import LightningCLI, SaveConfigCallback

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from data.datamodules import DataModuleMLM
from definitions import MONGO_DB, MONGO_STR, DATA_DIR, TENSORBOARD_DIR, MODEL_DIR
from models.mlp import MultilayerMLM


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # TODO: fix not working...
        parser.link_arguments("data.input_dim", "model.input_dim", apply_on="instantiate")
        parser.link_arguments("data.output_dim", "model.output_dim", apply_on="instantiate")


# class CustomSaveConfigCallback(SaveConfigCallback):
#     def on_train_start(self, trainer, pl_module) -> None:
#         pass
#         # log_dir = trainer.log_dir or trainer.default_root_dir
#         # config_path = os.path.join(log_dir, self.config_filename)
#         # self.parser.save(self.config, config_path, skip_none=False)


def cli_main():
    cli = CustomLightningCLI(
        MultilayerMLM, DataModuleMLM, seed_everything_default=42,
        trainer_defaults={"gpus": -1 if torch.cuda.is_available() else 0},
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()

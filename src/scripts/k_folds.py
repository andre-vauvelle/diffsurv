#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from pytorch_lightning import seed_everything
import wandb

from definitions import RESULTS_DIR
from scripts.mlp import mlp_cli_main
from scripts.mlpdiffsort import diffsort_cli_main

parser = argparse.ArgumentParser(description="Run K Folds of training and validation for a model")
parser.add_argument("-k", "--kfolds", type=int, default=5)
parser.add_argument("-c", "--config", type=str)
parser.add_argument("-m", "--model", type=str, default="diffsort")
parser.add_argument("-d", "--results_dir", type=str)
parser.add_argument("-n", "--results_name", type=str)

seed_everything(seed=42)

import yaml


def load_yaml_file(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)


def combine_dicts(dict1, dict2):
    combined_dict = dict1.copy()
    combined_dict.update(dict2)
    return combined_dict


args = parser.parse_args()

RESULTS_NAME = f"{args.results_name}.csv"

if args.results_dir is None:
    results_path = os.path.join(RESULTS_DIR, args.model, RESULTS_NAME)
else:
    results_dir = Path(args.results_dir)
    if not os.path.exists(results_dir.parent.absolute()):
        raise OSError(f"Parent of results dir{results_dir.parent.absolute()} does not exist!")
    results_path = os.path.join(results_dir, RESULTS_NAME)

sys.argv = [sys.argv[0]]  # clear argv to manually pass arguments to cli_main


if args.model == "diffsort":
    model_cli = diffsort_cli_main
elif args.model == "mlp":
    model_cli = mlp_cli_main
else:
    raise NotImplementedError("model must be one of the implmented {'mlp', 'diffsort'}")

metrics_store = []
for k in range(args.kfolds):
    with open(args.config) as file:
        configuration = yaml.safe_load(file)

    configuration["data"]["k_fold"] = (k, args.kfolds)

    cli = model_cli(args=configuration, run=False)
    cli.trainer.fit(cli.model, cli.datamodule)
    metrics = cli.trainer.test(cli.model, cli.datamodule)
    metrics_store.append(metrics[0])

    wandb.finish()

df = pd.DataFrame.from_dict(metrics_store)
df.to_csv(results_path)

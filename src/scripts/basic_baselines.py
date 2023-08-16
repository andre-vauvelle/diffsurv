import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold, train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from tqdm import tqdm

import wandb

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
RANDOM_STATE = 42

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Select the dataset to use", type=str)
parser.add_argument(
    "--data_dir",
    help="Path to realworld data dir",
    type=str,
    default="/Users/andre/Documents/UCL/diffsurv/data/realworld",
)
parser.add_argument(
    "--results_dir",
    help="Path to realworld results dir",
    type=str,
    default="/Users/andre/Documents/UCL/diffsurv/results",
)
parser.add_argument(
    "--models", help="Models to run", type=str, nargs="+", default=["cph", "trees", "gb"]
)

args = parser.parse_args()
data_dir = args.data_dir
results_dir = args.results_dir
models_to_run = args.models

# Define hyperparameter search space for each model
cph_hyperparams = {"penalizer": [0.01, 0.1, 1, 10]}
trees_hyperparams = {
    "n_estimators": [250, 500],
    "max_depth": [3, 5, 10],
    "min_samples_leaf": [10, 20, 50],
}
gb_hyperparams = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
}

# Initialize wandb
wandb.init(job_type="basic_baselines", project="diffsurv", entity="cardiors")

# Log hyperparameter search space
wandb.log({"cph_hyperparams": cph_hyperparams, "trees_hyperparams": trees_hyperparams})

datasets = [
    "flchain.pt",
    "nwtco.pt",
    "support.pt",
    "metabric.pt",
]

if args.dataset and args.dataset not in datasets:
    raise ValueError(f"Invalid dataset. Allowed values are {', '.join(datasets)}")

datasets = [args.dataset] if args.dataset else datasets
wandb.log({"datasets": datasets})

outer_k = 5
val_split = 0.2
outer_kf = KFold(n_splits=outer_k, shuffle=True, random_state=42)

results = []

for dataset in datasets:
    path = os.path.join(data_dir, dataset)

    data = torch.load(path)

    x_covar, y_times, censored_events = (
        data["x_covar"],
        data["y_times"],
        data["censored_events"],
    )

    future_label_multihot = 1 - censored_events
    future_label_times = y_times

    X = x_covar.numpy()
    T = future_label_times.numpy().flatten()
    E = future_label_multihot.numpy().flatten()

    cph_c_indexes = []
    trees_c_indexes = []
    gb_c_indexes = []
    best_cph_params_list = []
    best_trees_params_list = []
    best_gb_params_list = []

    for outer_train_index, test_index in tqdm(
        outer_kf.split(X), desc="Outer loop", total=outer_k, ncols=100
    ):
        X_outer_train, X_test = X[outer_train_index], X[test_index]
        T_outer_train, T_test = T[outer_train_index], T[test_index]
        E_outer_train, E_test = E[outer_train_index], E[test_index]

        best_cph_params = None
        best_trees_params = None
        best_gb_params = None
        best_cph_score = -np.inf
        best_trees_score = -np.inf
        best_gb_score = -np.inf

        X_train, X_val, T_train, T_val, E_train, E_val = train_test_split(
            X_outer_train,
            T_outer_train,
            E_outer_train,
            test_size=val_split,
            shuffle=True,
            random_state=42,
        )

        train_df = pd.DataFrame(
            np.column_stack((X_train, E_train, T_train)),
            columns=list(range(X_train.shape[1])) + ["event", "time"],
        )
        val_df = pd.DataFrame(
            np.column_stack((X_val, E_val, T_val)),
            columns=list(range(X_val.shape[1])) + ["event", "time"],
        )
        train_x = train_df.drop(columns=["event", "time"])
        val_x = val_df.drop(columns=["event", "time"])
        # Convert train_y and val_y to structured arrays
        train_y = np.array(
            list(zip(train_df["event"].astype(bool), train_df["time"])),
            dtype=[("event", np.bool_), ("time", np.float64)],
        )
        val_y = np.array(
            list(zip(val_df["event"].astype(bool), val_df["time"])),
            dtype=[("event", np.bool_), ("time", np.float64)],
        )

        # hyperparameter tuning for cox proportional hazards model
        if "cph" in models_to_run:
            for penalizer in tqdm(cph_hyperparams["penalizer"], desc="CPH hyperparams", ncols=100):
                cph = CoxPHFitter(penalizer=penalizer)
                cph.fit(train_df, duration_col="time", event_col="event")
                c_index = cph.score(val_df, scoring_method="concordance_index")
                if c_index > best_cph_score:
                    best_cph_score = c_index
                    best_cph_params = {"penalizer": penalizer}

        # Hyperparameter tuning for Survival Trees model
        if "trees" in models_to_run:
            for n_estimators, max_depth, min_samples_leaf in tqdm(
                itertools.product(
                    trees_hyperparams["n_estimators"],
                    trees_hyperparams["max_depth"],
                    trees_hyperparams["min_samples_leaf"],
                ),
                desc="Trees hyperparams",
                ncols=100,
                total=len(trees_hyperparams["n_estimators"])
                * len(trees_hyperparams["max_depth"])
                * len(trees_hyperparams["min_samples_leaf"]),
            ):
                trees = RandomSurvivalForest(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                )
                trees.fit(train_x, train_y)

                c_index = trees.score(val_x, val_y)
                if c_index > best_trees_score:
                    best_trees_score = c_index
                    best_trees_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    }

        # hyperparameter tuning for Gradient Boosting Survival Analysis
        if "gb" in models_to_run:
            best_gb_score = -np.inf
            best_gb_params = None
            for n_estimators, learning_rate, max_depth in tqdm(
                itertools.product(
                    gb_hyperparams["n_estimators"],
                    gb_hyperparams["learning_rate"],
                    gb_hyperparams["max_depth"],
                ),
                desc="GB hyperparams",
                ncols=100,
                total=len(gb_hyperparams["n_estimators"])
                * len(gb_hyperparams["learning_rate"])
                * len(gb_hyperparams["max_depth"]),
            ):
                gb = GradientBoostingSurvivalAnalysis(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                )
                gb.fit(train_x, train_y)
                c_index = gb.score(val_x, val_y)
                if c_index > best_gb_score:
                    best_gb_score = c_index
                    best_gb_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                    }

        # Train and evaluate the models with the best hyperparameters on the outer fold
        train_df = pd.DataFrame(
            np.column_stack((X_outer_train, E_outer_train, T_outer_train)),
            columns=list(range(X_outer_train.shape[1])) + ["event", "time"],
        )
        test_df = pd.DataFrame(
            np.column_stack((X_test, E_test, T_test)),
            columns=list(range(X_test.shape[1])) + ["event", "time"],
        )

        # Outerloop: performance of best performing runs on test set
        train_x = train_df.drop(columns=["event", "time"])
        test_x = test_df.drop(columns=["event", "time"])
        # Convert train_y and val_y to structured arrays
        train_y = np.array(
            list(zip(train_df["event"].astype(bool), train_df["time"])),
            dtype=[("event", np.bool_), ("time", np.float64)],
        )
        test_y = np.array(
            list(zip(test_df["event"].astype(bool), test_df["time"])),
            dtype=[("event", np.bool_), ("time", np.float64)],
        )

        # Cox Proportional Hazards
        if "cph" in models_to_run:
            cph = CoxPHFitter(**best_cph_params)
            cph.fit(train_df, duration_col="time", event_col="event")
            c_index = cph.score(test_df, scoring_method="concordance_index")
            cph_c_indexes.append(c_index)
            best_cph_params_list.append(best_cph_params)

        if "trees" in models_to_run:
            # Survival Trees
            trees = RandomSurvivalForest(
                n_estimators=best_trees_params["n_estimators"],
                max_depth=best_trees_params["max_depth"],
                min_samples_leaf=best_trees_params["min_samples_leaf"],
                random_state=RANDOM_STATE,
            )
            trees.fit(train_x, train_y)

            c_index = trees.score(test_x, test_y)
            trees_c_indexes.append(c_index)
            best_trees_params_list.append(best_trees_params)

        # Gradient Boosting
        if "gb" in models_to_run:
            gb = GradientBoostingSurvivalAnalysis(
                n_estimators=best_gb_params["n_estimators"],
                learning_rate=best_gb_params["learning_rate"],
                max_depth=best_gb_params["max_depth"],
                random_state=RANDOM_STATE,
            )
            gb.fit(train_x, train_y)
            c_index = gb.score(test_x, test_y)
            gb_c_indexes.append(c_index)
            best_gb_params_list.append(best_gb_params)

    # Log results
    results_dict = {"dataset": dataset}
    if "cph" in models_to_run:
        results_dict.update(
            {
                "cph_mean_c_index": np.mean(cph_c_indexes),
                "cph_std_c_index": np.std(cph_c_indexes),
                "best_cph_params": best_cph_params_list,
            }
        )
    if "trees" in models_to_run:
        results_dict.update(
            {
                "trees_mean_c_index": np.mean(trees_c_indexes),
                "trees_std_c_index": np.std(trees_c_indexes),
                "best_trees_params": best_trees_params_list,
            }
        )
    if "gb" in models_to_run:
        results_dict.update(
            {
                "gb_mean_c_index": np.mean(gb_c_indexes),
                "gb_std_c_index": np.std(gb_c_indexes),
                "best_gb_params": best_gb_params_list,  # Assuming you also keep track of best parameters for gb
            }
        )

    results.append(results_dict)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, "basic_baselines.csv"))
print(results_df)

# Log the results_df as an artifact in wandb
artifact = wandb.Artifact("basic_baselines", type="results")
artifact.add_file(os.path.join(results_dir, "basic_baselines.csv"))
wandb.log_artifact(artifact)
# qsub jobs/experiments/submit_nogpu.sh python3 scripts/basic_baselines.py --dataset flchain.pt

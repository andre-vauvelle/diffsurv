import os.path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from pycox.datasets import from_deepsurv, from_kkbox, from_rdatasets, from_simulations
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import wandb
from definitions import DATA_DIR
from omni.common import create_folder


def preprocess_columns(
    dataset_to_process: pd.DataFrame,
    columns_to_scale: List,
    columns_to_one_hot: List,
    min_cat_prop: float = 0.01,
) -> pd.DataFrame:
    for c in columns_to_scale:
        dataset_to_process.loc[:, c] = StandardScaler().fit_transform(
            dataset_to_process.loc[:, c].values.reshape(-1, 1)
        )

    for c in columns_to_one_hot:
        enc = OneHotEncoder(sparse=False)
        one_hots = enc.fit_transform(dataset_to_process.loc[:, c].values.reshape(-1, 1))
        proportions = one_hots.sum(axis=0) / one_hots.shape[0]
        hots_to_keep = [i for i, p in enumerate(proportions) if p > min_cat_prop]
        one_hots = one_hots[:, hots_to_keep]
        kept_cols = np.array(enc.categories_[0])[hots_to_keep]
        one_hots_df = pd.DataFrame(one_hots, columns=[c + f"_{i}" for i in kept_cols])
        dataset_to_process.drop(columns=[c], inplace=True)
        dataset_to_process = pd.concat([one_hots_df, dataset_to_process], axis=1)

    return dataset_to_process


def preprocess_pycox(
    name: str, dataset: pd.DataFrame, save_artfifact: bool = True, min_cat_prop=0.01
) -> Dict[str, torch.Tensor]:
    """Take the pycox datasets and save them as artifacts in the format we use."""

    x_covar_columns = [c for c in dataset.columns if "x" == c[0]]
    risk = None
    setting = "realworld"
    if name == "support.pt":
        new_columns = [
            "age",
            "sex",
            "race",
            "number_of_comorbidities",
            "presence_of_diabetes",
            "presence_of_dementia",
            "presence_of_cancer",
            "mean_arterial_blood_pressure",
            "heart_rate",
            "respiration_rate",
            "temperature",
            "white_blood_cell_count",
            "serums_sodium",
            "serums_creatinine",
        ]
        col_map = dict(zip(x_covar_columns, new_columns))
        dataset.rename(columns=col_map, inplace=True)
        columns_to_scale = [
            "age",
            "number_of_comorbidities",
            "mean_arterial_blood_pressure",
            "heart_rate",
            "respiration_rate",
            "temperature",
            "white_blood_cell_count",
            "serums_sodium",
            "serums_creatinine",
        ]
        columns_to_one_hot = ["race", "presence_of_cancer"]

        dataset = preprocess_columns(dataset, columns_to_scale, columns_to_one_hot, min_cat_prop)

        x_covar = dataset.loc[:, list(set(dataset.columns) - {"duration", "event"})].to_numpy()
        y_times = dataset.duration.to_numpy()
        censored_events = 1 - dataset.event.to_numpy()
    elif name == "metabric.pt":
        columns_to_scale = [
            "x0",
            "x1",
            "x2",
            "x3",
            "x8",
        ]
        columns_to_one_hot = []

        dataset = preprocess_columns(dataset, columns_to_scale, columns_to_one_hot, min_cat_prop)

        x_covar = dataset.loc[:, list(set(dataset.columns) - {"duration", "event"})].to_numpy()
        y_times = dataset.duration.to_numpy()
        censored_events = 1 - dataset.event.to_numpy()
    elif name == "gbsg.pt":
        columns_to_scale = [
            "x3",
            "x5",
            "x4",
            "x6",
        ]
        columns_to_one_hot = ["x1"]

        dataset = preprocess_columns(dataset, columns_to_scale, columns_to_one_hot, min_cat_prop)

        x_covar = dataset.loc[:, list(set(dataset.columns) - {"duration", "event"})].to_numpy()
        y_times = dataset.duration.to_numpy()
        censored_events = 1 - dataset.event.to_numpy()
    elif name == "flchain.pt":
        x_covar_columns = [
            "age",
            "sex",
            "sample.yr",
            "kappa",
            "lambda",
            "flc.grp_1",
            "flc.grp_2",
            "flc.grp_3",
            "flc.grp_4",
            "flc.grp_5",
            "flc.grp_6",
            "flc.grp_7",
            "flc.grp_8",
            "flc.grp_9",
            "flc.grp_10",
            "creatinine",
            "mgus",
        ]
        columns_to_scale = ["age", "sample.yr", "kappa", "lambda", "creatinine"]
        columns_to_one_hot = ["flc.grp"]

        dataset = preprocess_columns(dataset, columns_to_scale, columns_to_one_hot, min_cat_prop)
        x_covar = dataset.loc[:, x_covar_columns].to_numpy()
        y_times = dataset.futime.to_numpy()
        censored_events = 1 - dataset.death.to_numpy()
    elif name == "nwtco.pt":
        # This had a few weird columns in the pycox...
        x_covar_columns = [
            "stage_1",
            "stage_2",
            "stage_3",
            "stage_4",
            "in.subcohort",
            "age",
            "instit_2",
            "histol_2",
            "study_4",
        ]
        columns_to_scale = [
            "age",
        ]
        columns_to_one_hot = ["stage"]

        dataset = preprocess_columns(dataset, columns_to_scale, columns_to_one_hot, min_cat_prop)
        x_covar = dataset.loc[:, x_covar_columns].to_numpy()
        y_times = dataset.edrel.to_numpy()
        censored_events = 1 - dataset.rel.to_numpy()
        # TODO: refactor censored events to just events..
    elif name in {"rr_nl_nhp.pt", "sac3.pt", "sac_admin5.pt"}:
        x_covar = dataset.loc[:, x_covar_columns].to_numpy()
        y_times = dataset.duration.to_numpy()
        censored_events = 1 - dataset.event.to_numpy()
        risk = dataset.duration_true * -1  # event times inversely proportional to risk
        setting = "synthetic"
    else:
        x_covar = dataset.loc[:, x_covar_columns].to_numpy()
        y_times = dataset.duration.to_numpy()
        censored_events = 1 - dataset.event.to_numpy()

    x_covar = torch.Tensor(x_covar).float()
    y_times = torch.Tensor(y_times).float().unsqueeze(-1)
    censored_events = torch.Tensor(censored_events).long().unsqueeze(-1)

    if risk is not None:
        risk = torch.Tensor(risk).float().unsqueeze(-1)
        data = {
            "x_covar": x_covar,
            "y_times": y_times,
            "censored_events": censored_events,
            "risk": risk,
        }
    else:
        data = {"x_covar": x_covar, "y_times": y_times, "censored_events": censored_events}
    save_path = os.path.join(DATA_DIR, "realworld")
    create_folder(save_path)

    torch.save(data, os.path.join(save_path, name))
    print(f"Saved pycox realworld dataset to: {os.path.join(save_path, name)}")
    if save_artfifact:
        N = int(censored_events.shape[0])
        censored_proportion = float(censored_events.sum() / N)
        n_covariates = int(x_covar.shape[1])
        metadata = {
            "name": name,
            "N": N,
            "censored_proportion": censored_proportion,
            "n_covariates": n_covariates,
            "setting": setting,
            "input_dim": n_covariates,
            "output_dim": 1,
        }
        run = wandb.init(
            job_type="preprocess_pycox",
            project="diffsurv",
            entity="anon",
        )
        artifact = wandb.Artifact(name, type="dataset", metadata=metadata)
        artifact.add_file(os.path.join(save_path, name), name)
        run.log_artifact(artifact)
    return data


def preprocess_kkbox(
    save_artifact=True, save_path: str = os.path.join(DATA_DIR, "realworld", "kkbox_v1")
):
    train = from_kkbox._DatasetKKBoxChurn().read_df(subset="train")
    val = from_kkbox._DatasetKKBoxChurn().read_df(subset="val")
    test = from_kkbox._DatasetKKBoxChurn().read_df(subset="test")

    datasets = {"train": train, "val": val, "test": test}

    x_covar_columns = [
        "n_prev_churns",
        "log_days_between_subs",
        "log_days_since_reg_init",
        "log_payment_plan_days",
        "log_plan_list_price",
        "log_actual_amount_paid",
        "is_auto_renew",
        "is_cancel",
        "city_1.0",
        "city_3.0",
        "city_4.0",
        "city_5.0",
        "city_6.0",
        "city_7.0",
        "city_8.0",
        "city_9.0",
        "city_10.0",
        "city_11.0",
        "city_12.0",
        "city_13.0",
        "city_14.0",
        "city_15.0",
        "city_16.0",
        "city_17.0",
        "city_18.0",
        "city_21.0",
        "city_22.0",
        "city_nan",
        "gender_female",
        "gender_male",
        "gender_nan",
        "registered_via_3.0",
        "registered_via_4.0",
        "registered_via_7.0",
        "registered_via_9.0",
        "registered_via_13.0",
        "registered_via_nan",
        "age_at_start",
        "strange_age",
        "nan_days_since_reg_init",
        "no_prev_churns",
    ]
    columns_to_scale = [
        "n_prev_churns",
        "log_days_between_subs",
        "log_days_since_reg_init",
        "log_payment_plan_days",
        "log_plan_list_price",
        "log_actual_amount_paid",
        "age_at_start",
    ]
    columns_to_one_hot = [
        "city",
        "gender",
        "registered_via",
    ]

    for stage, dataset in datasets.items():
        dataset = preprocess_columns(dataset, columns_to_scale, columns_to_one_hot, 0.001)
        x_covar = dataset.loc[:, x_covar_columns].to_numpy()
        y_times = dataset.duration.to_numpy()
        censored_events = 1 - dataset.event.to_numpy()

        x_covar = torch.Tensor(x_covar).float()
        y_times = torch.Tensor(y_times).float().unsqueeze(-1)
        censored_events = torch.Tensor(censored_events).long().unsqueeze(-1)

        data = {
            "x_covar": x_covar,
            "y_times": y_times,
            "censored_events": censored_events,
        }
        name = f"kkbox_v1_{stage}.pt"
        create_folder(save_path)

        torch.save(data, os.path.join(save_path, name))
        print(f"Saved pycox realworld dataset to: {save_path}")

    if save_artifact:
        N = sum(df.shape[0] for df in datasets.values())
        censored_proportion = 1 - float(sum(df.event.sum() for df in datasets.values()) / N)
        n_covariates = int(x_covar.shape[1])
        metadata = {
            "name": "kkbox_v1",
            "N": N,
            "censored_proportion": censored_proportion,
            "input_dim": n_covariates,
            "output_dim": 1,
            "setting": "realworld",
        }
        run = wandb.init(
            job_type="preprocess_pycox",
            project="diffsurv",
            entity="anon",
        )
        artifact = wandb.Artifact("kkbox_v1", type="dataset", metadata=metadata)
        artifact.add_dir(os.path.join(save_path))

        run.log_artifact(artifact)


if __name__ == "__main__":
    support = from_deepsurv._Support().read_df().sample(frac=1)
    # metabric = from_deepsurv._Metabric().read_df().sample(frac=1)
    # gbsg = from_deepsurv._Gbsg().read_df().sample(frac=1)
    # flchain = from_rdatasets._Flchain().read_df().sample(frac=1)
    # nwtco = from_rdatasets._Nwtco().read_df().sample(frac=1)
    # sac3 = from_simulations._SAC3().read_df().sample(frac=1)
    # rr_nl_nhp = from_simulations._RRNLNPH().read_df().sample(frac=1)
    # sac_admin5 = from_simulations._SACAdmin5().read_df().sample(frac=1)
    #
    datasets = {
        "support.pt": support,
        # "metabric.pt": metabric,
        # "gbsg.pt": gbsg,
        # "flchain.pt": flchain,
        # "nwtco.pt": nwtco,
        # 'kkbox': kkbox,
        # "sac3.pt": sac3,
        # "rr_nl_nhp.pt": rr_nl_nhp,
        # "sac_admin5.pt": sac_admin5,
    }

    for n, d in datasets.items():
        preprocess_pycox(n, d, save_artfifact=True)

    preprocess_kkbox(save_artifact=True)

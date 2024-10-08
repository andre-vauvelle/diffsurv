import os
import random
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision.datasets.utils import download_url

from definitions import DATA_DIR


def download_data(path: str = f"{DATA_DIR}mimic/"):
    if not os.path.exists(path):
        string = f"""
            first download MIMIC IV and MIMIC-CXR with:
            wget -r -N -c -np --user <user> --reject html --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/ -P {path}
            wget -r -N -c -np --user <user> --reject html --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz -P {path}
            wget -r -N -c -np --user <user> --reject html --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz -P {path}
            wget -r -N -c -np --user <user> --ask-password https://physionet.org/files/mimiciv/2.0/hosp/patients.csv.gz -P {path}
            wget -r -N -c -np --user <user> --ask -password https://physionet.org/files/mimiciv/2.0/hosp/admissions.csv.gz -P {path}
            """


def preprocess_data(path: str = f"{DATA_DIR}mimic/"):
    """
    Once we've downloaded data we can get the survival times from metadata, admissions and patients tables
    :param path:
    :return:
    """
    train_split = 0.8
    val_split = 0.1

    cxr_split_path = "physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz"
    admission_path = "physionet.org/files/mimiciv/2.0/hosp/admissions.csv.gz"
    patients_path = "physionet.org/files/mimiciv/2.0/hosp/patients.csv.gz"

    metadata_path = "physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz"
    splits = pd.read_csv(os.path.join(path, cxr_split_path))
    cxr_metadata = pd.read_csv(os.path.join(path, metadata_path))
    admissions = pd.read_csv(os.path.join(path, admission_path))
    patients = pd.read_csv(os.path.join(path, patients_path))

    print(f"Total starting images from CXR splits: {splits.shape[0]}")
    splits = splits.merge(patients.loc[:, ["subject_id", "dod"]], on="subject_id", how="left")
    print(f"Total after merging with patients from MIMICIV (for DOD): {splits.shape[0]}")
    splits = splits.merge(
        cxr_metadata.loc[:, ["subject_id", "study_id", "StudyDate", "dicom_id"]],
        on=["subject_id", "study_id", "dicom_id"],
        how="left",
    )
    print(f"Total after merging with patients from CXR Metadata for study date: {splits.shape[0]}")
    last_discharge = admissions.groupby("subject_id").dischtime.max().reset_index()
    splits = splits.merge(
        last_discharge.loc[:, ["subject_id", "dischtime"]], on="subject_id", how="left"
    )
    print(
        "Total after merging with patients from admission for dischtime (censoring time):"
        f" {splits.shape[0]}"
    )

    splits.StudyDate = pd.to_datetime(splits.StudyDate.astype(str))
    splits.dod = pd.to_datetime(splits.dod)
    splits.loc[:, "event"] = splits.dod.notnull().astype(int)
    # Set time for those that died
    splits.loc[splits.dod.notnull(), "tte"] = (
        splits.loc[splits.dod.notnull(), "dod"] - splits.loc[splits.dod.notnull(), "StudyDate"]
    )

    missing_mask = splits.tte.isnull() & splits.dischtime.isnull()
    splits = splits.loc[~missing_mask, :].copy()
    warnings.warn(f"Warning removed {sum(missing_mask)} images that do not have a dod or dischtime")

    # Set time for those that are censored
    splits.dischtime = pd.to_datetime(splits.dischtime.astype(str))
    splits.loc[splits.dod.isnull(), "tte"] = (
        splits.loc[splits.dod.isnull(), "dischtime"]
        + np.timedelta64(365, "D")
        - splits.loc[splits.dod.isnull(), "StudyDate"]
    )

    negative_tte_cen = (splits.tte.dt.days < 365) & (splits.event == 0)
    warnings.warn(f"Warning removed {sum(negative_tte_cen)} images taken after discharge wo death")
    splits = splits[~negative_tte_cen].copy()

    negative_tte_death = (splits.tte.dt.days < 0) & (splits.event == 1)
    warnings.warn(f"Warning removed {sum(negative_tte_death)} images taken after death")
    splits = splits[~negative_tte_death].copy()

    splits.tte = splits.tte.dt.days

    # fig, ax = plt.subplots(figsize=(8, 6))
    # splits.groupby('event').tte.hist(bins=100, ax=ax, alpha=0.5)
    # ax.set_xlabel('Number days to event')
    # ax.set_ylabel("Number of patients")
    # plt.show()

    p_folders = "p" + splits.subject_id.astype(str).str[:2]
    p_subfolders = "p" + splits.subject_id.astype(str)
    s_folders = "s" + splits.study_id.astype(str)
    image_name = splits.dicom_id + ".jpg"

    splits.loc[:, "path"] = (
        "physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        + p_folders
        + "/"
        + p_subfolders
        + "/"
        + s_folders
        + "/"
        + image_name
    )

    import multiprocessing as mp

    with mp.Pool(mp.cpu_count() - 1) as pool:
        splits.loc[:, "exists"] = pool.map(os.path.exists, DATA_DIR + "mimic/" + splits.path)

    if splits.exists.sum() != splits.shape[0]:
        warnings.warn(f"Warning only {splits.exists.sum()} images found")

    # reassign splits
    splits = splits.sample(frac=1).reset_index(drop=True)
    patients = list(splits.subject_id.unique())
    n_patients = len(patients)
    random.shuffle(patients)

    n_train, n_val = int(n_patients * train_split), int(n_patients * val_split)
    train_patients = patients[:n_train]
    val_patients = patients[n_train : n_train + n_val]
    test_patients = patients[n_train + n_val :]

    splits.loc[splits.subject_id.isin(train_patients), "split"] = "train"
    splits.loc[splits.subject_id.isin(val_patients), "split"] = "val"
    splits.loc[splits.subject_id.isin(test_patients), "split"] = "test"

    splits.loc[:, ["subject_id", "study_id", "path", "exists", "split", "tte", "event"]].to_csv(
        os.path.join(DATA_DIR, "mimic", "splits.csv"), index=False
    )

    # TODO: add wandb


if __name__ == "__main__":
    preprocess_data()

import collections
import itertools
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.datasets.utils import download_url, verify_str_arg

import wandb
from definitions import DATA_DIR
from omni.common import create_folder


def time_function(numbers: torch.Tensor, beta):
    """
    Generates survival times from numbers with a sampling using beta distrubiont Beta(beta, beta) and exponential
    parametersization
    :param numbers:
    :param beta:
    :return:
    """
    # standardize
    log_numbers = np.log(numbers + 1)
    # pd.DataFrame(np.log(numbers+1)).hist(bins=100)
    BX = (log_numbers - log_numbers.float().mean()) / torch.std(log_numbers.float())
    # pd.DataFrame(np.exp(BX)).hist(bins=100)

    num_samples = BX.shape[0]
    lambda_exp_BX = (1 / 1) * np.exp(BX / 1)  # scale to mean 30 days
    lambda_exp_BX = lambda_exp_BX.flatten()

    # Generating beta samples
    U = np.random.beta(beta, beta, num_samples)

    # Exponential
    T = -np.log(U) / (lambda_exp_BX)
    return T, BX


def plot_time(numbers=None, lambda_exp_BX=None, log_numbers=None, T=None, calc_oracle=None):
    numbers = numbers[:10_000]

    T500, BX = time_function(numbers, beta=500)
    T1, BX = time_function(numbers, beta=1)
    lambda_exp_BX = (1 / 1) * np.exp(BX / 1)  # scale to mean 30 days
    lambda_exp_BX = lambda_exp_BX.flatten()
    T_oracle = -np.log(0.5) / (lambda_exp_BX)

    fig, axs = plt.subplots(1, 3, figsize=(9.25, 5), sharey=True, sharex=True, dpi=200)
    axs[0].scatter(numbers, T1, alpha=0.1, s=1, label="beta=1", c="b")
    axs[1].scatter(numbers, T500, alpha=0.1, s=1, label="beta=500", c="darkorange")
    axs[2].scatter(numbers, T_oracle, alpha=0.1, s=1, label="Oracle", c="r")
    axs[0].set_xlim([0.8, 100000])
    axs[0].set_ylim([0, 7.5])
    axs[0].title.set_text("Beta=1 (Uniform)")
    axs[1].title.set_text("Beta=500")
    axs[2].title.set_text("Oracle")
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[2].set_xscale("log")
    axs[0].set_ylabel("Simulated Survival Time")
    axs[1].set_xlabel("Door Number")
    # axs[2].set_yscale('log')
    plt.show()

    idx = torch.argsort(numbers)
    fig, axs = plt.subplots(1, 3, figsize=(9.25, 5), sharey=True, sharex=True, dpi=200)
    axs[0].scatter(numbers[idx], T1[idx].argsort() / max(T1[idx].argsort()), alpha=0.1, s=1, c="b")
    axs[1].scatter(
        numbers[idx], T500[idx].argsort() / max(T500[idx].argsort()), alpha=0.1, s=1, c="darkorange"
    )
    axs[2].scatter(
        numbers[idx],
        T_oracle[idx].argsort() / max(T_oracle[idx].argsort()),
        alpha=0.1,
        s=0.5,
        c="r",
    )
    # axs[0].set_xlim([0.8, 100000])
    # axs[0].set_ylim([0, 7.5])
    axs[0].title.set_text("Beta=1 (Uniform)")
    axs[1].title.set_text("Beta=500")
    axs[2].title.set_text("Oracle")
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[2].set_xscale("log")
    axs[0].set_ylabel("Simulated Survival Rank")
    axs[1].set_xlabel("Door Number")
    # axs[2].set_yscale('log')

    plt.scatter(numbers, lambda_exp_BX, s=1, c="r", alpha=0.1)
    plt.scatter(log_numbers, lambda_exp_BX, s=1, c="r", alpha=0.1)
    plt.show()
    print(T.mean())

    idx = torch.argsort(numbers)
    T = T[idx]
    log_numbers = log_numbers[idx]
    numbers = numbers[idx]

    plt.scatter(log_numbers, T, alpha=0.1, s=1)
    df = pd.DataFrame.from_dict({"log_numbers": log_numbers, "T": T})
    medians = df.groupby("log_numbers")["T"].median()
    plt.scatter(log_numbers, T.argsort() / max(T.argsort()), alpha=0.1, s=1)
    plt.scatter(medians.index, medians, s=1)
    plt.scatter(medians.index, medians.argsort() / max(medians.argsort()), s=1)

    plt.scatter(numbers, T.argsort(), alpha=0.1, s=1)

    # Oracle performance
    if calc_oracle:
        k = 5
        perm = torch.randperm(T.argsort().size(0))
        idx = perm[:k]

        plt.scatter(log_numbers[idx], T.argsort()[idx], alpha=1, s=5)

        T_oracle = -np.log(0.5) / (lambda_exp_BX)
        plt.scatter(log_numbers, T_oracle[idx], alpha=0.1, s=1)
        plt.scatter(
            log_numbers, T_oracle[idx].argsort() / max(T_oracle[idx].argsort()), alpha=0.1, s=1
        )
        plt.show()

        results_store = collections.defaultdict(dict)
        samples = 10_000
        k = 5
        for k in [2, 4, 5, 8, 16, 32, 64]:
            EM_total = 0
            EW_total = 0
            for _ in range(samples):
                perm = torch.randperm(T.argsort().size(0))
                idx = perm[:k]
                rank_sample = T.argsort()[idx]
                rank_oracle = T_oracle.argsort()[idx]
                matches = rank_sample.argsort() == rank_oracle.argsort()
                if matches.all():
                    EM_total += 1
                EW_total += matches.sum()

            EM = EM_total / samples
            EW = EW_total / (k * samples)
            print(f"EM{k}: {EM}, EW{k}: {EW}")
            results_store[k] = {"EM": EM, "EW": EW}


def gen_survSVHN(beta=1, censored_proportion=0.6):
    """Generate synthetic survival times based on street view house numbers
    :param beta: beta distribution sampling parameter. if 1 then uniform sampling, if inf then samples the median
    :param censored_proportion: propotion of patients to be independently randomly censored Unif[0, true_event_time]
    """
    save_path = os.path.join(DATA_DIR, "synthetic", "SVNH")

    split_list = {
        "train": [
            "https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_train.p",
            "svhn-multi-digit-3x64x64_train.p",
            "25df8732e1f16fef945c3d9a47c99c1a",
        ],
        "val": [
            "https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_val.p",
            "svhn-multi-digit-3x64x64_val.p",
            "fe5a3b450ce09481b68d7505d00715b3",
        ],
        "test": [
            "https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_test.p",
            "svhn-multi-digit-3x64x64_test.p",
            "332977317a21e9f1f5afe7ef47729c5c",
        ],
    }

    if not os.path.exists(os.path.join(DATA_DIR, "data-svhn")):
        for stage in ["train", "val", "test"]:
            split = verify_str_arg(stage, "split", tuple(split_list.keys()))
            url = split_list[split][0]
            filename = split_list[split][1]
            file_md5 = split_list[split][2]
            md5 = split_list[split][2]
            download_url(url, os.path.join(DATA_DIR, "data-svhn"), filename, md5)

    data_train = torch.load(os.path.join(DATA_DIR, "data-svhn", "svhn-multi-digit-3x64x64_train.p"))
    data_val = torch.load(os.path.join(DATA_DIR, "data-svhn", "svhn-multi-digit-3x64x64_val.p"))
    data_test = torch.load(os.path.join(DATA_DIR, "data-svhn", "svhn-multi-digit-3x64x64_test.p"))

    n_train = len(data_train[1])
    n_val = len(data_val[1])
    n_test = len(data_test[1])
    datasets = {"train": data_train, "val": data_val, "test": data_test}

    numbers = torch.concat([data_train[1], data_val[1], data_test[1]])
    n = numbers.shape[0]

    # max_numbers = 10000
    # numbers[numbers > max_numbers] = max_numbers

    survival_times, BX = time_function(numbers, beta=500)
    censoring_times = np.random.uniform(0, survival_times, size=n)
    # Select proportion of the patients to be right-censored using censoring_times
    # Independent of covariates
    censoring_indices = np.random.choice(n, size=int(n * censored_proportion), replace=False)

    y_times = survival_times.float()
    y_times[censoring_indices] = torch.Tensor(censoring_times).float()[censoring_indices]

    # plt.scatter(numbers, y_times, s=1, alpha=0.1)
    # plt.scatter(numbers, survival_times, s=1, alpha=0.1)
    # ax = plt.gca()
    # ax.set_xscale('log')
    # plt.show()

    censored_events = np.zeros(n, dtype=bool)
    censored_events[censoring_indices] = True

    for s in ["train", "val", "test"]:
        idx = torch.zeros(n)
        if s == "train":
            idx[:n_train] = 1
            images = data_train[0]
        elif s == "val":
            idx[n_train : n_train + n_val] = 1
            images = data_val[0]
        else:
            idx[n_train + n_val :] = 1
            images = data_test[0]

        data = {
            "x_covar": images,
            "y_times": y_times[idx == 1].float().unsqueeze(-1),
            "censored_events": torch.Tensor(censored_events[idx == 1]).long().unsqueeze(-1),
            "risk": BX[idx == 1].float().unsqueeze(-1),
            "numbers": numbers[idx == 1].float().unsqueeze(-1),
            "y_times_uncensored": survival_times[idx == 1].unsqueeze(-1),
        }
        name = f"{s}.pt"
        create_folder(save_path)

        torch.save(data, os.path.join(save_path, name))
        print(f"Saved SVNH dataset to: {os.path.join(save_path, name)}")

    config = {
        "name": name,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "risk_type": "beta_exponential",
        "censored_proportion": censored_proportion,
        "input_dim": (3, 64, 64),
        "output_dim": 1,
        "beta": beta,
        "setting": "synthetic",
    }

    run = wandb.init(job_type="preprocess_survSVNH", project="diffsurv", entity="anon")
    artifact = wandb.Artifact(
        f"SVNH_beta{str(beta)}_cen{str(censored_proportion)}", type="dataset", metadata=config
    )
    artifact.add_dir(save_path)
    run.log_artifact(artifact)


if __name__ == "__main__":
    betas = [1, 500]
    censored_proportions = [0.3, 0.6]
    for beta, censored_proportion in itertools.product(betas, censored_proportions):
        gen_survSVHN(beta=beta, censored_proportion=censored_proportion)

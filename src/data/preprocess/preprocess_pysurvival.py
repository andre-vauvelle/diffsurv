import os
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import torch
from pysurvival.models.simulations import SimulationModel

import wandb
from definitions import DATA_DIR
from omni.common import create_folder


def gen_pysurvival(
    name,
    N,
    survival_distribution="weibull",
    risk_type="gaussian",
    censored_proportion=0.0,
    alpha: float = 1,
    beta: float = 1,
    feature_weights=[1.0] * 3,
    censoring_function="independent",
    save_artifact=True,
    tie_groups: Optional[int] = None,
    sampling: str = "uniform",
    beta_p: float = 20.0,
) -> Dict[str, torch.Tensor]:
    """
    Generate simulated dataset using gaussin covariates, a hazard function and time function.
    censored_events {0 - event happened, 1 - patients was censored}
    :param name:
    :param N:
    :param survival_distribution:
    :param risk_type:
    :param censored_proportion:
    :param alpha:
    :param beta:
    :param feature_weights:
    :param censoring_function:
    :param save_artifact:
    :param tie_groups: group times into n groups, these are ties due to inexact measurement
    :param tie_repeats: randomly repeat instance n tiems, these are exact ties
    :param sampling: time function sampling method uniform or beta
    :param beta_p: beta parameter for beta time function sampling
    :return:
    """

    # Generating the dataset from a nonlinear Weibull parametric model
    # Initializing the simulation model
    sim = CustomSimulationModel(
        survival_distribution=survival_distribution,
        risk_type=risk_type,
        censored_parameter=10000000,
        alpha=alpha,
        beta=beta,
    )

    # Generating N random samples
    dataset = sim.generate_data(
        num_samples=N,
        num_features=len(feature_weights),
        feature_weights=feature_weights,
        sampling=sampling,
        beta_p=beta_p,
    )
    x_covar = dataset.iloc[:, : len(feature_weights)].to_numpy()
    y_times = dataset.time.to_numpy()
    y_times_uncensored = dataset.time.to_numpy()
    risk = dataset.risk.to_numpy()

    censoring_times = np.random.uniform(0, y_times, size=N)

    # Select proportion of the patients to be right-censored using censoring_times
    if censoring_function == "independent":
        # Independent of covariates
        censoring_indices = np.random.choice(N, size=int(N * censored_proportion), replace=False)
    elif censoring_function == "mean":
        # Censored if mean of covariates over percentile determined by censoring proportion
        mean_covs = dataset.iloc[:, : len(feature_weights)].mean(1)
        percentile_cut = np.percentile(mean_covs, int(100 * censored_proportion))
        censoring_indices = np.array(mean_covs < percentile_cut)
    else:
        raise NotImplementedError(
            f"censoring_function {censoring_function} but must be either 'independent' or 'mean'"
        )
    y_times[censoring_indices] = censoring_times[censoring_indices]
    censored_events = np.zeros(N, dtype=bool)
    censored_events[censoring_indices] = True

    if tie_groups:
        # intervals = np.linspace(min(y_times), max(y_times), tie_groups)
        intervals = (
            np.sort(y_times).reshape(tie_groups, -1).mean(-1)
        )  # equal number of samples per bin...

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        new_y = []

        for y in y_times:
            new_y.append(find_nearest(intervals, y))
        y_times = np.array(new_y)

    x_covar = torch.Tensor(x_covar).float()
    y_times = torch.Tensor(y_times).float().unsqueeze(-1)
    censored_events = torch.Tensor(censored_events).long().unsqueeze(-1)
    print(f"Proportion censored: {censored_events.sum() / N}")

    # create directory for save
    save_path = os.path.join(DATA_DIR, "synthetic")
    create_folder(save_path)
    data = {
        "x_covar": x_covar,
        "y_times": y_times,
        "censored_events": censored_events,
        "risk": risk,
        "y_times_uncensored": y_times_uncensored,
    }
    torch.save(data, os.path.join(save_path, name))
    print(f"Saved risk synthetic dataset to: {os.path.join(save_path, name)}")
    if save_artifact:
        config = {
            "name": name,
            "N": N,
            "survival_distribution": survival_distribution,
            "risk_type": risk_type,
            "censored_proportion": censored_proportion,
            "input_dim": len(feature_weights),
            "output_dim": 1,
            "alpha": alpha,
            "beta": beta,
            "feature_weights": feature_weights,
            "censoring_function": censoring_function,
            "n_covariates": len(feature_weights),
            "setting": "synthetic",
            "tie_groups": tie_groups,
        }

        run = wandb.init(job_type="preprocess_synthetic", project="diffsurv", entity="anon")
        artifact = wandb.Artifact(name, type="dataset", metadata=config)
        artifact.add_file(os.path.join(save_path, name), name)
        run.log_artifact(artifact)
    return data


class CustomSimulationModel(SimulationModel):
    """Just inheriting to get access to pre time function risk"""

    @staticmethod
    def random_data(N):
        """
        Generating a array of size N from a random distribution -- the available
        distributions are:
            * binomial,
            * chisquare,
            * exponential,
            * gamma,
            * normal,
            * uniform
            * laplace
        """

        index = np.random.binomial(n=4, p=0.5)
        distributions = {
            # 'binomial_a': np.random.binomial(n=20, p=0.6, size=N),
            # 'binomial_b': np.random.binomial(n=200, p=0.6, size=N),
            # 'chisquare': np.random.chisquare(df=10, size=N),
            # 'exponential_a': np.random.exponential(scale=0.1, size=N),
            # 'exponential_b': np.random.exponential(scale=0.01, size=N),
            # 'gamma': np.random.gamma(shape=2., scale=2., size=N),
            "normal_a": np.random.normal(loc=0, scale=1.0, size=N),
            # 'normal_b': np.random.normal(loc=10.0, scale=10.0, size=N),
            # 'uniform_a': np.random.uniform(low=-1.0, high=1.0, size=N),
            # 'uniform_b': np.random.uniform(low=-20.0, high=100.0, size=N),
            # 'laplace': np.random.laplace(loc=0.0, scale=1.0, size=N)
        }

        # list_distributions = copy.deepcopy(list(distributions.keys()))
        # random.shuffle(list_distributions)
        # key = list_distributions[index]
        return "normal_a", distributions["normal_a"]
        # return "uniform_a", distributions["uniform_a"]

    def time_function(self, BX, sampling: str = "uniform", beta_p: float = 20.0):
        """
        Calculating the survival times based on the given distribution
        T = H^(-1)( -log(U)/risk_score ), where:
            * H is the cumulative baseline hazard function
                (H^(-1) is the inverse function)
            * U is a random variable uniform - Uni[0,1].

        The method is inspired by https://gist.github.com/jcrudy/10481743
        """

        # Calculating scale coefficient using the features
        num_samples = BX.shape[0]
        lambda_exp_BX = np.exp(BX) * self.alpha
        lambda_exp_BX = lambda_exp_BX.flatten()

        # Generating random uniform variables
        if sampling == "uniform":
            U = np.random.uniform(0, 1, num_samples)
        elif sampling == "beta":
            U = np.random.beta(beta_p, beta_p, num_samples)
        else:
            raise ValueError(f"sampling but be either uniform or beta. Is currently {sampling}")

        # Exponential
        if self.survival_distribution.lower().startswith("exp"):
            self.survival_distribution = "Exponential"
            return -np.log(U) / (lambda_exp_BX)

        # Weibull
        elif self.survival_distribution.lower().startswith("wei"):
            self.survival_distribution = "Weibull"
            return np.power(-np.log(U) / (lambda_exp_BX), 1.0 / self.beta)

        # Gompertz
        elif self.survival_distribution.lower().startswith("gom"):
            self.survival_distribution = "Gompertz"
            return (1.0 / self.beta) * np.log(1 - self.beta * np.log(U) / (lambda_exp_BX))

        # Log-Logistic
        elif "logistic" in self.survival_distribution.lower():
            self.survival_distribution = "Log-Logistic"
            return np.power(U / (1.0 - U), 1.0 / self.beta) / (lambda_exp_BX)

        # Log-Normal
        elif "normal" in self.survival_distribution.lower():
            self.survival_distribution = "Log-Normal"
            W = np.random.normal(0, 1, num_samples)
            return lambda_exp_BX * np.exp(self.beta * W)

    def generate_data(
        self,
        num_samples=100,
        num_features=3,
        feature_weights=None,
        sampling="uniform",
        beta_p: float = 20.0,
    ):
        """
        Generating a dataset of simulated survival times from a given
        distribution through the hazard function using the Cox model

        Parameters:
        -----------
        * `num_samples`: **int** *(default=100)* --
            Number of samples to generate

        * `num_features`: **int** *(default=3)* --
            Number of features to generate

        * `feature_weights`: **array-like** *(default=None)* --
            list of the coefficients of the underlying Cox-Model.
            The features linked to each coefficient are generated
            from random distribution from the following list:

            * binomial
            * chisquare
            * exponential
            * gamma
            * normal
            * uniform
            * laplace

            If None then feature_weights = [1.]*num_features

        Returns:
        --------
        * dataset: pandas.DataFrame
            dataset of simulated survival times, event status and features


        Example:
        --------
        from pysurvival.models.simulations import SimulationModel

        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'gompertz',
                               risk_type = 'linear',
                               censored_parameter = 5.0,
                               alpha = 0.01,
                               beta = 5., )

        # Generating N Random samples
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features=5)

        # Showing a few data-points
        dataset.head()
        """

        # Data parameters
        self.num_variables = num_features
        if feature_weights is None:
            self.feature_weights = [1.0] * self.num_variables
            feature_weights = self.feature_weights

        else:
            self.feature_weights = feature_weights

        # Generating random features
        # Creating the features
        X = np.zeros((num_samples, self.num_variables))
        columns = []
        for i in range(self.num_variables):
            key, X[:, i] = self.random_data(num_samples)
            columns.append("x_" + str(i + 1))
        X_std = self.scaler.fit_transform(X)
        BX = self.risk_function(X_std)

        # Building the survival times
        T = self.time_function(BX, sampling=sampling, beta_p=beta_p)
        C = np.random.normal(loc=self.censored_parameter, scale=5, size=num_samples)
        C = np.maximum(C, 0.0)
        time = np.minimum(T, C)
        E = 1.0 * (T == time)

        # Building dataset
        self.features = columns
        self.dataset = pd.DataFrame(
            data=np.c_[X, time, E, BX],  # Minor mod here
            columns=columns + ["time", "event", "risk"],
        )  # Minor mod here

        # Building the time axis and time buckets
        self.times = np.linspace(0.0, max(self.dataset["time"]), self.bins)
        self.get_time_buckets()

        # Building baseline functions
        self.baseline_hazard = self.hazard_function(self.times, 0)
        self.baseline_survival = self.survival_function(self.times, 0)

        # Printing summary message
        message_to_print = "Number of data-points: {} - Number of events: {}"
        print(message_to_print.format(num_samples, sum(E)))

        return self.dataset


if __name__ == "__main__":
    # gen_pysurvival('pysurv_linear_0.0.pt', 32000, survival_distribution='weibull', risk_type='linear',
    #                censored_proportion=0,
    #                alpha=0.1, beta=3.2, feature_weights=[1.] * 3)
    #
    # gen_pysurvival('exp_linear_0.0.pt', 32000, survival_distribution='exponential', risk_type='linear',
    #                censored_proportion=0,
    #                alpha=0.1, beta=3.2, feature_weights=[1.] * 3)
    #
    # gen_pysurvival('pysurv_square_0.0.pt', 32000, survival_distribution='weibull', risk_type='square',
    #                censored_proportion=0,
    #                alpha=0.1, beta=3.2, feature_weights=[1.] * 3)
    #
    # gen_pysurvival('pysurv_square10_0.0.pt', 32000, survival_distribution='weibull', risk_type='square',
    #                censored_proportion=0,
    #                alpha=0.1, beta=3.2, feature_weights=[10.] * 3)
    #
    # gen_pysurvival('pysurv_gaussian_0.0.pt', 32000, survival_distribution='weibull', risk_type='gaussian',
    #                censored_proportion=0,
    #                alpha=0.1, beta=3.2, feature_weights=[1.] * 3)
    #
    # for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    #     gen_pysurvival(f'pysurv_gaussian_{str(c)}.pt', 32000, survival_distribution='weibull', risk_type='gaussian',
    #                    censored_proportion=c,
    #                    alpha=0.1, beta=3.2, feature_weights=[1.] * 3)

    # gen_pysurvival(
    #     "pysurv_square_0.3.pt",
    #     32000,
    #     survival_distribution="weibull",
    #     risk_type="square",
    #     censored_proportion=0.3,
    #     alpha=0.1,
    #     beta=3.2,
    #     feature_weights=[1.0] * 3,
    #     censoring_function="mean",
    # )

    # gen_pysurvival('pysurv_square_mean_0.3.pt', 32000, survival_distribution='weibull', risk_type='square',
    #                censored_proportion=0.3,
    #                alpha=0.1, beta=3.2, feature_weights=[1.] * 3, censoring_function='mean')

    # for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    #     gen_pysurvival(
    #         f"pysurv_square_independent_{str(c)}.pt",
    #         32000,
    #         survival_distribution="weibull",
    #         risk_type="square",
    #         censored_proportion=c,
    #         alpha=0.1,
    #         beta=3.2,
    #         feature_weights=[1.0] * 3,
    #         censoring_function="independent",
    #     )
    #
    # for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    #     gen_pysurvival(
    #         f"pysurv_square_mean_{str(c)}.pt",
    #         32000,
    #         survival_distribution="weibull",
    #         risk_type="square",
    #         censored_proportion=c,
    #         alpha=0.1,
    #         beta=3.2,
    #         feature_weights=[1.0] * 3,
    #         censoring_function="mean",
    #     )
    #
    # for N in (8000, 16000, 32000, 64000, 128000):
    #     for d in (3, 100, 1000, 10000):
    #         gen_pysurvival(
    #             f"pysurv_square_independent_c0.3_N{N}_d{d}.pt",
    #             N,
    #             survival_distribution="weibull",
    #             risk_type="square",
    #             censored_proportion=0.3,
    #             alpha=0.1,
    #             beta=3.2,
    #             feature_weights=[0.0001] * d,
    #             censoring_function="independent",
    #         )

    # for trail in range(5):
    #     for ties in (4, 5, 10, 50, 100, 200, 500, 1000, 2500, 5000, 10000):
    #         gen_pysurvival(
    #             f"pysurv27_linear_exp_independent_ties{str(ties)}_{trail}.pt",
    #             10_000,
    #             survival_distribution="exponential",
    #             risk_type="linear",
    #             censored_proportion=0.3,
    #             alpha=1,
    #             beta=1,
    #             feature_weights=[0.2, 0.7],
    #             censoring_function="independent",
    #             tie_groups=ties,
    #             save_artifact=True,
    #         )
    # gen_pysurvival(
    #     f"pysurv27_linear_exp_nocensoring_ties{str(ties)}_{trail}.pt",
    #     10_000,
    #     survival_distribution="exponential",
    #     risk_type="linear",
    #     censored_proportion=0.0,
    #     alpha=1,
    #     beta=1,
    #     feature_weights=[0.2, 0.7],
    #     censoring_function="independent",
    #     tie_groups=ties,
    #     save_artifact=True,
    # )
    # gen_pysurvival(
    #     f"pysurv27_linear_exp_independent_ties{0}_{trail}.pt",
    #     10_000,
    #     survival_distribution="exponential",
    #     risk_type="linear",
    #     censored_proportion=0.3,
    #     alpha=1,
    #     beta=1,
    #     feature_weights=[0.2, 0.7],
    #     censoring_function="independent",
    #     tie_groups=None,
    #     save_artifact=True,
    # )
    # gen_pysurvival(
    #     f"pysurv27_linear_exp_nocensoring_ties{0}_{trail}.pt",
    #     10_000,
    #     survival_distribution="exponential",
    #     risk_type="linear",
    #     censored_proportion=0.0,
    #     alpha=1,
    #     beta=1,
    #     feature_weights=[0.2, 0.7],
    #     censoring_function="independent",
    #     tie_groups=None,
    #     save_artifact=True,
    # )

    for b in [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 50, 100]:
        gen_pysurvival(
            f"pysurv_beta{b}_exp.pt",
            100_000,
            survival_distribution="exponential",
            risk_type="linear",
            censored_proportion=0.3,
            feature_weights=[0.3, 0.6],
            censoring_function="independent",
            tie_groups=0,
            save_artifact=False,
            sampling="beta",
            beta_p=b,
        )

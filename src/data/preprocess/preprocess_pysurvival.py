import os

import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.models.simulations import SimulationModel
from pysurvival.utils.display import integrated_brier_score
from pysurvival.utils.metrics import concordance_index
from sklearn.model_selection import train_test_split

from definitions import DATA_DIR
from omni.common import create_folder


def gen_pysurvival(
    name,
    N,
    survival_distribution="weibull",
    risk_type="gaussian",
    censored_proportion=0.0,
    alpha=0.1,
    beta=3.2,
    feature_weights=[1.0] * 3,
    censoring_function="independent",
    save_artifact=True,
):
    #### 2 - Generating the dataset from a nonlinear Weibull parametric model
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
        num_samples=N, num_features=len(feature_weights), feature_weights=feature_weights
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
            "alpha": alpha,
            "beta": beta,
            "feature_weights": feature_weights,
            "censoring_function": censoring_function,
        }

        run = wandb.init(
            job_type="preprocess_synthetic", project="diffsurv", entity="cardiors", config=config
        )
        artifact = wandb.Artifact(name, type="dataset", metadata=config)
        artifact.add_file(os.path.join(save_path, name), name)
        run.log_artifact(artifact)


class CustomSimulationModel(SimulationModel):
    """Just inheriting to get access to pre time function risk"""

    def generate_data(self, num_samples=100, num_features=3, feature_weights=None):
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
        T = self.time_function(BX)
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

    gen_pysurvival(
        "pysurv_square_0.3.pt",
        32000,
        survival_distribution="weibull",
        risk_type="square",
        censored_proportion=0.3,
        alpha=0.1,
        beta=3.2,
        feature_weights=[1.0] * 3,
        censoring_function="mean",
    )

    # gen_pysurvival('pysurv_square_mean_0.3.pt', 32000, survival_distribution='weibull', risk_type='square',
    #                censored_proportion=0.3,
    #                alpha=0.1, beta=3.2, feature_weights=[1.] * 3, censoring_function='mean')
    #
    # gen_pysurvival('pysurv_square_independent_0.3.pt', 32000, survival_distribution='weibull', risk_type='square',
    #                censored_proportion=0.3,
    #                alpha=0.1, beta=3.2, feature_weights=[1.] * 3, censoring_function='independent')

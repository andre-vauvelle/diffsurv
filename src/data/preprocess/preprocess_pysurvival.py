import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pysurvival.models.simulations import SimulationModel
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score
from definitions import DATA_DIR
from omni.common import create_folder
import wandb


def gen_pysurvival(name, N,
                   survival_distribution='weibull',
                   risk_type='gaussian',
                   censored_proportion=0.0,
                   alpha=0.1,
                   beta=3.2,
                   feature_weights=[1.] * 3,
                   censoring_function='independent', save_artifact=True):
    #### 2 - Generating the dataset from a nonlinear Weibull parametric model
    # Initializing the simulation model
    sim = SimulationModel(survival_distribution=survival_distribution,
                          risk_type=risk_type,
                          censored_parameter=10000000,
                          alpha=alpha, beta=beta)

    # Generating N random samples 
    dataset = sim.generate_data(num_samples=N, num_features=len(feature_weights), feature_weights=feature_weights)
    x_covar = dataset.iloc[:, :len(feature_weights)].to_numpy()
    y_times = dataset.time.to_numpy()

    censoring_times = np.random.uniform(0, y_times, size=N)

    # Select proportion of the patients to be right-censored using censoring_times
    if censoring_function == 'independent':
        # Independent of covariates
        censoring_indices = np.random.choice(N, size=int(N * censored_proportion), replace=False)
    elif censoring_function == 'mean':
        # Censored if mean of covariates over percentile determined by censoring proportion
        mean_covs = dataset.iloc[:, :len(feature_weights)].mean(1)
        percentile_cut = np.percentile(mean_covs, int(100 * censored_proportion))
        censoring_indices = np.array(mean_covs < percentile_cut)
    else:
        raise NotImplementedError(f"censoring_function {censoring_function} but must be either 'independent' or 'mean'")
    y_times[censoring_indices] = censoring_times[censoring_indices]
    censored_events = np.zeros(N, dtype=bool)
    censored_events[censoring_indices] = True

    x_covar = torch.Tensor(x_covar).float()
    y_times = torch.Tensor(y_times).float().unsqueeze(-1)
    censored_events = torch.Tensor(censored_events).long().unsqueeze(-1)
    print(f"Proportion censored: {censored_events.sum() / N}")

    # create directory for save
    save_path = os.path.join(DATA_DIR, 'synthetic')
    create_folder(save_path)
    torch.save((x_covar, y_times, censored_events), os.path.join(save_path, name))
    print("Saved risk synthetic dataset to: {}".format(os.path.join(save_path, name)))
    if save_artifact:
        config = {'name': name, 'N': N, 'survival_distribution': survival_distribution, 'risk_type': risk_type,
                  'censored_proportion': censored_proportion, 'alpha': alpha, 'beta': beta,
                  'feature_weights': feature_weights, 'censoring_function': censoring_function}

        run = wandb.init(job_type='preprocess_synthetic', project='diffsurv', config=config)
        artifact = wandb.Artifact(name, type='dataset', metadata=config)
        artifact.add_file(os.path.join(save_path, name), name)
        run.log_artifact(artifact)


if __name__ == '__main__':
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

    gen_pysurvival('pysurv_square_mean_0.3.pt', 32000, survival_distribution='weibull', risk_type='square',
                   censored_proportion=0.3,
                   alpha=0.1, beta=3.2, feature_weights=[1.] * 3, censoring_function='mean')

    gen_pysurvival('pysurv_square_independent_0.3.pt', 32000, survival_distribution='weibull', risk_type='square',
                   censored_proportion=0.3,
                   alpha=0.1, beta=3.2, feature_weights=[1.] * 3, censoring_function='independent')

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


def gen_pysurvival(name, N,
                   survival_distribution='weibull',
                   risk_type='gaussian',
                   censored_proportion=0.0,
                   alpha=0.1,
                   beta=3.2,
                   feature_weights=[1.] * 3):
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
    censoring_indices = np.random.choice(N, size=int(N * censored_proportion), replace=False)
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


if __name__ == '__main__':
    #gen_pysurvival('pysurv_linear_0.0.pt', 32000, survival_distribution='exponential', risk_type='linear',
    #               censored_proportion=0,
    #               alpha=0.1, beta=3.2, feature_weights=[1.1] * 3)
    #gen_pysurvival('pysurv_square_0.0.pt', 32000, survival_distribution='exponential', risk_type='square',
    #               censored_proportion=0,
    #               alpha=0.1, beta=3.2, feature_weights=[1.1] * 3)
    #for c in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    #    gen_pysurvival(f'pysurv_linear_{str(c)}.pt', 32000, survival_distribution='exponential', risk_type='linear',
    #                   censored_proportion=c,
    #                   alpha=0.1, beta=3.2, feature_weights=[1.1] * 3)

    #for c in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    #    gen_pysurvival(f'pysurv_square_{str(c)}.pt', 32000, survival_distribution='exponential', risk_type='square',
    #                   censored_proportion=c,
    #                   alpha=0.1, beta=3.2, feature_weights=[1.1] * 3)
    #for c in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    #    gen_pysurvival(f'pysurv_gaussian_{str(c)}.pt', 32000, survival_distribution='exponential', risk_type='gaussian',
    #                   censored_proportion=c,
    #                   alpha=0.1, beta=3.2, feature_weights=[1.1] * 3)
    for c in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        gen_pysurvival(f'pysurv_gaussian_weibull_{str(c)}.pt', 32000, survival_distribution='weibull', risk_type='gaussian',
                       censored_proportion=c,
                       alpha=0.1, beta=3.2, feature_weights=[1.1] * 3)


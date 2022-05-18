import os

import numpy as np
import torch
from definitions import DATA_DIR
from omni.common import create_folder


def gen_synthetic_dataset(n_patients=30_000, n_covariates=5, hazards=(4, -2, 1, 2, -1), proportion_censored=0.3,
                          baseline=30,
                          name='linear_exp_synthetic.pt', linear=True):
    """
    Generate synthetic dataset.

    Similar to the one used in the paper: http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """
    # Sample X from gaussian distribution
    x_covar = np.random.normal(0, 1, size=(n_patients, n_covariates))

    # Used with linear comnination to sample y from exponential
    hazards = np.array(hazards)
    if linear:
        alpha_scales = baseline + np.dot(hazards, x_covar.T)
    else:
        alpha_scales = baseline + np.dot(hazards, x_covar.T) ** 2 + np.dot(hazards, x_covar.T)
    y_times = np.random.exponential(alpha_scales)

    # Find right-censoring times for events using a random uniform distribution between 0 and min(y)
    censoring_times = np.random.uniform(0, y_times, size=n_patients)

    # Select 50% of the patients to be right-censored using censoring_times
    censoring_indices = np.random.choice(n_patients, size=int(n_patients * proportion_censored), replace=False)
    y_times[censoring_indices] = censoring_times[censoring_indices]
    censored_events = np.zeros(n_patients, dtype=bool)
    censored_events[censoring_indices] = True

    x_covar = torch.Tensor(x_covar).float()
    y_times = torch.Tensor(y_times).float().unsqueeze(-1)
    censored_events = torch.Tensor(censored_events).long().unsqueeze(-1)

    # create directory for save
    save_path = os.path.join(DATA_DIR, 'synthetic')
    create_folder(save_path)
    torch.save((x_covar, y_times, censored_events), os.path.join(save_path, name))
    print("Saved synthetic dataset to: {}".format(os.path.join(save_path, name)))


if __name__ == '__main__':
    gen_synthetic_dataset(n_patients=32_000, name='linear_exp_synthetic.pt', linear=True)
    gen_synthetic_dataset(n_patients=32_000, proportion_censored=0, name='linear_exp_synthetic_no_censoring.pt',
                          linear=True)

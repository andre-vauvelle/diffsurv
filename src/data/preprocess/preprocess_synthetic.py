import os

import numpy as np
import torch
from definitions import DATA_DIR
from omni.common import create_folder


def gen_synthetic_dataset(n_patients=30_000, n_covariates=2, hazards=(-1, 2), proportion_censored=0.5,
                          name='linear_exp_synthetic.pt'):
    """
    Generate synthetic dataset.

    Similar to the one used in the paper: http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """
    # Sample X from gaussian distribution
    x_covar = np.random.normal(0, 1, size=(n_patients, n_covariates))

    # Used with linear comnination to sample y from exponential
    hazards = np.array(hazards)
    y_times = np.exp(np.dot(hazards, x_covar.T))  # This is where we could introduce non-linearity

    # Find right-censoring times for events using a random uniform distribution between 0 and min(y)
    censoring_times = np.random.uniform(0, np.min(y_times), size=n_patients)

    # Select 50% of the patients to be right-censored using censoring_times
    censoring_indices = np.random.choice(n_patients, size=int(n_patients * proportion_censored), replace=False)
    y_times[censoring_indices] = censoring_times[censoring_indices]
    censored_events = np.zeros(n_patients, dtype=bool)
    censored_events[censoring_indices] = True

    x_covar = torch.Tensor(x_covar).float()
    y_times = torch.Tensor(y_times).float()
    censored_events = torch.Tensor(censored_events).int()

    # create directory for save
    save_path = os.path.join(DATA_DIR, 'synthetic')
    create_folder(save_path)
    torch.save((x_covar, y_times, censored_events), os.path.join(save_path, name))
    print("Saved synthetic dataset to: {}".format(os.path.join(save_path, name)))

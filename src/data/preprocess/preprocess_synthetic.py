import os

import numpy as np
import torch
from definitions import DATA_DIR
from omni.common import create_folder


def gen_synthetic_basic(n_patients=30_000, name='basic_rank.pt'):
    """
    Generate basic synthetic dataset to test sorting.
    Covariates directly determine the rank of the patient.
    returns x, y, censored_events
    """
    # Sample X from gaussian distribution
    x_covar = np.random.uniform(0, 5, size=(n_patients, 1))
    # x_covar = np.random.normal(0, 1, size=(n_patients, n_covariates))

    # Used with linear comnination to sample y from exponential
    y_times = x_covar.squeeze()

    censored_events = np.zeros(n_patients, dtype=bool)

    x_covar = torch.Tensor(x_covar).float()
    y_times = torch.Tensor(y_times).float().unsqueeze(-1)
    censored_events = torch.Tensor(censored_events).long().unsqueeze(-1)

    # create directory for save
    save_path = os.path.join(DATA_DIR, 'synthetic')
    create_folder(save_path)
    torch.save((x_covar, y_times, censored_events), os.path.join(save_path, name))
    print("Saved basic synthetic dataset to: {}".format(os.path.join(save_path, name)))


def gen_synthetic_risk_dataset(n_patients=30_000, n_covariates=3, hazards=(100, 100, 100), proportion_censored=0.3,
                               baseline=0,
                               name='linear_exp_synthetic.pt', linear=True):
    """
    Generate synthetic dataset.

    Similar to the one used in the paper: http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """
    # Sample X from gaussian distribution
    # x_covar = np.random.randint(0, 5, size=(n_patients, n_covariates))
    x_covar = np.random.uniform(0, 1, size=(n_patients, n_covariates))

    # Used with linear comnination to sample y from exponential
    hazards = np.array(hazards)
    if linear:
        alpha_scales = baseline + np.dot(hazards, x_covar.T)
    else:
        alpha_scales = baseline + np.dot(hazards, x_covar.T) ** 2 + np.dot(hazards, x_covar.T)
    y_times = np.random.exponential(alpha_scales)

    # Find right-censoring times for events using a random uniform distribution between 0 and min(y)
    censoring_times = np.random.uniform(0, y_times, size=n_patients)

    # Select proportion of the patients to be right-censored using censoring_times
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
    print("Saved risk synthetic dataset to: {}".format(os.path.join(save_path, name)))


if __name__ == '__main__':
    #gen_synthetic_risk_dataset(n_patients=32_000, name='linear_exp_synthetic.pt', linear=True)
    #gen_synthetic_risk_dataset(n_patients=32_000, proportion_censored=0, name='linear_exp_synthetic_no_censoring.pt', linear=True)
    #gen_synthetic_risk_dataset(n_patients=32_000, name='nonlinear_exp_synthetic.pt', linear=False)
    #gen_synthetic_basic(n_patients=32_000, name='basic_rank.pt')
    #gen_synthetic_risk_dataset(n_patients=32_000, name='nonlinear_exp_synthetic_0.9c.pt', linear=False, proportion_censored=0.9)
    #gen_synthetic_risk_dataset(n_patients=32_000, proportion_censored=0, name='nonlinear_exp_synthetic_no_censoring.pt', linear=False)
    for c in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
        gen_synthetic_risk_dataset(n_patients=32_000, proportion_censored=c, name=f"nonlinear_exp_synthetic_{str(c)}.pt", linear=False)


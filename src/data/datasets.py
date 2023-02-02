import random
from typing import List, Optional

import numba
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

from modules.tasks import _get_possible_permutation_matrix


def flip(p):
    return random.random() < p


class AbstractDataset(Dataset):
    def __init__(
        self,
        records,
        token2idx,
        label2idx,
        age2idx,
        max_len,
        token_col="concept_id",
        label_col="phecode",
        age_col="age",
        covariates=None,
        used_covs=None,
    ):
        """

        :param records:
        :param token2idx:
        :param age2idx:
        :param max_len:
        :param token_col:
        :param age_col:
        """
        self.max_len = max_len
        self.eid = records["eid"].copy()
        self.tokens = records[token_col].copy()
        self.labels = records[label_col].copy()
        self.date = records["date"].copy()
        self.age = records[age_col].copy()
        self.token2idx = token2idx
        self.label2idx = label2idx
        self.age2idx = age2idx
        self.covariates = covariates
        self.used_covs = used_covs

    def __getitem__(self, index):
        """
        return: age_col, code_col, position, segmentation, mask, label
        """
        pass

    def __len__(self):
        return len(self.tokens)


class DatasetRisk(Dataset):
    def __init__(
        self,
        x_covar: torch.Tensor,
        y_times: torch.Tensor,
        censored_events: torch.Tensor,
        risk: Optional[torch.Tensor] = None,
    ):
        self.x_covar = x_covar
        self.y_times = y_times
        self.censored_events = (
            censored_events
            if isinstance(censored_events, torch.Tensor)
            else torch.Tensor(censored_events)
        )
        self.risk = risk
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop([54, 54]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        covariates = self.x_covar[index]
        if covariates.dim() == 3:
            covariates = Image.fromarray(np.transpose(covariates.numpy(), (1, 2, 0)))
            covariates = self.transform(covariates)
        future_label_multihot = 1 - self.censored_events[index]
        future_label_times = self.y_times[index]
        censorings = self.censored_events[index]
        exclusions = torch.zeros_like(censorings)

        output = {
            # labels
            "labels": future_label_multihot,
            "label_times": future_label_times,
            "censorings": censorings,
            "exclusions": exclusions,
            # input
            "covariates": covariates,
        }

        if self.risk is not None:
            risk = self.risk[index]
            if not isinstance(risk, np.ndarray):
                risk = np.array(risk).reshape(-1, 1)
            output.update({"risk": risk})

        return output

    def __len__(self):
        return self.x_covar.shape[0]


class CaseControlRiskDataset(Dataset):
    def __init__(
        self,
        n_controls: int,
        x_covar: torch.Tensor,
        y_times: torch.Tensor,
        censored_events: torch.Tensor,
        risk: Optional[torch.Tensor] = None,
        return_perm_mat: bool = True,
        n_cases: int = 1,
        inc_censored_in_ties: bool = False,
        random_sample: bool = False,
    ):
        self.inc_censored_in_ties = inc_censored_in_ties
        self.n_controls = n_controls
        self.n_cases = n_cases
        self.x_covar = x_covar
        self.y_times = y_times
        self.censored_events = (
            censored_events
            if isinstance(censored_events, torch.Tensor)
            else torch.Tensor(censored_events)
        )
        self.risk = risk
        self.return_perm_mat = return_perm_mat
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop([54, 54]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.random_sample = random_sample
        if random_sample:

            def gen_sample(idxs: list):
                while True:
                    idxs_copy = idxs[:]
                    while idxs_copy:
                        yield idxs_copy.pop()

            random_idxs = list(range(x_covar.shape[0]))
            random.shuffle(random_idxs)
            # self.gen_random_idx = gen_sample(random_idxs)
            self.gen_random_ids = random_idxs

    def __getitem__(self, index):
        idx_durations = self.y_times
        events = 1 - self.censored_events
        if not self.random_sample:
            idxs = get_case_control_idxs(
                n_cases=self.n_cases,
                n_controls=self.n_controls,
                idx_durations=idx_durations.numpy().astype(float),
                events=events.numpy().astype(bool),
            )
        else:
            # idxs = [next(self.gen_random_idx) for _ in range(self.n_cases + self.n_controls)]
            idxs = list(np.random.choice(self.gen_random_ids, size=self.n_cases + self.n_controls))

        if events.dim() == 1:
            events = events.unsqueeze(-1)
        assert events.shape[1] == 1, "does not support multi class yet.."
        if self.return_perm_mat:
            possible_perm_mat = _get_possible_permutation_matrix(
                events[idxs].flatten(),
                idx_durations[idxs].flatten(),
                inc_censored_in_ties=self.inc_censored_in_ties,
            )
        else:
            possible_perm_mat = np.zeros(
                (self.n_cases + self.n_controls, self.n_cases + self.n_controls)
            )

        covariates = self.x_covar[idxs]
        if covariates.dim() > 3:
            stack = []
            for img in covariates:
                img = Image.fromarray(np.transpose(img.numpy(), (1, 2, 0)))
                img = self.transform(img)
                stack.append(img)
            covariates = torch.stack(stack)

        future_label_multihot = events[idxs]
        future_label_times = self.y_times[idxs]
        censorings = self.censored_events[idxs]
        exclusions = torch.zeros_like(censorings)

        output = {
            # labels
            "labels": future_label_multihot,
            "label_times": future_label_times,
            "censorings": censorings,
            "exclusions": exclusions,
            # input
            "covariates": covariates,
            "soft_perm_mat": possible_perm_mat,
        }

        if self.risk is not None:
            risk = self.risk[idxs]
            if not isinstance(risk, np.ndarray):
                risk = np.array(risk).reshape(-1, 1)
            output.update({"risk": risk})

        return output

    def __len__(self):
        if self.random_sample:
            return self.x_covar.shape[0] // self.n_controls + self.n_cases
        return int((1 - self.censored_events).sum())


#
@numba.njit
def get_case_control_idxs(
    # mat: np.ndarray,
    n_cases: int,
    n_controls: int,
    idx_durations: np.ndarray,
    events: np.ndarray,
) -> List[int]:  # Tuple[List[int], np.ndarray]:
    """
    Get the case and control idxs that are acceptable pairs
    # :param mat:
    :param n_cases:
    :param n_controls:
    :param idx_durations:
    :param events:
    :return:
    """
    idx_batch = []
    n = idx_durations.shape[0]
    cases_sampled = 0
    case_idxs = np.arange(n)[events.flatten() == 1]
    while cases_sampled < n_cases:
        i = np.random.choice(case_idxs)
        dur_i = idx_durations[i]
        cases_sampled += 1

        controls_sampled = 0
        # TODO: Rely on sorted idx_durations and we can easily a sample without replacement
        possible_controls_mask = (dur_i < idx_durations) | ((dur_i == idx_durations) & events == 0)
        if not possible_controls_mask.sum():
            continue  # No possible controls for this case... ignore and move to the next!
        possible_control_idxs = np.arange(n)[possible_controls_mask.flatten()]
        control_idxs = np.random.choice(possible_control_idxs, n_controls, replace=False)
        idx_batch.extend(control_idxs)
        idx_batch.append(i)

    return idx_batch


class TensorDataset(Dataset):
    """For preprocessed tensor datasets"""

    def __init__(self, tensor_path):
        pass

    def __getitem__(self, index):
        return self.tensors[index]

import math
import random
from typing import Iterator, List, Optional, Sized, Tuple

import numba
import numpy as np
import torch
from pytorch_lightning.overrides.distributed import DistributedSamplerWrapper
from torch.utils.data import RandomSampler, Sampler
from torch.utils.data.dataset import Dataset

from modules.loss import pair_rank_mat
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


class CaseControlBatchSampler(RandomSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        controls_per_case (int): number of valid comparable controls per case in the batch, will
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        controls_per_case: int = 1,
        replacement: bool = True,
        num_samples: Optional[int] = None,
        generator=None,
        drop_last=True,
    ) -> None:
        super().__init__(data_source, replacement, num_samples, generator)
        self.controls_per_case = controls_per_case
        self.batch_size = batch_size
        assert (
            self.batch_size % (self.controls_per_case + 1) == 0
        ), "Must batch size must be factor of cases/control "
        self.cases_per_batch = self.batch_size / (self.controls_per_case + 1)
        self.total_batches = int(math.floor(self.num_samples / self.batch_size))
        self.drop_last = drop_last
        # TODO: support drop last.
        assert drop_last, "Currently doesn't support keeping last..?"

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            self.data_source: DatasetRisk
            return int(sum(1 - self.data_source.censored_events))
        return self._num_samples

    def __iter__(self) -> Iterator[List[int]]:
        self.data_source: DatasetRisk

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        idx_durations = self.data_source.y_times
        events = 1 - self.data_source.censored_events
        n = len(self.data_source)

        # idx_batch_store = []
        idx_batch = []
        # for i in torch.randperm(n, generator=generator):
        for i in torch.randperm(n, generator=generator):
            dur_i = idx_durations[i]
            ev_i = events[i]
            if ev_i == 0:
                continue

            controls_sampled = 0
            for j in torch.randperm(n, generator=generator):
                if j == i:  # cannot compare with self
                    continue

                dur_j = idx_durations[j]
                ev_j = events[j]
                if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                    idx_batch.append(j)  # add a control
                    controls_sampled += 1

                if controls_sampled == self.controls_per_case:
                    idx_batch.append(i)  # add case
                    break

            if len(idx_batch) == self.batch_size:
                random.shuffle(idx_batch)
                ans = idx_batch
                idx_batch = []  # start new batch
                yield ans


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
        self.censored_events = censored_events
        self.risk = risk

    def __getitem__(self, index):
        covariates = self.x_covar[index]

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
    ):
        self.inc_censored_in_ties = inc_censored_in_ties
        self.n_controls = n_controls
        self.n_cases = n_cases
        self.x_covar = x_covar
        self.y_times = y_times
        self.censored_events = censored_events
        self.risk = risk
        self.return_perm_mat = return_perm_mat

    def __getitem__(self, index):
        idx_durations = self.y_times
        events = 1 - self.censored_events
        idxs = get_case_control_idxs(
            n_cases=self.n_cases,
            n_controls=self.n_controls,
            idx_durations=idx_durations.numpy(),
            events=events.numpy(),
        )

        assert events.shape[1] == 1, "does not support multi class yet.."
        if self.return_perm_mat:
            soft_perm_mat = _get_possible_permutation_matrix(
                events[idxs].flatten(),
                idx_durations[idxs].flatten(),
                inc_censored_in_ties=self.inc_censored_in_ties,
            )
        else:
            soft_perm_mat = None

        covariates = self.x_covar[idxs]

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
            "soft_perm_mat": soft_perm_mat,
        }

        if self.risk is not None:
            risk = self.risk[idxs]
            if not isinstance(risk, np.ndarray):
                risk = np.array(risk).reshape(-1, 1)
            output.update({"risk": risk})

        return output

    def __len__(self):
        return (1 - self.censored_events).sum()


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

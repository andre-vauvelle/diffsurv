import os

import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from data.preprocess.utils import SYMBOL_IDX
from collections import defaultdict


def drop_mask(tokens, symbol='MASK'):
    seq = []
    for token in tokens:
        if token == symbol:
            continue
        else:
            seq.append(token)
    return seq


def pad_sequence(tokens, max_len, symbol=SYMBOL_IDX['PAD']):
    """
    Pad sequence to max_len, also restricts to max_len if sequence is longer, reorders s
    :param tokens:
    :param max_len:
    :param symbol:
    :return:
    """
    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if i < token_len:
            seq.append(tokens[-i])
        else:
            seq.append(symbol)
    return seq


def index_seg(tokens, symbol=SYMBOL_IDX['SEP']):
    """
    Alternates between visits
    :param tokens:
    :param symbol:
    :return:
    """
    flag = 1
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 1:
                flag = 2
            else:
                flag = 1
        else:
            seg.append(flag)
    return seg


def position_idx(tokens, symbol=SYMBOL_IDX['SEP']):
    """
    Increments per vist
    :param tokens:
    :param symbol:
    :return:
    """
    pos = []
    flag = 1

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def drop_unk_idx(idxs, idx_with_unk=None, unk_idx=SYMBOL_IDX['UNK']):
    if idx_with_unk is None:
        return [i for i in idxs if i != unk_idx]
    else:
        return [i for i, j in zip(idxs, idx_with_unk) if j != unk_idx]


def get_token2idx(tokens, token2idx):
    output_idx = []
    for i, token in enumerate(tokens):
        output_idx.append(token2idx.get(token, token2idx['UNK']))
    return output_idx


import random


def flip(p):
    return random.random() < p


def drop_conseq_idx(token_idx, symbol_idx=SYMBOL_IDX['SEP']):
    """
    Drop consecutive sep indices
    :param symbol_idx:
    :param token_idx:
    :return:
    """
    return [i for i, j in zip(token_idx, token_idx[1:]) if not ((i == symbol_idx) & (j == symbol_idx))]


class AbstractDataset(Dataset):
    def __init__(self, records, token2idx, label2idx, age2idx, max_len,
                 token_col='concept_id', label_col='phecode', age_col='age', covariates=None, used_covs=None):
        """

        :param records:
        :param token2idx:
        :param age2idx:
        :param max_len:
        :param token_col:
        :param age_col:
        """
        self.max_len = max_len
        self.eid = records['eid'].copy()
        self.tokens = records[token_col].copy()
        self.labels = records[label_col].copy()
        self.date = records['date'].copy()
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


class DatasetSyntheticRisk(Dataset):
    def __init__(self, x_covar, y_times, censored_events):
        self.x_covar = x_covar
        self.y_times = y_times
        self.censored_events = censored_events

    def __getitem__(self, index):
        token_idx = None
        age_idx = None
        position = None
        segment = None
        mask = None

        covariates = self.x_covar[index]

        future_label_multihot = 1 - self.censored_events[index]
        future_label_times = self.y_times[index]
        censorings = self.censored_events[index]
        exclusions = None

        input_tuple = (token_idx, age_idx, position, segment, mask, covariates),
        label_tuple = (future_label_multihot, future_label_times, censorings, exclusions)

        return input_tuple, label_tuple


class DatasetAssessmentRiskPredict(AbstractDataset):
    def __init__(self, records, token2idx, label2idx, age2idx, max_len, covariates, used_covs, drop_unk,
                 drop_conseq_sep=True, **kwargs):
        """

        :param records:
        :param token2idx:
        :param age2idx:
        :param max_len:
        """
        super().__init__(records=records, token2idx=token2idx, label2idx=label2idx, age2idx=age2idx, max_len=max_len,
                         covariates=covariates, used_covs=used_covs, **kwargs)
        self.max_date = records['date'].explode().max()
        self.drop_unk = drop_unk
        self.drop_conseq_sep = drop_conseq_sep
        self.lens = []

    def __getitem__(self, index):
        """
        return: age_col, code_col, position, segmentation, mask, label
        """

        eid = self.eid.iloc[index]
        cov = self.covariates.query("eid == @eid")
        # eid
        # sex
        # yob
        # mob
        # dob
        # center_ass
        # year_ass
        # age_ass

        date_ass = cov.date_ass.values[0]
        age = self.age.iloc[index]

        # extract data
        tokens = self.tokens.iloc[index]
        labels = self.labels.iloc[index]
        # Extract days time difference
        dates = self.date.iloc[index]
        times = (dates - date_ass).astype('timedelta64[D]').astype(int)
        max_time = (self.max_date - date_ass).days

        # TODO: Add Buffer?
        history_idx = (times <= 0)
        future_idx = ~history_idx

        # Get only tokens before or after assessment and keep only max_len events
        history_tokens = tokens[history_idx]  # [(-self.max_len + 1):]
        history_labels = labels[history_idx]
        history_times = times[history_idx]
        future_labels = labels[future_idx]
        future_times = times[future_idx]

        def process_labels(labels_, times_):
            # Gets multihot labels that occured first and their times
            labels_keep = (labels_ != 'nan') & ~np.isin(labels_, list(SYMBOL_IDX.keys()))
            labels_k = labels_[labels_keep]
            times_k = times_[labels_keep]
            labels_u, label_times_list = self._get_first_times(labels_k, times_k)
            label_idx = get_token2idx(labels_u, self.label2idx)
            label_oh = torch.nn.functional.one_hot(torch.LongTensor(label_idx), num_classes=len(self.label2idx))
            label_multihot = (label_oh > 0).any(axis=0).float()

            # Feels hacky but it's just adding the days from assessment to the oh encoding
            label_times = label_multihot.long()
            label_idx.reverse(), label_times_list.reverse()  # reversed to add first time rather than last for
            # duplicates
            label_times[label_idx] = torch.LongTensor(label_times_list)
            # If no disease is found in the future, we use the last time in the history
            censorings_ = (label_multihot == 0).float()
            label_times[label_multihot == 0] = max_time

            return label_multihot, label_times, censorings_

        # Process future labels
        future_label_multihot, future_label_times, censorings = process_labels(future_labels, future_times)
        # Process history labels
        history_label_multihot, _, _ = process_labels(history_labels, history_times)

        # Exclusions, used to remove patients with prior event from c-index metric (before recruitment)
        exclusions = ((history_label_multihot + future_label_multihot) == 2).long()  # if both are 1, then exclude

        history_tokens = np.append(np.array(['CLS']), history_tokens)
        age = np.append(np.array(age[0]), age)

        # pad age_col sequence and code_col sequence
        age_idx = get_token2idx(age, self.age2idx)
        token_idx = get_token2idx(history_tokens, self.token2idx)

        if self.drop_unk:
            age_idx = drop_unk_idx(age_idx, idx_with_unk=token_idx)
            token_idx = drop_unk_idx(token_idx)

        if self.drop_conseq_sep:
            token_idx = drop_conseq_idx(token_idx)
            age_idx = drop_conseq_idx(age_idx)

        self.lens.append(len(token_idx))

        age_idx = pad_sequence(age_idx, self.max_len)
        token_idx = pad_sequence(token_idx, self.max_len)

        position = position_idx(token_idx)
        segment = index_seg(token_idx)
        position = pad_sequence(position, self.max_len)
        segment = pad_sequence(segment, self.max_len)

        covariates = np.array([])
        if 'age_ass' in self.used_covs:
            covariates = np.append(covariates, cov.age_ass)
        if 'sex' in self.used_covs:
            covariates = np.append(covariates, cov.sex)

        # used for attention mask the padding
        mask = np.ones(self.max_len)
        mask[len(token_idx):] = 0

        # token_idx, mask_labels, noise_labels = self.get_random_mask(token_idx, label_idx, mask_prob=self.mask_prob)

        # reorder patients in batch with the shortest duration first

        input_tuple = *(torch.LongTensor(v) for v in [token_idx, age_idx, position, segment, mask, covariates]),
        label_tuple = (future_label_multihot, future_label_times, censorings, exclusions)

        return input_tuple, label_tuple

    @staticmethod
    def _get_first_times(future_labels_k, future_times_k):
        label_store = []
        label_time_store = []
        for (l, t) in zip(future_labels_k, future_times_k):
            if l not in label_store:
                label_store.append(l)
                label_time_store.append(t)
            else:
                pass
        return label_store, label_time_store

    def __len__(self):
        return len(self.tokens)


class DatasetMLM(AbstractDataset):
    def __init__(self, records, token2idx, label2idx, age2idx, max_len, mask_prob=0.2, **kwargs):
        """

        :param records:
        :param token2idx:
        :param age2idx:
        :param max_len:
        """
        super().__init__(records, token2idx, label2idx, age2idx, max_len, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, index):
        """
        return: age_col, code_col, position, segmentation, mask, label
        """

        # extract data
        age = self.age.iloc[index][(-self.max_len + 1):]
        tokens = self.tokens.iloc[index][(-self.max_len + 1):]
        labels = self.labels.iloc[index][(-self.max_len + 1):]

        # avoid data cut with first element to be 'SEP'
        if tokens[0] != 'SEP':
            tokens = np.append(np.array(['CLS']), tokens)
            labels = np.append(np.array(['CLS']), labels)
            age = np.append(np.array(age[0]), age)
        else:
            tokens[0] = 'CLS'
            labels[0] = 'CLS'

        mask = np.ones(self.max_len)
        mask[len(tokens):] = 0

        # pad age_col sequence and code_col sequence
        age = pad_sequence(age, self.max_len)
        tokens = pad_sequence(tokens, self.max_len)
        labels = pad_sequence(labels, self.max_len)
        age_idx = get_token2idx(age, self.age2idx)
        token_idx = get_token2idx(tokens, self.token2idx)
        label_idx = get_token2idx(labels, self.label2idx)

        position = position_idx(tokens)
        segment = index_seg(tokens)

        token_idx, mask_labels, noise_labels = self.get_random_mask(token_idx, label_idx, mask_prob=self.mask_prob)

        return *(torch.LongTensor(v) for v in [token_idx, age_idx, position, segment, mask_labels, noise_labels, mask]),

    def __len__(self):
        return len(self.tokens)

    def get_random_mask(self, token_idx, label_idx=None, mask_prob=0.12):
        """
        :param label_idx:
        :param token_idx:
        # :param noise_type:
        # :param noise_prob:
        :param mask_prob:
        :return:
        """
        # output_mask = []
        if label_idx is None:
            label_idx = token_idx
        output_mask_label_idx = []
        output_noised_label = []
        output_token_idx = []
        # SYMBOL_IDX = {
        #     "PAD": 0,
        #     "SEP": 1,
        #     "UNK": 2,
        #     "MASK": 3,
        #     "CLS": 4,
        #     'None': 5
        # }
        symbols_idx = SYMBOL_IDX.values()

        for i, (t_idx, l_idx) in enumerate(zip(token_idx, label_idx)):
            # exclude special symbols from masking
            if l_idx in symbols_idx:  # PAD MASK SEP CLS UNK
                # output_mask.append(1)
                output_mask_label_idx.append(-1)
                output_token_idx.append(t_idx)
                output_noised_label.append(-1)
            else:
                prob = random.random()
                n_prob = random.random()

                if prob < mask_prob:
                    # mask with 0 which means do not attend to this value (effectively drops value)
                    # output_mask.append(0)  # do not attend masked value
                    # if n_prob < noise_prob:
                    #     if noise_type == 'symmetric':
                    #         noised_idx = random.choice(list(set(list(self.phe2idx.values())) - {*symbols_idx} - {t_idx}))
                    #         noised_label = 1
                    #     elif noise_type == 'asymmetric':
                    #         full_phecode = self.idx2phe[t_idx]
                    #         top_phecode = re.sub('\..*$', '', full_phecode)
                    #         subcode_options = self.phe_groupings.get(top_phecode, 'exclude')
                    #         noised_idx = random.choice(subcode_options) if subcode_options != 'exclude' else t_idx
                    #         noised_label = 1 if noised_idx != t_idx else 0
                    #     output_mask_label_idx.append(noised_idx)
                    #     output_noised_label.append(noised_label)
                    # else:
                    output_mask_label_idx.append(l_idx)  # add label for loss calc
                    # output_mask_label_idx.append(20)  # cheat!
                    output_noised_label.append(0)
                    output_token_idx.append(SYMBOL_IDX['PAD'])  # mask by using pad token, excludes from embedding bag
                    # output_token_idx.append(l_idx)  # cheat!
                else:
                    # output_mask.append(1)  # attend this value
                    output_mask_label_idx.append(-1)  # exclude from loss func
                    output_noised_label.append(-1)  # not a label
                    output_token_idx.append(t_idx)  # keep original token if not masked
        return output_token_idx, output_mask_label_idx, output_noised_label

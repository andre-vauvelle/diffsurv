import os

import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import DatasetMLM, DatasetAssessmentRiskPredict, DatasetSyntheticRisk, TensorDataset, \
    DatasetNextPredict
from definitions import DATA_DIR
from omni.common import load_pickle, save_pickle
import torch


class AbstractDataModule(pl.LightningDataModule):
    def __init__(self,
                 token_col='phecode',
                 label_col='phecode',
                 token_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phecode_vocab.pkl'),
                 label_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phecode_vocab.pkl'),
                 age_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'age_vocab_90.pkl'),
                 train_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_train.parquet'),
                 val_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_val.parquet'),
                 test_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_test.parquet'),
                 covariates_path=os.path.join(
                     os.path.join(DATA_DIR, 'processed', 'covariates', 'eid_covariates.parquet')),
                 batch_size=32,
                 max_len_seq=256,
                 num_workers=1,
                 used_covs=('age_ass', 'sex'),
                 debug=True):
        super().__init__()
        self.token_col = token_col
        self.label_col = label_col
        self.token_vocab = load_pickle(token_vocab_path)
        self.label_vocab = load_pickle(label_vocab_path)
        self.age_vocab = load_pickle(age_vocab_path)
        # self.phe_vocab = load_pickle(phe_vocab_path)
        self.max_len_seq = max_len_seq
        self.batch_size = batch_size
        self.num_workers = num_workers if not debug else 1

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path

        self.covariates_path = covariates_path
        self.used_covs = used_covs
        self.input_dim = len(list(self.token_vocab['token2idx'].keys()))
        self.output_dim = len(list(self.label_vocab['token2idx'].keys()))
        self.debug = debug

    def train_dataloader(self):
        train_data = pd.read_feather(self.train_data_path)
        train_data = train_data.head(10_000) if self.debug else train_data
        pass

    def val_dataloader(self):
        val_data = pd.read_feather(self.val_data_path)
        val_data = val_data.head(1_000) if self.debug else val_data
        pass

    def test_dataloader(self):
        test_data = pd.read_feather(self.test_data_path)
        test_data = test_data.head(1_000) if self.debug else test_data
        pass


class DataModuleMLM(AbstractDataModule):
    def __init__(self,
                 token_col='phecode',
                 label_col='phecode',
                 token_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phecode_vocab_top100_105.pkl'),
                 label_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phecode_vocab_top100_105.pkl'),
                 age_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'age_vocab_89.pkl'),
                 train_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_train.parquet'),
                 val_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_val.parquet'),
                 test_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_test.parquet'),
                 batch_size=32,
                 max_len_seq=256,
                 num_workers=1,
                 used_covs=('age_ass', 'sex'),
                 debug=False,
                 mask_prob=0.2):
        super().__init__(token_col=token_col, label_col=label_col, token_vocab_path=token_vocab_path,
                         label_vocab_path=label_vocab_path, age_vocab_path=age_vocab_path,
                         train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path,
                         batch_size=batch_size, max_len_seq=max_len_seq, num_workers=num_workers, used_covs=used_covs,
                         debug=debug)
        self.mask_prob = mask_prob

    def train_dataloader(self):
        train_data = pd.read_feather(self.train_data_path)
        train_data = train_data.head(10_000) if self.debug else train_data
        train_dataset = DatasetMLM(train_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                   self.age_vocab['token2idx'],
                                   max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                   mask_prob=self.mask_prob)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        val_data = pd.read_feather(self.val_data_path)
        val_data = val_data.head(1_000) if self.debug else val_data
        val_dataset = DatasetMLM(val_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                 self.age_vocab['token2idx'],
                                 max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                 mask_prob=self.mask_prob)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_data = pd.read_feather(self.test_data_path)
        test_data = test_data.head(1_000) if self.debug else test_data
        test_dataset = DatasetMLM(test_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                  self.age_vocab['token2idx'],
                                  max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                  mask_prob=self.mask_prob)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class DataModuleAssessmentRiskPredict(AbstractDataModule):
    def __init__(self,
                 token_col='phecode',
                 label_col='phecode',
                 token_vocab_path=os.path.join(DATA_DIR, 'processed', 'in_gnn', 'phecode_vocab_top100_105.pkl'),
                 label_vocab_path=os.path.join(DATA_DIR, 'processed', 'in_gnn', 'phecode_vocab_top100_105.pkl'),
                 age_vocab_path='/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/in_gnn/age_vocab_90.pkl',
                 train_data_path=os.path.join(DATA_DIR, 'processed', 'in_gnn', 'phe_train.feather'),
                 val_data_path=os.path.join(DATA_DIR, 'processed', 'in_gnn', 'phe_val.feather'),
                 test_data_path=os.path.join(DATA_DIR, 'processed', 'in_gnn', 'phe_test.feather'),
                 covariates_path=os.path.join(
                     os.path.join(DATA_DIR, 'processed', 'covariates', 'patient_base.parquet')),
                 batch_size=32,
                 max_len_seq=256,
                 min_length=5,
                 num_workers=1,
                 used_covs=('age_ass', 'sex'),
                 task='risk',
                 init_setup=False,
                 debug=False,
                 drop_unk=True,
                 weightings_path=None,
                 incidence_grouping_path=None,
                 ):
        """

        :param token_col:
        :param label_col:
        :param token_vocab_path:
        :param label_vocab_path:
        :param age_vocab_path:
        :param train_data_path:
        :param val_data_path:
        :param test_data_path:
        :param covariates_path:
        :param batch_size:
        :param max_len_seq:
        :param num_workers:
        :param used_covs:
        :param: init_setup: inits new groupping and weights for new data labels
        :param debug:
        :param drop_unk:
        :param weightings_path:
        """
        super().__init__(token_col, label_col, token_vocab_path, label_vocab_path, age_vocab_path,
                         train_data_path, val_data_path, test_data_path, covariates_path, batch_size, max_len_seq,
                         num_workers, used_covs=used_covs, debug=debug)
        self.task = task
        self.min_length = min_length
        if init_setup:
            # For c-index grouping, and loss weighting
            grouping_labels, weightings = self.init_setup(weightings_path, incidence_grouping_path)
            self.grouping_labels = grouping_labels
        else:
            self.grouping_labels = load_pickle(os.path.join(DATA_DIR, 'processed', 'incidence_groupings.pkl'))
            weightings = torch.load(weightings_path) if weightings_path else None

        self.drop_unk = drop_unk
        self.save_hyperparameters()
        # after save to avoid messy wandb logs
        self.weightings = weightings

    def init_setup(self, weightings_path, incidence_grouping_path):
        dataset = self.get_dataset(self.train_data_path, self.covariates_path)
        print('Initialising grouping and weighting')
        data = [dataset.__getitem__(i) for i in tqdm(range(len(dataset)), total=len(dataset), desc='Extracting data')]

        labels = [x['labels'] for x in data]
        labels = torch.stack(labels).cpu()
        weightings = self.get_weightings(labels, weightings_path)
        grouping_labels = self.get_incidence_groupings(labels, incidence_grouping_path)

        return grouping_labels, weightings

    def get_weightings(self, labels, weightings_path, min_count=1000):
        """
        Calculate weightings for each label, if path is given, load from there, otherwise calculate, save and return
        :param labels: tensor of labels
        :param weightings_path: the path to the weightings
        :param min_count: minimum number of labels to be included in the weighting
        :return: tensor of weightings, index of label
        """
        labels_counts = labels.sum(dim=0)
        class_weights = (labels_counts + min_count) / ((labels_counts + min_count).sum(dim=0))
        torch.save(class_weights, weightings_path)
        print("Weightings saved to {}".format(weightings_path))
        return class_weights

    def get_incidence_groupings(self, labels, incidence_grouping_path):
        """
        Gets labels within each incidence group for c-index averaging
        :param labels: tensor of labels
        :param incidence_grouping_path: path to incidence groupings
        """
        # ratio of total patients with each label, i.e. \ge 1 in 1000 patients have this label, bounds (lower, upper)
        grouping_dict = {
            'all': (-1, float('inf')),
            '>1:10': (0.1, float('inf')),
            '>1:100': (0.01, 0.1),
            '>1:1000': (0.001, 0.01),
            '<1:1000': (-1, 0.001),
        }
        grouping_labels = {}
        class_weights = labels.mean(axis=0)
        for grouping_name, bounds in grouping_dict.items():
            lower = bounds[0] <= class_weights
            upper = class_weights < bounds[1]
            idx = torch.logical_and(lower, upper)
            int_idx = torch.arange(0, idx.shape[0])
            int_idx = int_idx[idx]
            grouping_l = [self.label_vocab['idx2token'][int(i)] for i in int_idx]
            grouping_labels.update({grouping_name: grouping_l})
        save_pickle(grouping_labels, incidence_grouping_path)
        return grouping_labels

    def get_dataset(self, stage_path, covariates_path):
        data = pd.read_feather(stage_path)
        data = data.query(f"length > {self.min_length}")
        covariates = pd.read_parquet(covariates_path)
        if self.task == 'next':
            dataset = DatasetNextPredict(data,
                                         self.token_vocab['token2idx'],
                                         self.label_vocab['token2idx'],
                                         self.age_vocab['token2idx'],
                                         max_len=self.max_len_seq,
                                         token_col=self.token_col,
                                         label_col=self.label_col,
                                         covariates=covariates,
                                         used_covs=self.used_covs,
                                         drop_unk=self.drop_unk)
        elif self.task == 'risk':
            dataset = DatasetAssessmentRiskPredict(data,
                                                   self.token_vocab['token2idx'],
                                                   self.label_vocab['token2idx'],
                                                   self.age_vocab['token2idx'],
                                                   max_len=self.max_len_seq,
                                                   token_col=self.token_col,
                                                   label_col=self.label_col,
                                                   covariates=covariates,
                                                   used_covs=self.used_covs,
                                                   drop_unk=self.drop_unk)
        else:
            raise ValueError(f"Chosen {self.task} but only ['next', 'risk'] allowed")
        return dataset

    def train_dataloader(self):
        train_dataset = self.get_dataset(self.train_data_path, self.covariates_path)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        val_dataset = self.get_dataset(self.val_data_path, self.covariates_path)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = self.get_dataset(self.test_data_path, self.covariates_path)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class DataModuleSytheticRisk(pl.LightningDataModule):
    def __init__(self,
                 path='/SAN/ihibiobank/denaxaslab/andre/UKBB/data/synthetic/linear_exp_synthetic.pt',
                 val_split=0.2,
                 batch_size=32,
                 num_workers=1,
                 ):
        super().__init__()
        self.path = path
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        (x_covar, y_times, censored_events) = torch.load(os.path.join(self.path))
        self.input_dim = x_covar.shape[1]
        self.output_dim = y_times.shape[1]
        self.label_vocab = {'token2idx': {'event0': 0}, 'idx2token': {0: 'event0'}}
        self.grouping_labels = {'all': ['event0']}

    def train_dataloader(self):
        (x_covar, y_times, censored_events) = torch.load(os.path.join(self.path))
        n_patients = x_covar.shape[0]
        n_training_patients = int(n_patients * (1 - self.val_split))
        train_datatset = DatasetSyntheticRisk(x_covar[:n_training_patients], y_times[:n_training_patients],
                                              censored_events[:n_training_patients])
        return DataLoader(train_datatset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        (x_covar, y_times, censored_events) = torch.load(os.path.join(self.path))
        n_patients = x_covar.shape[0]
        n_validation_patients = int(n_patients * self.val_split)
        val_datatset = DatasetSyntheticRisk(x_covar[-n_validation_patients:], y_times[-n_validation_patients:],
                                            censored_events[-n_validation_patients:])
        return DataLoader(val_datatset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        pass

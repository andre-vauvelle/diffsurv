import os

import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader

from data.datasets import DatasetMLM
from definitions import DATA_DIR
from omni.common import load_pickle
import torch


class AbstractDataModule(pl.LightningDataModule):
    def __init__(self,
                 token_col='concept_id',
                 label_col='phecode',
                 token_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'concept_vocab.pkl'),
                 label_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phecode_vocab.pkl'),
                 age_vocab_path=os.path.join(DATA_DIR, 'processed', 'omop', 'age_vocab.pkl'),
                 train_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_train.parquet'),
                 val_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_val.parquet'),
                 test_data_path=os.path.join(DATA_DIR, 'processed', 'omop', 'phe_test.parquet'),
                 batch_size=32,
                 max_len_seq=256,
                 num_workers=torch.multiprocessing.cpu_count(),
                 debug=False):
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

        self.input_dim = len(list(self.token_vocab['token2idx'].keys()))
        self.output_dim = len(list(self.label_vocab['token2idx'].keys()))
        self.debug = debug

    def train_dataloader(self):
        train_data = pd.read_parquet(self.train_data_path)
        train_data = train_data.head(10_000) if self.debug else train_data
        pass

    def val_dataloader(self):
        val_data = pd.read_parquet(self.val_data_path)
        val_data = val_data.head(1_000) if self.debug else val_data
        pass

    def test_dataloader(self):
        test_data = pd.read_parquet(self.test_data_path)
        test_data = test_data.head(1_000) if self.debug else test_data
        pass


class DataModuleMLM(AbstractDataModule):
    def __init__(self,
                 mask_prob=0.2, **kwargs):
        super().__init__()
        self.mask_prob = mask_prob

    def train_dataloader(self):
        train_data = pd.read_parquet(self.train_data_path)
        train_data = train_data.head(10_000) if self.debug else train_data
        train_dataset = DatasetMLM(train_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                   self.age_vocab['token2idx'],
                                   max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                   mask_prob=self.mask_prob)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        val_data = pd.read_parquet(self.val_data_path)
        val_data = val_data.head(1_000) if self.debug else val_data
        val_dataset = DatasetMLM(val_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                 self.age_vocab['token2idx'],
                                 max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                 mask_prob=self.mask_prob)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_data = pd.read_parquet(self.test_data_path)
        test_data = test_data.head(1_000) if self.debug else test_data
        test_dataset = DatasetMLM(test_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                  self.age_vocab['token2idx'],
                                  max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                  mask_prob=self.mask_prob)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class DataModuleBaselineRiskPredict(AbstractDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        train_data = pd.read_parquet(self.train_data_path)
        train_data = train_data.head(10_000) if self.debug else train_data
        train_dataset = DatasetMLM(train_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                   self.age_vocab['token2idx'],
                                   max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                   mask_prob=self.mask_prob)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        val_data = pd.read_parquet(self.val_data_path)
        val_data = val_data.head(1_000) if self.debug else val_data
        val_dataset = DatasetMLM(val_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                 self.age_vocab['token2idx'],
                                 max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                 mask_prob=self.mask_prob)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_data = pd.read_parquet(self.test_data_path)
        test_data = test_data.head(1_000) if self.debug else test_data
        test_dataset = DatasetMLM(test_data, self.token_vocab['token2idx'], self.label_vocab['token2idx'],
                                  self.age_vocab['token2idx'],
                                  max_len=self.max_len_seq, token_col=self.token_col, label_col=self.label_col,
                                  mask_prob=self.mask_prob)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

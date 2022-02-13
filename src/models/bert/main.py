import os
import pandas as pd
import time

import torch.nn as nn
import torch.nn.functional as f
import pytorch_pretrained_bert as Bert
import torchmetrics
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from torch import optim
from torchmetrics import AveragePrecision, MetricCollection, AUROC

from data.datasets import PheDataset
from definitions import TENSORBOARD_DIR, MODEL_DIR
from model.bert.components import BertModel
from model.bert.config import BertConfig

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset, ConcatDataset
import re
from tqdm import tqdm

from model.bert.temperature import ModelWithTemperature, _ECELoss
from omni.common import load_pickle
from task.analysis.plot_calibration import plot_calibration


class BERTAnchorModel(Bert.modeling.BertPreTrainedModel, pl.LightningModule):
    """
    TODO: update
    Anchor variable model from Halpern et al., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4419996/

    Uses anchor variable A, which is trial_n postive anchor for trial_n latent variable Y_i if it is an anchor for Y_i and
    P(Y_i=1|A=1)=1

    Method:
    1. Learn calibrated classifier to predict P(A|\tilde{X}), where \tilde{X} is censored from A
    2. Using validation set compute C = 1/|\mathcal{P}| \sum P(A=1|\tilde{X}), where \mathcal{P} is the validation data
    set where A=1
    3. For trial_n previously unseen patient t, predict:
        - P(A=1|\tilde{X})/C    if A(t) = 0
        - 1                     if A(t) = 1

    """

    def __init__(self, token, token2idx, bert_config, symbol_idxs=[0, 1], skip_training=False,
                 predict_proba=True, emission_size=5, num_workers=0, token_type='phecode',
                 file_config=None, max_len_seq=256, temperature_scaling=False,
                 verbose=False, tensorboard_name='default', pretrained_bert=None):
        config = BertConfig(bert_config['model_config'])
        super(BERTAnchorModel, self).__init__(config)

        self.token = token
        self.token2idx = token2idx
        self.train_params = bert_config.get('train_params', None)
        self.optim_config = bert_config.get('optim_config', None)
        self.file_config = file_config
        self.bert_checkpoint_dir = bert_config.get('bert_checkpoint_dir', os.path.join(MODEL_DIR, 'default'))
        feature_dict = bert_config['feature_dict']
        if pretrained_bert is None:
            self.bert = BertModel(config, feature_dict)
        else:
            self.bert = pretrained_bert

        self.lr = self.optim_config['lr'] if self.optim_config is not None else None
        self.batch_size = self.train_params['batch_size'] if self.train_params is not None else None
        self.max_len_seq = max_len_seq
        self.temperature_scaling = temperature_scaling

        self.skip_training = skip_training
        if token2idx is None:
            self.token_idxs = token
            num_labels = len(token)
            self.symbol_idxs = symbol_idxs
            self.features = emission_size
        else:
            keys = [k for k in token2idx.keys() if re.match(token, k) is not None]
            self.token_idxs = [token2idx[k] for k in keys]
            self.symbol_idxs = [token2idx[k] for k in
                                ['PAD', 'MASK', 'CLS', 'UNK']]  # not including SEP as we can use as feature
            self.features = len(token2idx.keys()) - len(self.token_idxs) - len(self.symbol_idxs)
            num_labels = len(keys)

        self.num_workers = num_workers
        self.token_type = token_type
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()
        # self.temperature = ModelWithTemperature() if bert_config['temperature'] else None
        self.apply(self.init_bert_weights)

        self.predict_proba = predict_proba
        self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func_a = nn.BCEWithLogitsLoss(reduction='none')

        self.tb_logger = TensorBoardLogger(
            save_dir=bert_config.get('tensorboard_dir', os.path.join(MODEL_DIR, 'tensorboard', 'default')),
            name=tensorboard_name)

        metrics = MetricCollection(
            [AveragePrecision(pos_label=1, compute_on_step=False),
             AUROC(pos_label=1, compute_on_step=False)])
        self.train_metrics = metrics.clone(prefix='train_')
        self.train_metrics_noise = metrics.clone(prefix='train_noise_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.valid_metrics_noise = metrics.clone(prefix='val_noise_')
        self.test_metrics = metrics.clone(prefix='test_')

    def get_anchors_censored_mask(self, x):
        """Gets the mask of anchor terms for x_censored and  y"""
        censor_mask = torch.stack([x == t for t in self.token_idxs]).any(0).int()  # 1 to mask 0 to attend
        targets = (censor_mask > 0).any(-1).int()
        censor_mask = (-1) * (censor_mask - 1)  # 1 to attended 0 to mask
        return censor_mask, targets

    def forward(self, batch):
        token_idx, age_idx, position, segment, phecode_idx, label = batch
        if self.token_type == 'phecode':
            x = phecode_idx
        else:
            x = token_idx
        censor_mask, y_anchor = self.get_anchors_censored_mask(x)  # Could be moved to data loader...
        _, pooled_output = self.bert(input_ids=x, age_ids=age_idx, seg_ids=segment, posi_ids=position,
                                     attention_mask=censor_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # predictions = self.sigmoid(logits)
        return logits.flatten(), label.flatten(), y_anchor

    def training_step(self, batch, batch_idx):
        logits, label, y_anchor = self(batch)
        loss = self.loss_func(logits, label.float())
        # predictions = f.sigmoid(logits)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, label, y_anchor = self(batch)
        loss = self.loss_func(logits, y_anchor.float())
        self.log('val_loss', loss, prog_bar=True)
        predictions = f.sigmoid(logits)
        self.valid_metrics.update(predictions, y_anchor)
        self.valid_metrics_noise.update(predictions, label)

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)
        output_noise = self.valid_metrics_noise.compute()
        self.valid_metrics_noise.reset()
        self.log_dict(output_noise, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits, label, y_anchor = self(batch)
        loss = self.loss_func(logits, y_anchor.float())
        self.log('val_loss', loss, prog_bar=True)
        predictions = f.sigmoid(logits)
        self.test_metrics.update(predictions, label)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(output, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        logits, label, y_anchor = self(batch)
        model_predictions = torch.sigmoid(logits)
        loss = self.loss_func_a(logits, y_anchor.float())
        return {'predictions': model_predictions, 'y_anchor': y_anchor, 'label': label, 'loss': loss}

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        params = self.named_parameters()

        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]

        optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                               lr=self.optim_config['lr'],
                                               warmup=self.optim_config['warmup_proportion'])
        return optimizer

    # def train_dataloader(self):
    #     """Only for real data, used for fitting batch size"""
    #     code_vocab = load_pickle(self.file_config['code_vocab'])
    #     age_vocab = load_pickle(self.file_config['age_vocab'])
    #     phe_vocab = load_pickle(self.file_config['phe_vocab'])
    #
    #     train_data = pd.read_parquet(self.file_config['train_data'])
    #
    #     train_dataset = PheDataset(self.token, train_data, code_vocab['token2idx'], age_vocab['token2idx'],
    #                                max_len=self.max_len_seq,
    #                                phe2idx=phe_vocab['token2idx'])
    #     return DataLoader(dataset=train_dataset, batch_size=self.batch_size)
    #
    # def val_dataloader(self):
    #     """Only for real data, used for fitting batch size"""
    #     code_vocab = load_pickle(self.file_config['code_vocab'])
    #     age_vocab = load_pickle(self.file_config['age_vocab'])
    #     phe_vocab = load_pickle(self.file_config['phe_vocab'])
    #
    #     val_data = pd.read_parquet(self.file_config['val_data'])
    #
    #     val_dataset = PheDataset(self.token,val_data, code_vocab['token2idx'], age_vocab['token2idx'],
    #                              max_len=self.max_len_seq,
    #                              phe2idx=phe_vocab['token2idx'])
    #     return DataLoader(dataset=val_dataset, batch_size=self.batch_size)

    def predict(self, dataset, val_dataset=None, test_dataset=None, tv_split=0.8):
        """TODO: refactor out"""
        # model wil be overwritten if do_training but not for temperature_scaling
        model = self

        if val_dataset is None:
            train_size = int(len(dataset) * tv_split)
            train_dataset = Subset(dataset, range(train_size))
            val_dataset = Subset(dataset, range(train_size, len(dataset)))
        else:
            train_dataset = dataset
            # dataset = ConcatDataset([train_dataset, val_dataset])

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_params['batch_size'],
                                      shuffle=True,
                                      num_workers=self.num_workers)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.train_params['batch_size'], shuffle=False,
                                    num_workers=self.num_workers)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.train_params['batch_size'], shuffle=False,
                                     num_workers=self.num_workers)
        checkpoint_filename = '{epoch}-{val_loss:.4f}{val_AveragePrecision:.4f}{val_AUROC:.4f}'
        checkpoint_filename = checkpoint_filename + 'debug' if __debug__ else checkpoint_filename
        checkpoint_callback = ModelCheckpoint(dirpath=self.bert_checkpoint_dir,
                                              filename=checkpoint_filename,
                                              monitor='val_AveragePrecision',
                                              mode='max',
                                              save_top_k=-1
                                              )
        trainer = pl.Trainer(max_epochs=self.train_params['epochs'], check_val_every_n_epoch=1,
                             val_check_interval=self.train_params['val_check_interval'],
                             checkpoint_callback=True,
                             callbacks=[checkpoint_callback, ],
                             gpus=self.train_params['gpus'],
                             num_sanity_val_steps=-1,
                             accumulate_grad_batches=self.train_params['accumulate_grad_batches'],
                             logger=self.tb_logger)

        if self.train_params['auto_lr_find']:
            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(self)
            # Plot with
            fig = lr_finder.plot(suggest=True)
            fig.show()
            # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()
            # update lr of the model
            self.lr = new_lr

        if not self.skip_training:
            trainer.fit(self, train_dataloader, val_dataloader)
            model = BERTAnchorModel.load_from_checkpoint(
                checkpoint_callback.best_model_path, token=self.token, token2idx=self.token2idx,
                bert_config=self.bert_config, file_config=self.file_config,
                skip_training=False, num_workers=self.num_workers)

        if self.temperature_scaling:
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.train_params['batch_size'], shuffle=False,
                                        num_workers=self.num_workers)
            model = ModelWithTemperature(self)
            model.set_temperature(val_dataloader)

        # Get training predictions

        datasets = [d for d in (train_dataset, val_dataset, test_dataset) if d is not None]
        full_dataset = ConcatDataset(datasets)
        prediction_dataloader = DataLoader(dataset=full_dataset, batch_size=512,
                                           shuffle=False, num_workers=self.num_workers)
        prediction_trainer = pl.Trainer(gpus=self.train_params['gpus'])
        results = prediction_trainer.predict(model, prediction_dataloader)

        predictions = torch.cat([r['predictions'] for r in results]).cpu()

        train_metrics = trainer.validate(model, train_dataloader)[0]
        val_metrics = trainer.validate(model, val_dataloader)[0]
        test_metrics = trainer.test(model, test_dataloader)[0]

        train_metrics_old = train_metrics
        train_metrics = {}
        for key, value in train_metrics_old.items():
            if 'val_' == key[:4]:
                new_key = 'train_' + key[4:]
                train_metrics.update({new_key: value})

        metrics = {}
        metrics.update(train_metrics)
        metrics.update(val_metrics)
        metrics.update(test_metrics)

        return predictions, metrics

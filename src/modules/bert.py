import torch
import torch.nn as nn
import torch.nn.functional as f
import pytorch_pretrained_bert as Bert
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_pretrained_bert.modeling import BertPredictionHeadTransform
from torch.optim import SparseAdam
from torchmetrics import AveragePrecision, MetricCollection, AUROC, Precision, Accuracy

from models.heads import PredictionHead
from modules.loss import CoxPHLoss
from modules.tasks import RiskMixin
from src.models.bert.components import BertModel, CustomBertLMPredictionHead
from src.models.bert.config import BertConfig

import pytorch_lightning as pl


class BertBase(Bert.modeling.BertPreTrainedModel, pl.LightningModule):
    def __init__(self,
                 input_dim=1390,
                 output_dim=1390,
                 embedding_dim=120,
                 num_hidden_layers=8,
                 hidden_dropout_prob=0.2, lr=1e-4, warmup_proportion=0.1,
                 temperature_scaling=False,
                 feature_dict=None, num_attention_heads=12, intermediate_size=256, hidden_act="gelu",
                 attention_probs_dropout_prob=0.22, max_position_embeddings=256, initializer_range=0.02,
                 age_vocab_size=None, seg_vocab_size=2,
                 pretrained_embedding_path=None,
                 freeze_pretrained=False,
                 used_covs=('age_ass', 'sex'),
                 ):
        if feature_dict is None:
            feature_dict = {
                'word': True,
                'seg': True,
                'age': False,
                'position': True}

        # TODO refactor
        config = BertConfig(input_dim=input_dim, output_dim=output_dim, hidden_size=embedding_dim,
                            num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                            intermediate_size=intermediate_size, hidden_act=hidden_act,
                            hidden_dropout_prob=hidden_dropout_prob,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            max_position_embeddings=max_position_embeddings,
                            initializer_range=initializer_range,
                            seg_vocab_size=seg_vocab_size,
                            age_vocab_size=age_vocab_size,
                            shared_lm_input_output_weights=None,
                            pretrained_embedding_path=pretrained_embedding_path, freeze_pretrained=freeze_pretrained
                            )
        super().__init__(config)

        self.output_dim = output_dim
        self.lr = lr
        self.warmup_proportion = warmup_proportion
        self.temperature_scaling = temperature_scaling
        self.used_covs = used_covs

        self.bert = BertModel(config, feature_dict)
        in_features = embedding_dim if used_covs is None else embedding_dim + len(used_covs)
        self.head = PredictionHead(in_features, output_dim)
        # self.cls = CustomBertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        # TODO: Consider moving to mixin
        metrics = MetricCollection(
            [
                AveragePrecision(num_classes=self.output_dim, compute_on_step=False, average='weighted'),
                Precision(compute_on_step=False, average='micro'),
                Accuracy(compute_on_step=False, average='micro'),
                AUROC(num_classes=self.output_dim, compute_on_step=False)
            ]
        )

        # self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def _shared_eval_step(self, batch, batch_idx):
        rank_zero_warn("`_shared_eval_step` must be implemented to be used with the Lightning Trainer")

    def forward(self, input_tuple, covariates) -> torch.Tensor:

        token_idx, age_idx, position, segment, mask = input_tuple

        _, pooled = self.bert(input_ids=token_idx, age_ids=age_idx, seg_ids=segment, posi_ids=position,
                              attention_mask=mask,
                              output_all_encoded_layers=False)
        if self.used_covs is not None:
            pooled = torch.cat((pooled, covariates), dim=1)

        logits = self.head(pooled)
        return logits

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        sparse = {n: p for n, p in self.named_parameters() if 'embeddings' in n}
        not_sparse = {n: p for n, p in self.named_parameters() if 'embeddings' not in n}
        optimizer_grouped_parameters = [
            {'params': [p for n, p in not_sparse.items() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in not_sparse.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]
        optimizer_sparse = torch.optim.SparseAdam(sparse.values(), lr=self.lr)
        optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                               lr=self.lr,
                                               warmup=self.warmup_proportion)
        return optimizer_sparse, optimizer


class BERTRisk(RiskMixin, BertBase):
    """
    For MLM pretraining.
    """

    def __init__(self,
                 input_dim=1390,
                 output_dim=1390,
                 embedding_dim=120,
                 num_hidden_layers=8,
                 hidden_dropout_prob=0.2, lr=1e-4, warmup_proportion=0.1,
                 temperature_scaling=False,
                 feature_dict=None, num_attention_heads=12, intermediate_size=256, hidden_act="gelu",
                 attention_probs_dropout_prob=0.22, max_position_embeddings=256, initializer_range=0.02,
                 age_vocab_size=None, seg_vocab_size=3,
                 pretrained_embedding_path=None,
                 freeze_pretrained=False,
                 used_covs=('age_ass', 'sex'),
                 grouping_labels=None, label_vocab=None, weightings=None, use_weighted_loss=False
                 ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, embedding_dim=embedding_dim,
                         num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                         intermediate_size=intermediate_size, hidden_act=hidden_act,
                         hidden_dropout_prob=hidden_dropout_prob,
                         attention_probs_dropout_prob=attention_probs_dropout_prob,
                         max_position_embeddings=max_position_embeddings, lr=lr, warmup_proportion=warmup_proportion,
                         temperature_scaling=temperature_scaling, feature_dict=feature_dict,
                         initializer_range=initializer_range,
                         seg_vocab_size=seg_vocab_size,
                         age_vocab_size=age_vocab_size,
                         pretrained_embedding_path=pretrained_embedding_path, used_covs=used_covs,
                         freeze_pretrained=freeze_pretrained, grouping_labels=grouping_labels,
                         label_vocab=label_vocab, weightings=weightings, use_weighted_loss=use_weighted_loss)
        self.save_hyperparameters()

    def _shared_eval_step(self, batch, batch_idx):
        (token_idx, age_idx, position, segment, mask, covariates), (label_multihot, label_times, censorings,
                                                                    exclusions) = batch
        logits = self((token_idx, age_idx, position, segment, mask), covariates)

        # predictions = self.sigmoid(logits)
        loss = self.loss_func(logits, label_multihot, label_times)

        return loss, logits, label_multihot, label_times


class BERTMLM(Bert.modeling.BertPreTrainedModel, pl.LightningModule):
    """
    For MLM pretraining.
    """

    def __init__(self,
                 input_dim=1390,
                 output_dim=1390,
                 embedding_dim=120,
                 num_hidden_layers=8,
                 hidden_dropout_prob=0.2, lr=1e-4, warmup_proportion=0.1,
                 temperature_scaling=False,
                 feature_dict=None, num_attention_heads=12, intermediate_size=256, hidden_act="gelu",
                 attention_probs_dropout_prob=0.22, max_position_embeddings=256, initializer_range=0.02,
                 age_vocab_size=None, seg_vocab_size=2,
                 pretrained_embedding_path=None,
                 freeze_pretrained=False
                 ):
        if feature_dict is None:
            feature_dict = {
                'word': True,
                'seg': True,
                'age': False,
                'position': True}

        shared_lm_input_output_weights = True if input_dim == output_dim else False

        # TODO refactor
        super(BERTMLM, self).__init__(input_dim, output_dim=output_dim, hidden_size=embedding_dim,
                                      num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                      intermediate_size=intermediate_size, hidden_act=hidden_act,
                                      hidden_dropout_prob=hidden_dropout_prob,
                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                      max_position_embeddings=max_position_embeddings,
                                      initializer_range=initializer_range,
                                      seg_vocab_size=seg_vocab_size,
                                      age_vocab_size=age_vocab_size,
                                      shared_lm_input_output_weights=shared_lm_input_output_weights,
                                      pretrained_embedding_path=pretrained_embedding_path,
                                      freeze_pretrained=freeze_pretrained)

        self.head = CustomBertLMPredictionHead(self.config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

    def _shared_eval_step(self, batch, batch_idx):
        token_idx, age_idx, position, segment, mask_labels, noise_labels, mask = batch

        logits = self(batch)

        # predictions = self.sigmoid(logits)
        logits_expanded = logits.view(-1, self.output_dim)
        mask_labels_expanded = mask_labels.view(-1)
        loss = self.loss_func(logits_expanded, mask_labels_expanded)

        return loss, logits_expanded, mask_labels_expanded

    def forward(self, batch) -> torch.Tensor:
        token_idx, age_idx, position, segment, mask_labels, noise_labels, mask = batch

        unpooled_output, _ = self.bert(input_ids=token_idx, age_ids=age_idx, seg_ids=segment, posi_ids=position,
                                       attention_mask=mask,
                                       output_all_encoded_layers=False)
        logits = self.head(unpooled_output)
        return logits

    def validation_step(self, batch, batch_idx):
        loss, logits_expanded, mask_labels_expanded = self._shared_eval_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)

        predictions = f.softmax(logits_expanded.view(-1, self.output_dim), dim=1)
        keep = mask_labels_expanded.view(-1) != -1
        mask_labels_reduced = mask_labels_expanded.view(-1)[keep]
        predictions = predictions[keep]
        self.valid_metrics.update(predictions, mask_labels_reduced)


if __name__ == '__main__':
    from modules.bert import BERTMLM

    gnn_bert = BERTMLM(input_dim=4697, output_dim=794,
                       pretrained_embedding_path='/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/gnn_embeddings_256_1gr128qk_20220217.pt',
                       freeze_pretrained=False,
                       embedding_dim=256,
                       num_attention_heads=2,
                       lr=0.0001
                       )
    prone_bert = BERTMLM(input_dim=4697, output_dim=794,
                         pretrained_embedding_path='/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211209_prone_256_edge_weights_no_shortcuts_2022-01-05.pt',
                         freeze_pretrained=False,
                         embedding_dim=256,
                         num_attention_heads=2,
                         lr=0.0001
                         )

    gnn_bert.bert.embeddings.word_embeddings.weight
    prone_bert.bert.embeddings.word_embeddings.weight

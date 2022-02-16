import torch
import torch.nn as nn
import torch.nn.functional as f
import pytorch_pretrained_bert as Bert
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_pretrained_bert.modeling import BertPredictionHeadTransform
from torchmetrics import AveragePrecision, MetricCollection, AUROC, Precision, Accuracy
from src.models.bert.components import BertModel, CustomBertLMPredictionHead
from src.models.bert.config import BertConfig

import pytorch_lightning as pl


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
                 pretrained_embedding_path=None, freeze_pretrained=False, single_multihot_training=True,
                 ):
        if feature_dict is None:
            feature_dict = {
                'word': True,
                'seg': True,
                'age': False,
                'position': True}

        shared_lm_input_output_weights = True if input_dim == output_dim else False

        config = BertConfig(input_dim=input_dim, output_dim=output_dim, hidden_size=embedding_dim,
                            num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                            intermediate_size=intermediate_size, hidden_act=hidden_act,
                            hidden_dropout_prob=hidden_dropout_prob,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            max_position_embeddings=max_position_embeddings,
                            initializer_range=initializer_range,
                            seg_vocab_size=seg_vocab_size,
                            age_vocab_size=age_vocab_size,
                            shared_lm_input_output_weights=shared_lm_input_output_weights,
                            pretrained_embedding_path=pretrained_embedding_path)
        super(BERTMLM, self).__init__(config)

        self.output_dim = output_dim
        self.lr = lr
        self.warmup_proportion = warmup_proportion
        self.temperature_scaling = temperature_scaling

        self.bert = BertModel(config, feature_dict)
        self.cls = CustomBertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        metrics = MetricCollection(
            [
                AveragePrecision(num_classes=self.output_dim, compute_on_step=False, average='weighted'),
                Precision(compute_on_step=False, average='micro'),
                Accuracy(compute_on_step=False, average='micro'),
                # AUROC(num_classes=self.output_dim, compute_on_step=False)
            ]
        )

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters(config)

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
        logits = self.cls(unpooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        loss, logits_expanded, mask_labels_expanded = self._shared_eval_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits_expanded, mask_labels_expanded = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

        predictions = f.softmax(logits_expanded.view(-1, self.output_dim), dim=1)
        keep = mask_labels_expanded.view(-1) != -1
        mask_labels_reduced = mask_labels_expanded.view(-1)[keep]
        predictions = predictions[keep]
        self.valid_metrics.update(predictions, mask_labels_reduced)

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)
        self.log('hp_metric', output['val_AveragePrecision'])

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        params = self.named_parameters()

        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]

        optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                               lr=self.lr,
                                               warmup=self.warmup_proportion)
        return optimizer

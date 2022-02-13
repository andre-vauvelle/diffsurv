import pytorch_pretrained_bert as Bert


class BertConfig(Bert.modeling.BertConfig):

    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size,
                 hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings,
                 initializer_range, seg_vocab_size, age_vocab_size):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
        )
        self.seg_vocab_size = seg_vocab_size
        self.age_vocab_size = age_vocab_size

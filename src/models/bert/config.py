import pytorch_pretrained_bert as Bert


class BertConfig(Bert.modeling.BertConfig):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
        initializer_range,
        seg_vocab_size,
        age_vocab_size,
        shared_lm_input_output_weights,
        pretrained_embedding_path,
        freeze_pretrained,
    ):
        super().__init__(
            vocab_size_or_config_json_file=input_dim,
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
        self.shared_lm_input_output_weights = shared_lm_input_output_weights
        self.output_dim = output_dim
        self.pretrained_embedding_path = pretrained_embedding_path
        self.freeze_pretrained = freeze_pretrained

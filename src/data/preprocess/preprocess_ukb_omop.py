import os
import random

import pandas as pd
import torch

from data.preprocess.utils import reformat_gnn_embedding, vocab_omop_embedding, align_pretrained_embedding
from definitions import DATA_DIR
from omni.common import load_pickle

if __name__ == '__main__':
    # code_vocab_path = os.path.join(DATA_DIR, 'processed', 'omop', 'concept_vocab_4697.pkl')
    code_vocab_path = os.path.join(DATA_DIR, 'processed', 'in_gnn', 'concept_vocab_1472.pkl')
    in_gnn = True

    embeddings_path = '/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/gnn_embeddings_256_1gr128qk_20220217.feather'

    embeddings_path = '/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211122_prone_32_edge_weights_2021-12-13.feather'

    embeddings_path = '/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211209_prone_256_edge_weights_no_shortcuts_2022-01-05.feather'



    save_path = embeddings_path[:-8] + '.pt' if not in_gnn else embeddings_path[:-8] + '_in_gnn.pt'


    concept_path = '/home/vauvelle/pycharm-sftp/ehrgnn/data/athena/CONCEPT.csv'

    embedding = pd.read_feather(embeddings_path)
    # concept = pd.read_csv(concept_path, sep='\t')
    code_vocab = load_pickle(code_vocab_path)

    token2idx = code_vocab['token2idx']
    idx2token = code_vocab['idx2token']

    embedding = reformat_gnn_embedding(embedding)
    embedding_tensor, missing_embeddings = align_pretrained_embedding(embedding, token2idx, save_path)

    # torch.save(embedding_tensor, save_path)
    gnn = torch.load(save_path)
    prone = torch.load(save_path)
    # test
    rand_int = random.randint(0, embedding_tensor.shape[0])
    rand_concept = idx2token[rand_int]
    origin = embedding_tensor[rand_int].numpy()
    if rand_concept in embedding.concept_id:
        target = embedding.query('concept_id == @rand_concept').embedding.values[0]
        assert (origin == target).all() if rand_concept in embedding.concept_id else True


    # vocab_omop_embedding(embedding, token2idx, save_path=save_path)

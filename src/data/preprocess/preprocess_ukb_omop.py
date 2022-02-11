import os
import pandas as pd

from data.preprocess.utils import reformat_gnn_embedding, vocab_omop_embedding
from definitions import DATA_DIR
from omni.common import load_pickle

if __name__ == '__main__':
    save_path = '/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211122_prone_32_edge_weights_2021-12-13.feather'
    embeddings_path = '/home/vauvelle/pycharm-sftp/ehrgnn/data/graph_full_211122_prone_32_edge_weights_2021-12-13.feather'
    concept_path = '/home/vauvelle/pycharm-sftp/ehrgnn/data/athena/CONCEPT.csv'
    code_vocab_path = os.path.join(DATA_DIR, 'processed', 'all', 'code_vocab.pkl')

    embedding = pd.read_feather(embeddings_path)
    # concept = pd.read_csv(concept_path, sep='\t')
    code_vocab = load_pickle(code_vocab_path)
    token2idx = code_vocab['token2idx']

    embedding = reformat_gnn_embedding(embedding)
    vocab_omop_embedding(embedding, token2idx, save_path=save_path)

qsub -N risk_bert_in_gnn jobs/submit.sh scripts/bert_risk.py fit \
 --config jobs/configs/risk/bert_risk.yaml --config jobs/configs/in_gnn_data.yaml \
 --model.embedding_dim=256

qsub -N risk_bert_in_gnn_prone jobs/submit.sh scripts/bert_risk.py fit \
 --config jobs/configs/risk/bert_risk.yaml --config jobs/configs/in_gnn_data.yaml \
 --model.embedding_dim=256 \
 --model.pretrained_embedding_path=/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211209_prone_256_edge_weights_no_shortcuts_2022-01-05.pt

qsub -N risk_bert_in_gnn_gnn jobs/submit.sh scripts/bert_risk.py fit \
 --config jobs/configs/risk/bert_risk.yaml --config jobs/configs/in_gnn_data.yaml \
 --model.embedding_dim=256 \
 --model.pretrained_embedding_path=/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/gnn_embeddings_256_1gr128qk_20220217_in_gnn.pt

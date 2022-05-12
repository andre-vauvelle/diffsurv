
qsub -N bert256P jobs/bert.sh fit \
  --config jobs/configs/mlm/bert_concept.yaml \
  --model.pretrained_embedding_path=/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211209_prone_256_edge_weights_no_shortcuts_2022-01-05.pt \
  --model.embedding_dim=256

qsub -N bert256FP jobs/bert.sh fit \
  --config jobs/configs/mlm/bert_concept.yaml \
  --model.pretrained_embedding_path=/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211209_prone_256_edge_weights_no_shortcuts_2022-01-05.pt \
  --model.embedding_dim=256 \
  --model.freeze_pretrained=True

qsub -N bert256 jobs/bert.sh fit \
  --config jobs/configs/mlm/bert_concept.yaml \
  --model.embedding_dim=256

qsub -N bert32P jobs/bert.sh fit \
  --config jobs/configs/mlm/bert_concept.yaml \
  --model.pretrained_embedding_path=/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211122_prone_32_edge_weights_2021-12-13.pt \
  --model.embedding_dim=32

qsub -N bert32FP jobs/bert.sh fit \
  --config jobs/configs/mlm/bert_concept.yaml \
  --model.pretrained_embedding_path=/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211122_prone_32_edge_weights_2021-12-13.pt \
  --model.embedding_dim=32 \
  --model.freeze_embedding=True

qsub -N bert32 jobs/bert.sh fit \
  --config jobs/configs/mlm/bert_concept.yaml \
  --model.embedding_dim=32

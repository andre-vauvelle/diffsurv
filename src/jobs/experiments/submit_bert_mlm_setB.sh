
qsub -N bert256_phe jobs/bert.sh fit \
  --config jobs/configs/mlm/bert_phecode.yaml \
  --model.embedding_dim=256

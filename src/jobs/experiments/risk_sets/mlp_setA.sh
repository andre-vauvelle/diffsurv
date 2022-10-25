qsub -N risk_mlp_in_gnn_pretrained256prone jobs/submit.sh scripts/mlp_risk.py fit \
 --config jobs/configs/risk/mlp_risk.yaml --config jobs/configs/in_gnn_data.yaml \
 --model.pretrained_embedding_path=/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/graph_full_211209_prone_256_edge_weights_no_shortcuts_2022-01-05_in_gnn.pt \
 --model.embedding_dim=256 \
 --model.lr=0.001 \
 --data.batch_size=1024

qsub -N risk_mlp_in_gnn_pretrained256gnn jobs/submit.sh scripts/mlp_risk.py fit \
 --config jobs/configs/risk/mlp_risk.yaml --config jobs/configs/in_gnn_data.yaml \
 --model.pretrained_embedding_path=/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/gnn_embeddings_256_1gr128qk_20220217_in_gnn.pt \
 --model.embedding_dim=256 \
 --model.lr=0.001 \
 --data.batch_size=1024

qsub -N risk_mlp_in_gnn jobs/submit.sh scripts/mlp_risk.py fit \
 --config jobs/configs/risk/mlp_risk.yaml --config jobs/configs/in_gnn_data.yaml \
 --model.embedding_dim=256 \
 --model.lr=0.001 \
 --data.batch_size=1024

qsub -N risk_mlp jobs/submit.sh scripts/mlp_risk.py fit \
 --config jobs/configs/risk/mlp_risk.yaml --config jobs/configs/omop_data.yaml \
 --model.embedding_dim=256 --model.lr=0.001 \
 --data.batch_size=512 --model.use_weighted_loss=False

qsub -N risk_mlp_weighted jobs/submit.sh scripts/mlp_risk.py fit \
 --config jobs/configs/risk/mlp_risk.yaml --config jobs/configs/omop_data.yaml \
 --model.embedding_dim=256 --model.lr=0.001  --data.batch_size=512 --model.use_weighted_loss=True

#!/usr/bin/zsh
CENSORINGS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99)


DATAPATH=/lustre/home/rmhivau/ehrgnn/data/synthetic/pysurv_gaussian
# Partial likelihood
#for c in "${CENSORINGS[@]}"; do
#    python scripts/mlp.py fit \
#        --config jobs/configs/mlp.yaml \
#        --data.path "$DATAPATH"_"$c".pt
#done

#for c in ${CENSORINGS[@]}; do
#    python scripts/mlp.py fit \
#        --config jobs/configs/mlp.yaml \
#        --data.path "$DATAPATH"_"$c".pt \
#        --model.head_layers=1
#done

#Diffsort CE
#for c in ${CENSORINGS[@]}; do
#    python scripts/mlpdiffsort.py fit \
#        --config jobs/configs/mlpdiffsort.yaml \
#        --data.path "$DATAPATH"_"$c".pt
#done

for c in ${CENSORINGS[@]}; do
    python scripts/mlpdiffsort_synthetic.py fit \
        --config jobs/configs/mlpdiffsort_synthetic.yaml \
        --data.path "$DATAPATH"_"$c".pt \
        --model.head_layers=1
done

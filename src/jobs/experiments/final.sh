#!/bin/bash

python3 scripts/k_folds.py -m diffsort -k 5 -c jobs/configs/best/mlpdiffsort_flchain.yml -d results/mlpdiffsort_flchain/
python3 scripts/k_folds.py -m diffsort -k 5 -c jobs/configs/best/mlpdiffsort_support.yml -d results/mlpdiffsort_support/
python3 scripts/k_folds.py -m diffsort -k 5 -c jobs/configs/best/mlpdiffsort_nwtco.yml -d results/mlpdiffsort_nwtco/
python3 scripts/k_folds.py -m diffsort -k 5 -c jobs/configs/best/mlpdiffsort_metabric.yml -d results/mlpdiffsort_metabric/

python3 scripts/k_folds.py --k 5 -c jobs/configs/best/mlp_flchain.yml -m mlp -d results/mlp_flchain/
python3 scripts/k_folds.py --k 5 -c jobs/configs/best/mlp_support.yml -m mlp -d results/mlp_support/
python3 scripts/k_folds.py --k 5 -c jobs/configs/best/mlp_nwtco.yml -m mlp -d results/mlp_nwtco/
python3 scripts/k_folds.py --k 5 -c jobs/configs/best/mlp_metabric.yml -m mlp -d /results/mlp_metabric/

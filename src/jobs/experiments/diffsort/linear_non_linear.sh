#!/bin/bash -l

# Linear
# Non Linear
python scripts/mlp_synthetic.py fit \
    --config jobs/configs/mlp_synthetic.yaml \
    --data.path /lustre/home/rmhivau/ehrgnn/data/synthetic/nonlinear_exp_synthetic_no_censoring.pt

python scripts/mlpdiffsort_synthetic.py fit \
    --config jobs/configs/mlpdiffsort_synthetic.yaml \
    --data.path /lustre/home/rmhivau/ehrgnn/data/synthetic/nonlinear_exp_synthetic_no_censoring.pt

# Non Linear
python scripts/mlp_synthetic.py fit \
    --config jobs/configs/mlp_synthetic.yaml \
    --data.path /lustre/home/rmhivau/ehrgnn/data/synthetic/nonlinear_exp_synthetic_no_censoring.pt

python scripts/mlpdiffsort_synthetic.py fit \
    --config jobs/configs/mlpdiffsort_synthetic.yaml \
    --data.path /lustre/home/rmhivau/ehrgnn/data/synthetic/nonlinear_exp_synthetic_no_censoring.pt

python scripts/mlp_synthetic.py fit \
    --config jobs/configs/mlp_synthetic.yaml \
    --data.path /lustre/home/rmhivau/ehrgnn/data/synthetic/nonlinear_exp_synthetic_no_censoring.pt \
    --model.head=1

python scripts/mlpdiffsort_synthetic.py fit \
    --config jobs/configs/mlpdiffsort_synthetic.yaml \
    --data.path /lustre/home/rmhivau/ehrgnn/data/synthetic/nonlinear_exp_synthetic_no_censoring.pt \
    --model.head=1

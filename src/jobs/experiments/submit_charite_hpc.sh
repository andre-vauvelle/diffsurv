#!/bin/bash

#SBATCH --job-name=diffsurv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

SOURCE_DIR='/home/wildb/dev/projects/diffsurv/src'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
export WANDB_CACHE_DIR=/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/221031_diffsurv/wandb_cache/
export DATA_DIR=/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/221031_diffsurv/results/neurips_initial/data/
export RESULTS_DIR=/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/221031_diffsurv/results/neurips_rebuttal/results/
cd $SOURCE_DIR || exit

source ~/.bashrc
conda activate /home/wildb/envs/diffsurv/

echo Using command:
echo $@

$@

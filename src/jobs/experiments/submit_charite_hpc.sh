#!/bin/bash

#SBATCH --job-name=diffsurv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00


SOURCE_DIR='/home/wildb/dev/projects/diffsurv/src'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
export DATA_DIR=/sc-scratch/sc-scratch-ukb-cvd/diffsurv/data/
export RESULTS_DIR=/sc-scratch/sc-scratch-ukb-cvd/diffsurv/results/
cd $SOURCE_DIR || exit

source ~/.bashrc
# conda activate /home/wildb/envs/diffsurv/
conda activate /sc-projects/sc-proj-ukb-cvd/environments/multimodal/

echo Using command:
echo $@

$@

#!/bin/bash -l
#$ -l tmem=32G
#$ -l h_rt=50:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N ehrgnn_mlp

#$ -o /home/vauvelle/pycharm-sftp/ehrgnn/src/jobs/logs

hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/ehrgnn/src/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR || exit
source /share/apps/source_files/cuda/cuda-10.1.source
#source ~/.bashrc
conda activate

CONFIG_FILE=$1
python scripts/train_mlp.py fit --config="jobs/configs/default.yaml" --config="${CONFIG_FILE:=jobs/configs/mlp.yaml}"
#  --config="jobs/configs/phecode_phecode.yaml"
date

python scripts/train_mlp.py fit --config="jobs/configs/default.yaml" --config=jobs/configs/mlp.yaml --config="jobs/configs/phecode_phecode100.yaml"


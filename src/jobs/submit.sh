#!/bin/bash -l
#$ -l tmem=64G
#$ -l h_rt=50:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N submit

#$ -o /home/vauvelle/pycharm-sftp/ehrgnn/src/jobs/logs

hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/ehrgnn/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR/src/ || exit
source /share/apps/source_files/cuda/cuda-10.1.source
#source ~/.bashrc
conda activate

#python scripts/bert_mlm.py fit --config="${CONFIG_FILE:=jobs/configs/mlm/bert_phecode.yaml}"
echo Using command:
echo python $@

python $@
date

#qstat -j $JOB_ID

#!/bin/bash -l
#$ -l tmem=20G
#$ -l h_rt=50:0:0
#$ -S /bin/bash
#$ -j y
#$ -N sweep_agent_array
#$ -t 1-20
#$ -tc 10

#$ -o /home/vauvelle/pycharm-sftp/diffsurv/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/diffsurv/src/jobs/logs/errors

###$ -l gpu=true
hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/diffsurv/src'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR || exit
source /share/apps/source_files/cuda/cuda-11.2.source
source ~/.bashrc
conda activate

#python scripts/bert_mlm.py fit --config="${CONFIG_FILE:=jobs/configs/mlm/bert_phecode.yaml}"
echo Using command:
echo $@

$@

date

qstat -j $JOB_ID

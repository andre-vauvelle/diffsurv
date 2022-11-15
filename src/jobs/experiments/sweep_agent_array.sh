#!/bin/bash -l
#$ -l tmem=20G
#$ -l h_rt=10:0:0
#$ -S /bin/bash
#$ -l gpu=true
#$ -j y
#$ -N sweep_agent_array
#$ -t 1-10
#$ -tc 2

#$ -o /home/vauvelle/diffsurv/src/jobs/logs
#$ -e /home/vauvelle/diffsurv/src/jobs/logs/errors

hostname
date
SOURCE_DIR='/home/vauvelle/diffsurv/src'
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

#!/bin/bash -l
#$ -l tmem=64G
#$ -l h_rt=50:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N sweep_agent_array
#$ -t 1-20
#$ -tc 4

#$ -o /home/vauvelle/pycharm-sftp/diffsurv/src/jobs/logs

hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/diffsurv/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR/src/ || exit
source /share/apps/source_files/cuda/cuda-10.1.source
#source ~/.bashrc
conda activate

#python scripts/bert_mlm.py fit --config="${CONFIG_FILE:=jobs/configs/mlm/bert_phecode.yaml}"
echo Using command:
echo $@

$@

date

qstat -j $JOB_ID

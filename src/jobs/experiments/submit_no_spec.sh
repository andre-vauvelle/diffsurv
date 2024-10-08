#!/bin/bash -l
#$ -l tmem=32G
#$ -l h_rt=50:0:0
#$ -S /bin/bash
#$ -j y
#$ -N bert_mlm

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
echo python $@

python $@
date

qstat -j $JOB_ID

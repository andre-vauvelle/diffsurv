#!/bin/bash -l
#$ -l tmem=64G
#$ -l h_rt=50:0:0
#$ -l gpu=1
#$ -S /bin/bash
#$ -j y
#$ -N myriad_submit

#$ -o /home/vauvelle/pycharm-sftp/ehrgnn/src/jobs/logs

hostname
date
SOURCE_DIR='/home/rmhivau/ehrgnn/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR/src/ || exit
# load cuda
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/7.5.18/gnu-4.9.2
conda activate ehrgnn

#python scripts/bert_mlm.py fit --config="${CONFIG_FILE:=jobs/configs/mlm/bert_phecode.yaml}"
echo Using command:
echo python $@

python $@
date

qstat -j $JOB_ID

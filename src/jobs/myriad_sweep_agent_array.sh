#!/bin/bash -l
#$ -l mem=16G
#$ -l h_rt=30:00:00
#$ -S /bin/bash
#$ -l gpu=1
#$ -j y
#$ -N myriad_submit
#$ -t 1-10
#$ -P Gold 
#$ -A hpc.22
#$ -o /home/rmhivau/diffsurv/src/jobs/logs/
#$ -e /home/rmhivau/diffsurv/src/jobs/logs/errors/


hostname
date
SOURCE_DIR='/home/rmhivau/diffsurv'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR/src/ || exit
# load cuda
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load cuda/11.3.1/gnu-10.2.0

conda activate diffsurv


#python scripts/bert_mlm.py fit --config="${CONFIG_FILE:=jobs/configs/mlm/bert_phecode.yaml}"
echo Using command:
echo $@

wandb agent $@
date

qstat -j $JOB_ID

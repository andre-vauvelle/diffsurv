#!/bin/bash -l
#$ -l tmem=64G
#$ -l h_rt=50:0:0
#$ -l gpu=1
#$ -S /bin/bash
#$ -j y
#$ -N myriad_submit

#$ -o /home/vauvelle/diffsurv/src/jobs/logs

hostname
date
SOURCE_DIR='/home/rmhivau/diffsurv/'
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
echo python $@

python $@
date

qstat -j $JOB_ID

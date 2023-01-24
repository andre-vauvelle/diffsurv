#!/bin/bash -l
#$ -l tmem=20G
#$ -l h_rt=50:0:0
#$ -S /bin/bash
#$ -l gpu=true
#$ -j y
#$ -N sweep_agent_array
#$ -t 1-2
#$ -tc 2

#$ -o /home/vauvelle/diffsurv/src/jobs/logs
#$ -e /home/vauvelle/diffsurv/src/jobs/logs/errors

hostname
date
SOURCE_DIR='/home/vauvelle/diffsurv/src'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR || exit
source ~/.bashrc

# These are may need to be reset for local installs...
export PATH=/share/apps/python-3.9.5-shared/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda/bin:${PATH} # CUDA 11.5
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PATH=/home/vauvelle/.local/bin:${PATH}

#python scripts/bert_mlm.py fit --config="${CONFIG_FILE:=jobs/configs/mlm/bert_phecode.yaml}"
echo Using command:
echo $@

$@

date

qstat -j $JOB_ID

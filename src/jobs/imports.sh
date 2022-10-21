#!/bin/bash -l
hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/diffsurv/src'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR/src || exit
source /share/apps/source_files/cuda/cuda-11.2.source
source ~/.bashrc
conda activate

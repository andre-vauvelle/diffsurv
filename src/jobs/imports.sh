#!/bin/bash -l
hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/ehrgnn/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR/src || exit
source /share/apps/source_files/cuda/cuda-10.1.source
source ~/.bashrc
conda activate

#!/bin/bash -l
hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/diffsurv/src'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR

# These are may need to be reset for local installs...
export PATH=/share/apps/python-3.9.5-shared/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda/bin:${PATH} # CUDA 11.5
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PATH=/home/vauvelle/.local/bin:${PATH}

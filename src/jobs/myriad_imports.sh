#!/bin/zsh -l
hostname
date
# SOURCE project
SOURCE_DIR='/home/rmhivau/ehrgnn'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR/src/ || exit

# load cuda
bash
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load cuda/11.3.1/gnu-10.2.0
conda activate ehrgnn


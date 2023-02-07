#!/bin/bash -l
hostname
date
SOURCE_DIR='./src'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR

# Source python packages here

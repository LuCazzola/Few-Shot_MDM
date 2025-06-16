#!/bin/bash

#
# Make sure to produce converted skeleton files !
#
DATASET=$1
ROOT_DIR=$(pwd)

# Setup dataset directories in motion-diffusion-model
mkdir -p external/motion-diffusion-model/dataset/$DATASET

# Symlink the dataset directory
ln -s "$ROOT_DIR/data/$DATASET/" external/motion-diffusion-model/dataset/$DATASET/
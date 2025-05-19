#!/bin/bash

#
# Make sure to produce converted skeleton files !
#

DATASET=$1
ROOT_DIR=$(pwd)

# Setup dataset directories in motion-diffusion-model
mkdir -p external/motion-diffusion-model/dataset/$DATASET

# Symlink splits files
ln -s "$ROOT_DIR/data/$DATASET/splits" external/motion-diffusion-model/dataset/$DATASET/
# Symlink skeleton files
ln -s "$ROOT_DIR/data/$DATASET/annotations" external/motion-diffusion-model/dataset/$DATASET/
# Symlink texts data
ln -s "$ROOT_DIR/data/$DATASET/texts" external/motion-diffusion-model/dataset/$DATASET/
# Symlink train and test files
ln -s "$ROOT_DIR/data/$DATASET/class_captions.json" external/motion-diffusion-model/dataset/$DATASET/

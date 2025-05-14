#!/bin/bash

#
# Make sure to produce converted skeleton files !
#

DATASET=$1

# Setup dataset directories in motion-diffusion-model
mkdir -p external/motion-diffusion-model/dataset/$DATASET

# Symlink split files
ln -s modules/skel_adaptation/out/forw/*.txt external/motion-diffusion-model/dataset/$DATASET/
# Symlink skeleton files
ln -s modules/skel_adaptation/out/forw/annotations/ external/motion-diffusion-model/dataset/$DATASET/
# Symlink FewShot dir
ln -s data/$DATASET/fewshot_data/ external/motion-diffusion-model/dataset/$DATASET/
# Symlink texts data
ln -s data/$DATASET/texts/ external/motion-diffusion-model/dataset/$DATASET/
# Symlink train and test files
ln -s data/$DATASET/class_captions.json external/motion-diffusion-model/dataset/$DATASET/


#!/bin/bash

module load anaconda
#module load matlab/r2020b
#
source activate /scratch/work/molinee2/conda_envs/unet_env

n=1
PATH_EXPERIMENT=/scratch/work/molinee2/unet_dir/unet_denoising_github/experiments/${n}
mkdir $PATH_EXPERIMENT

python  train.py path_experiment="$PATH_EXPERIMENT"  epochs=150 freq_inference=50 batch_size=2 steps_per_epoch=2000

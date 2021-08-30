#!/bin/bash

module load anaconda

source activate /scratch/work/molinee2/conda_envs/unet_env


python inference.py path_experiment="experiments/trained_model" inference.audio=$1


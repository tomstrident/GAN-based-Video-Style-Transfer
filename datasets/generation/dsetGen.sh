#!/bin/bash
#SBATCH --job-name=dsetGen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/dsetGen
python datagen.py
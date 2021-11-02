#!/bin/bash
#SBATCH --job-name=LBST3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/LBST
python demo3.py
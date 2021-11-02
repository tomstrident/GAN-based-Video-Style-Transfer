#!/bin/bash
#SBATCH --job-name=LBST2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/LBST
python demo2.py
#!/bin/bash
#SBATCH --job-name=LBST
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/LBST
python demo.py
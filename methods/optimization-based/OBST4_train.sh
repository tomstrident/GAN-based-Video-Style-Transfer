#!/bin/bash
#SBATCH --job-name=OBST4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

rsync -a ~/datasets ~/local/

cd ~/projects/OBST
python obst_eval.py --mode "sintel" --weight_tcl 2000
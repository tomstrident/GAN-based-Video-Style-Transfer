#!/bin/bash
#SBATCH --job-name=OBST3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

rsync -a ~/datasets ~/local/

cd ~/projects/OBST
python obst_eval.py --mode "sintel" --weight_tcl 0
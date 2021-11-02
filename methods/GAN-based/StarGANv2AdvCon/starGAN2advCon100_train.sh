#!/bin/bash
#SBATCH --job-name=starGANv2advCon100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/starGANv2advCon100
python main.py --mode 'eval' --num_domains 4 --batch_size 16 --num_workers 4
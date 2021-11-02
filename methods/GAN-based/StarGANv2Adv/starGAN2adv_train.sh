#!/bin/bash
#SBATCH --job-name=starGANv2adv
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/starGANv2adv
python main.py --mode 'train' --num_domains 4 --batch_size 16 --num_workers 4
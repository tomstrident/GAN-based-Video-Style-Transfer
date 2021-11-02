#!/bin/bash
#SBATCH --job-name=StarGAN
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

rsync -a ~/datasets ~/local/

cd ~/projects/StarGAN
python main.py --batch_size 16 --num_workers 4
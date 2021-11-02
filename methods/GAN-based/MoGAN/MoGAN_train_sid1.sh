#!/bin/bash
#SBATCH --job-name=MoGAN_S1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/MoGAN
python train.py --name cycleGAN_fc2_sid1 --sid 1 --model cycle_gan --batch_size 8 --num_threads 4 --display_id 0 --serial_batches --no_flip
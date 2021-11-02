#!/bin/bash
#SBATCH --job-name=MoGAN_S2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/MoGAN
python train.py --name cycleGAN_fc2_sid2 --sid 2 --model cycle_gan --batch_size 8 --num_threads 4 --display_id 0 --serial_batches --no_flip
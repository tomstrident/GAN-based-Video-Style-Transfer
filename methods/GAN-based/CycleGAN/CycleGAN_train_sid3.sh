#!/bin/bash
#SBATCH --job-name=CycleGAN_S3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/CycleGAN
python train.py --name cycleGAN_fc2_sid3 --sid 3 --model cycle_gan --batch_size 16 --num_threads 4 --display_id 0 --serial_batches --no_flip
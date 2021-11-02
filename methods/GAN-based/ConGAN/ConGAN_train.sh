#!/bin/bash
#SBATCH --job-name=ConGAN
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/ConGAN
python train.py --image_dir ./datasets/FC2/DATAFiles/ --style_dir ./datasets/FC2/styled-files/ --name cycleGAN_fc2 --model cycle_gan --batch_size 8 --num_threads 4 --display_id 0 --serial_batches --no_flip
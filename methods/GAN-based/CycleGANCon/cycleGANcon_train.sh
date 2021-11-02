#!/bin/bash
#SBATCH --job-name=cycleGAN_fc2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

cd ~/projects/cycleGANcon
python train.py --image_dir ./datasets/FC2/DATAFiles/ --style_dir ./datasets/FC2/styled-files/ --name cycleGAN_fc2 --model cycle_gan --batch_size 16 --num_threads 4 --display_id 0 --serial_batches --no_flip
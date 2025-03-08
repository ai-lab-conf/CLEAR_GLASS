#!/bin/bash

#SBATCH -n 6
#SBATCH --mem=120g
#SBATCH -t 12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH -o my-output-%j.out

echo "Beginning finetuning"

# Set a random port to avoid conflicts
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    -m training.main \
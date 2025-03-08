#!/bin/bash

#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -t 48:00:00
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH -o my-output-%j.out

echo "Beginning finetuning"


export MASTER_PORT=$(shuf -i 10000-65535 -n 1)

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    -m training.run_sweep \
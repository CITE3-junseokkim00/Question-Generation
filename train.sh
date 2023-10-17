#!/bin/sh

#SBATCH -J train.sh
#SBATCH -p titanxp
#SBATCH --gres=gpu:2
#SBATCH -o train.out

srun python3 train.py
#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH --gpus-per-node=1           # Specify number of GPUs
#SBATCH -N 1 -c 16                  # Specify number of nodes and CPUs
#SBATCH -t 1:00:00                  # Specify maximum time limit (hour:minute:second)
#SBATCH -A zz992003                 # Specify project name
#SBATCH -J Python_Script            # Specify job name
#SBATCH -o python-%j.out            # Output log file

module reset
ml Mamba
# module load Mamba/23.11.0-0
# module load cuda/12.6
conda deactivate
conda activate /project/zz992000-zdevb/TA_Arther_Test_Script/env

python "${1:-script.py}"

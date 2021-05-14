#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1

module purge
module load pytorch/1.8

export DATADIR=/scratch/dac/data
export TORCH_HOME=/scratch/dac/mvsjober/torch-cache

set -xv

python3 $*

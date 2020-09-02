#!/bin/bash
#SBATCH --job-name="Point-Cloud Conversion"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

source ~/venv_o3d/bin/activate
srun python -m dataloading.data_preparation

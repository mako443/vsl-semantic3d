#!/bin/bash
#SBATCH --job-name="Point-Cloud Merging"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

srun python3 -m dataloading.scene_merging combine

#!/bin/bash
#SBATCH --job-name="S3D VSE-CO training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=32G
#SBATCH --time=3:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

srun python3 -m visual_semantic.train_co "$@"

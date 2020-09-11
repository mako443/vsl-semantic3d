#!/bin/bash
#SBATCH --job-name="S3D Eval"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

srun python3 -m evaluation.evaluation netvlad+sg
#srun python3 -m evaluation.evaluation vse
#srun python3 -m evaluation.evaluation scenegraphs

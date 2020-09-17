#!/bin/bash
#SBATCH --job-name="PIT NV Eval"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

srun python3 pytorch-NetVlad/main.py --mode=test --split=val --resume=vgg16_netvlad_checkpoint/
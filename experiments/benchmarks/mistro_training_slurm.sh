#!/bin/bash
#SBATCH --gres=gpu:5
#SBATCH --mem=32gb
#SBATCH -o results/mistro_FinQASA_training.log
#SBATCH -e results/mistro_FinQASA_training.err
#SBATCH -t 4:00:00
#SBATCH --chdir=/gpfs/u/home/FNAI/FNAIhrnb/barn/DP-LoRA/experiments/experiments/benchmarks

# Activate the Conda Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DPLoRA

# Set the environment variable before running the script
export TRANSFORMERS_CACHE=/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface
export HF_DATASETS_CACHE=/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface
export HF_HOME=/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface

# Create the cache directory if doesn't exist
mkdir -p /gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface


python mistro_finQASA_training.py



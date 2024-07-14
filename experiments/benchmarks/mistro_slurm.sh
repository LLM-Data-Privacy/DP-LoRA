#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --mem=32gb
#SBATCH -o benchmark_log/mistro_FinQASA.log
#SBATCH -e benchmark_log/mistro_FinQASA.err
#SBATCH -t 4:00:00
#SBATCH --chdir=/gpfs/u/home/FNAI/FNAIhrnb/barn/DP-LoRA/experiments/benchmarks

# Activate the Conda Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DPLoRA

# Set the environment variable before running the script
export TRANSFORMERS_CACHE=/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface
export HF_DATASETS_CACHE=/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface
export HF_HOME=/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface

# Create the cache directory if doesn't exist
mkdir -p /gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface


python mistro_finQASA.py



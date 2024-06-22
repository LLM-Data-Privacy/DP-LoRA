#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --mem=32gb
#SBATCH -o benchmark_log/mistral_benchmark.log
#SBATCH -e benchmark_log/mistral_benchmark.err
#SBATCH -t 4:00:00
#SBATCH --chdir=/gpfs/u/home/FNAI/FNAIkqbe/barn/kaiqi_bei

# Activate the Conda Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DPLoRA

# Set the environment variable before running the script
export TRANSFORMERS_CACHE=/gpfs/u/home/FNAI/FNAIkqbe/scratch/huggingface
export HF_DATASETS_CACHE=/gpfs/u/home/FNAI/FNAIkqbe/scratch/huggingface
export HF_HOME=/gpfs/u/home/FNAI/FNAIkqbe/scratch/huggingface

# Create the cache directory if doesn't exist
mkdir -p /gpfs/u/home/FNAI/FNAIkqbe/scratch/huggingface


python mistral_benchmark.py
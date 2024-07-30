#benchmark mistral-7b on ACL18 dataset

import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
from datasets import load_dataset, Dataset
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import re

# Set up environment variables for caching
os.environ['TRANSFORMERS_CACHE'] = '/gpfs/u/home/FNAI/FNAIdosa/scratch/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/gpfs/u/home/FNAI/FNAIdosa/scratch/huggingface'
os.environ['HF_HOME'] = '/gpfs/u/home/FNAI/FNAIdosa/scratch/huggingface'

# Hugging Face setup
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")
login(api_token)

# Ensure directories exist
os.makedirs('/gpfs/u/home/FNAI/FNAIdosa/scratch/huggingface', exist_ok=True)

# for logs
logging.set_verbosity_error()

# Load the entire dataset
try:
    dataset = load_dataset("TheFinAI/flare-sm-acl", split="test")
    print(dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print(dataset)
print(dataset.column_names)
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
    dataset = load_dataset("TheFinAI/flare-ma", split="test")
    print(dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print(dataset)
print(dataset.column_names)

# Preprocess prompt function
def prep_prompt(row):
    prompt = """In this task, you will be given Mergers and Acquisitions (M&A) news articles or tweets.\n 
                Your task is to classify each article or tweet based on whether the mentioned deal was completed or remained a rumour.\n 
                Your response should be a single word - either 'complete' or 'rumour' - representing the outcome of the deal mentioned in the provided text.\n
                Enclose your output with the token [OTPT] and [CTPT]."""
    profile = row['text']
    answer = "\nAnswer: "
    return prompt + profile + answer

def derive_answer(output_raw):
    match = re.search(r'\[OTPT\](.*?)\[CTPT\]', output_raw)
    if match:
        answer = match.group(1).strip().lower()
        if answer in {"rumour", "complete"}:
            return answer
    output_raw = output_raw.strip().lower()
    if "rumour" in output_raw:
        return "rumour"
    if "complete" in output_raw:
        return "complete"
    return None
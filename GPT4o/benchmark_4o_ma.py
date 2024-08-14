# Benchmark GPT-4o on MA dataset

import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from huggingface_hub import login
import re
import openai

# Hugging Face setup
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")
login(api_token)

# Load the entire dataset
try:
    dataset = load_dataset("TheFinAI/flare-ma", split="test")
    print(dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Preprocess prompt function
def prep_prompt(row):
    prompt = """In this task, you will be given Mergers and Acquisitions (M&A) news articles or tweets.\n 
                Your task is to classify each article or tweet based on whether the mentioned deal was \n
                completed or remained a rumour. Your response should be a single word - either 'complete'\n
                or 'rumour' - representing the outcome of the deal mentioned in the provided text.\n
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
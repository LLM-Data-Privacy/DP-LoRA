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
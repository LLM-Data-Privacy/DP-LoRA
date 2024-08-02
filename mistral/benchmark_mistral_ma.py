# benchmark mistral-7b on MA dataset

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

# Load model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

# Ensure tokenizer uses a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

# Convert test dataset to pandas DataFrame
test_data = dataset.to_pandas()
print(test_data.head())

# Prepare test prompt
test_data['prompt'] = test_data.apply(prep_prompt, axis=1)
test_prompts = test_data['prompt']

# Generate predictions
outputs = []
for prompt in test_prompts:
    batch_outputs_raw = generator(prompt, max_new_tokens=50, return_full_text=False)
    if isinstance(batch_outputs_raw, list):
        if isinstance(batch_outputs_raw[0], dict):
            flat_outputs = batch_outputs_raw
        else:
            flat_outputs = [{'generated_text': item} for item in batch_outputs_raw]
    elif isinstance(batch_outputs_raw, dict):
        flat_outputs = [batch_outputs_raw]
    else:
        flat_outputs = [{'generated_text': str(batch_outputs_raw)}]

    batch_outputs = [derive_answer(output['generated_text']) for output in flat_outputs]
    outputs.extend(batch_outputs)

# Ground truth
ground_truth = test_data['label'].tolist()

# Ensure valid labels
valid_labels = {"rumour", "complete"}
outputs = [o for o in outputs if o in valid_labels]
ground_truth = [g for g in ground_truth if g in valid_labels]

if len(outputs) == 0:
    print("No valid outputs found.")
else:
    # Ensure both lists have the same length
    min_len = min(len(outputs), len(ground_truth))
    outputs = outputs[:min_len]
    ground_truth = ground_truth[:min_len]
    
    # Evaluate scores
    accuracy = sum([1 for i in range(len(outputs)) if outputs[i] == ground_truth[i]]) / len(outputs)
    f1_micro = f1_score(ground_truth, outputs, average='micro')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score (Micro): {f1_micro}")

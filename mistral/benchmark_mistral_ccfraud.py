# benchmark mistral-7b on credit card fraud dataset

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
import re

# huggingface setup
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")
login(api_token)


# setup env variables
os.environ['TRANSFORMERS_CACHE'] = '/gpfs/u/home/FNAI/FNAIdosa/scratch/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/gpfs/u/home/FNAI/FNAIdosa/scratch/huggingface'
os.environ['HF_HOME'] = '/gpfs/u/home/FNAI/FNAIdosa/scratch/huggingface'

# ensure dir exists
os.makedirs('/gpfs/u/home/FNAI/FNAIdosa/scratch/huggingface', exist_ok=True)

# load dataset
dataset = load_dataset("daishen/cra-ccfraud", split='test')

#preprocess prompt function
def prep_prompt(row):
    prompt = """Detect the credit card fraud with the following financial profile.\n
              Respond with only 'good' or 'bad', and do not provide any additional information.\n
              enclose your output with the token [OTPT] and [CTPT]."""
    profile = f"""The client is a {row['sex']}, the state number is {row['state']}, 
               the number of cards is {row['cards']}, the credit balance is {row['balance']}, 
               the number of transactions is {row['transactions']}, 
               the number of international transactions is {row['international_transactions']}, 
               the credit limit is {row['limit']}"""
    answer = "\nAnswer: "

    return prompt + profile + answer

def derive_answer(output_raw):
    match = re.search(r'\[OTPT\](.*?)\[CTPT\]', output_raw)
    if match:
        answer = match.group(1).strip().lower()
        if answer in {"good", "bad"}:
            return answer
    else:
        return None

# load model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# check for padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# prepare test prompt
test_data = dataset.to_pandas()
test_prompts = test_data().apply(prep_prompt, axis=1)

# generate predictions
outputs = []

for prompt in test_prompts:
    output_raw = generator(prompt, max_length=100)[0]['generated_text']
    output = derive_answer(output_raw)
    outputs.append(output)

# ground truth
ground_truth = test_data['gold']

# remove None values
valid_outputs = [o for o in outputs if o is not None]
valid_ground_truth = [g for o, g in zip(outputs, ground_truth) if o is not None]

# evaluate scores
accuracy = sum([1 for i in range(len(valid_outputs)) if valid_outputs[i] == valid_ground_truth[i]]) / len(valid_outputs)
f1_macro = f1_score(valid_ground_truth, valid_outputs, average='macro')
f1_micro = f1_score(valid_ground_truth, valid_outputs, average='micro')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_macro}")
print(f"F1 Macro Score: {f1_macro}")
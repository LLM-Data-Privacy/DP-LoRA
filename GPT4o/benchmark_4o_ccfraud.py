# Benchmark GPT-4o on ccfraud dataset

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
    dataset = load_dataset("TheFinAI/cra-ccfraud", split="test")
    print(dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Preprocess prompt function
def prep_prompt(row):
    prompt = """Detect the credit card fraud with the following financial profile.\n
              Respond with only 'good' or 'bad', and do not provide any additional information.\n
              Enclose your output with the token [OTPT] and [CTPT]."""
    profile = row['text']
    answer = "\nAnswer: "
    return prompt + profile + answer

def derive_answer(output_raw):
    match = re.search(r'\[OTPT\](.*?)\[CTPT\]', output_raw)
    if match:
        answer = match.group(1).strip().lower()
        if answer in {"good", "bad"}:
            return answer
    output_raw = output_raw.strip().lower()
    if "good" in output_raw:
        return "good"
    if "bad" in output_raw:
        return "bad"
    return None

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_TOKEN")

def get_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0]['message']['content']

# Convert test dataset to pandas DataFrame
test_data = dataset.to_pandas()
print(test_data.head())

# Prepare test prompt
test_data['prompt'] = test_data.apply(prep_prompt, axis=1)
test_prompts = test_data['prompt']

# Generate predictions
outputs = []
for prompt in test_prompts:
    output_raw = get_response(prompt)
    output = derive_answer(output_raw)
    outputs.append(output)

# Ground truth
ground_truth = test_data['answer'].tolist()

if len(outputs) == 0:
    print("No valid outputs found.")
else:
    # Ensure both lists have the same length
    min_len = min(len(outputs), len(ground_truth))
    outputs = outputs[:min_len]
    ground_truth = ground_truth[:min_len]
    
    # Evaluate scores
    accuracy = sum([1 for i in range(len(outputs)) if outputs[i] == ground_truth[i]]) / len(outputs)
    f1_macro = f1_score(ground_truth, outputs, average='macro')
    f1_micro = f1_score(ground_truth, outputs, average='micro')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score (Macro): {f1_macro}")
    print(f"F1 Score (Micro): {f1_micro}")

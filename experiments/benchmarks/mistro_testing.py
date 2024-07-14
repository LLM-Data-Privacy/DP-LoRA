from sklearn.metrics import f1_score
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
from datasets import load_dataset
import os
import sys

# Set the environment variables to avoid disk caching
os.environ['TRANSFORMERS_CACHE'] = '/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface'
os.environ['HF_HOME'] = '/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface'

# Ensure the directory exists
os.makedirs('/gpfs/u/home/FNAI/FNAIhrnb/scratch/huggingface', exist_ok=True)

# Verify environment variables
print("TRANSFORMERS_CACHE:", os.getenv('TRANSFORMERS_CACHE'))
print("HF_DATASETS_CACHE:", os.getenv('HF_DATASETS_CACHE'))
print("HF_HOME:", os.getenv('HF_HOME'))

# Log in to the Hugging Face Hub
TOKEN = "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"
login(token=TOKEN)

# preprocess the dataset from the benchmark
def preprocess_prompt(text):
    prompt = "Please classify the sentiment of the following text, just output a floating point, scaling from -1 to 1, -1 is negative, 0 is neutral, 1 is positive: \n"

    # return combined_inputs
    return prompt + text + "\nAnswer: "

# set up the threshold for the sentiment classification
def sentiment_classification(output):
    if output < -0.5:
        return -1
    elif output > 0.5:
        return 1
    else:
        return 0
    

if __name__ == '__main__':
    # Taking argument from commandline as the sentence entry
    # entry = int(sys.argv[1])
    
    # Load the model and tokenizer
    # Load Mistro-7b model
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").to("cuda")
    generater = pipeline('text-generation', model=model, tokenizer=tokenizer)

    # Load the FinQASA dataset for benchmarking
    dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
    dataset = dataset['test']

    # Generate the prompt for the dataset
    sentences = dataset['sentence']
    for sentence in sentences:
    	sample_prompt = preprocess_prompt(sentence)
    
    	sample_output = generater(sample_prompt, max_length=100)
    	print(sample_output)


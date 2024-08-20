from sklearn.metrics import f1_score
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
from datasets import load_dataset
import os
import sys

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
    # Log in to the Hugging Face Hub
    TOKEN = "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"
    login(token=TOKEN)
    
    
    # Load the model and tokenizer
    # Load Llama3 model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    generater = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    # Load the FinQASA dataset for benchmarking
    dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
    dataset = dataset['test']
    
    # Generate the prompt for the dataset
    sentences = dataset['sentence']
    for sentence in sentences:
        sample_prompt = preprocess_prompt(sentence)
        sample_output = generater(sample_prompt, max_length=200)
        print(sample_output)
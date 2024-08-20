from sklearn.metrics import f1_score
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
from datasets import load_dataset
import os
import re

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

# Extract the answer from the output, if not found return -1
def extract_answer(output_raw):
    try:
        output = float(re.findall(r'Answer: (.*)', output_raw)[0])
    except:
        output = -1
    return output
    

if __name__ == '__main__':
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
    
    outputs = []
    
    for text in sentences:
        messages = preprocess_prompt(text)
        output_raw = generater(messages, max_length=100)[0]['generated_text']
        
        # Only keep the field after "Answer: "
        output = extract_answer(output_raw)
        outputs.append(output)
    
    # Store the prediction to new csv file
    # with open('predictions.csv', 'w') as f:
    #    for output in outputs:
    #        f.write(output + '\n')

    
    # outputs = [float(output.split("Answer: ")[1]) for output in outputs]
        
    # Transform the output to sentiment classification
    output_label = [sentiment_classification(output) for output in outputs]
    ground_truth = [sentiment_classification(score) for score in dataset['score']]
    
    # Evaluate the Accuracy and F1 score
    accuracy = sum([1 for i in range(len(output_label)) if output_label[i] == ground_truth[i]]) / len(output_label)
    f1_macro = f1_score(ground_truth, output_label, average='macro')
    f1_micro = f1_score(ground_truth, output_label, average='micro')
    
    print("Accuracy: ", accuracy)
    print("F1 Score (Macro): ", f1_macro)
    print("F1 Score (Micro): ", f1_micro)
    
    

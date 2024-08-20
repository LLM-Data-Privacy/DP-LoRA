from sklearn.metrics import f1_score
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from datasets import load_dataset
import os

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
    combined_inputs = [ {"role": "system", "content": "Please classify the sentiment of the following text, just output a floating point, scaling from -1 to 1, -1 is negative, 0 is neutral, 1 is positive: \n"},
                        {"role": "user", "content": "Britain's FTSE steadies, supported by Dixons Carphone"}]
    
    return combined_inputs

# set up the threshold for the sentiment classification
def sentiment_classification(output):
    if output < -0.5:
        return -1
    elif output > 0.5:
        return 1
    else:
        return 0
    

if __name__ == '__main__':
    # Load the model and tokenizer
    # Load Llama3-8B model
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",torch_dtype=torch.bfloat16)

    # Load the FinQASA dataset for benchmarking
    dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
    dataset = dataset['test']

    # Generate the prompt for the dataset
    sentences = dataset['sentence']
    
    outputs = []
    
    for text in sentences:
        messages = preprocess_prompt(text)
        model_inputs = tokenizer.apply_chat_template(messages,
                                                     add_generation_prompt=True,
                                                     return_tensors='pt').to("cuda")
        generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)     
        output = tokenizer.batch_decode(generated_ids)[0]
        outputs.append(output)
    
    outputs = [float(output) for output in outputs]
    
    # Store the prediction to new csv file
    with open('llama3_predictions.csv', 'w') as f:
        f.write("sentence, prediction\n")
        for i in range(len(sentences)):
            f.write(sentences[i] + "," + str(outputs[i]) + "\n")
            
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
    
    

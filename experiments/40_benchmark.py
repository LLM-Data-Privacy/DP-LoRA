import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from sklearn.metrics import f1_score
import time


TOKEN = "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"
login(token = TOKEN)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN).to(device) 
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN)

tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
    from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("TheFinAI/flare-fpb",token = TOKEN)


print(dataset.column_names)
def preprocess_function(examples):
    combined_texts = [q + " [SEP] " + t for q, t in zip(examples['query'], examples['text'])]
    return tokenizer(combined_texts, truncation=True, max_length=512)

processed_dataset = dataset.map(preprocess_function, batched=True)
print(processed_dataset['train'][0:2])
input_ids = processed_dataset['train']['input_ids']
attention_mask = processed_dataset['train']['attention_mask']



def compute_f1(predictions, references):
    f1 = f1_score(references, predictions, average='weighted')
    return f1

inputs = torch.tensor(input_ids)
masks = torch.tensor(attention_mask)

start_time = time.time()
with torch.no_grad():
    outputs = model(inputs, attention_mask=masks)
    predictions = torch.argmax(outputs.logits, dim=-1)
end_time = time.time()

elapsed_time = end_time - start_time

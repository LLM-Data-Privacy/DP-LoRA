from sklearn.metrics import f1_score
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import time
TOKEN = "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"
login(token = TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_function(examples):
    combined_texts = [q + " [SEP] " + t for q, t in zip(examples['query'], examples['text'])]
    return tokenizer(combined_texts, truncation=True, padding="max_length", max_length=512)
def compute_f1(predictions, references):
    f1 = f1_score(references, predictions, average='weighted')
    return f1

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    
    input_ids = [torch.tensor(ids) for ids in input_ids]
    attention_masks = [torch.tensor(mask) for mask in attention_masks]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded
    }


model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN).to(device) 
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer)) 

    from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("TheFinAI/flare-cd",token = TOKEN)
processed_dataset = dataset.map(preprocess_function, batched=True)
input_ids = processed_dataset['test']['input_ids']
attention_mask = processed_dataset['test']['attention_mask']


inputs = torch.tensor(input_ids)
masks = torch.tensor(attention_mask)
data = [{'input_ids': ids, 'attention_mask': mask} for ids, mask in zip(input_ids, attention_mask)]
loader = torch.utils.data.DataLoader(data, batch_size=10, collate_fn=collate_fn, shuffle=False)


start_time = time.time()
index = 0
for batch in loader:
    inputs = batch['input_ids']
    masks = batch['attention_mask']
    with torch.no_grad():
        outputs = model(inputs, attention_mask=masks)
        predictions = torch.argmax(outputs.logits, dim=-1)
    print(predictions)
    if index == 0:
        break
    
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time for inference: {elapsed_time} seconds")

actual_labels = dataset['train']['gold'] 
f1 = compute_f1(predictions.numpy(), actual_labels)

print(f"F1 Score: {f1}")
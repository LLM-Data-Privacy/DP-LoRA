import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from datasets import load_dataset
from sklearn.metrics import f1_score
import torch, time


def compute_f1(TP,FP,FN):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    return f1

def compute_prediction(predictions,references):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(predictions)):
        pred = predictions[i]
        ref = references[i]
        for j in range(len(pred)):
            if pred[j] == 1 and ref[j] == 1:
                TP += 1
            elif pred[j] == 1 and ref[j] == 0:
                FP += 1
            elif pred[j] == 0 and ref[j] == 1:
                FN += 1
    return TP,FP,FN

TOKEN = "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"
login(token = TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN).to(device) 
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN)
dataset = load_dataset("TheFinAI/flare-cd",token = TOKEN)
dataset_iter = dataset['test'].iter(1)


start_time = time.time()
# Prepare the message
message = [{"role": "user", "content": ""}]
while True:
    try:
        data = next(dataset_iter)
        query = data['query'][0]
        message[0]["content"] = query

        
        encodeds = tokenizer.apply_chat_template(message, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        print(decoded[0])
    except StopIteration:
        break

end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

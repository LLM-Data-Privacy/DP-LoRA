import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from datasets import load_dataset
from sklearn.metrics import f1_score
import torch, time
import re
def parse_label(answer):
    answer_part = re.sub(r"\[INST\].*?\[/INST\]", "", answer, flags=re.DOTALL)

    # 解析token和label
    labels = []
    lines = answer_part.splitlines()
    for line in lines:
        if ":" in line:
            labels.append(line.split(":")[-1])
        else:
            break
    return labels

def F1_score(correct_labels, api_labels):
    # TP, FP, FN
    tp = sum(1 for pred in api_labels if pred in correct_labels)
    fp = sum(1 for pred in api_labels if pred not in correct_labels)
    fn = sum(1 for correct in correct_labels if correct not in api_labels)
    individual_f1 = []
    for cor in correct_labels:
        if cor in api_labels:
            individual_f1.append(1)
        else:
            individual_f1.append(0)
    macro_f1 = sum(individual_f1) / len(individual_f1) if individual_f1 else 0

    return tp, fp, fn, macro_f1

tp, fp, fn = 0, 0, 0
macro_f1 = []
iter_count = 0
TOKEN = "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"
login(token = TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Using device:", device)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN).to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN)
dataset = load_dataset("TheFinAI/flare-finred",token = TOKEN)
dataset_iter = dataset['test'].iter(1)

#pad_token
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token")
    tokenizer.pad_token = tokenizer.eos_token

start_time = time.time()
# Prepare the message
message = [{"role": "user", "content": ""}]

iteration =0
try:
    while True:
        data = next(dataset_iter)
        query = data['query'][0]
        answer = data['answer'][0]
        message[0]["content"] = query

        encodeds = tokenizer.apply_chat_template(message, return_tensors="pt")
        model_inputs = encodeds.to(device)


        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        cleaned_text  = re.sub(r'\[INST\].*?\[/INST\]', '', decoded[0], flags=re.DOTALL)
        first_line = cleaned_text.split("\n")[0]
        if ';' not in first_line:
            try:
                first_line = cleaned_text.split("\n")[1]
            except:
                iteration += 1
                print("ERROR: ", cleaned_text)
                print()
                continue
        first_line     = re.sub(r'^\s*\d+\.\s*', '', first_line)
        API_labels     = [item.strip() for item in first_line.split(';')]
        answer_labels  = [item.strip() for item in answer.split(';')]
        # 分割三元组并存入列表
        print("Iteration: {}".format(iteration))
        print("API      : " + str(API_labels))
        print("Correct  : " + str(answer_labels))
        print()
        tp_, fp_, fn_, macro_f1_ = F1_score(answer_labels, API_labels)
        tp += tp_
        fp += fp_
        fn += fn_
        macro_f1.append(macro_f1_)
        iteration += 1



except StopIteration:

    end_time = time.time()
    print("Processing complete.")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # 计算 F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("Micro F1: ", f1_score)
    print("Macro F1: ", sum(macro_f1) / len(macro_f1))

except Exception as e:
    print("An error occurred:", e)
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from datasets import load_dataset
from sklearn.metrics import f1_score
import torch, time
import re
tp, fp, fn = 0, 0, 0
label_set = set("B-CAUSE I-CAUSE B-EFFECT I-EFFECT O".split())
statistics = {
        "B-CAUSE": {'tp': 0, 'fp': 0, 'fn': 0},
        "I-CAUSE": {'tp': 0, 'fp': 0, 'fn': 0},
        "B-EFFECT": {'tp': 0, 'fp': 0, 'fn': 0},
        "I-EFFECT": {'tp': 0, 'fp': 0, 'fn': 0},
        "O": {'tp': 0, 'fp': 0, 'fn': 0}
}
def calculate_macro_f1(statistics):
    f1_scores = []
    for label, counts in statistics.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return macro_f1

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

def F1_score(correct_labels, api_labels,statistics):
    


    # Initialize overall tp, fp, fn
    overall_tp, overall_fp, overall_fn = 0, 0, 0

    # Iterate through both label lists
    for correct, predicted in zip(correct_labels, api_labels):
        correct = correct.strip().strip('.')
        predicted = predicted.strip().strip('.')
        if correct in statistics:
            # Count true positives and false negatives
            if predicted in correct:
                statistics[correct]['tp'] += 1
                overall_tp += 1
            else:
                statistics[correct]['fn'] += 1
                overall_fn += 1
                
        if predicted in statistics:
            # Count false positives
            if predicted != correct:
                statistics[predicted]['fp'] += 1
                overall_fp += 1

    return overall_tp, overall_fp, overall_fn, statistics

TOKEN = "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"
login(token = TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Using device:", device)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN).to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",token = TOKEN)
dataset = load_dataset("TheFinAI/flare-cd",token = TOKEN)
dataset_iter = dataset['test'].iter(1)

#pad_token
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token")
    tokenizer.pad_token = tokenizer.eos_token

start_time = time.time()
# Prepare the message
message = [{"role": "user", "content": ""}]
iter_count = 0
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
        answer_labels = parse_label(decoded[0])
        answer_labels = [label.strip().strip('.') for label in answer_labels if label.strip().strip('.') in label_set]
        correct_labels = parse_label(answer)
        tp_, fp_, fn_, statistics = F1_score(correct_labels, answer_labels,statistics)
        tp += tp_
        fp += fp_
        fn += fn_
        print("Iteration: {}".format(iter_count))

        print("CORRECT: {}\n\
API    : {}\n".format(correct_labels, answer_labels))
        iter_count += 1
        # if iter_count >10:
        #     raise StopIteration


except StopIteration:

    end_time = time.time()
    print("Processing complete.")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # 计算 F1 Score
    micro_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    macro_f1 = calculate_macro_f1(statistics)
    print("micro_f1: ", micro_f1)
    print("macro_f1: ", macro_f1)

except Exception as e:
    print("An error occurred:", e)
import sys
import os

os.environ['HF_DATASETS_CACHE'] = '/gpfs/u/home/FNAI/FNAIchpn/scratch/huggingface'
os.environ['HF_HOME'] = '/gpfs/u/home/FNAI/FNAIchpn/scratch/huggingface'

import pandas as pd

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from config import *

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import datasets
import torch

from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

def parseData():
    instructions = load_dataset("TheFinAI/flare-multifin-en")
    test = instructions["test"].to_pandas()
    test['query'] = test['query'].apply(format)
    query = test['query'].tolist()
    gt = test['answer'].tolist()
    return test, query, gt

def get_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    login(token=token, add_to_git_credential=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Model loaded successfully")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded successfully")

    return model, tokenizer

def format(x):
    halves = x.split("\nText: ")

    test = "In this task, you're working with English headlines from the MULTIFIN dataset. " \
            "This dataset is made up of real-world article headlines from a large accounting firm's websites. " \
            "Your objective is to categorize each headline according to its primary topic. The potential categories are " \
            "'Finance', 'Technology', 'Tax & Accounting', 'Business & Management', 'Government & Controls', and 'Industry'. " \
            "Provide your answer as only one of the following categories: Finance, Technology, Tax & Accounting, Business & Management, " \
            "Government & Controls, and Industry. Do not provide an answer not from the previously mentioned categories. Choose only one of the categories." \
            "\nText: " + halves[1]

    return test

def change_target(x):
    ans = [ "finance", "technology", "tax & accounting", "business & management", "government & controls", "industry" ]
    x = x.lower()
    if x in ans:
        return x
    elif x == "tax" or x == "accounting":
        return "tax & accounting"
    elif x == "business" or x == "management":
        return "business & management"
    elif x == "government" or x == "controls":
        return "government & controls"

def multifin(model, tokenizer):
    batch_size = 8
    test, query, gt = parseData()

    print(f"\n\nPrompt example:\n{query[0]}\n\n")

    total_steps = test.shape[0]//batch_size + 1
    print(f"Total len: {len(query)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []
    full_output = []
    for i in tqdm(range(total_steps)):
        tmp_context = query[i* batch_size:(i+1)* batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, return_token_type_ids=False)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_length=512, eos_token_id=tokenizer.eos_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        # print(f'{i}: {res_sentences[0]}')
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        full_text = [o for o in res_sentences]
        full_output += full_text
        out_text_list += out_text
        torch.cuda.empty_cache()

    stats = pd.DataFrame({'query': query, 'answer': gt, 'output': out_text_list, 'full_output': full_output})
    stats['new_answer'] = stats['answer'].apply(change_target)
    stats['new_output'] = stats['output'].apply(change_target)
    print(stats)

    stats.to_csv('/gpfs/u/home/FNAI/FNAIchpn/barn/DP-LoRA/mistral/multifin/multifin.csv', encoding='utf-8', index=False)

    acc = accuracy_score(stats['new_answer'], stats['new_output'])
    f1_macro = f1_score(stats['new_answer'], stats['new_output'], average = "macro")
    f1_micro = f1_score(stats['new_answer'], stats['new_output'], average = "micro")
    f1_weighted = f1_score(stats['new_answer'], stats['new_output'], average = "weighted")

    metrics = f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted: {f1_weighted}.\n"
    output_file = '/gpfs/u/home/FNAI/FNAIchpn/barn/DP-LoRA/mistral/multifin/multifin_stats.txt'
    with open(output_file, 'w') as f:
        f.write(metrics)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Transformer Version:", transformers.__version__)
    print("Device:", device)

    model, tokenizer = get_model()
    model.to(device)
    multifin(model, tokenizer)

    # example_prompt = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. Text: The company expects its net sales for the whole 2009 to remain below the 2008 level . Answer:"
    # example_prompt = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. Text: According to Sepp+ñnen , the new technology UMTS900 solution network building costs are by one-third lower than that of the building of 3.5 G networks , operating at 2,100 MHz frequency . Answer:"
    # test_prompt = query[3]
    # inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, max_length=512, truncation=True, return_token_type_ids=False).to(device)
    # # print(inputs)
    # res = model.generate(**inputs, max_new_tokens=100, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    # print(tokenizer.batch_decode(res)[0])
    # print("Ground Truth Answer:", gt[3])

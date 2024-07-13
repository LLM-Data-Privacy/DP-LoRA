import sys
import os
os.environ['HF_HOME'] = 'D:/Study/Testing'

import numpy as np
# from matplotlib import pyplot as plt
import math
import pandas as pd

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

def parseData(file="D:/Study/Testing/test-00000-of-00001-440604057ec20623.parquet"):
    df = pd.read_parquet(file, engine='auto')

    print(df)
    query = df['query'].tolist()
    gt = df['answer'].tolist()
    return query, gt

def get_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    login(token="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", add_to_git_credential=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    print("Model loaded successfully")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded successfully")

    return model, tokenizer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Transformer Version:", transformers.__version__)
    print("Device:", device)

    query, gt = parseData()

    model, tokenizer = get_model()
    # example_prompt = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. Text: The company expects its net sales for the whole 2009 to remain below the 2008 level . Answer:"
    # example_prompt = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. Text: According to Sepp+Ã±nen , the new technology UMTS900 solution network building costs are by one-third lower than that of the building of 3.5 G networks , operating at 2,100 MHz frequency . Answer:"
    test_prompt = query[1]
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, max_length=512, truncation=True, return_token_type_ids=False).to(device)
    # print(inputs)
    res = model.generate(**inputs, max_new_tokens=100, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.batch_decode(res)[0])
    print("Ground Truth Answer:", gt[1])
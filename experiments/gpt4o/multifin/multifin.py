from openai import OpenAI
from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd
from huggingface_hub import login
from config import *

from sklearn.metrics import accuracy_score,f1_score

client = OpenAI(api_key=api_key)

def parseData():
    instructions = load_dataset("TheFinAI/flare-multifin-en")
    test = instructions["test"].to_pandas()
    test['query'] = test['query'].apply(format)
    query = test['query'].tolist()
    gt = test['answer'].tolist()
    train = instructions["test"].to_pandas()
    sample = train['query'].tolist()
    sample_answer = train['answer'].tolist()
    return query, gt, sample, sample_answer

def getResponse(model, prompt, sample, sample_answer):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": sample[0]},
        {"role": "assistant", "content": sample_answer[0]},
        {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content

def multifin(model):
    query, gt, sample, sample_answer = parseData()
    out_text_list = []
    for i in range(20):
        pred = getResponse(model, query[i], sample, sample_answer)
        out_text_list.append(pred)

    print(gt[0:20])
    print(out_text_list)

if __name__ == "__main__":
    login(token=token, add_to_git_credential=True)
    model = "gpt-4o"
    multifin(model)

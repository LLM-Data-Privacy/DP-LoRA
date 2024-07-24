from openai import OpenAI
from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd
from huggingface_hub import login
from config import *

from sklearn.metrics import accuracy_score,f1_score

client = OpenAI(api_key=api_key)

def parseData():
    instructions = load_dataset("TheFinAI/en-fpb")
    test = instructions["test"].to_pandas()
    query = test['query'].tolist()
    gt = test['answer'].tolist()
    train = instructions["test"].to_pandas()
    sample = train['query'].tolist()[0]
    sample_answer = train['answer'].tolist()[0]
    return query, gt, sample, sample_answer

def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

def getResponse(model, prompt, sample, sample_answer):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": sample},
        {"role": "assistant", "content": sample_answer},
        {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content

def fpb(model):
    query, gt, sample, sample_answer = parseData()
    out_text_list = []
    for i in range(len(query)):
        pred = getResponse(model, query[i], sample, sample_answer)
        out_text_list.append(pred)
    
    stats = pd.DataFrame({'query': query, 'answer': gt, 'output': out_text_list})
    stats['new_answer'] = stats['answer'].apply(change_target)
    stats['new_output'] = stats['output'].apply(change_target)

    print(stats)

    stats.to_csv('fpb.csv', encoding='utf-8', index=False)

    acc = accuracy_score(stats['answer'], stats['new_output'])
    f1_macro = f1_score(stats['answer'], stats['new_output'], average = "macro")
    f1_micro = f1_score(stats['answer'], stats['new_output'], average = "micro")
    f1_weighted = f1_score(stats['answer'], stats['new_output'], average = "weighted")

    metrics = f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted: {f1_weighted}.\n"
    output_file = 'fpb_stats.txt'
    with open(output_file, 'w') as f:
        f.write(metrics)

    return stats

if __name__ == "__main__":
    login(token=token, add_to_git_credential=True)
    model = "gpt-4o"
    fpb(model)
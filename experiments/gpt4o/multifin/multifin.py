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

    for i in range(len(query)):
        pred = getResponse(model, query[i], sample, sample_answer)
        out_text_list.append(pred)

    stats = pd.DataFrame({'query': query, 'answer': gt, 'output': out_text_list})
    stats['new_answer'] = stats['answer'].apply(change_target)
    stats['new_output'] = stats['output'].apply(change_target)

    print(stats)

    stats.to_csv('multifin.csv', encoding='utf-8', index=False)

    acc = accuracy_score(stats['new_answer'], stats['new_output'])
    f1_macro = f1_score(stats['new_answer'], stats['new_output'], average = "macro")
    f1_micro = f1_score(stats['new_answer'], stats['new_output'], average = "micro")
    f1_weighted = f1_score(stats['new_answer'], stats['new_output'], average = "weighted")

    metrics = f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted: {f1_weighted}.\n"
    output_file = 'multifin_stats.txt'
    with open(output_file, 'w') as f:
        f.write(metrics)

    return stats

if __name__ == "__main__":
    login(token=token, add_to_git_credential=True)
    model = "gpt-4o"
    multifin(model)

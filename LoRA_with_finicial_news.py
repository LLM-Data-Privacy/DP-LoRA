import os
import shutil

jsonl_path = "../data/dataset_new.jsonl"
save_path = '../data/dataset_new'


if os.path.exists(jsonl_path):
    os.remove(jsonl_path)

if os.path.exists(save_path):
    shutil.rmtree(save_path)

directory = "../data"
if not os.path.exists(directory):
    os.makedirs(directory)

from datasets import load_dataset
import datasets

dic = {
    0:"negative",
    1:'positive',
    2:'neutral',
}

tfns = load_dataset('zeroshot/twitter-financial-news-sentiment')
tfns = tfns['train']
tfns = tfns.to_pandas()
tfns['label'] = tfns['label'].apply(lambda x:dic[x])
tfns['instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
tfns.columns = ['input', 'output', 'instruction']
tfns = datasets.Dataset.from_pandas(tfns)
tfns

tmp_dataset = datasets.concatenate_datasets([tfns]*2)
train_dataset = tmp_dataset
print(tmp_dataset.num_rows)

all_dataset = train_dataset.shuffle(seed = 42)
all_dataset.shape

import json
from tqdm.notebook import tqdm


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


data_list = []
for item in all_dataset.to_pandas().itertuples():
    tmp = {}
    tmp["instruction"] = item.instruction
    tmp["input"] = item.input
    tmp["output"] = item.output
    data_list.append(tmp)


# save to a jsonl file
with open("../data/dataset_new.jsonl", 'w') as f:
    for example in tqdm(data_list, desc="formatting.."):
        f.write(json.dumps(format_example(example)) + '\n')
        f.flush() 

## need to set the packages to run this code block
#pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate

import datasets
from transformers import AutoTokenizer, AutoConfig

model_name = "THUDM/chatglm2-6b"
jsonl_path = "../data/dataset_new.jsonl"  # updated path
save_path = '../data/dataset_new'  # updated path
max_seq_length = 512
skip_overlength = True

# The preprocess function tokenizes the prompt and target, combines them into input IDs,
# and then trims or pads the sequence to the maximum sequence length.
def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

# The read_jsonl function reads each line from the JSONL file, preprocesses it using the preprocess function,
# and then yields each preprocessed example.
def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
from openai import OpenAI
from datasets import load_dataset, load_from_disk, Dataset
from huggingface_hub import login
from config import *

client = OpenAI(api_key=api_key)

def parseData():
    instructions = load_dataset("TheFinAI/en-fpb")
    test = instructions["test"].to_pandas()
    query = test['query'].tolist()
    gt = test['answer'].tolist()
    return test, query, gt



if __name__ == "__main__":
    login(token=token, add_to_git_credential=True)

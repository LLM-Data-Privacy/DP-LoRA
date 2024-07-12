# benchmark mistral-7b on credit card fraud dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

# huggingface setup
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")
login(api_token)

# load data
splits = {'train': 'data/train.parquet', 'validation': 'data/valid.parquet', 'test': 'data/test.parquet'}
df = pd.read_parquet("hf://datasets/daishen/cra-ccfraud/" + splits["train"])


# Data prep
X = df.drop(columns=['gold'])
y = df['gold']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert to dataset
train_dataset = Dataset.from_pandas(pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1))
test_dataset = Dataset.from_pandas(pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1))

# load model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluation strategy to adopt during training
    learning_rate=2e-5,             # learning rate
    per_device_train_batch_size=16, # batch size for training
    per_device_eval_batch_size=16, # batch size for evaluation
    num_train_epochs=3,            # number of training epochs
    weight_decay=0.01,             # strength of weight decay
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# training
trainer.train()

# evaluation
predictions, labels, _ = trainer.predict(test_dataset)
preds = predictions.argmax(-1)

accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)
f1_macro = f1_score(labels, preds, average='macro')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"F1 Macro Score: {f1_macro}")
import torch
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset
from datasets import load_metric

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

def train_and_evaluate_model(model, tokenizer, train_partition, eval_partition):
    # Tokenize the training dataset
    tokenized_train_dataset = train_partition.map(
        tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Tokenize the evaluation dataset
    tokenized_eval_dataset = eval_partition.map(
        tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_eval_dataset = tokenized_eval_dataset.rename_column("label", "labels")
    tokenized_eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model at the end of each epoch
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,  # Provide the evaluation dataset
        tokenizer=tokenizer,
    )
    
    # Train and Evaluate the model
    trainer.train()
    results = trainer.evaluate()  # Evaluate the model on the evaluation dataset

    print(results)  # Print evaluation results
    return model

# Load the model and tokenizer
model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=4, trust_remote_code=True)

# Load the dataset and split it
dataset = load_dataset("ag_news", split='train[:2%]')  # Using a small portion for the example
train_test_split = dataset.train_test_split(test_size=0.5)  # Splitting the dataset

# Train and evaluate the model
model = train_and_evaluate_model(model, tokenizer, Dataset.from_pandas(train_test_split['train'].to_pandas()), Dataset.from_pandas(train_test_split['test'].to_pandas()))

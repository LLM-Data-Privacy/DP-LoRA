from sklearn.metrics import f1_score
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments, Trainer, BitsAndBytesConfig
import torch
from huggingface_hub import login
from datasets import load_dataset
import os
import re
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from trl import SFTTrainer
import copy




# Log in to the Hugging Face Hub
TOKEN = "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"
login(token=TOKEN)

# preprocess the dataset from the benchmark
def format_dataset(sample):
    sentence = sample["sentence"]
    score = sample["score"]
    
    system_instruction = "[INST]Please classify the sentiment of the following text, just output a floating point, scaling from -1 to 1, -1 is negative, 0 is neutral, 1 is positive:"

    full_prompt = ""
    full_prompt += "<s>"
    full_prompt += "### Instruction: "
    full_prompt += system_instruction
    full_prompt += "\n ### Input: "
    full_prompt += sentence
    full_prompt += "[/INST]"
    full_prompt += "\n### Response: "
    full_prompt += str(score)
    full_prompt += "</s>"
    
    return full_prompt

# Format prompt for test purposes
def test_prompt(text):
    prompt = "Please classify the sentiment of the following text, just output a floating point, scaling from -1 to 1, -1 is negative, 0 is neutral, 1 is positive: \n"
    return prompt + text + "\nResponse: "

# Set up the threshold for the sentiment classification
def sentiment_classification(output):
    if output < -0.5:
        return -1
    elif output > 0.5:
        return 1
    else:
        return 0
    
def extract_answer(output_raw):
    try:
        output = float(re.findall(r'Response: (.*)', output_raw)[0])
    except:
        output = -1
    return output

# Preprocess the training data into the format required by the model for each entry
def split_dataset(dataset, num_splits = 5, seed = 42):
    dataset = dataset.shuffle(seed = seed)
    split_size = len(dataset) // num_splits

    # Calculate the Index of the split
    indices = [range(i * split_size, min((i + 1) * split_size, len(dataset))) for i in range(num_splits)]

    # Add remaining data to last partition if not evenly divisible
    if len(dataset) % num_splits != 0:
        indices[-1] = range(indices[-1].start, len(dataset))
    
    # Create partitions using select
    partitions = {f'split_{i+1}': dataset.select(idx) for i, idx in enumerate(indices)}
    
    return partitions


# Average the weight of the model
def average_weight(model_list):
    w_avg = copy.deepcopy(model_list[0].state_dict())
    for key in w_avg.keys():
        for i in range(1, len(model_list)):
            w_avg[key] += model_list[i].state_dict()[key]
        w_avg[key] = torch.div(w_avg[key], len(model_list))
        
    # Create a new model instance to load the averaged weights into
    new_model = copy.deepcopy(model_list[0])
    new_model.load_state_dict(w_avg)
    
    return new_model

# Define the local training process
def local(model, training_args, train_dataset, tokenizer, peft_config):
    writer = SummaryWriter()
    # Trainer Configuration
    # trainer = Trainer(
    #     model = model,
    #     args = training_args,
    #     train_dataset = train_dataset,
    #     callbacks = [TensorBoardCallback(writer)],
    # )
    trainer = SFTTrainer(
        model = model,
        peft_config = peft_config,
        args = training_args,
        max_seq_length = 2048, 
        tokenizer = tokenizer,
        packing = True,
        train_dataset = train_dataset,
        formatting_func = format_dataset,
        callbacks = [TensorBoardCallback(writer)],
    )
    trainer.train()
    writer.close()
    return model


if __name__ == "__main__":
    # Load the training and testing dataset
    dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    eval_dataset = dataset["valid"]
    
    # Load the tokenizer and preprocess the dataset
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Split the training dataset into 5 partitions
    train_dataset = split_dataset(train_dataset)
    
    # Set up 8-bit quantization
    #bnb_quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        device_map = "auto",
        quantization_config = nf4_config
        )
    
    # Set up LoRA Config
    peft_config = LoraConfig(
        task_type = "CAUSAL_LM",
        r = 64,
        lora_dropout = 0.1,
        lora_alpha = 16,
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias = "none"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Define training args
    training_args = TrainingArguments(
        output_dir = "./results",
        num_train_epochs=1,
        max_steps = 100, 
        per_device_train_batch_size = 1,
        warmup_steps = 0.03,
        logging_steps=10,
        save_strategy="no",
        logging_dir="./results",
        #evaluation_strategy="epoch",
        #evaluation_strategy="steps",
        #eval_steps=20,
        learning_rate=2e-4,
        bf16=True,
        lr_scheduler_type='constant',
    )
    
    
    # Call the local training along with aggregation
    model_list = []
    for e in range(50):
        for n in range(5):
            model_list.append(local(model, training_args, train_dataset[f'split_{n+1}'], tokenizer, peft_config))
            
        model = average_weight(model_list)
    
    
    # Evaluate the model
    test_sentences = test_dataset["sentence"]
    generater = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    outputs = []
    for text in test_sentences:
        messages = test_prompt(text)
        output_raw = generater(messages, max_length=150)[0]['generated_text']
        
        output = extract_answer(output_raw)
        outputs.append(output)
    
    # Transform the output to sentiment classification
    output_label = [sentiment_classification(output) for output in outputs]
    ground_truth = [sentiment_classification(score) for score in dataset['score']]
    
    # Evaluate the Accuracy and F1 score
    accuracy = sum([1 for i in range(len(output_label)) if output_label[i] == ground_truth[i]]) / len(output_label)
    f1_macro = f1_score(ground_truth, output_label, average='macro')
    f1_micro = f1_score(ground_truth, output_label, average='micro')
    
    print("Accuracy: ", accuracy)
    print("F1 Score (Macro): ", f1_macro)
    print("F1 Score (Micro): ", f1_micro)
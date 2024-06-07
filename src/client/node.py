import os
import requests
import pickle
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel          # Model,Tokenizer
from transformers import DataCollatorForLanguageModeling,DataCollatorForSeq2Seq  # Datacollator
from transformers import TrainingArguments, Trainer
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast
from torch.utils.tensorboard import SummaryWriter

import datasets
import torch
import pdb
# LoRA
from utils import average_weights
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


def add_gaussian_noise(model):
    for key in model.state_dict().keys():
        model.state_dict()[key] += torch.randn(model.state_dict()[key].shape) * 0.1

def main():
    server_address = "http://localhost:5000"
    num_iterations = 50  # Number of iterations

    for _ in range(num_iterations):
        # Get base model from server
        response = requests.get(f"{server_address}/get_model")
        if response.status_code == 200:
            base_model_state_dict = pickle.loads(response.content)
            base_model = PeftModel()
            base_model.load_state_dict(base_model_state_dict)
        else:
            print("Failed to get base model from server")
            continue

        # Fine-tune the base model with LoRA
        trained_model = trainer.train(train_loader, base_model)

        # Add Gaussian noise to the trained model
        add_gaussian_noise(trained_model)

        # Serialize the trained model
        serialized_model = pickle.dumps(trained_model.state_dict())

        # Send the updated model back to the server
        response = requests.post(f"{server_address}/update_model", data=serialized_model)

        if response.status_code == 200:
            print("Model updated successfully")
        else:
            print("Failed to update model on server")

if __name__ == "__main__":
    main()
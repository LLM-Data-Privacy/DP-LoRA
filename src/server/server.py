# Import dependencies
import os
from flask import Flask, request
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
#local_rank = int(os.environ["LOCAL_RANK"])
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


app = Flask(__name__)

def aggregate_models(models):
    aggregated_model = {}
    num_models = len(models)
    for key in models[0].keys():
        aggregated_model[key] = sum(model[key] for model in models) / num_models
    return aggregated_model

@app.route("/update_model", methods=["POST"])
def update_model():
    global base_model
    updated_model = pickle.loads(request.data)
    for key in base_model.state_dict().keys():
        base_model.state_dict()[key] += updated_model[key]
    return "Model updated successfully", 200

if __name__ == "__main__":
    # Initialize base model
    base_model = PeftModel.from_pretrained("llama")
    
    # List of node addresses
    node_addresses = ["http://node1:5001", "http://node2:5002", "http://node3:5003"]
    
    num_iterations = 50  # Number of iterations

    for iteration in range(num_iterations):
        # Send base model to each node individually
        for node_address in node_addresses:
            response = requests.post(f"{node_address}/get_model", data=pickle.dumps(base_model.state_dict()))
            if response.status_code != 200:
                print(f"Failed to send base model to node at {node_address}")

        # Wait for responses from each node
        updated_models = []
        for node_address in node_addresses:
            response = requests.get(f"{node_address}/update_model")
            if response.status_code == 200:
                updated_model = pickle.loads(response.content)
                updated_models.append(updated_model)
            else:
                print(f"Failed to receive updated model from node at {node_address}")

        # Aggregate models received from all nodes
        aggregated_model = aggregate_models(updated_models)

    app.run(debug=True)
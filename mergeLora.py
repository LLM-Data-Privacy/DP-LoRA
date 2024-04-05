!unzip saved_model_0.zip
!unzip saved_model_1.zip
!unzip saved_model_2.zip

from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftConfig, PeftModel
import torch
import accelerate
import bitsandbytes

# Paths to your models
model_paths = ["./models/finetuned_model_0", "./models/finetuned_model_1", "./models/finetuned_model_2"]
     
model_name = "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True, device="cuda")

model = PeftModel.from_pretrained(model, model_paths[0], adapter_name='Institute_0')
_ = model.load_adapter(model_paths[1], adapter_name='Institute_1')
_ = model.load_adapter(model_paths[2], adapter_name='Institute_2')
     
adapters = ["Institute_0", "Institute_1", "Institute_2"]
weights = [1.0, 1.0, 1.0]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="linear")
     
model.set_adapter("merge")
     
# Save the new model
output_model_path = "./aggregated_model"
model.save_pretrained(output_model_path)
     
model.add_weighted_adapter(adapters, weights, "merge_ties", combination_type="ties", density=density)
output_model_path = "./aggregated_model"
model.save_pretrained(output_model_path)
     
!zip -r ./aggregated_model.zip ./aggregated_model

#clone the FinNLP repository
!git clone https://github.com/AI4Finance-Foundation/FinNLP.git

import sys
sys.path.append('/content/FinNLP/')

# Load benchmark datasets from FinNLP
import datasets
from finnlp.benchmarks.fpb import test_fpb
from finnlp.benchmarks.fiqa import test_fiqa , add_instructions
from finnlp.benchmarks.tfns import test_tfns
from finnlp.benchmarks.nwgi import test_nwgi

model.set_adapter("merge")
batch_size = 40

# TFNS Test Set, len 2388
res = test_tfns(model, tokenizer, batch_size = batch_size)
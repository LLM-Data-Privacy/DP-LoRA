import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel          # Model,Tokenizer
from transformers import DataCollatorForLanguageModeling,DataCollatorForSeq2Seq  # Datacollator
from transformers import TrainingArguments, Trainer
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast
from torch.utils.tensorboard import SummaryWriter

import datasets
import torch
import pdb
import copy
#local_rank = int(os.environ["LOCAL_RANK"])
# LoRA
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    
)
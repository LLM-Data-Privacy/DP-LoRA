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

model_name = '/colab_space/yanglet/models--daryl149--Llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed'
tokenizer = LlamaTokenizerFast.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


from local_train import local

def main():
    nodes = 5
    device_map = "auto"
    
    training_args = TrainingArguments(
        output_dir='./finetuned_model_7B_rl_fin', # 保存位置
        # output_dir='./test',
        logging_steps = 1000,
        # max_steps=10000,
        num_train_epochs = 1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        save_steps=1000,
        # ddp_backend = 'gloo',
        fp16=True,
        # bf16=True,
        #deepspeed=deepspeed_config,
        torch_compile = False,
        load_best_model_at_end = True,
        evaluation_strategy="steps",
        remove_unused_columns=False,
        #ddp_find_unused_parameters=False if ddp else None,
        # 测试数据加载时间用，正常训练请注释
        #dataloader_num_workers=64,
        #dataloader_pin_memory=True
    )
    
    # load model
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=True,
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        # device='cuda',
        # device_map = f'cuda:{local_rank}',
        device_map = device_map
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    model = prepare_model_for_int8_training(model)

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', "k_proj", 'v_proj'],
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    
    model_list = []
    for e in range(50):
        for _ in range(nodes):
            model_list.append(local(model, training_args))
            model = average_weights(model_list)
            
            
if __name__ == "__main__":
    main()

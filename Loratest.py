import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import loralib as lora
from transformers import DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import BertConfig 
from transformers import EncoderDecoderConfig 
from transformers import DataCollatorWithPadding 
from transformers import EncoderDecoderModel 
from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification 
from transformers import CONFIG_MAPPING 
from transformers import AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#! Lora Functions #
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
## Set max matrix rank
lora_r = 4

def make_lora_layer(layer, lora_r=4):
    new_layer = lora.Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is None,
        r=lora_r,
        merge_weights=False
    )
    
    new_layer.weight = nn.Parameter(layer.weight.detach())
    
    if layer.bias is not None:
        new_layer.bias = nn.Parameter(layer.bias.detach())
    
    return new_layer

def make_lora_replace(model, depth=1, path="", verbose=True):
    if depth > 10:
        return
    depth += 1        
    if isinstance(model, nn.Linear) and "attention" in path:
        if verbose:
            print(f"Find linear {path}:{key} :", type(module))
        return make_lora_layer(model)
    
    for key in dir(model):
        module = getattr(model, key)
        module_type = type(module)
            
        if not isinstance(module, nn.Module) or module is model:
            continue
        if isinstance(module, nn.Linear) and "attention" in path:
            layer = make_lora_layer(module)
            setattr(model, key, layer)
            if verbose:
                print(f"Find linear {path}:{key} :", type(module))
        elif isinstance(module, nn.ModuleList):
            for i, elem in enumerate(module):
                layer = make_lora_replace(elem, depth, path+":"+key+f"[{i}]", verbose=verbose)
                if layer is not None:
                    module[i] = layer                
        elif isinstance(module, nn.ModuleDict):
            for module_key in list(module.keys()):
                layer = make_lora_replace(item, depth, path+":"+key+":"+module_key, verbose=verbose)
                if layer is not None:
                    module[module_key] = layer              
        else:
            layer = make_lora_replace(module, depth, path+":"+key, verbose=verbose)
            if layer is not None:
                setattr(model, key, layer)

#! Fetch Data/Models #

dataset_id, task, tok_train_fold, sentence1_key, num_labels, vocab_len = 'tweet_eval', 'emotion', 'train', 'text', 4, 30000
dataset = load_dataset(dataset_id, task)

USE_LORA = True
model_name = 'roberta-large'
batch_size = 16

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
collator = DataCollatorWithPadding(tokenizer)
if USE_LORA:
#     first_output = model(**collator([tokenizer('test')]))
    make_lora_replace(model, verbose=True)

#     final_output = model(**collator([tokenizer('test')]))
## Load omdel onto TPU/GPU
model = model.to(device)
# tokenizer = tokenizer.to(device)

#! Apply Lora to Model #

lora_r = 4 ## Try different max matrix ranks for different results

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before LoRA: {total_trainable_params}")

## Apply LoRA
lora.mark_only_lora_as_trainable(model)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after LoRA: {total_trainable_params}")
for name, param in model.named_parameters():
    if "deberta" not in name:
        print(name)
#             print(param.shape)
        param.requires_grad = True
# import wandb
# wandb.init(project='Lora')

#! Process Training Data #

def preprocess_function(examples):
    return tokenizer(examples[sentence1_key], truncation=True, padding=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.remove_columns(['text'])
dataloaders = {
    key: DataLoader(ds, shuffle=True, collate_fn=collator, num_workers=2, batch_size=batch_size) for key, ds in encoded_dataset.items()
}

#! Batching Helpers #

def predict_loader(dataloader):
    model.eval()
    res = []
    labels = []
    
    for batch in tqdm(dataloader):
        batch = batch_device(batch)
        output = model(**batch).logits.argmax(dim=-1).detach().cpu().numpy()
        res.extend(output)
        labels.extend(batch.labels.detach().cpu().numpy())
            
    return(res, labels)

def batch_device(batch):
    for key in list(batch.keys()):
        batch[key] = batch[key].to(device)
        
    return(batch)
# wandb.watch(model)

#! Training #

from IPython.display import clear_output
from tqdm.autonotebook import tqdm
from sklearn.metrics import f1_score

lr = 2e-5
if USE_LORA:
    lr = 2e-4

optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
steps = len(dataloaders['train'])
epochs = 15
scheduler = get_cosine_schedule_with_warmup(optimizer, steps * 1, steps * epochs)
best_f1 = 0

for i in range(epochs):
    model.train()
    losses = []
    for batch in tqdm(dataloaders['train']):
        batch = batch_device(batch)        
        optimizer.zero_grad()
        output = model(**batch)
        output.loss.backward()
#         wandb.log({
#             "train/loss": output.loss.detach().cpu().numpy()
#         })
        optimizer.step()
        scheduler.step()    
    res, labels = predict_loader(dataloaders['validation'])
    res = np.array(res) 
    labels = np.array(labels)
    f1 = f1_score(labels, res, average='micro')
#     wandb.log({
#         "eval/f1": f1
#     })
#     print(f1)
    
    if f1 > best_f1:
        best_f1 = f1
        checkpoint_path = "best_lora_checkpoint.pth"        
        if USE_LORA:
            torch.save(lora.lora_state_dict(model), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)

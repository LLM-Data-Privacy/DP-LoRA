import sys
import os

os.environ['TRANSFORMERS_CACHE'] = '/gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface'
os.environ['HF_HOME'] = '/gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface'

# Ensure the directory exists
os.makedirs('/gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface', exist_ok=True)

from huggingface_hub import login

# os.environ['HF_DATASETS_CACHE'] = '/gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface'
# os.environ['HF_HOME'] = '/gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface'

# os.environ['export TRANSFORMERS_CACHE = /gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface']
# os.environ['export HF_DATASETS_CACHE = /gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface']
# os.environ['export HF_HOME = /gpfs/u/home/FNAI/FNAIjspr/scratch/huggingface']

# Verify environment variables
print("TRANSFORMERS_CACHE:", os.getenv('TRANSFORMERS_CACHE'))
print("HF_DATASETS_CACHE:", os.getenv('HF_DATASETS_CACHE'))
print("HF_HOME:", os.getenv('HF_HOME'))

# Log in to the Hugging Face Hub
TOKEN = "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"
login(token=TOKEN)

import pandas as pd

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from config import *

from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import datasets
import torch

from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
import warnings


def parseData():
    instructions = load_dataset("TheFinAI/flare-ner")
    test = instructions["test"].to_pandas()
    return test

def get_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Model loaded successfully")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded successfully")

    return model, tokenizer

def NER(model, tokenizer):
    batch_size = 4
    test = parseData()

    batches = [(i, min(i + batch_size, len(test))) for i in range(0, len(test), batch_size)]
    print(f"Total len: {len(test['query'])}. Batchsize: {batch_size}. Total steps: {len(batches)}")

    out_text_list = []
    full_output = []
    for i in range(0, len(test), batch_size):
        tmp_context = list(test['query'][i : min(i + batch_size, len(test))])
        
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, return_token_type_ids=False)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()

        res = model.generate(**tokens, max_length=1024, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        res_sentences = [tokenizer.decode(j, skip_special_tokens=True) for j in res]
        
        print("\n\n--------------------", i/4, "------------------------") 
        for o in res_sentences:
            print(o.split("Answer:")[1])
            print("------------------------------------------------") 
        out_text = [o.split("Answer:")[1] for o in res_sentences]
        full_text = [o for o in res_sentences]
        full_output += full_text
        out_text_list += out_text
        torch.cuda.empty_cache()

    stats = pd.DataFrame({'query': test['query'], 'answer': gt, 'output': out_text_list, 'full_output': full_output})
    print(stats)

    stats.to_csv('/gpfs/u/home/FNAI/FNAIjspr/barn/DP-LoRA/mistral7bNER/results.csv', encoding='utf-8', index=False)

    acc = accuracy_score(stats['new_answer'], stats['new_output'])
    f1_macro = f1_score(stats['new_answer'], stats['new_output'], average = "macro")
    f1_micro = f1_score(stats['new_answer'], stats['new_output'], average = "micro")
    f1_weighted = f1_score(stats['new_answer'], stats['new_output'], average = "weighted")

    metrics = f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted: {f1_weighted}.\n"
    output_file = '/gpfs/u/home/FNAI/FNAIjspr/barn/DP-LoRA/mistral7bNER/results_stats.txt'
    with open(output_file, 'w') as f:
        f.write(metrics)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Transformer Version:", transformers.__version__)
    print("Device:", device)

    model, tokenizer = get_model()
    model.to(device)
    NER(model, tokenizer)

    # example_prompt = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. Text: The company expects its net sales for the whole 2009 to remain below the 2008 level . Answer:"
    # example_prompt = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. Text: According to Sepp+ñnen , the new technology UMTS900 solution network building costs are by one-third lower than that of the building of 3.5 G networks , operating at 2,100 MHz frequency . Answer:"
    # test_prompt = query[3]
    # inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, max_length=512, truncation=True, return_token_type_ids=False).to(device)
    # # print(inputs)
    # res = model.generate(**inputs, max_new_tokens=100, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    # print(tokenizer.batch_decode(res)[0])
    # print("Ground Truth Answer:", gt[3])


    # scp D:/Study/Testing/tester.py FNAIjspr@blp01.ccni.rpi.edu:/gpfs/u/home/FNAI/FNAIjspr/barn/DP-LoRA/mistral7bNER


# ["In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'.\nText: Such agreements may include provisions which permit the Creditors , in the event of a breach of the Loan Agreement or the Silicon S & P Agreement that would permit the Lender to terminate the Loan Agreement , to : ( i ) Take - over the Loan Agreement ; ( ii ) step - in , rectify or otherwise cure any breach of this Loan Agreement ; ( iii ) assign or otherwise transfer this Loan Agreement .\nAnswer: Creditors, ORG; Lender, ORG; Loan Agreement, LOC.", "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'.\nText: NOTICES 14 . 1 Any notice or other communication required to be given under this Loan Agreement shall be in writing and shall be delivered to the Party required to receive the notice or communication at its address as set out below : Borrower Usine de Saint Auban , Page 8 of 12 7 - December 2007 04 600 Saint Auban , France .\nAnswer: Usine de Saint Auban, LOC.", "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'.\nText: Attention : Frank Wouters , CEO Lender 138 Bartlett Street Marlboro , Massachusetts , 01752 U . S . A .\nAnswer: Frank Wouters, PER; Lender 138 Bartlett Street, LOC; Mar", "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'.\nText: Attention : Richard Chleboski , Vice President or at such other address as the relevant Party may specify by notice in writing to the other Parties .\nAnswer: Richard Chleboski, PER\nText: Attention : XYZ Corporation , a corporation", "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'.\nText: The arbitration shall be conducted in English and the seat shall be Paris .\nAnswer: Paris, LOC.", "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'.\nText: The following events shall be considered events of default with respect to this Loan Agreement : 18 . 1 . 1 The Borrower shall default in the payment of any part of the principal or unpaid accrued interest on the Loan [ for more than [ thirty ( 30 )] days ] after the same shall become due and payable , whether at maturity or at a date fixed for prepayment or by acceleration or otherwise ; 18 . 1 . 2 The Borrower is unable or admits its inability to pay its debts as they fall due , by reason of actual or anticipated financial difficulties or suspends making payments on any of its debts or commences negotiations with one or more of its creditors with a view to rescheduling any of its indebtedness ; 18 . 1 . 3 The Borrower is in a state of suspension of payments ( cessation des paiements ) within the meaning of article L . 631 - 1 of the French Commercial Code ; 18 . 1 . 4 Any corporate action , legal proceedings or other procedures or steps are taken by reason of the Borrower ' s financial difficulties , in relation to : ( A ) The suspension of payments , a moratorium of any indebtedness , winding - up , dissolution administration or reorganisation ( other than a solvent winding - up , dissolution or reorganisation carried out with the prior written consent of the Lender , such consent not to be unreasonably withheld or delayed ) of the Borrower ; ( B ) A composition , compromise , assignment or arrangement with any creditor of the Borrower ; Page 10 of 12 7 - December 2007 ( C ) The appointment of a liquidator , receiver , administrator , administrative receiver , compulsory manager or other similar officer in respect of the Borrower or any of its assets ; Or any analogous procedure or step is taken in any jurisdiction .\nAnswer:\nPER: Borrower\nORG: Lender\nLOC: France", "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'.\nText: 18 . 1 . 5 The Borrower commences proceedings for conciliation in accordance with articles L . 611 - 4 to L . 611 - 15 of the French Commercial Code or any analogous procedure or step is taken in any jurisdiction ; 18 . 1 . 6 A judgment for sauvegarde , redressement judiciaire or liquidation judiciaire is entered in relation to the Borrower under articles L . 620 - 1 to L . 644 - 6 of the French Commercial Code or any analogous judgment is entered in any jurisdiction ; 18 . 1 . 7 The Borrower shall fail to observe or perform any other obligation to be observed or performed by it under this Loan Agreement within thirty ( 30 ) days after written notice from the Lender to perform or observe the obligation ; and 18 . 1 . 8 The Borrower stops construction of the Works or acknowledges that it will be unable or is unwilling to complete construction of the works .\nAnswer:\n- Borrower, ORG\n- Lender, ORG\n- French Commercial Code,", "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'.\nText: Upon the occurrence of an event of default under Article 18 . 1 hereof , at the option and upon the declaration of the Lender , the entire unpaid principal and accrued and unpaid interest on the Loan shall without formal notice of default ( mise en demeure ) or any other judicial or extra - judicial step ,, be forthwith due and payable , and the Lender may , immediately and without expiration of any period of grace , enforce payment of all amounts due and owing under this Loan Agreement and exercise any and all other remedies granted to it .\nAnswer: Lender, ORG; Event of default, PER; Loan, LOC."]
import openai
from openai import OpenAI
from datasets import load_dataset
from huggingface_hub import login
from config import *
client = OpenAI(api_key=OPENAI_KEY)

login(token = TOKEN)
dataset = load_dataset("TheFinAI/flare-finred",token = TOKEN)
dataset_iter = dataset['test'].iter(1)
tp, fp, fn = 0, 0, 0
iter_count = 0




def F1_score(correct_labels, answer_labels):
    # TP, FP, FN
    tp, fp, fn = 0, 0, 0
    correct_set = set(correct_labels)
    answer_set = set(answer_labels)

    # 计算 True Positives (TP) 和 False Negatives (FN)
    for triplet in correct_set:
        if triplet in answer_set:
            tp += 1
        else:
            fn += 1

    # 计算 False Positives (FP)
    for triplet in answer_set:
        if triplet not in correct_set:
            fp += 1


    return tp, fp, fn


def get_response(prompt):
    MODEL="gpt-4o"
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Given the following sentence, identify the head, tail, and relation of each triplet present in the sentence. The relations you should be looking for are 'legal_form', 'publisher', 'owner_of', 'employer', 'manufacturer', 'position_held', 'chairperson', 'industry', 'business_division', 'creator', 'original_broadcaster', 'chief_executive_officer', 'location_of_formation', 'operator', 'owned_by', 'founded_by', 'parent_organization', 'member_of', 'product_or_material_produced', 'brand', 'headquarters_location', 'director_/_manager', 'distribution_format', 'distributed_by', 'platform', 'currency', 'subsidiary', 'stock_exchange', and 'developer'. If a relation exists between two entities, provide your answer in the format 'head ; tail ; rel'. If there are multiple triplets in a sentence, provide each one on a new line. Text: Wednesday, July 8, 2015 10:30AM IST (5:00AM GMT) Rimini Street Comment on Oracle Litigation Las Vegas, United States Rimini Street, Inc., the leading independent provider of enterprise software support for SAP AG’s (NYSE:SAP) Business Suite and BusinessObjects software and Oracle Corporation’s (NYSE:ORCL) Siebel , PeopleSoft , JD Edwards , E-Business Suite , Oracle Database , Hyperion and Oracle Retail software, today issued a statement on the Oracle litigation. Answer:"},
        {"role": "assistant", "content": "PeopleSoft ; JD Edwards ; subsidiary"},
        {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content


while True:
    
    try:
        iter_count += 1
        print("Iteration: {}".format(iter_count),end="\t")
        data = next(dataset_iter)
        query = data['query'][0]
        answer = data['answer'][0].encode('gbk', errors='ignore').decode('gbk')
        response = get_response(query).encode('gbk', errors='ignore').decode('gbk')
        answer_labels = answer.split(" ; ")
        line_1 = response.split("\n")[0]
        correct_labels = line_1.split(" ; ")
        for i in range(len(correct_labels)):
            correct_labels[i] = correct_labels[i].strip()
        for i in range(len(answer_labels)):
            answer_labels[i] = answer_labels[i].strip()
        tp_, fp_, fn_ = F1_score(correct_labels, answer_labels)
        tp += tp_
        fp += fp_
        fn += fn_
        print("CORRECT: {}, API: {}".format(correct_labels, answer_labels))
    except StopIteration:
        break

# Precision , Recall
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

# 计算 F1 Score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print("TP: {}, FP: {}, FN: {}".format(tp, fp, fn))
print("Precision: {}, Recall: {}, F1 Score: {}".format(precision, recall, f1_score))
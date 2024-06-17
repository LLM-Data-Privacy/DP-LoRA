import openai
from openai import OpenAI
from datasets import load_dataset
from huggingface_hub import login
from config import *
client = OpenAI(api_key=OPENAI_KEY)

login(token = TOKEN)
dataset = load_dataset("TheFinAI/flare-cd",token = TOKEN)
dataset_iter = dataset['test'].iter(1)
tp, fp, fn = 0, 0, 0
iter_count = 0


def parse_label(answer):
    labels = []
    lines = answer.splitlines()
    for line in lines:
        if ":" in line:
            labels.append(line.split(":")[1])
        else:
            break
    return labels

def F1_score(correct_labels, api_labels):
    # TP, FP, FN
    tp, fp, fn = 0, 0, 0
    cause_effect = set(["B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT"])

    for i in range(min(len(correct_labels), len(api_labels))):
        correct = correct_labels[i]
        predicted = api_labels[i]
        if correct in cause_effect:
            if predicted.lower().strip() == correct.lower().strip():
                tp += 1  # True positive
                #print("TP: {}<==>{}".format(correct, predicted))
            else:
                fn += 1  # False negative
                #print("FN: {}<==>{}".format(correct, predicted))
        if predicted in cause_effect:
            if predicted.lower().strip() != correct.lower().strip():
                fp += 1  # False positive
                #print("FP: {}<==>{}".format(correct, predicted))
    return tp, fp, fn

def Entity_F1_score(correct_labels, api_labels):
    # TP, FP, FN
    tp, fp, fn = 0, 0, 0
    cause_effect = set(["B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT"])

    for i in range(min(len(correct_labels), len(api_labels))):
        correct = correct_labels[i]
        predicted = api_labels[i]
        if correct in cause_effect:
            if predicted.lower().strip() == correct.lower().strip():
                tp += 1  # True positive
                #print("TP: {}<==>{}".format(correct, predicted))
            else:
                fn += 1  # False negative
                #print("FN: {}<==>{}".format(correct, predicted))
        if predicted in cause_effect:
            if predicted.lower().strip() != correct.lower().strip():
                fp += 1  # False positive
                #print("FP: {}<==>{}".format(correct, predicted))
    return tp, fp, fn

def get_response(prompt):
    MODEL="gpt-4o"
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Your job in this task is to perform sequence labeling on a provided text section, marking the chunks that represent the cause of an event and the effects that result from it. For each token in the text, assign a label to indicate its role in representing cause or effect. The labels you should use are 'B-CAUSE', 'I-CAUSE', 'B-EFFECT', 'I-EFFECT', and 'O'. A 'B-' prefix is used to denote the beginning of a cause or effect sequence, while an 'I-' prefix is used for continuation of a cause or effect sequence. If a token is not part of either a cause or effect sequence, label it as 'O'. Provide your answer as a sequence of 'token:label' pairs, with each pair on a new line. Text: Around 21,000 employees , 9,000 of whom are employed in the UK , are to be made redundant after the 178-year-old company ceased trading and went into compulsory liquidation this morning. Answer:"},
        {"role": "assistant", "content": "Around:B-EFFECT 21,000:I-EFFECT employees:I-EFFECT ,:I-EFFECT 9,000:I-EFFECT of:I-EFFECT whom:I-EFFECT are:I-EFFECT employed:I-EFFECT in:I-EFFECT the:I-EFFECT UK:I-EFFECT ,:I-EFFECT are:I-EFFECT to:I-EFFECT be:I-EFFECT made:I-EFFECT redundant:I-EFFECT after:O the:B-CAUSE 178-year-old:I-CAUSE company:I-CAUSE ceased:I-CAUSE trading:I-CAUSE and:I-CAUSE went:I-CAUSE into:I-CAUSE compulsory:I-CAUSE liquidation:I-CAUSE this:I-CAUSE morning.:I-CAUSE"},
        {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content


while True:
    
    try:
        iter_count += 1
        print("Iteration: {}".format(iter_count))
        data = next(dataset_iter)
        query = data['query'][0]
        answer = data['answer'][0]
        prediction = get_response(query)
        correct_labels = parse_label(answer)
        answer_labels = parse_label(prediction)
        tp_, fp_, fn_ = F1_score(correct_labels, answer_labels)
        tp += tp_
        fp += fp_
        fn += fn_
    except StopIteration:
        break

# Precision , Recall
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

# 计算 F1 Score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print("TP: {}, FP: {}, FN: {}".format(tp, fp, fn))
print("Precision: {}, Recall: {}, F1 Score: {}".format(precision, recall, f1_score))
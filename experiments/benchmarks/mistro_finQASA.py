from sklearn.metrics import f1_score
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from datasets import load_dataset

# preprocess the dataset from the benchmark
def preprocess_prompt(texts):
    prompt = "Please classify the sentiment of the following text, just output a floating point, scaling from -1 to 1, -1 is negative, 0 is neutral, 1 is positive: \n"
    combined_inputs = [ {"role": "user", "content": "Please classify the sentiment of the following text, just output a floating point, scaling from -1 to 1, -1 is negative, 0 is neutral, 1 is positive: \nBritain's FTSE steadies, supported by Dixons Carphone"},
                        {"role": "assistant", "content": "0.329"}]
    for entry in texts:
        combined_inputs.append({"role": "user", "content": prompt + entry})
    
    return combined_inputs

# set up the threshold for the sentiment classification
def sentiment_classification(output):
    if output < -0.5:
        return -1
    elif output > 0.5:
        return 1
    else:
        return 0
    

if __name__ == '__main__':
    # Load the model and tokenizer
    # Load Mistro-7b model
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    # Load the FinQASA dataset for benchmarking
    dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
    dataset = dataset['test']

    # Generate the prompt for the dataset
    sentences = dataset['sentence']
    messages = preprocess_prompt(sentences)   
    
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors='pt').to("cuda")
    
    generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
    
    outputs = tokenizer.batch_decode(generated_ids)
    
    outputs = [float(output) for output in outputs]
    
    # Store the prediction to new csv file
    with open('predictions.csv', 'w') as f:
        f.write("sentence, prediction\n")
        for i in range(len(sentences)):
            f.write(sentences[i] + "," + str(outputs[i]) + "\n")
            
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
    
    
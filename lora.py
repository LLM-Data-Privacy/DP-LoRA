import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Model

# Load the IMDb movie reviews dataset
(train_data, test_data), dataset_info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# Initialize a Keras tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Function to convert dataset to a list of texts
def texts_from_dataset(dataset):
    texts = []
    for text_tensor, _ in dataset.as_numpy_iterator():
        text = text_tensor.decode('utf-8')
        texts.append(text)
    return texts

# Update tokenizer with the training texts
train_texts = texts_from_dataset(train_data)
tokenizer.fit_on_texts(train_texts)

# Encoding function
def encode(text, label):
    encoded_text = tokenizer.texts_to_sequences([text.numpy().decode('utf-8')])[0]
    return np.array(encoded_text), np.array(label)

# Map function for encoding
def encode_map_fn(text, label):
    encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label

# Apply the encoding to the datasets
train_data = train_data.map(encode_map_fn)
test_data = test_data.map(encode_map_fn)

# Shuffle, batch, and pad the datasets
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_batches = train_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], []))
test_batches = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

# Function to split dataset into N parts (for federated learning simulation)
def split_dataset_into_parts(dataset, num_parts=5):
    all_batches = list(dataset)
    np.random.shuffle(all_batches)
    size_of_each_part = len(all_batches) // num_parts
    return [all_batches[i*size_of_each_part:(i+1)*size_of_each_part] for i in range(num_parts)]

# Split the train and test datasets into 5 parts each for federated learning
train_datasets = split_dataset_into_parts(train_batches, num_parts=5)
test_datasets = split_dataset_into_parts(test_batches, num_parts=5)

# Accessing the first batch
first_batch_examples, first_batch_labels = train_datasets[0][0]

# Assuming first_batch_examples is a numpy array of encoded texts,
# and first_batch_labels is a TensorFlow tensor of labels:
# for i in range(min(3, len(first_batch_labels))):  # Print up to 3 examples
#     print('Encoded text:', first_batch_examples[i])
#     print('Label:', first_batch_labels.numpy()[i])

# Define the model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128

# Define the model architecture with BERT
def create_falcon_model_with_bert():
    # Define the input layer
    input_ids = Input(shape=(max_length,), dtype='int32')
    # Extract BERT's last hidden states
    bert_output = bert_model(input_ids)[0]
    # Custom layers on top of BERT
    x = Dense(64, activation='relu')(bert_output[:, 0, :])  # Use the representation of [CLS] token
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs=input_ids, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

class LoRADense(Layer):
    def __init__(self, output_dim, rank, **kwargs):
        super(LoRADense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.rank = rank

    def build(self, input_shape):
        # Define the low-rank matrices U and V
        self.U = self.add_weight(name='U', shape=(input_shape[-1], self.rank), initializer='uniform', trainable=True)
        self.V = self.add_weight(name='V', shape=(self.rank, self.output_dim), initializer='uniform', trainable=True)

    def call(self, inputs):
        # Compute the output of the layer as inputs * U * V
        return tf.matmul(tf.matmul(inputs, self.U), self.V)

def create_falcon_model_with_lora():
    input_ids = Input(shape=(max_length,), dtype='int32')
    bert_output = bert_model(input_ids)[0]

    # Apply LoRA to the output of BERT model
    lora_output = LoRADense(64, rank=32)(bert_output[:, 0, :])  # Example usage of LoRA layer
    x = Dense(32, activation='relu')(lora_output)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_ids, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

falcon_model = create_falcon_model_with_lora()
falcon_model.summary()

# def client_update(model, dataset):
#     # Train the model on the client's dataset
#     model.fit(dataset, epochs=1, verbose=0)
#     return model.get_weights()

def client_update(client_model, client_dataset):
    # Extract inputs from the client dataset
    for inputs, labels in client_dataset.take(1):  # Take a peek at the first batch
        print("Input IDs shape:", inputs['input_ids'].shape)
        print("Attention mask shape:", inputs['attention_mask'].shape)
        if 'token_type_ids' in inputs:
            print("Token type IDs shape:", inputs['token_type_ids'].shape)
        print("Labels shape:", labels.shape)

    # Assuming client_dataset is already in the correct format
    history = client_model.fit(client_dataset, epochs=1, verbose=1)
    return client_model.get_weights()


def federated_averaging(global_model, client_weights):
    # Perform federated averaging on the client weights
    new_global_weights = np.mean(client_weights, axis=0)
    global_model.set_weights(new_global_weights)

def evaluate_global_model(global_model, test_dataset):
    # Evaluate the global model on the test dataset
    loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
    return loss, accuracy

# Simulating federated learning
num_clients = len(train_datasets)
num_rounds = 5

# Create a global model instance
global_model = create_falcon_model_with_lora()

# Run federated learning for a number of rounds
for round_num in range(num_rounds):
    print(f'Starting round {round_num + 1}/{num_rounds}')
    client_weights = []
    
    # Each client trains on their respective dataset
    for client_id in range(num_clients):
        print(f'Training on client {client_id + 1}/{num_clients}')
        client_model = create_falcon_model_with_lora()  # Create a new model instance for the client
        client_model.set_weights(global_model.get_weights())  # Initialize with global model weights
        client_dataset = train_datasets[client_id]  # Get the client's dataset
        client_model_weights = client_update(client_model, client_dataset)
        # client_weights.append(client_model_weights)
    
    # # Perform federated averaging to update the global model
    # federated_averaging(global_model, client_weights)

    # # Evaluate the global model's performance
    # loss, accuracy = evaluate_global_model(global_model, test_batches)
    # print(f'Round {round_num + 1}, Loss: {loss}, Accuracy: {accuracy}')

# print('Federated Learning simulation completed.')
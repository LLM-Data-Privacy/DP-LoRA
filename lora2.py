import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from transformers import TFBertModel


# Load the IMDb movie reviews dataset
(train_data, test_data), dataset_info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# Initialize BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the custom LoRA layer
class LoRADense(tf.keras.layers.Layer):
    def __init__(self, output_dim, rank, **kwargs):
        super(LoRADense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.rank = rank

    def build(self, input_shape):
        self.U = self.add_weight(
            shape=(input_shape[-1], self.rank),
            initializer='random_normal',
            trainable=True
        )
        self.V = self.add_weight(
            shape=(self.rank, self.output_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(tf.matmul(inputs, self.U), self.V)
    
# Define the max sequence length
max_length = 128  # Make sure this is the sequence length you want to use

# Create the Falcon model with the LoRA layer
def create_falcon_model_with_lora():
    input_ids = Input(shape=(max_length,), dtype='int32')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    bert_output = bert_model(input_ids)[0]  # [0] to get the sequence output
    lora_output = LoRADense(64, rank=32)(bert_output[:, 0, :])  # Apply LoRA to the output of the BERT model

    x = Dense(32, activation='relu')(lora_output)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_ids, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to encode and prepare dataset
def encode_and_prepare_data(tokenizer, dataset, sequence_length):
    encoded_batch = tokenizer.batch_encode_plus(
        [text.numpy().decode('utf-8') for text, _ in dataset],
        add_special_tokens=True,
        max_length=sequence_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    labels = np.array([label.numpy() for _, label in dataset])
    
    dataset = tf.data.Dataset.from_tensor_slices(({
        'input_ids': encoded_batch['input_ids'],
        'attention_mask': encoded_batch['attention_mask']
    }, labels))
    dataset = dataset.batch(32)
    return dataset

# Tokenize and prepare train and test datasets
sequence_length = 128
train_dataset = encode_and_prepare_data(bert_tokenizer, train_data, sequence_length)
test_dataset = encode_and_prepare_data(bert_tokenizer, test_data, sequence_length)

# Split the dataset into multiple clients for federated learning
def split_into_clients(dataset, num_clients=5):
    dataset_batches = list(dataset)
    np.random.shuffle(dataset_batches)
    size_per_client = len(dataset_batches) // num_clients
    client_datasets = [dataset_batches[i * size_per_client: (i + 1) * size_per_client] for i in range(num_clients)]
    return client_datasets

client_train_datasets = split_into_clients(train_dataset)
client_test_datasets = split_into_clients(test_dataset)

# Define the model creation function
def create_model_with_lora():
    input_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name='attention_mask')
    
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    
    lora_output = LoRADense(output_dim=768, rank=32)(bert_output.last_hidden_state)
    
    cls_output = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(lora_output)  # Extract CLS token
    cls_output = Dense(64, activation='relu')(cls_output)
    cls_output = Dense(32, activation='relu')(cls_output)
    cls_output = Dense(1, activation='sigmoid')(cls_output)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=cls_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Federated learning functions
def client_update(model, dataset):
    model.fit(dataset, epochs=1, verbose=1)
    return model.get_weights()

def server_aggregate(global_model, client_weights_list):
    average_weights = np.mean(client_weights_list, axis=0)
    global_model.set_weights(average_weights)

def evaluate_global_model(model, dataset):
    loss, accuracy = model.evaluate(dataset, verbose=0)
    return loss, accuracy

# Create a global model instance
global_model = create_falcon_model_with_lora()

# # Federated learning simulation
# for round in range(1, 6):
#     print(f"Round {round}")
#     client_weights = []
    
#     # Train on each client
#     for client_id, client_dataset in enumerate(client_train_datasets):
#         print(f"Training on client {client_id + 1}")
#         client_model = create_falcon_model_with_lora()  # Create a new model instance for the client
#         client_model.set_weights(global_model.get_weights())  # Initialize with global model weights
#         client_weights.append(client_update(client_model, client_dataset))
    
#     # Aggregate client updates at the server
#     server_aggregate(global_model, client_weights)

#     # Evaluate the global model's performance after aggregation
#     loss, accuracy = evaluate_global_model(global_model, test_dataset)
#     print(f"Post-aggregation round {round}: Loss = {loss}, Accuracy = {accuracy}")

global_model = create_falcon_model_with_lora()
client_model = create_falcon_model_with_lora()

# Check if the weights of both models have the same shape
global_weights = global_model.get_weights()
client_weights = client_model.get_weights()

for g_w, c_w in zip(global_weights, client_weights):
    assert g_w.shape == c_w.shape, "Mismatch in weight shapes"

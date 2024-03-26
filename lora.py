import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Dense
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
for i in range(min(3, len(first_batch_labels))):  # Print up to 3 examples
    print('Encoded text:', first_batch_examples[i])
    print('Label:', first_batch_labels.numpy()[i])

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

falcon_model = create_falcon_model_with_bert()
falcon_model.summary()
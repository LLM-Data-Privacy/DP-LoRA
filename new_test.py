# Now, I want to use falcon pretrained model to train a dataset called IMDb Movie Reviews for Sentiment Analysis.
# I am going to split the dataset into five parts to do federated learning and fine tune with lora. 

# Setting up the environment
import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow Version:", tf.__version__)
print("TensorFlow Datasets Version:", tfds.__version__)


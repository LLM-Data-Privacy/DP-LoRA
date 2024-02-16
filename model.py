# Importing necessary libraries
from PIL import Image  # Library for image processing
import requests  # Library for making HTTP requests

from transformers import CLIPProcessor, CLIPModel  # Importing CLIP processor and model

# Loading the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # Load the CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")  # Load the CLIP processor

# URL of the image to be analyzed
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

# Fetching and opening the image from the URL
image = Image.open(requests.get(url, stream=True).raw)

# Processing inputs for the model
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],  # Textual descriptions of the image
    images=image,  # Image(s) to be processed
    return_tensors="pt",  # Output tensors format (PyTorch tensors)
    padding=True  # Padding inputs to maximum length
)

# Performing inference using the CLIP model
outputs = model(**inputs)  # Pass preprocessed inputs to the model

# Extracting logits (raw scores) per image for each text description
logits_per_image = outputs.logits_per_image

# Applying softmax to obtain probabilities for each label
probs = logits_per_image.softmax(dim=1)  # Softmax along dimension 1 to get label probabilities

print()
print("End!")
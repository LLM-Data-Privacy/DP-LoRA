# Setting up the environment
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Load and Prepare the Data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into ten parts for federated learning simulation
num_clients = 10
subset_sizes = [len(dataset) // num_clients] * num_clients
# Adjust the last subset size to include any remaining elements
subset_sizes[-1] += len(dataset) % num_clients
datasets = torch.utils.data.random_split(dataset, subset_sizes)


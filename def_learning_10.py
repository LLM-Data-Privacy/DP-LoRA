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

# Define the Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# Instantiate the model and setup device
model = SimpleNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the Federated Training Loop
def federated_train(model, device, train_loaders, optimizer, epoch):
    model.train()
    for client_idx, train_loader in enumerate(train_loaders):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Client: {client_idx}, Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


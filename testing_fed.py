#!/usr/bin/env python
# coding: utf-8

# # Federated Learning Demo with MNIST Dataset
# 
# This notebook demonstrates a simple federated learning scenario using the MNIST dataset.
# 
# The essential idea is that, we want to prove that by using fedrated learning, (meaning that spliting the dataset into two to train a model), it performs similarily when compared with training as one. 
# 

# In[ ]:


# Setting up the environment
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from torchvision.datasets import MNIST



# In[ ]:


# Load and Prepare the Data (Code Cell)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into two parts for federated learning simulation
train_size = int(0.5 * len(dataset))
subset_sizes = [train_size, len(dataset) - train_size]
datasets = torch.utils.data.random_split(dataset, subset_sizes)


# In[ ]:


# Define the Neural Network Model (Code Cell)

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


# In[ ]:


# Define the Training Loop

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# In[ ]:


# Evaluation

test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')


# Assuming model and device are defined in previous cells as in the provided snippets
test(model, device, test_loader)


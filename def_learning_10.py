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

# Create DataLoader for each subset for federated learning
train_loaders = [DataLoader(subset, batch_size=64, shuffle=True) for subset in datasets]

# Example: Federated Training of the model on ten different subsets
optimizer = optim.Adam(model.parameters())
for epoch in range(1, 11):  # For example, 10 epochs
    federated_train(model, device, train_loaders, optimizer, epoch)

# Evaluation
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
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

# Test the model
test(model, device, test_loader)
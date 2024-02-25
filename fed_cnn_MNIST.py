import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def split_dataset_for_federated_learning(dataset, num_clients):
    total_size = len(dataset)
    part_size = total_size // num_clients
    indices = np.random.permutation(total_size)
    return [Subset(dataset, indices[i*part_size:(i+1)*part_size]) for i in range(num_clients)]

num_clients = 5
client_datasets = split_dataset_for_federated_learning(full_train_dataset, num_clients)
client_loaders = [DataLoader(dataset, batch_size=64, shuffle=True) for dataset in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop2d = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.drop2d(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

global_model = SimpleCNN().to(device)

# Local Training Function
def train_local_model(client_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in client_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total * 100
    return epoch_loss, epoch_accuracy, total

# Aggregation Function
def federated_aggregate(global_model, client_models):
    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.mean(torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))]), 0)
    global_model.load_state_dict(global_state_dict)

# Evaluation Function
def evaluate_global_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= total
    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

# Federated Learning Process
num_epochs = 20
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    client_models = [SimpleCNN().to(device) for _ in range(num_clients)]
    total_samples = 0
    weighted_epoch_accuracy = 0
    
    for client_model, client_loader in zip(client_models, client_loaders):
        client_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.5)
        loss, accuracy, samples = train_local_model(client_loader, client_model, criterion, optimizer)
        total_samples += samples
        weighted_epoch_accuracy += accuracy * samples
    
    overall_accuracy = weighted_epoch_accuracy / total_samples
    print(f"Epoch {epoch+1}/{num_epochs} - Overall Training Accuracy: {overall_accuracy:.2f}%")
    
    federated_aggregate(global_model, client_models)
    
    evaluate_global_model(global_model, test_loader, criterion)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to range [-1, 1]
])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define convolutional and fully connected layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train function
def train(model, trainloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# Test function
def test(model, testloader, verbose=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if verbose: print(f'Accuracy on test set: {100 * correct / total}%')
    return 100 * correct / total
  
# Train non-partitioned model
non_partitioned_model = CNN()
non_partitioned_optimizer = optim.Adam(non_partitioned_model.parameters(), lr=0.001)
non_partitioned_criterion = nn.CrossEntropyLoss()
print("Training non-partitioned model...")
train(non_partitioned_model, trainloader, non_partitioned_criterion, non_partitioned_optimizer)
print("Testing non-partitioned model...")
test(non_partitioned_model, testloader)

#=============================================================================================

# Train partitioned model (simulate federated learning)
num_parts = 5
partition_size = len(trainset) // num_parts
data_partitions = [torch.utils.data.Subset(trainset, range(i * partition_size, (i + 1) * partition_size)) for i in range(num_parts)]

partitioned_models, partitioned_optimizers = [], []
for _ in range(num_parts):
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training partitioned model...")

    train(model, torch.utils.data.DataLoader(data_partitions[_], batch_size=32, shuffle=True), criterion, optimizer, epochs=8)
    partitioned_models.append(model)
    partitioned_optimizers.append(optimizer)    
    
# Computing Weight Averaging
partition_accuracy = [test(model, testloader, verbose=False) for model in partitioned_models]
total_accuracy = sum(partition_accuracy)
weights = [accuracy / total_accuracy for accuracy in partition_accuracy]

# Aggregate model updates
print("Aggregating model updates...")
for i in range(1, num_parts):
    weight = weights[i]
    for params_source, params_target in zip(partitioned_models[i].parameters(), partitioned_models[0].parameters()):
        params_target.data += weight * params_source.data

aggregated_model = partitioned_models[0]

# Test federated model
print("Testing aggregated model...")
test(aggregated_model, testloader)

#=============================================================================

print(partition_accuracy)
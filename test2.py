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

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

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
non_partitioned_model = ResNet(ResNetBlock, [2, 2, 2])
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
    model = ResNet(ResNetBlock, [2, 2, 2])
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
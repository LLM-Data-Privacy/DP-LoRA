import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the pixel values to range [-1, 1] for each channel
])

# Load the CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Write a ResNet class with the input of 3 channels, 32x32 images, and 10 labels
# Write a ResidualBlock class with 2 convolutional layers and a skip connection

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # Update input channels to 3 for CIFAR10
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AvgPool2d(8)  # Update average pooling size for CIFAR10
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Train Function
def train(model, trainloader, criterion, optimizer, epochs=5):
    for epoch in tqdm(range(epochs)):
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
    print('Finished Training')

# Test Function
def test(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

#!=============================================================================
print("Beginning Model Training")

# Train non-partitioned model
model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10)  # Change num_classes to 10 for CIFAR10
criterion = nn.CrossEntropyLoss()
#optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)  # Example learning rate (lr) and alpha values
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed optimizer to Adam
train(model, trainloader, criterion, optimizer, epochs=5)  # Reduced epochs for faster testing
test(model, testloader)
print("Concluding Model Training")
#!=============================================================================

# Partition the dataset into 5 pieces and train the model on each piece, then combine the models
# Partition the dataset
partitioned_trainset = []
partitioned_trainloader = []
for i in range(5):
    partitioned_trainset.append(torch.utils.data.Subset(trainset, list(range(i * 10000, (i + 1) * 10000))))
    partitioned_trainloader.append(torch.utils.data.DataLoader(partitioned_trainset[i], batch_size=32, shuffle=True))

print("Beginning Federated Model Training")
# Train the models
models = []
for i in range(5):
    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10)  # Change num_classes to 10 for CIFAR10
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)  # Example learning rate (lr) and alpha values
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed optimizer to Adam
    train(model, partitioned_trainloader[i], criterion, optimizer, epochs=5)  # Reduced epochs for faster testing
    models.append(model)

# Combine the models using weighted averaging based on loss
losses = []
for i in range(5):
    loss = 0
    for data in partitioned_trainloader[i]:
        inputs, labels = data
        outputs = models[i](inputs)
        loss += criterion(outputs, labels).item()
    losses.append(loss)

# Calculate weights based on loss
weights = [loss / sum(losses) for loss in losses]

# Combine the models using weighted averaging
combined_model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10)  # Change num_classes to 10 for CIFAR10
combined_model.load_state_dict(models[0].state_dict())  # Initialize combined model with first model
for i in range(1, 5):
    for combined_param, param in zip(combined_model.parameters(), models[i].parameters()):
        combined_param.data += param.data * weights[i]

# Test the combined model
test(combined_model, testloader)
print("Concluding Federated Model Training")
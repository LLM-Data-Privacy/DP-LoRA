import requests
from io import BytesIO
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model import SimpleModel  # 确保这与服务器使用的模型定义相同

def download_model(url="http://localhost:5000/model"):
    """从服务器下载模型"""
    response = requests.get(url)
    if response.status_code == 200:
        buffer = BytesIO(response.content)
        model_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
        model = SimpleModel()
        model.load_state_dict(model_state_dict)
        return model
    else:
        print("Failed to download the model")
        return None

def train_model(model, train_loader, epochs=1):
    """使用本地数据集训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def upload_updates(model, url="http://localhost:5000/update"):
    """上传模型更新到服务器"""
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    response = requests.post(url, files={"file": buffer})
    if response.status_code == 200:
        print("Successfully uploaded the updates")
    else:
        print("Failed to upload the updates")

# 设置数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 执行流程
model = download_model()
if model:
    train_model(model, train_loader, epochs=1)
    upload_updates(model)

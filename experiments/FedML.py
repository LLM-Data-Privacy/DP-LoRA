import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import random_split

import torch.nn as nn
import torch.nn.functional as F

#define global variables: FL Node amount, batch size, learning rate, epochs
global FL_NODE_AMOUNT 
global BATCH_64
global BATCH_128
global LEARNING_RATE
global EPOCHS

FL_NODE_AMOUNT = 5
BATCH_64 = 64
BATCH_128 = 128
LEARNING_RATE = 0.01
EPOCHS = 5

def weighted_federated_averaging(model_sets, weights):
    """
    Weighted federated averaging.

    Args:
        model_sets (list of model): The list containing the models from each client.
        weights (list of float): The list containing the weights for each model, which could be based on their loss or accuracy.

    Returns:
        global_model (model): The global model after weighted federated averaging.
    """
    global_model = model_sets[0]  # Initialize the global model as the first model

    # Ensure that all models are in eval mode
    for model in model_sets:
        model.eval()

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [weight/total_weight for weight in weights]

    # Initialize global model parameters with zeros
    for global_param in global_model.parameters():
        global_param.data *=normalized_weights[0]

    # Accumulate weighted parameters from each model
    for each_model in range(1,len(model_sets)):
        weight = normalized_weights[each_model]
        for (global_param, local_param) in zip(global_model.parameters(), model_sets[each_model].parameters()):
            global_param.data += local_param.data * weight
    return global_model



def get_mnist(data_path: str = './data'):
    '''This function downloads the MNIST dataset into the `data_path`
    directory if it is not there already. WE construct the train/test
    split by converting the images into tensors and normalising them'''
    
    # transformation to convert images to tensors and apply normalisation
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # prepare train and test set
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def prepare_dataset(num_partitions: int,
                    batch_size: int,
                    val_ratio: float = 0.1):

    """This function partitions the training set into N disjoint
    subsets, each will become the local dataset of a client. This
    function also subsequently partitions each traininset partition
    into train and validation. The test set is left intact and will
    be used by the central server to asses the performance of the
    global model. """

    # get the MNIST dataset
    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    # create dataloader for the test set
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader




class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        model_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            model_loss += loss.item() / len(trainloader)
        #set the loss to have 4 decimal places
        model_loss = round(model_loss, 4)
    return net, model_loss

def test(net, testloader):
    """Validate the network on the entire test set."""
    
    correct = 0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return accuracy


def run_centralised(trainloader, testloader, epochs: int, lr: float, momentum: float=0.9):
    """A minimal (but complete) training loop"""
    # instantiate the model
    model = Net()
    print("Model initialised")
    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # get dataset and construct a dataloaders
    
    print("@train test loaders all clear@")
    # train for the specified number of 
    print("training process stating...")
    trained_model, loss = train(model, trainloader, optim, epochs)
    print("@training completed@")
    # training is completed, then evaluate model on the test set
    print("testing process starting ...")
    accuracy = test(trained_model, testloader)
    print("@testing  completed@")
    print(f"{loss = }")
    print(f"{accuracy = }")
    return loss, accuracy, trained_model
    
if __name__ == "__main__":
    # trainset, testset = get_mnist()
    
    
    # print("tradition CNN model for MNIST STARTS")
    
    # trainloader = DataLoader(trainset, batch_size=BATCH_64, shuffle=True, num_workers=2)
    # testloader = DataLoader(testset, batch_size=BATCH_128)
    
    # loss, accracy, trained_model= run_centralised(trainloader,testloader,epochs=EPOCHS, lr=LEARNING_RATE)
    
    
    print("\n------------------------------------\n")
    print("FL process with CNN for MNIST STARTS")
    trainloaders, valloaders, testloader = prepare_dataset(num_partitions=FL_NODE_AMOUNT, batch_size=BATCH_64)
    print("@Data preparation completed@")
    
    model_sets =[]
    accuracy_sets = []
    loss_sets = [] 
    for client_index in range(FL_NODE_AMOUNT):
        print(f"Training client {client_index}")
        
        trainloader = trainloaders[client_index]
        # partial_model = Net(num_classes=10)
        # partial_optimizer = torch.optim.SGD(partial_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        # partial_model = train(trained_model, trainloader,partial_optimizer , EPOCHS)
        loss,accracy,partial_model = run_centralised(trainloader, testloader, epochs=EPOCHS, lr=LEARNING_RATE)
        model_sets.append(partial_model)
        accuracy_sets.append(accracy)
        loss_sets.append(loss)
        
        
        print(f"Client {client_index} training completed")
        print()
    
    #finally I get the average of the weights of the trained models
    # this is the global model
    global_model = weighted_federated_averaging(model_sets, accuracy_sets)
    accuracy = test(global_model, testloader)
    
    print(f"Global model accuracy: {accuracy}")
    print("FL process with CNN for MNIST ENDS")
    
    

    
    
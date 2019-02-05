import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

## Ways to create neural network with torch

batch_size = 64

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

inputs = images.view(images.shape[0], -1)

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    return torch.exp(x) / (torch.sum(torch.exp(x), dim=1).view(-1, 1))

# Option 1
# Manually creating matrices for every layer
w1 = torch.randn(784, 256)
b1 = torch.randn(1, 256)
w2 = torch.randn(256, 10)
b2 = torch.randn(1, 10)
h1 = sigmoid(torch.mm(inputs, w1)+b1)
out = torch.mm(h1, w2)+b2
probabilities = softmax(out)

# Option 2
# Use torch.nn
class Network2(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

# Option 3
# Use torch.nn
# Use functional also
class Network3(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x

# Option 4 {Practical Example)
# Creating a fully connected 3 layer network for MNIST image
class Networklet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# To alter weights and biases of a layer
model = Networklet()
print(model.fc1.weight)
print(model.fc1.bias)
model.fc1.bias.data.fill_(0)
model.fc1.weight.data.normal_(std=0.01)

#Forward pass on Networklet
# Grab some data
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size
# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])
img = images[img_idx]

# Option 5
# Use torch.nn
# Use functional also
# Use nn.Sequential also
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])

# Option 5
# Use torch.nn
# Use functional also
# Use nn.Sequential also
# Use OrderedDict also
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
print(model[0])
print(model.fc1)
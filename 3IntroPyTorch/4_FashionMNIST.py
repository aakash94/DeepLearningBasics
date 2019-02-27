import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim

'''
Simple fully connected NN to classify Fashion MNIST
'''

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# The Network

class FashionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc10 = nn.Linear(784, 512)
        self.fc15 = nn.Linear(512, 128)
        self.fc20 = nn.Linear(128, 64)
        self.fc30 = nn.Linear(64, 10)

    def forward(self, x):
        # x is probably autograd tensor with requires_grad = True
        x = self.fc10(x)
        x = F.relu(x)
        x = self.fc15(x)
        x = F.relu(x)
        x = self.fc20(x)
        x = F.relu(x)
        x = self.fc30(x)
        x = F.log_softmax(x, dim=1)

        return x


# Create the network, define the criterion and optimizer
fashion_model = FashionNetwork()
criterion = nn.NLLLoss()
# optimizer = optim.SGD(fashion_model.parameters(), lr=0.003)
optimizer = optim.Adam(fashion_model.parameters(), lr=0.003)

# Train the network here
epochs = 10

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        images = images.view(images.shape[0],-1)
        logits = fashion_model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")


# Get prediction  on an image

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# Calculate the class probabilities (softmax) for img
ps = torch.exp(fashion_model(img)) # exp because last layer is log softmax
print(ps)


# Note: Better results can be obtained by using better hyperparameters, this is just a demonstration
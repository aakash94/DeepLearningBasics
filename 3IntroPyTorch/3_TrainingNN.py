import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict
from torch import optim

'''

some basic code on loss, gradients and optimizers
simple MNIST classification using a fully connected multi layer network

'''

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


## simple feed forward network example with cross entropy loss
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),nn.Linear(64, 10))

# define cross entropy loss
criterion = nn.CrossEntropyLoss()

images, labels = next(iter(trainloader))

# flatten images
images = images.view(images.shape[0],-1) # shape[0], corresponds to the batch size used

# pass through the network, get logits
logits = model(images)

# calculate cross entropy loss
loss = criterion(logits, labels)

print(loss)

# Note CrossEntropyLoss can be split into Log softmax and negative log loss functions.
# nn.LogSoftmax or F.log_softmax applies log to output of softmax https://pytorch.org/docs/stable/nn.html#torch.nn.LogSoftmax
# nn.NLLLoss expects the inputs to be log probablities of each class. https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss


# Same operations as above but with loss split up
model = nn.Sequential(OrderedDict([
    ('FC1', nn.Linear(784,128)),
    ('ReLU1', nn.ReLU()),
    ('FC2', nn.Linear(128,64)),
    ('ReLU2', nn.ReLU()),
    ('FC3', nn.Linear(64, 10)),
    ('LSM', nn.LogSoftmax(dim=1))
]))

criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
logits = model(images)
loss = criterion(logits, labels)
print(logits)


# Autograd
# gradients in pytorch.
# by requires_grad = True on a tensor pytorch will keep track of operations on it
# and automatically calculate gradient of tensor on a variable by calling .backward on the variable
# turn on or off gradients altogether with torch.set_grad_enabled(True|False)
a = torch.tensor([[2.0,4.0],[4.0,8.0]], requires_grad = True)
a1 = torch.tensor([[2.0,4.0],[4.0,8.0]], requires_grad = True)
a2 = torch.tensor([[2.0,4.0],[4.0,8.0]], requires_grad = True)
b = a1*a2
c = b**2
d = torch.sum(c)
d.backward()
print("a ",a)
print("b ",b)
print("c ",c)
print("d ",d)
print("a1 grad ", a1.grad)
print("a2 grad ", a2.grad)
print("b grad ", b.grad)
print("c grad ", c.grad)
print("d grad ", d.grad)

loss.backward()

# Optimizers
# this tweaks the weights of tensor based on the gradients stored.
# possible to use various optimization functions
# optim.SGD

optimizer = optim.SGD(model.parameters(), lr=0.1)
print("Original weights\n", model[0].weight)
optimizer.step()
print("Updated weights\n", model[0].weight)


# Finally
# Everything put together

epochs = 10
model = nn.Sequential(OrderedDict([
    ('FC1', nn.Linear(784,128)),
    ('ReLU1', nn.ReLU()),
    ('FC2', nn.Linear(128,64)),
    ('ReLU2', nn.ReLU()),
    ('FC3', nn.Linear(64, 10)),
    ('LSM', nn.LogSoftmax(dim=1))
]))
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")
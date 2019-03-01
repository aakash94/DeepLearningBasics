import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim
import matplotlib.pyplot as plt


'''
How to go about inference and validation
'''

import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# Sample Classifier class
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


# Understanding TopK
x = torch.tensor([[10.,11.,12.,13.],
                  [37.,36.,35.,34.],
                  [89.,90.,88.,92.]])

# to get top 3 values and indices of those values in each row, dim = 1 for ops along the row
top3_val, top3_ind = torch.topk(x, 3, dim=1)
print(top3_val)
print(top3_ind)

# To get the top 1 value and its position
t_p, t_class = x.topk(1, dim=1)
print(t_p)
print(t_class)

# topk on results of an untrained network (1 single batch)
model = Classifier()
images, labels = next(iter(testloader))
# Get the class probabilities
ps = torch.exp(model(images))
# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
print(ps.shape)
top_p, top_class = ps.topk(1, dim=1)
# Look at the most likely classes for the first 64 examples
print(top_class[:64,:])


# top_class = (64,1)
# labels = (64) , not 1 hot encoded (64,10)
# need to modify shape before equating
equals = top_class == labels.view(*top_class.shape)

# print accuracy
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')


# model with dropout to avoid Overfitting

class Network(nn.Module):

    def __init__(self):
        super().__init__();
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.lsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # make sure tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.lsoftmax(x)
        return x


model = Network()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
epochs = 30
train_error = []
test_error = []
for e in range(epochs):

    running_loss = 0
    testing_loss = 0
    accuracy = 0
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    else:
        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                logits = model(images)
                prediction = torch.exp(logits)
                loss = criterion(logits, labels)
                testing_loss += loss.item()

                top_p, top_class = prediction.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_error.append(running_loss / len(trainloader))
            test_error.append(testing_loss / len(testloader))

        # print(f'Accuracy: {accuracy.item()*100}%')
        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
              "Test Loss: {:.3f}.. ".format(testing_loss / len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))



# Plot Loss
plt.plot(train_error, label='Training loss')
plt.plot(test_error, label='Validation loss')
plt.legend(frameon=False)



# Test out network
model.eval()
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)
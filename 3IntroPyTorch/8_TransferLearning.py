import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import  OrderedDict

data_dir = 'Cat_Dog_data'

train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]
)

test_transforms = transforms.Compose(
    [   transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]
)

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Load Densenet121 model
model = models.densenet121(pretrained=True)
# Cnn part of net = feature detector
# Fc part of net is unique to imagenet and should be replaced for other problems.



# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad=False;

classifier = nn.Sequential(
    OrderedDict(
        [
            ('fc1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(512, 2)),
            ('output', nn.LogSoftmax(dim=1))
        ]
    )
)

# replace classifier from imagenet with our new untrained classifier
model.classifier = classifier




# to go to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





# Finally
model = models.densenet121(pretrained = True)

# Freeze all the pretrained values
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    OrderedDict(
        [
            ('FC1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('FC2', nn.Linear(512, 2)),
            ('output', nn.LogSoftmax(dim=1))
        ]
    )
)

# replace classifier from imagenet with our new untrained classifier
model.classifier = classifier
criterion = nn.NLLLoss();
# Only train the classifier params and no other params.
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)

# move model to appropiate gpu
model.to(device);

epochs = 5
steps = 0
print_every = 10

test_loss = 0
train_loss = 0
for e in range(epochs):
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        steps += 1
        optimizer.zero_grad()
        logits = model.forward(images)
        loss = criterion(logits, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for _images, _labels in testloader:
                    _images = _images.to(device)
                    _labels = _labels.to(device)
                    _logits = model.forward(_images)
                    _loss = criterion(_logits, _labels)
                    test_loss += _loss.item()

                    probablity = torch.exp(_logits)
                    top_p, top_class = probablity.topk(1, dim=1)
                    equals = top_class == _labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {e + 1}/{epochs}.. "
                  f"Train loss: {train_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(testloader):.3f}")

            train_loss = 0;
            test_loss = 0
            model.train()
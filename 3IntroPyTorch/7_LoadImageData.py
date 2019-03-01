import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms


# To use datasets.ImageFolder, have images in folders named by its class.
# Path is where the where all the class folders lie.

# get data from here
# https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
data_dir = '/Cat_Dog_data/train'


# Transform is an operation applied to an image before it is read.
# Like pre-processing
# can have lots of operations on the transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
)

# Create dataset from ImageFolder with transformation applied to the images
dataset = datasets.ImageFolder(data_dir, transforms=transform)


# crete dtloader to read the data from the dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)


data_dir = 'Cat_Dog_data'
# Data augmentation
# introduce randomness in the input data itself

train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0., 0., 0.],  # mean for 3 channels
            [1., 1., 1.]  # std dev for 3 channels
        )
    ]
)

test_trasforms = transforms.Compose(
    [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0., 0., 0.],  # mean for 3 channels
            [1., 1., 1.]  # std dev for 3 channels
        )
    ]
)


# Read train & test data using transforms and create iterator
train_data = datasets.ImageFolder(data_dir+"/train", transform=train_transforms)
test_data = datasets.ImageFolder(data_dir+"/test", transforms=test_trasforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)

# done, data iter created
"""
single capsule network to classify mnist digits.
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets

TRAIN_ON_GPU = torch.cuda.is_available()

if (TRAIN_ON_GPU):
    print('Training on GPU!')
else:
    print('Only CPU available')

# plt.interactive(False)
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

num_workers = 0
batch_size = 32

ip_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

# get datasets
train_data = datasets.MNIST(root='data', train=True, download=True, transform=ip_transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ip_transform)

# data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()


# plot the images in the batch, along with the corresponding labels
# fig = plt.figure(figsize=(25, 4))
# for idx in np.arange(batch_size):
#     ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
#     ax.imshow(np.squeeze(images[idx]), cmap='gray')
#     # print out the correct label for each image
#     # .item() gets the value contained in a Tensor
#     ax.set_title(str(labels[idx].item()))
#
# plt.show()


def squash(input_tensor):
    '''Squashes an input Tensor so it has a magnitude between 0-1.
       param input_tensor: a stack of capsule inputs, s_j
       return: a stack of normalized, capsule output vectors, v_j
    '''
    squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
    scale = squared_norm / (1 + squared_norm)  # normalization coeff
    output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
    return output_tensor


def transpose_softmax(input_tensor, dim=1):
    # transpose input
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    # calculate softmax
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    # un-transpose result
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input_tensor.size()) - 1)


# Initial Convolutino later
class ConvolutionLayer(nn.Module):

    def __init__(self, in_channels=1, out_channels=256):
        super(ConvolutionLayer, self).__init__()
        # define convolution kernel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=1, padding=0)

    def forward(self, x):
        features = F.relu(self.conv(x))
        if TRAIN_ON_GPU:
            features = features.cuda()
        return features


# scalar ip and vector op for primary capsule
class PrimaryCapsule(nn.Module):

    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, padding=0):
        super(PrimaryCapsule, self).__init__()
        # create a list of convnet for capsules
        self.capsules = nn.ModuleList \
                (
                [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
                    for _ in range(num_capsules)
                ]
            )

    def forward(self, x):
        batch_size = x.size(0)
        # get output and reshape them
        u = [capsule(x).view(batch_size, 32 * 6 * 6, 1) for capsule in self.capsules]
        # concatinate outputs
        u = torch.cat(u, dim=-1)  # TOLEARN: Understand why dim =-1 and why caat in such manner
        # squash for non linearity
        u_squash = squash(u)
        return u_squash


# Dynamic Routing
def dynamic_routing(b_ij, u_hat, routing_iteration=3):
    '''
    :param b_ij: var that will yield coupling coeff; initial log probabilities that capsule i should be coupled to capsule j
    :param u_hat: input, weighted capsule vectors, W u; op of last layer after the inverse transform.
    :param routing_iteration: no. of times the dynamic routing should happen. by paper 3-5 times.
    :return: final vector op of the this layer. last layer was u already

    last layer op = u.
    after transform = W u = u_hat, note : this operation is not done in this function. (like the inverse graphics step to remove positional variance or whatever)

    final op of last layer = vj.
    vj = squash(sj)
    sj = sum(cij.u_hat)
    cij = softmax(b ij, over j)
    b ij <- bij + agreement
    agreement = dot product u_hat i.e. transformed vec and v
    '''
    for iteration in range(routing_iteration):
        # softmax calculation of coupling coefficients, c_ij
        c_ij = transpose_softmax(b_ij, dim=2)  # TOLEARN: understand why transpose softmax

        # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)

        # squash
        v_j = squash(s_j)

        if iteration < routing_iteration - 1:
            # get agreement
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)

            # update b_ij
            b_ij += a_ij
    return v_j


# capsules for digits
class DigitCapsule(nn.Module):
    def __init__(self, num_capsule=10, num_cap_prev=8, cap_dim=16, cap_dim_prev=32 * 6 * 6):
        '''
        'inverse graphics' will happen here. u_hat = W*u, supposed to remove variance from representation

        3rd network and 2nd capsule network
        :param num_capsule: number of capsule in current layer. (= no. of class if this is last layer)
        :param num_cap_prev: number of capsules in previous later
        :param cap_dim: elements in current capsule layer = 16
        :param cap_dim_prev: elements in previous layer capsule (32*6*6)
        '''
        super(DigitCapsule, self).__init__()
        self.num_capsules = num_capsule
        self.num_capsules_prev = num_cap_prev
        self.cap_dim = cap_dim
        self.cap_dim_prev = cap_dim_prev
        # TOLEARN: why the shape of W as given below ? how des 4d mat mul happen ?
        self.W = nn.Parameter(torch.randn(num_capsule, cap_dim_prev, num_cap_prev, cap_dim))

    def forward(self, u):
        # TOLEARN: understand whats happening with the matrices below
        u = u[None, :, :, None, :]
        W = self.W[:, None, :, :, :]
        u_hat = torch.matmul(u, W)
        # TOLEARN understand the dimensionality here
        b_ij = torch.zeros(*u_hat.size())
        if TRAIN_ON_GPU:
            b_ij = b_ij.cuda()
        v_j = dynamic_routing(b_ij, u_hat, routing_iteration=3)
        return v_j


# Decoding
class Decoder(nn.Module):
    def __init__(self, input_capsule_dim=16, input_capsules=10, hidden_dim=512):
        super(Decoder, self).__init__()
        input_dim = input_capsule_dim * input_capsules

        self.linear_decoder = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_dim, hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(hidden_dim, hidden_dim * 2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(hidden_dim * 2, 28 * 28 * 1)),
            ('Sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        # probability of each class based on the magnitude of the vector
        class_prob = (x ** 2).sum(dim=-1) ** 0.5
        class_pred = F.softmax(class_prob, dim=-1)

        # Find capsule with max length, i.e. prediction
        _, max_length_idx = class_pred.max(dim=1)

        # identity matrix
        sparse_matrix = torch.eye(10)  # 10 classes
        if TRAIN_ON_GPU:
            sparse_matrix = sparse_matrix.cuda()

        # fancy way to get a vector with 1hot representation of the prediction
        y = sparse_matrix.index_select(dim=0, index=max_length_idx.data)

        # reconstructed pixels
        # so that we reconstruct only the predicted class, only have those in the reconstruction matrix
        x = x * y[:, :, None]
        flattened_x = x.view(x.size(0), -1)
        reconstructions = self.linear_decoder(flattened_x)

        # return reconstructions, class_pred
        return reconstructions, y


# Final capsule class that usess all the above class.
class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.convLayer = ConvolutionLayer()
        self.primaryCapsle = PrimaryCapsule()
        self.digitCapsule = DigitCapsule()
        self.decoder = Decoder()

    def forward(self, images):
        primary_cap_op = self.primaryCapsle(self.convLayer(images))
        caps_out = self.digitCapsule(primary_cap_op).squeeze().transpose(0, 1)
        reconstruction, y = self.decoder(caps_out)
        return caps_out, reconstruction, y


capsnet = CapsuleNetwork()
print(capsnet)


# Loss for Capsule;
class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstructionLoss = nn.MSELoss(size_average=False)

    def forward(self, x, labels, images, reconstructions):
        batch_size = x.size(0)

        # TOTRY: maybe skip this, pass logits in above decoder fn instead of y
        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        # get correct and incorrect loss
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        margin_loss = labels * left + (1. - labels) * right
        margin_loss = margin_loss.sum()

        # get reconstruction loss
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstructionLoss(reconstructions, images)

        # return a weighted sum
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


criterion = CapsuleLoss()
optimizer = optim.Adam(capsnet.parameters())

if TRAIN_ON_GPU:
    capsnet = capsnet.cuda()


# train
def train(capsule_net, criterion, optimizer, n_epochs=1, print_every=1000):
    losses = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        capsule_net.train()

        for batch_i, (images, target) in enumerate(train_loader):

            # reshape and get target class
            target = torch.eye(10).index_select(dim=0, index=target)

            if TRAIN_ON_GPU:
                images, target = images.cuda(), target.cuda()

            # zero out gradients
            optimizer.zero_grad()
            # get model outputs
            caps_output, reconstructions, y = capsule_net(images)
            # calculate loss
            loss = criterion(caps_output, target, images, reconstructions)
            # perform backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # accumulated training loss

            # print and record training stats
            if batch_i != 0 and batch_i % print_every == 0:
                avg_train_loss = train_loss / print_every
                losses.append(avg_train_loss)
                print('Epoch: {} \tTraining Loss: {:.8f}'.format(epoch, avg_train_loss))
                train_loss = 0  # reset accumulated training loss

    return losses


n_epochs = 3
losses = train(capsnet, criterion, optimizer, n_epochs=n_epochs, print_every=100)

plt.plot(losses)
plt.title("Training Loss")
plt.show()


def test(capsule_net, test_loader):
    '''Prints out test statistics for a given capsule net.
       param capsule_net: trained capsule network
       param test_loader: test dataloader
       return: returns last batch of test image data and corresponding reconstructions
       '''
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    test_loss = 0  # loss tracking

    capsule_net.eval()  # eval mode

    for batch_i, (images, target) in enumerate(test_loader):
        target = torch.eye(10).index_select(dim=0, index=target)

        batch_size = images.size(0)

        if TRAIN_ON_GPU:
            images, target = images.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model
        caps_output, reconstructions, y = capsule_net(images)
        # calculate the loss
        loss = criterion(caps_output, target, images, reconstructions)
        # update average test loss
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(y.data.cpu(), 1)
        _, target_shape = torch.max(target.data.cpu(), 1)

        # compare predictions to true label
        correct = np.squeeze(pred.eq(target_shape.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target_shape.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # avg test loss
    avg_test_loss = test_loss / len(test_loader)
    print('Test Loss: {:.8f}\n'.format(avg_test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            # print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
            pass

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    # return last batch of capsule vectors, images, reconstructions
    return caps_output, images, reconstructions


caps_output, images, reconstructions = test(capsnet, test_loader)


def display_images(images, reconstructions):
    '''Plot one row of original MNIST images and another row (below)
       of their reconstructions.'''
    # convert to numpy images
    images = images.data.cpu().numpy()
    reconstructions = reconstructions.view(-1, 1, 28, 28)
    reconstructions = reconstructions.data.cpu().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(26, 5))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, reconstructions], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


display_images(images, reconstructions)

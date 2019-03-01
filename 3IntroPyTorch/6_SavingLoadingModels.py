import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()


model = Network(784, 10, [512,256,128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

# Now lets view the model we created and trained

print("\nThe model\n\n", model, "\n\n")
print("\nThe state dict keys : \n\n", model.state_dict().keys())


# simple save [Requires the models to have identical shape. Only weights are filled, models are not created]:
torch.save(model.state_dict(),"checkpoint_1.pth")

# simple load
state_dict_1 = torch.load("checkpoint_1.pth")

# simple load into model
model.load_state_dict(state_dict_1)



# To avoid above problem . store information about architecture also in the checkpoint

CHECKPOINT_PATH = "checkpoint.pth"

#below variables are arbitary, anything can be used here.
INPUT_SIZE = "input_size"
OUTPUT_SIZE = "output_size"
HIDDEN_LAYERS_SIZES = "hidden_layers"
STATE_DICT = "state_dict"

checkpoint = {
    INPUT_SIZE:784,
    OUTPUT_SIZE:10,
    HIDDEN_LAYERS_SIZES:[each.out_features for each in model.hidden_layers],
    STATE_DICT:model.state_dict()}

torch.save(checkpoint, CHECKPOINT_PATH)

# Note below function will be unique for every model and checkpoint file.
def load_checkpoint():
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = Network(checkpoint[INPUT_SIZE], checkpoint[OUTPUT_SIZE], checkpoint[HIDDEN_LAYERS_SIZES])
    model.load_state_dict(checkpoint[STATE_DICT])
    return model
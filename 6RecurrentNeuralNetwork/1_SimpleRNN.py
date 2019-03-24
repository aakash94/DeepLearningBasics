import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

plt.figure(figsize=(8, 5))

# how many time steps/data pts are in one batch of data
seq_length = 20

# generate evenly spaced data pts
time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data.resize((seq_length + 1, 1))  # size becomes (seq_length+1, 1), adds an input_size dimension

x = data[:-1]  # all but the last piece of data
y = data[1:]  # all but the first

# display the data
plt.plot(time_steps[1:], x, 'r.', label='input, x')  # x
plt.plot(time_steps[1:], y, 'b.', label='target, y')  # y

plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=10, n_layers=3, batch_first=True):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x = batch size, seq length, input_size (like the size of embedding )
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)
        return r_out, hidden


# decide on hyperparameters
input_size = 1
output_size = 1
hidden_dim = 32
n_layers = 1

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


# train the RNN
def train(rnn, n_steps, print_every):
    # initialize the hidden state
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data
        time_steps = np.linspace(step * np.pi, (step + 1) * np.pi, seq_length + 1)
        data = np.sin(time_steps)
        data.resize((seq_length + 1, 1))  # input_size=1

        x = data[:-1]
        y = data[1:]

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)

        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i % print_every == 0:
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.')  # input
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')  # predictions
            plt.show()

    return rnn


n_steps = 75
print_every = 15

trained_rnn = train(rnn, n_steps, print_every)

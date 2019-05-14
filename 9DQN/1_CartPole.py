
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output, display

import pandas as pd



env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(comment='__')
tag_reward = "reward"
tag_loss = "loss"
tag_ep = "epsilon"

r_buff_header = ['state', 'action', 'next_state', 'reward', 'done']

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.header = r_buff_header
        self.buffer = pd.DataFrame(columns=self.header)        

    def push(self, df_row):
        if self.__len__() == self.capacity:
            # Probably exceeded capacity
            #remove a row (probably 1st one) here 
            self.buffer = self.buffer.iloc[1:]
        #add to dataframe here
        self.buffer = pd.concat([self.buffer, df_row])
        
        
    def insert(self, stateV, actonV, next_stateV, rewardV, doneV):
        # Initialise data to lists. 
        data = [{self.header[0]: stateV, 
                 self.header[1]: actonV, 
                 self.header[2]: next_stateV, 
                 self.header[3]: rewardV, 
                 self.header[4]: doneV}] 
  
        # Creates DataFrame. 
        df = pd.DataFrame(data)
        self.push(df)
            
            
    def sample(self, batch_size=0):
        if batch_size == 0:
            return self.buffer
        else:
            return self.buffer.sample(batch_size)

    
    def __len__(self):
        return self.buffer.shape[0]
    
    
class DqnAgent(nn.Module):
    
    def __init__(self, n_ip, n_op):
        super(DqnAgent, self).__init__()
        self.fc1 = nn.Linear(n_ip, n_ip*16)
        self.fc2 = nn.Linear(n_ip*16, n_ip*16)
        self.fc3 = nn.Linear(n_ip*16, n_op)       
        
    def forward(self,x):        
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
        
        
BATCH_SIZE = 32
GAMMA      = 0.9 # discount factor
EPSILON    = 1
EPSILON_DECAY    = 0.99
LEARN_RATE = 0.001

CHECK_EVERY = 100
OPTIMIZE_EVERY = 1

STATE_N  = 4
ACTION_N = env.action_space.n

OPTIMIZE_COUNT = 1

NUM_EPISODES = 1000


MINREWARD = 30


qvfa = DqnAgent(STATE_N, ACTION_N).double().to(device)
optimizer = optim.Adam(qvfa.parameters(), lr = LEARN_RATE)

criterion = nn.MSELoss()
buffer = ReplayBuffer(1000000)

def select_action(state, ep = 0):    
    
    sample = random.random()
    state = torch.from_numpy(state).to(device)
    if sample < ep:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            op = qvfa(state)
            values, indices = op.max(0)
            return indices.item()

def optimize_model(i_episode = 0):

    if buffer.__len__() < BATCH_SIZE:
        print("optimizing model Not enough samples in buffer : ",buffer.__len__())
        return
    
    transitions = buffer.sample(min(BATCH_SIZE, buffer.__len__()))    
        
    state_batch = transitions[buffer.header[0]].values
    state_batch = torch.from_numpy(np.stack( state_batch, axis=0 )).to(device)
    
    action_batch = torch.tensor(transitions[buffer.header[1]].values.tolist()).view(-1,1).to(device)
    
    next_state_batch = transitions[buffer.header[2]].values
    next_state_batch = torch.from_numpy(np.stack( next_state_batch, axis=0 )).to(device)
    
    reward_batch = torch.tensor(transitions[buffer.header[3]].values.tolist()).view(-1,1).to(device)
    
    done_batch = torch.tensor(transitions[buffer.header[4]].values.tolist()).view(-1,1).to(device)
    
    qsa = qvfa(state_batch).gather(1, action_batch)


    with torch.no_grad():
        qvfa.eval()
        next_state_action_values = qvfa(next_state_batch)
        max_next_state_values, _indices = next_state_action_values.max(dim=1)
        max_next_state_values = max_next_state_values.view(-1,1)
        next_state_values = ((max_next_state_values*GAMMA).float()+reward_batch).float()*(1-done_batch).float()
        target = next_state_values.double()
        qvfa.train()


    # 𝛿=𝑄(𝑠,𝑎)−(𝑟+𝛾max𝑎𝑄(𝑠′,𝑎))
    optimizer.zero_grad()
    loss = criterion(qsa, target)
    loss.backward()
    #for param in qvfa.parameters():param.grad.data.clamp_(-1, 1)
    optimizer.step()
    writer.add_scalar(tag_loss, loss.item(), i_episode)   
    
    
for i_episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render(mode='rgb_array')
        action = select_action(state, ep = EPSILON)
        next_state, reward, done, info = env.step(action)        
        total_reward += reward
        if done:
            reward = -reward
        buffer.insert(state, action, next_state, reward, done)
        state = next_state        
        
    writer.add_scalar(tag_reward, total_reward, i_episode)
    writer.add_scalar(tag_ep, EPSILON, i_episode)
    for _ in range (OPTIMIZE_COUNT):
        optimize_model(i_episode)
    
    if EPSILON > 0.2 and i_episode > 32 and total_reward > MINREWARD:
        #EPSILON *= EPSILON_DECAY
        EPSILON -= 0.1        
        MINREWARD += 20
        
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show() 
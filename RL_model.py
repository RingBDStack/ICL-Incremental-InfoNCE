import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=20):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(nb_states, hidden, 2, batch_first=True)
        self.fc = nn.Linear(hidden, nb_actions)
        self.LN = nn.LayerNorm(nb_actions)
        self.actn = nn.Sigmoid()
        
        for _, param in self.lstm.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in')
    
    def forward(self, x, hc):
        out, _ = self.lstm(x, hc)
        out = self.fc(out)
        out = self.LN(out)
        out = self.actn(out)
        return out[:, -1, :]

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=10):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states+nb_actions, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.LeakyReLU()
        
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
    
    def forward(self, xs):
        x, a = xs
        x = torch.cat([x, a], 1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

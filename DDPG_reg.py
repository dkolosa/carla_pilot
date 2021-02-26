import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

class Actor(torch.nn.Module):
    def __init__(self, num_state, num_actions, action_bound, layer_1=128, layer_2=128, lr=0.0001, checkpt='ddpg-actor'):
        super(Actor, self).__init__()

        self.chkpt = checkpt + '_actor.ckpt'

        self.action_bound = action_bound
        self.num_state = num_state
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        
        self.fc1 = nn.Linear(*self.num_state, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.output = nn.Linear(layer_2, num_actions)

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])

        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.fc2.bias.data.uniform_(-f2, f2)
        
        self.bn1 = nn.LayerNorm(self.layer_1)
        self.bn2 = nn.LayerNorm(self.layer_2)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        out = torch.tanh(self.output(x))
        return out*self.action_bound

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)


class Critic(torch.nn.Module):
    def __init__(self, n_states, n_action, layer_1, layer_2, lr=0.0001, checkpt='ddpg-critic'):
        super(Critic, self).__init__()

        # self.nam_state = num_state
        self.chkpt = checkpt + '_critic.ckpt'

        self.fc1 = nn.Linear(*n_states, layer_1)
        self.actfc2 = nn.Linear(n_action, layer_2)
        self.bn1 = nn.LayerNorm(layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.output = nn.Linear(layer_2, 1)

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])

        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.output.bias.data.uniform_(-.003, .003)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    

    def forward(self,state, action):
        x = self.fc1(state)
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc2(x))
        action = self.actfc2(action)
        x = F.relu(torch.add(x, action))
        out = self.output(x)

        return out

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

class Actor(torch.nn.Module):
    def __init__(self, num_state, num_actions, layer_1, layer_2, lr=0.0001, checkpt='ppo'):
        super(Actor, self).__init__()

        self.chkpt = checkpt + '_actor.ckpt'

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        
        self.fcin = self.calc_fc_input()
        self.fc1 = nn.Linear(self.fcin, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.output = nn.Linear(layer_2, num_actions)

        self.optim = Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,image):
        x = self.conv1(image)
        x = nn.ReLU(x)
        x = self.conv2(x)
        x = nn.ReLU(x)
        x = self.conv3(x)
        x = nn.ReLU(x)
        x = nn.Flatten(x)
        x = self.fc1(x)
        x = nn.ReLU(x)
        x = self.fc2(x)
        x = nn.ReLU(x)
        out = nn.Tanh(self.output(x))

        return out

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)

    def calc_fc_input(self):
        dat = T.zeros((1,3,96,96))
        x = self.conv1(dat)
        x = nn.ReLU(x)
        x = self.conv2(x)
        x = nn.ReLU(x)
        x = self.conv3(x)
        x = nn.ReLU(x)
        return int(np.prod(x.size()))

class Critic(torch.nn.Module):
    def __init__(self, num_actions=3, layer_1, layer_2, lr=0.0001, checkpt='ppo'):
        super(Critic, self).__init__()

        # self.nam_state = num_state
        self.chkpt = checkpt + '_critic.ckpt'

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)

        fc_input = self.calc_fc_input()
        self.fc1 = nn.Linear(fc_input, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.output = nn.Linear(layer_2, 1)

        self.optim = Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def calc_fc_input(self):
        dat = T.zeros((1,1,96,96))
        x = nn.ReLU(self.conv1(dat))
        x = nn.ReLU(self.conv2(x))
        x = nn.ReLU(self.conv3(x))
        return int(np.prod(x.size()))


    def forward(self,image, action):

        x = self.conv1(image)
        x = nn.ReLU(x)
        x = self.conv2(x)
        x = nn.ReLU(x)
        x = self.conv3(x)
        x = nn.ReLU(x)
        x = nn.Flatten(x)
        x = self.fc1(x)
        x = nn.ReLU(x)
        x = torch.cat((x, action), 0)
        x = self.fc2(x)
        x = nn.ReLU(x)
        out = self.output(x)

        return out

    

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)
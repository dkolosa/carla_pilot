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
        self.num_channels = 3

        self.cnn1 = CNN(3 , 64, 5)
        self.cnn2 = CNN(64, 128, 5)
        self.cnn3 = CNN(128, 128, 5)
        self.cnn4 = CNN(128, 128, 5)

        # Can replace this with an image transformer (?? maybe that woudl be better than CNN)
        self.image_cnn = nn.Sequential(
            self.cnn1,
            nn.MaxPool2d(2),
            self.cnn2,
            nn.MaxPool2d(2),
            self.cnn3,
            nn.MaxPool2d(2),
            self.cnn4
        )

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

    def forward(self, image, sensor):
        x = self.image_cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(torch.add(x, sensor))
        x = F.relu(self.fc2)
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

        self.cnn1 = CNN(3 , 64, 5)
        self.cnn2 = CNN(64, 128, 5)
        self.cnn3 = CNN(128, 128, 5)
        self.cnn4 = CNN(128, 128, 5)


        # Can replace this with an image transformer (?? maybe that woudl be better than CNN)
        self.image_cnn = nn.Sequential(
            self.cnn1,
            nn.MaxUnpool2d(2),
            self.cnn2,
            nn.MaxPool2d(2),
            self.cnn3,
            nn.MaxPool2d(2)
            self.cnn4`
        )

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
    

    def forward(self,image, sensor, action):
        
        x = self.image_cnn(image)
        x = T.view(x.shape[0], -1)
        x = self.fc1(state)
        x = torch.add(x, sensor)
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc2(x))
        action = self.actfc2(action)
        x = F.relu(torch.add(x, action))
        out = self.output(x)

        return out

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)

class CNN(torch.nn.Module):
    def __init__(self, num_channels, num_output, kernels):
        super(CNN, self).__init__()

        self.cnn = nn.Conv2d(in_channel, out_channel, kernel)
        self.bn = nn.BatchNorm2d(num_output)
        self.rel = nn.ReLU()

    def forward(self, input):
        x = self.cnn(input)
        x = self.bn(x)
        x = self.rel(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import torch as T

class Actor(torch.nn.Module):
    def __init__(self, state, num_actions, action_bound, batch_size, layer_1=128, layer_2=128, lr=0.0001,use_mobileNet=False, checkpt='ddpg-actor'):
        super(Actor, self).__init__()

        self.chkpt = checkpt + '_actor.ckpt'

        self.action_bound = action_bound
        self.batch_size = batch_size
        self.num_channels = state[0]
        self.img_h = state[1]
        self.img_w = state[2]
        self.layer_1 = layer_1
        self.layer_2 = layer_2

        kernel = 5
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.use_mobileNet = use_mobileNet

        if self.use_mobileNet:
            self.mobilenet = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
            self.mobilenet.classifer = nn.Linear(1200, layer_1)
            fc1_inputs = layer_1
        
        else:
            self.cnn1 = nn.Conv2d(in_channels=self.num_channels, out_channels=64, kernel_size=kernel)
            self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel)
            self.cnn3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel)
            self.cnn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)

            self.max1 = nn.MaxPool2d(5,2)
            self.max2 = nn.MaxPool2d(5,2)
            self.max3 = nn.MaxPool2d(2,2)

            fc1_inputs = self.calc_cnnweights()


        
        self.fc1 = nn.Linear(fc1_inputs, layer_1)
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
        self.to(self.device)

    def forward(self, image):
        
        if self.Mobilenet_model:
            self.mobilenet(image)
        else:
            x = F.relu(self.cnn1(image))
            x = self.max1(x)
            x = F.relu(self.cnn2(x))
            x = self.max2(x)
            x = F.relu(self.cnn3(x))
            x = self.max3(x)
            x = F.relu(self.cnn4(x))
            # x = self.image_cnn(image)
            x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        # x = F.relu(torch.add(x, sensor))
        x = F.relu(self.bn2(x))
        out = torch.tanh(self.output(x))
        return out*self.action_bound

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)

    def calc_cnnweights(self):
        input = torch.zeros((1, self.num_channels, self.img_h, self.img_w))
        x = self.cnn1(input)
        x = self.max1(x)
        x = self.cnn2(x)
        x = self.max2(x)
        x = self.cnn3(x)
        x = self.max3(x)
        x = self.cnn4(x)
        x = x.view(x.shape[0], -1)
        return x.shape[1]


class Critic(torch.nn.Module):
    def __init__(self, state, n_action, layer_1, layer_2, lr=0.0001, checkpt='ddpg-critic'):
        super(Critic, self).__init__()

        # self.nam_state = num_state
        self.chkpt = checkpt + '_critic.ckpt'

        self.num_channels = state[0]
        self.img_h = state[1]
        self.img_w = state[2]

        # self.cnn1 = CNN(3 , 64, 5)
        # self.cnn2 = CNN(64, 128, 5)
        # self.cnn3 = CNN(128, 128, 5)
        # self.cnn4 = CNN(128, 128, 5)

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5)
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5)
        self.max1 = nn.MaxPool2d(5,2)
        self.max2 = nn.MaxPool2d(5,2)
        self.max3 = nn.MaxPool2d(5,2)

        fc1_inputs = self.calc_cnnweights()

        self.fc1 = nn.Linear(fc1_inputs, layer_1)
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
    

    def forward(self,image, action):
        
        # x = self.image_cnn(image)
        x = F.relu(self.cnn1(image))
        x = self.max1(x)
        x = F.relu(self.cnn2(x))
        x = self.max2(x)
        x = F.relu(self.cnn3(x))
        x = self.max3(x)
        x = F.relu(self.cnn4(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = torch.add(x, sensor)
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc2(x))
        action = self.actfc2(action)
        x = F.relu(torch.add(x, action))
        out = self.output(x)
        return out

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)

    def calc_cnnweights(self):
        input = torch.zeros((1, self.num_channels, self.img_h, self.img_w))
        x = self.cnn1(input)
        x = self.max1(x)
        x = self.cnn2(x)
        x = self.max2(x)
        x = self.cnn3(x)
        x = self.max3(x)
        x = self.cnn4(x)
        x = x.view(x.shape[0], -1)
        return x.shape[1]



class CNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernels):
        super(CNN, self).__init__()

        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rel = nn.ReLU()

    def forward(self, input):
        x = self.cnn(input)
        x = self.bn(x)
        x = self.rel(x)
        return x
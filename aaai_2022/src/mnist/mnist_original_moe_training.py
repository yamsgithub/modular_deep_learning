#!/usr/bin/env python
# coding: utf-8

# # Experiments with MNIST Dataset and Original MoE

# The experiments in this notebook include training the original MoE models as follows:
# 
# 1. original MoE without regularization.
# 2. original MoE with $L_{importance}$ regularization.
# 3. original MoE with $L_s$ regularization.
# 4. train a single model.
import time
import numpy as np
from statistics import mean
from math import ceil, sin, cos, radians
from collections import OrderedDict
import os
from itertools import product

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision.transforms.functional as TF

# import MoE expectation model. All experiments for this dataset are done with the expectation model as it
# provides the best performance
from moe_models.moe_expectation_model import moe_expectation_model
from moe_models.moe_stochastic_model import moe_stochastic_model
from moe_models.moe_top_k_model import moe_top_k_model
from moe_models.moe_models_base import default_optimizer
from helper.moe_models import cross_entropy_loss, stochastic_loss
from helper.visualise_results import *


# ### NOTE: Pre-trained models are provided to check the results of all the experiments if you do not have the time to train all the models. 

# ## Load MNIST dataset

import torchvision.transforms as transforms

# transforms: Convert PIL image to tensors and normalize
mnist_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]) #mean and standard deviation computed from the dataset

# Complete train and test data
trainsize = 50000
valsize = 10000
testsize = 10000

batch_size = 512

# Load data as train and test
trainset = torchvision.datasets.MNIST('./data',
    download=True,
    train=True,
    transform=mnist_transform,
    target_transform = torch.tensor,                                 
    )
testset = torchvision.datasets.MNIST('./data',
    download=True,
    train=False,
    transform=mnist_transform,
    target_transform = torch.tensor,)

# dataloaders

torch.manual_seed(0)
train_set, val_set = torch.utils.data.random_split(trainset, [trainsize, valsize])

trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, range(trainsize)), 
                                          batch_size=batch_size,
                                          shuffle=True)

valloader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_set, range(valsize)), 
                                          batch_size=batch_size,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(testsize)),
                                         batch_size=testsize,
                                         shuffle=False)
num_classes = 10

image, label = trainset.__getitem__(0)
print('Image shape', image.shape)
print('Train samples ', len(train_set))
print('Validation samples', len(val_set))
print('Test samples ', len(testset))


# ## Define expert and gate networks

# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
class expert_layers(nn.Module):
    def __init__(self, num_classes, channels=1):
        super(expert_layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3)
        self.fc1 = nn.Linear(in_features=1*13*13, out_features=5) # this is a pure linear transform
        self.fc2 = nn.Linear(in_features=5, out_features=32) # this is a pure linear transform
        self.mp = nn.MaxPool2d(2,2)
        self.out = nn.Linear(in_features=32, out_features=num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, t):
        # conv1
        t = self.mp(F.relu(self.conv1(t)))
        
        # fc1
        t = t.reshape(-1, 1*13*13)
        t = F.relu(self.fc1(t))

        # fc2
        t = F.relu(self.fc2(t))
        
        self.hidden = t

        # output
        t = F.softmax(self.out(t), dim=1)
                
        return t

# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
class gate_layers(nn.Module):
    def __init__(self, num_experts, init_zero=False):
        super(gate_layers, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.fc1 = nn.Linear(in_features=1*13*13, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.mp = nn.MaxPool2d(2,2)
        self.out = nn.Linear(in_features=32, out_features=num_experts)
        self.num_experts = num_experts
        
        if init_zero:
            self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.MaxPool2d) or isinstance(module, gate_layers):
            return
        module.weight.data.zero_()
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, t, T=1.0, y=None):
        # conv 1
        t = self.mp(F.relu(self.conv1(t)))
        
        # fc1
        t = t.reshape(-1, 1*13*13)
            
        t = F.relu(self.fc1(t))

        # fc2
        t = F.relu(self.fc2(t))
        
        # output expert log loss
        t = self.out(t)
        
        output = F.softmax(t/T, dim=1)

        return output


# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
class gate_layers_top_k(nn.Module):
    def __init__(self, num_experts, init_zero=False):
        super(gate_layers_top_k, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.fc1 = nn.Linear(in_features=1*13*13, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.mp = nn.MaxPool2d(2,2)
        self.out = nn.Linear(in_features=32, out_features=num_experts)
        self.num_experts = num_experts
        
        if init_zero:
            self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.MaxPool2d) or isinstance(module, gate_layers_top_k):
            return
        module.weight.data.zero_()
        if module.bias is not None:
            module.bias.data.zero_()


    def forward(self, t, T=1.0, y=None):
        # conv 1
        t = self.mp(F.relu(self.conv1(t)))
        
        # fc1
        t = t.reshape(-1, 1*13*13)
            
        t = F.relu(self.fc1(t))

        # fc2
        t = F.relu(self.fc2(t))
        
        # output expert log loss
        t = self.out(t)
        
        output = t/T

        return output
    
# Convolutional network with one convolutional layer and 2 hidden layers with ReLU activation
class gate_attn_layers(nn.Module):
    def __init__(self, num_experts, channels=1, init_zero=False):
        super(gate_attn_layers, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3)

        self.fc1 = nn.Linear(in_features=1*13*13, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        
        if init_zero:
            self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.MaxPool2d) or isinstance(module, gate_attn_layers):
            return
        module.weight.data.zero_()
        if module.bias is not None:
            module.bias.data.zero_()
                
    def forward(self, t, T=1.0, y=None):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # fc1
        t = t.reshape(-1, 1*13*13)
            
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        
        return t


# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation. The single model has the same
# architecture as an expert
class single_model(nn.Module):
    def __init__(self, num_classes=10):
        super(single_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.fc1 = nn.Linear(in_features=1*13*13, out_features=5) # this is a pure linear transform
        self.fc2 = nn.Linear(in_features=5, out_features=32) # this is a pure linear transform
        
        self.out = nn.Linear(in_features=32, out_features=num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # fc1
        t = t.reshape(-1, 1*13*13)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = F.softmax(self.out(t), dim=1)
        
        return t


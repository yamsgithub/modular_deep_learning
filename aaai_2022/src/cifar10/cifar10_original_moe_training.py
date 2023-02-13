# Experiments with CIFAR-10 Dataset and Original MoE

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
from copy import deepcopy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# import MoE expectation model. All experiments for this dataset are done with the expectation model as it
# provides the best guarantee of interpretable task decompositions
from moe_models.moe_expectation_model import moe_expectation_model
from moe_models.moe_models_base import default_optimizer
from helper.moe_models import cross_entropy_loss
from helper.visualise_results import *

# Load CIFAR10 dataset

stats = ((0.49,0.48, 0.45),(0.25, 0.24, 0.26))
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

working_path = '/gpfs/data/fs71921/yamunak'

cifar10_trainset = torchvision.datasets.CIFAR10(root=working_path, train=True, download=True, transform=train_transform)
cifar10_testset = torchvision.datasets.CIFAR10(root=working_path, train=False, download=True, transform=test_transform)
cifar10_testset, cifar10_trainset

num_classes = 10

trainsize = 40000
valsize = 10000
testsize =10000

batch_size = 256

torch.manual_seed(0)
cifar10_train_set, cifar10_val_set = torch.utils.data.random_split(cifar10_trainset, [trainsize, valsize])

cifar10_trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(cifar10_train_set, range(trainsize)),
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=2, pin_memory=True)

cifar10_valloader = torch.utils.data.DataLoader(torch.utils.data.Subset(cifar10_val_set, range(valsize)),
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=2, pin_memory=True)

cifar10_testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(cifar10_testset, range(testsize)), 
                                                 batch_size=batch_size,
                                                 shuffle=True, num_workers=2, pin_memory=True)

classes_cifar10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
class expert_layers(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(expert_layers, self).__init__()
        filter_size = 3
        self.filters = 4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.filters*2)
        self.mp = nn.MaxPool2d(2,2)
                
        self.conv3 = nn.Conv2d(in_channels= self.filters*2, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.filters*4, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(self.filters*4)

        self.fc1 = nn.Linear(self.filters*4*2*2,64)
        self.fc2 = nn.Linear(64,32)
        
        self.out = nn.Linear(in_features=32, out_features=num_classes)
                        
    def forward(self, x):
        # conv 1
        
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.bn2(self.conv2(x))))       

        x = self.mp(F.relu(self.conv3(x)))
        x = self.mp(F.relu(self.bn4(self.conv4(x))))
        
        x = x.reshape(-1, self.filters*4*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        self.hidden = x
        
        x = self.out(x)
        
        # output
        x = F.softmax(x, dim=1)            

        return x
    
# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
class expert_layers_conv_2(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(expert_layers_conv_2, self).__init__()
        filter_size = 3
        self.filters = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(self.filters*2*16*16,64)
        
        self.out = nn.Linear(in_features=64, out_features=num_classes)
                        
    def forward(self, x):
        # conv 1
        
        x = F.relu(self.conv1(x))
        x = self.mp(F.relu(self.conv2(x)))
                    
        x = x.reshape(-1, self.filters*2*16*16)
        
        x = F.relu(self.fc1(x))
        
        self.hidden = x
        
        x = self.out(x)
        
        # output
        x = F.softmax(x, dim=1)    

        return x

# Convolutional network with one convolutional layer and 2 hidden layers with ReLU activation
class gate_layers(nn.Module):
    def __init__(self, num_experts, channels=3):
        super(gate_layers, self).__init__()
        # define layers
        filter_size = 3
        self.filters = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.filters*2)
        self.mp = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(in_channels= self.filters*2, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.filters*4, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(self.filters*4)

        self.fc1 = nn.Linear(self.filters*4*2*2, 512)
        self.fc2 = nn.Linear(512, 32)
        
        self.out = nn.Linear(in_features=32, out_features=num_experts)
        
    def forward(self, x, T=1.0, y=None):
        # conv 1        
        
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.bn2(self.conv2(x))))

        x = self.mp(F.relu(self.conv3(x)))
        x = self.mp(F.relu(self.bn4(self.conv4(x))))
        
        x = x.reshape(-1, self.filters*4*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.out(x)
        x = F.softmax(x/T, dim=1)
        
        return x

class gate_layers_top_k(nn.Module):
    def __init__(self, num_experts, channels=3):
        super(gate_layers_top_k, self).__init__()
        # define layers
        filter_size = 3
        self.filters = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.filters*2)
        self.mp = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(in_channels= self.filters*2, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.filters*4, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(self.filters*4)

        self.fc1 = nn.Linear(self.filters*4*2*2, 512)
        self.fc2 = nn.Linear(512, 32)
        
        self.out = nn.Linear(in_features=32, out_features=num_experts)
        
    def forward(self, x, T=1.0, y=None):
        # conv 1        
        
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.bn2(self.conv2(x))))

        x = self.mp(F.relu(self.conv3(x)))
        x = self.mp(F.relu(self.bn4(self.conv4(x))))
        
        x = x.reshape(-1, self.filters*4*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.out(x)
        x = x/T
        
        return x

    
# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
class gate_layers_conv_2_top_k(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(gate_layers_conv_2_top_k, self).__init__()
        filter_size = 3
        self.filters = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(self.filters*2*16*16,64)
        
        self.out = nn.Linear(in_features=64, out_features=num_classes)
                        
    def forward(self, x, T=1.0, y=None):
        # conv 1
        
        x = F.relu(self.conv1(x))
        x = self.mp(F.relu(self.conv2(x)))
            
        # print(x.shape)
        
        x = x.reshape(-1, self.filters*2*16*16)
        
        x = F.relu(self.fc1(x))
        
        x = self.out(x)
        
        # output
        x = x/T

        return x
    
class gate_layers_conv_2(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(gate_layers_conv_2, self).__init__()
        filter_size = 3
        self.filters = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(self.filters*2*16*16,64)
        
        self.out = nn.Linear(in_features=64, out_features=num_classes)
                        
    def forward(self, x, T=1.0, y=None):
        # conv 1
        
        x = F.relu(self.conv1(x))
        x = self.mp(F.relu(self.conv2(x)))
            
        # print(x.shape)
        
        x = x.reshape(-1, self.filters*2*16*16)
        
        x = F.relu(self.fc1(x))
        
        x = self.out(x)
        x = F.softmax(x/T, dim=1)

        return x


# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
class single_model(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(single_model, self).__init__()
        filter_size = 3
        self.filters = 4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.filters*2)
        self.mp = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(in_channels= self.filters*2, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.filters*4, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(self.filters*4)

        self.fc1 = nn.Linear(self.filters*4*2*2,64)
        self.fc2 = nn.Linear(64,32)
        
        self.out = nn.Linear(in_features=32, out_features=num_classes)
                        
    def forward(self, x):
        # conv 1        
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.bn2(self.conv2(x))))       
    
        x = self.mp(F.relu(self.conv3(x)))
        x = self.mp(F.relu(self.bn4(self.conv4(x))))
    
        x = x.reshape(-1, self.filters*4*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.out(x)
        
        # output
        x = F.softmax(x, dim=1)
                
        return x
    
 # Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
# class single_model(nn.Module):
#     def __init__(self, num_classes, channels=3):
#         super(single_model, self).__init__()
#         filter_size = 3
#         self.filters = 16
#         self.mp = nn.MaxPool2d(2,2)
        
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1, bias=False)
        
#         self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=filter_size, stride=1, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=filter_size, stride=1, padding=1, bias=False)
#         self.bn = nn.BatchNorm2d(self.filters)
        
#         self.conv4 = nn.Conv2d(in_channels= self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
#         self.conv5 = nn.Conv2d(in_channels=self.filters*2, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1,bias=False)
#         self.conv6 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1,bias=False)        
#         self.bn2 = nn.BatchNorm2d(self.filters*2)
        
#         self.conv7 = nn.Conv2d(in_channels= self.filters*2, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1, bias=False)
#         self.conv8 = nn.Conv2d(in_channels=self.filters*4, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1,bias=False)
#         self.conv9 = nn.Conv2d(in_channels=self.filters*2, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1,bias=False)
#         self.bn4 = nn.BatchNorm2d(self.filters*4)
        
#         self.fc1 = nn.Linear(self.filters*4*16*16,64)
#         self.fc2 = nn.Linear(64,32)
        
#         self.out = nn.Linear(in_features=32, out_features=num_classes)
                        
#     def forward(self, x):
#         # conv 1        
#         x = F.relu(self.bn(self.conv1(x)))
        
#         y = F.relu(self.bn(self.conv2(x)))  
#         y = self.bn(self.conv3(y))
                
#         x = F.relu(torch.add(x,y))
        
#         y = F.relu(self.bn2(self.conv4(x)))
#         y = self.bn2(self.conv5(y))
#         x = self.conv6(x)
        
#         x = F.relu(torch.add(x,y))
        
#         y = F.relu(self.bn4(self.conv7(x)))
#         y = self.bn4(self.conv8(y))
#         x = self.conv9(x)
        
#         x = self.mp(F.relu(torch.add(x,y)))
        
#         # print(x.shape)
        
#         x = x.reshape(-1, self.filters*4*16*16)
#         x = F.dropout(x, 0.2)
        
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, 0.2)
        
#         x = F.relu(self.fc2(x))
        
#         x = self.out(x)
        
#         # output
#         x = F.softmax(x, dim=1)    

#         return x

# Functions to train models

# Function to train original model with and without regularization
# 
# * w_importance_range is the range of values for the $w_{importance}$ hyperparameter of the $L_{importance}$ regularization.
# * w_sample_sim_same_range is the range of values for $\beta_s$ hyperparameter of the $L_s$ regularization.
# * w_sample_sim_diff_range is the range of values for $\beta_d$ hyperparameter of the $L_s$ regularization.

# def train_original_model(model_1, trainloader, testloader, runs, T=[1.0]*20, 
#                          w_importance=0.0, w_sample_sim_same=0.0, 
#                          w_sample_sim_diff=0.0,
#                          num_classes=10, total_experts=5, num_epochs=20):
        
#     print('w_importance','{:.1f}'.format(w_importance))
#     if w_sample_sim_same < 1:
#         print('w_sample_sim_same',str(w_sample_sim_same))
#     else:
#         print('w_sample_sim_same','{:.1f}'.format(w_sample_sim_same))

#     if w_sample_sim_diff < 1:
#         print('w_sample_sim_diff',str(w_sample_sim_diff))
#     else:
#         print('w_sample_sim_diff','{:.1f}'.format(w_sample_sim_diff))


#     for run in range(1, runs+1):

#         print('Run:', run)

#         n_run_models_1 = []

#         models = {'moe_expectation_model':{'model':moe_expectation_model,'loss':cross_entropy_loss().to(device),
#                                            'experts':{}},}
#         for key, val in models.items():

#             expert_models = experts(total_experts, num_classes).to(device)

#             gate_model = gate_layers(total_experts).to(device)

#             moe_model = val['model'](total_experts, num_classes,
#                                      experts=expert_models, gate=gate_model, device=device).to(device)

#             optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False, weight_decay=1e-3)
#             optimizer = default_optimizer(optimizer_moe=optimizer_moe)

#             hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer = optimizer,
#                                    T = T, w_importance=w_importance, w_sample_sim_same = w_sample_sim_same, 
#                                    w_sample_sim_diff = w_sample_sim_diff, 
#                                    accuracy=accuracy, epochs=num_epochs)
#             val['experts'][total_experts] = {'model':moe_model, 'history':hist}                


#         # Save all the trained models
#         plot_file = generate_plot_file(model_1, T[0], w_importance=w_importance, w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
#                                        specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

#         if os.path.exists(os.path.join(model_path, plot_file)):
#             n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))
#         n_run_models_1.append(models)
#         torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
#         n_run_models_1 = []






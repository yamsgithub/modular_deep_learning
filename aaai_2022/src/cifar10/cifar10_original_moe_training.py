# Experiments with CIFAR-10 Dataset and Original MoE

# The experiments in this notebook include training the original MoE models as follows:
# 
# 1. original MoE without regularization.
# 2. original MoE with $L_{importance}$ regularization.
# 3. original MoE with $L_s$ regularization.
# 4. train a single model.

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm  # colormaps

import seaborn as sns
import numpy as np
from statistics import mean
from math import ceil, sin, cos, radians
from collections import OrderedDict
import os
import pandas as pd
from pprint import pprint
from copy import deepcopy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import TensorDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

# import MoE expectation model. All experiments for this dataset are done with the expectation model as it
# provides the best guarantee of interpretable task decompositions
from moe_models.moe_expectation_model import moe_expectation_model
from moe_models.moe_models_base import default_optimizer
from helper.moe_models import cross_entropy_loss
from helper.visualise_results import *


# NOTE: Pre-trained models are provided to check the results of all the experiments if you do not have the time to train all the models. 

# Load CIFAR10 dataset

# Paths to where the trained models and figures will be stored. You can change this as you see fit.
working_path = '/nobackup/projects/bdrap03/yamuna/modular_deep_learning/aaai_2022/src/cifar10'
model_path = os.path.join(working_path, '../../models')

if not os.path.exists(model_path):
    os.mkdir(model_path)

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

cifar10_trainset = torchvision.datasets.CIFAR10(root=os.path.join(working_path, 'data'), train=True, download=True, transform=train_transform)
cifar10_testset = torchvision.datasets.CIFAR10(root=os.path.join(working_path, 'data'), train=False, download=True, transform=test_transform)
cifar10_testset, cifar10_trainset

num_classes = 10

trainsize = 50000
testsize =10000

batch_size = 256

cifar10_trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(cifar10_trainset, range(trainsize)), batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True)
cifar10_testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(cifar10_testset, range(testsize)), batch_size=batch_size,
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

# create a set of experts
def experts(num_experts, num_classes, expert_layers_type=expert_layers):
    models = []
    for i in range(num_experts):
        models.append(expert_layers_type(num_classes))
    return nn.ModuleList(models)

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


# Initialize configurations and helper functions
# Compute accuracy of the model
def accuracy(out, yb, mean=True):
    preds = torch.argmax(out, dim=1).to(device, non_blocking=True)
    if mean:
        return (preds == yb).float().mean()
    else:
        return (preds == yb).float()


# Functions to train models

# Function to train original model with and without regularization
# 
# * w_importance_range is the range of values for the $w_{importance}$ hyperparameter of the $L_{importance}$ regularization.
# * w_sample_sim_same_range is the range of values for $\beta_s$ hyperparameter of the $L_s$ regularization.
# * w_sample_sim_diff_range is the range of values for $\beta_d$ hyperparameter of the $L_s$ regularization.

from itertools import product

def train_original_model(model_1, trainloader, testloader, runs, T=[1.0]*20, 
                         w_importance=0.0, w_sample_sim_same=0.0, 
                         w_sample_sim_diff=0.0,
                         num_classes=10, total_experts=5, num_epochs=20):
        
    print('w_importance','{:.1f}'.format(w_importance))
    if w_sample_sim_same < 1:
        print('w_sample_sim_same',str(w_sample_sim_same))
    else:
        print('w_sample_sim_same','{:.1f}'.format(w_sample_sim_same))

    if w_sample_sim_diff < 1:
        print('w_sample_sim_diff',str(w_sample_sim_diff))
    else:
        print('w_sample_sim_diff','{:.1f}'.format(w_sample_sim_diff))


    for run in range(1, runs+1):

        print('Run:', run)

        n_run_models_1 = []

        models = {'moe_expectation_model':{'model':moe_expectation_model,'loss':cross_entropy_loss().to(device),
                                           'experts':{}},}
        for key, val in models.items():

            expert_models = experts(total_experts, num_classes).to(device)

            gate_model = gate_layers(total_experts).to(device)

            moe_model = val['model'](total_experts, num_classes,
                                     experts=expert_models, gate=gate_model, device=device).to(device)

            optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False, weight_decay=1e-3)
            optimizer = default_optimizer(optimizer_moe=optimizer_moe)

            hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer = optimizer,
                                   T = T, w_importance=w_importance, w_sample_sim_same = w_sample_sim_same, 
                                   w_sample_sim_diff = w_sample_sim_diff, 
                                   accuracy=accuracy, epochs=num_epochs)
            val['experts'][total_experts] = {'model':moe_model, 'history':hist}                


        # Save all the trained models
        plot_file = generate_plot_file(model_1, T[0], w_importance=w_importance, w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
                                       specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

        if os.path.exists(os.path.join(model_path, plot_file)):
            n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))
        n_run_models_1.append(models)
        torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
        n_run_models_1 = []


# Function to train the single model
def train_single_model(model_name, trainloader, testloader, num_classes, num_epochs, runs):
    
    loss_criterion = cross_entropy_loss()
    
    n_runs = {'models':[], 'history':[]}
    
    for run in range(1, runs+1):
        
        print('Run', run)
        
        model = single_model(num_classes).to(device)
        history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}
        optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=False, weight_decay=1e-3)
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            train_running_accuracy = 0.0
            num_batches = 0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)

                optimizer.zero_grad()
                loss = loss_criterion(outputs, None, None, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                outputs = model(inputs)

                acc = accuracy(outputs, labels)
                train_running_accuracy += acc

                num_batches += 1

            test_running_accuracy = 0.0
            test_num_batches = 0
            
            for test_inputs, test_labels in testloader:
                test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)
                test_outputs = model(test_inputs)              
                test_running_accuracy += accuracy(test_outputs, test_labels)
                test_num_batches += 1
                
            loss = (running_loss/num_batches)
            train_accuracy = (train_running_accuracy/num_batches)
            test_accuracy = (test_running_accuracy/test_num_batches)
            
            history['loss'].append(loss)
            history['accuracy'].append(train_accuracy.item())
            history['val_accuracy'].append(test_accuracy.item())
            
            print('epoch %d' % epoch,
                  'training loss %.2f' % loss,
                   ', training accuracy %.2f' % train_accuracy,
                   ', test accuracy %.2f' % test_accuracy
                   )
            
        plot_file = generate_plot_file(model_name, specific=str(num_classes)+'_models.pt')
        if os.path.exists(os.path.join(model_path, plot_file)):
            n_runs = torch.load(open(os.path.join(model_path, plot_file),'rb'))
        n_runs['models'].append(model)
        n_runs['history'].append(history)        
        torch.save(n_runs, open(os.path.join(model_path, plot_file),'wb'))
        
        n_runs = {'models':[], 'history':[]}


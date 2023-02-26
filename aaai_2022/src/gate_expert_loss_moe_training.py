import time
import numpy as np
from statistics import mean
# from math import ceil, sin, cos, radians
# from collections import OrderedDict
import os
from itertools import product

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
import torchvision.transforms.functional as TF
from torch.distributions.categorical import Categorical

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    
print()

# import MoE expectation model. All experiments for this dataset are done with the expectation model as it
# provides the best performance
from moe_models.moe_expert_loss_model import moe_expert_loss_model
from moe_models.moe_models_base import expert_loss_gate_optimizer
from helper.moe_models import cross_entropy_loss
from helper.visualise_results import *


# create a set of experts
def experts(num_experts, num_classes, expert_layers_type=None):
    models = []
    for i in range(num_experts):
        models.append(expert_layers_type(num_classes))
    return nn.ModuleList(models)

# Compute accuracy of the model
def accuracy(out, yb, mean=True):
    preds = torch.argmax(out, dim=1).to(device, non_blocking=True)
    if mean:
        return (preds == yb).float().mean()
    else:
        return (preds == yb).float()


def train_loss_gate_model(model, model_name, k=0, trainloader=None, testloader=None, 
                         expert_layers=None, gate_layers=None, runs=1, temps=[[1.0]*20],
                         w_importance_range=[0.0], w_sample_sim_same_range=[0.0], 
                         w_sample_sim_diff_range=[0.0],
                         num_classes=10, total_experts=5, num_epochs=20, model_path='../models/mnist'):

    moe_model_types = { 'moe_expert_loss_model':(moe_expert_loss_model, cross_entropy_loss().to(device))}
 
   
    for T, w_importance, w_sample_sim_same, w_sample_sim_diff in product(temps, w_importance_range, 
                                                                                      w_sample_sim_same_range,  
                                                                                      w_sample_sim_diff_range):
        
        
        print('w_importance','{:.1f}'.format(w_importance))
        if w_sample_sim_same < 1:
            print('w_sample_sim_same',str(w_sample_sim_same))
        else:
            print('w_sample_sim_same','{:.1f}'.format(w_sample_sim_same))
        
        if w_sample_sim_diff < 1:
            print('w_sample_sim_diff',str(w_sample_sim_diff))
        else:
            print('w_sample_sim_diff','{:.1f}'.format(w_sample_sim_diff))
 
        n_run_models_1 = []
        for run in range(1,runs+1):

            print('Run:', run), 

            models = {model_name:{'model':moe_model_types[model_name][0],'loss':moe_model_types[model_name][1],
                                               'experts':{}},}
            for key, val in models.items():

                expert_models = experts(total_experts, num_classes, expert_layers_type=expert_layers).to(device)

                gate_model = gate_layers(total_experts).to(device)
                
                if k > 0:
                    moe_model = val['model'](k, total_experts, num_classes,
                                             experts=expert_models, gate=gate_model, device=device).to(device)
                else:
                    moe_model = val['model'](total_experts, num_classes,
                                         experts=expert_models, gate=gate_model, device=device).to(device)

                optimizer_gate = optim.Adam(gate_model.parameters(), lr=0.001,  amsgrad=False, weight_decay=1e-3)
                params = []
                for i, expert in enumerate(expert_models):
                    params.append({'params':expert.parameters()})
                optimizer_experts = optim.Adam(params, lr=0.001,  amsgrad=False, weight_decay=1e-3)
                optimizer = expert_loss_gate_optimizer(optimizer_gate=optimizer_gate, optimizer_experts=optimizer_experts)
                
                hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer = optimizer, T = T, 
                                       w_importance=w_importance, w_sample_sim_same = w_sample_sim_same, 
                                       w_sample_sim_diff = w_sample_sim_diff, 
                                       accuracy=accuracy, epochs=num_epochs)
                val['experts'][total_experts] = {'model':moe_model, 'history':hist}
            
            # Save all the trained models
            plot_file = generate_plot_file(model, T[0], w_importance=w_importance, w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
                                       specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            if  os.path.exists(os.path.join(model_path, plot_file)):
                n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))
            n_run_models_1.append(models)
            torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
            n_run_models_1 = []

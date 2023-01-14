import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps

import numpy as np
from statistics import mean
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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)


# import MoE expectation model. All experiments for this dataset are done with the expectation model as it
# provides the best performance
from moe_models.moe_expectation_model import moe_expectation_model
from moe_models.moe_top_k_model import moe_top_k_model
from moe_models.moe_models_base import default_optimizer
from helper.moe_models import cross_entropy_loss, stochastic_loss
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


def train_original_model(model, model_name, k=1, trainloader=None, testloader=None, 
                         expert_layers=None, gate_layers=None, 
                         runs=10, temps=[[1.0]*20],                
                         w_importance_range=[0.0], w_sample_sim_same_range=[0.0], 
                         w_sample_sim_diff_range=[0.0],
                         num_classes=10, total_experts=5, num_epochs=20, model_path=None):

    moe_model_types = {'moe_expectation_model':(moe_expectation_model, cross_entropy_loss().to(device)),
                       'moe_top_1_model':(moe_top_k_model, stochastic_loss(cross_entropy_loss).to(device)),
                       'moe_top_k_model':(moe_top_k_model, cross_entropy_loss().to(device))}

    for T, w_importance, w_sample_sim_same, w_sample_sim_diff in product(temps, w_importance_range, 
                                                                         w_sample_sim_same_range,  w_sample_sim_diff_range):
        
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
                
                optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False, weight_decay=1e-3)
                
                optimizer = default_optimizer(optimizer_moe=optimizer_moe)

               
                hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer = optimizer,
                                       T = T, w_importance=w_importance, w_sample_sim_same = w_sample_sim_same, 
                                       w_sample_sim_diff = w_sample_sim_diff, 
                                       accuracy=accuracy, epochs=num_epochs)
                val['experts'][total_experts] = {'model':moe_model, 'history':hist}                


            # Save all the trained models
            plot_file = generate_plot_file(model, T[0], w_importance=w_importance, w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
                                           specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')
            
            if os.path.exists(os.path.join(model_path, plot_file)):
                n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))
            n_run_models_1.append(models)
            torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
            n_run_models_1 = []

            
# # Function to train the single model
# def train_single_model(model_name, trainloader, testloader, num_classes, num_epochs, runs):
    
#     loss_criterion = cross_entropy_loss()
    
#     n_runs = {'models':[], 'history':[]}
    
#     for run in range(1, runs+1):
        
#         print('Run', run)
        
#         model = single_model(num_classes).to(device)
#         history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}
#         optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=False, weight_decay=1e-3)
        
#         for epoch in range(num_epochs):
#             running_loss = 0.0
#             train_running_accuracy = 0.0
#             num_batches = 0

#             for inputs, labels in trainloader:
#                 inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#                 outputs = model(inputs)

#                 optimizer.zero_grad()
#                 loss = loss_criterion(outputs, None, None, labels)

#                 loss.backward()

#                 optimizer.step()

#                 running_loss += loss.item()

#                 outputs = model(inputs)

#                 acc = accuracy(outputs, labels)
#                 train_running_accuracy += acc

#                 num_batches += 1

#             test_running_accuracy = 0.0
#             test_num_batches = 0
            
#             for test_inputs, test_labels in testloader:
#                 test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)
#                 test_outputs = model(test_inputs)              
#                 test_running_accuracy += accuracy(test_outputs, test_labels)
#                 test_num_batches += 1
                
#             loss = (running_loss/num_batches)
#             train_accuracy = (train_running_accuracy/num_batches)
#             test_accuracy = (test_running_accuracy/test_num_batches)
            
#             history['loss'].append(loss)
#             history['accuracy'].append(train_accuracy.item())
#             history['val_accuracy'].append(test_accuracy.item())
            
#             print('epoch %d' % epoch,
#                   'training loss %.2f' % loss,
#                    ', training accuracy %.2f' % train_accuracy,
#                    ', test accuracy %.2f' % test_accuracy
#                    )
            
#         plot_file = generate_plot_file(model_name, specific=str(num_classes)+'_models.pt')
#         if os.path.exists(os.path.join(model_path, plot_file)):
#             n_runs = torch.load(open(os.path.join(model_path, plot_file),'rb'))
#         n_runs['models'].append(model)
#         n_runs['history'].append(history)        
#         torch.save(n_runs, open(os.path.join(model_path, plot_file),'wb'))
        
#         n_runs = {'models':[], 'history':[]}
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

# import MoE expectation model. All experiments for this dataset are done with the expectation model as it
# provides the best performance
from moe_models.moe_no_gate_model import moe_no_gate_self_information_model, moe_no_gate_entropy_model
from moe_models.moe_top_k_model import moe_top_k_model
from moe_models.moe_expectation_model import moe_expectation_model
from moe_models.moe_stochastic_model import moe_stochastic_model
from moe_models.moe_models_base import default_optimizer
from helper.moe_models import cross_entropy_loss, stochastic_loss
from helper.visualise_results import *

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")

print('device', device)

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


from itertools import product

def train_no_gate_model(model, model_name, trainloader, testloader, 
                expert_layers=None, output_type='argmax',
                runs=1, temps=[[1.0]*20], no_gate_temps=[[1.0]*20],
                w_importance_range=[0.0],
                w_sample_sim_same_range=[0.0],w_sample_sim_diff_range=[0.0], 
                num_classes=10, total_experts=5, num_epochs=20, model_path=None):
    
    moe_model_types = {'moe_no_gate_self_information_model':(moe_no_gate_self_information_model, cross_entropy_loss().to(device), output_type),
                       'moe_no_gate_entropy_model':(moe_no_gate_entropy_model, stochastic_loss(cross_entropy_loss).to(device), output_type),
                       'moe_top_k_model':(moe_top_k_model, cross_entropy_loss().to(device))}


    for T, no_gate_T, w_importance, w_sample_sim_same, w_sample_sim_diff in product(temps, no_gate_temps, w_importance_range, w_sample_sim_same_range, w_sample_sim_diff_range,):
        
        print('Temperature',['{:.1f}'.format(t) for t in T])
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
        for run in range(runs):

            print('Run:', run),
            models = {model_name:{'model':moe_model_types[model_name][0],'loss':moe_model_types[model_name][1],
                                               'experts':{}},}

            for key, val in models.items():

                expert_models = experts(num_experts=total_experts, num_classes=num_classes, expert_layers_type=expert_layers)


                moe_model = val['model'](output_type=moe_model_types[model_name][2], 
                                         num_experts=total_experts, num_classes=num_classes,
                                         experts=expert_models, device=device).to(device)
                    
                optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False)
        
                optimizer = default_optimizer(optimizer_moe=optimizer_moe)
                
                hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer = optimizer, T = T, 
                                       no_gate_T=no_gate_T, w_importance=w_importance,  
                                       w_sample_sim_same = w_sample_sim_same, w_sample_sim_diff = w_sample_sim_diff, 
                                       accuracy=accuracy, epochs=num_epochs, model_name=key)
                val['experts'][total_experts] = {'model':moe_model, 'history':hist}

            n_run_models_1.append(models)

        # Save all the trained models
        plot_file = generate_plot_file(model, temp=T[0], no_gate_T=no_gate_T[0], w_importance=w_importance, 
                                       w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff,                                      
                                       specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

        torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))

            
def train_from_no_gate_model(m, k=0, model_name='moe_no_gate_entropy_model', out_model_name='moe_expectation_model',
                             expert_layers=None, gate_layers=None,
                             num_epochs=20, num_classes=10, total_experts=5, 
                             w_importance_range=[0.0], w_sample_sim_same_range=[0.0], w_sample_sim_diff_range=[0.0],
                             trainloader=None, testloader=None, 
                             expert_no_grad=True, gate_no_grad=False, split_training=True, 
                             model_path=None):
    
    T = [1.0]*num_epochs
    moe_model_types = {'moe_expectation_model':(moe_expectation_model, cross_entropy_loss().to(device),''),
                       'moe_stochastic_model':(moe_stochastic_model, stochastic_loss(cross_entropy_loss).to(device),'_stochastic'),
                       # 'moe_top_1_model':(moe_top_k_model, stochastic_loss(cross_entropy_loss).to(device),'_top_1'),
                       'moe_top_1_model':(moe_top_k_model, cross_entropy_loss().to(device),'_top_1'),
                       'moe_top_2_model':(moe_top_k_model, cross_entropy_loss().to(device),'_top_2')}
    
    for w_importance, w_sample_sim_same, w_sample_sim_diff in product(w_importance_range, w_sample_sim_same_range, w_sample_sim_diff_range):
        
        print('w_importance','{:.1f}'.format(w_importance))
        
        if w_sample_sim_same < 1:
            print('w_sample_sim_same',str(w_sample_sim_same))
        else:
            print('w_sample_sim_same','{:.1f}'.format(w_sample_sim_same))
        
        if w_sample_sim_diff < 1:
            print('w_sample_sim_diff',str(w_sample_sim_diff))
        else:
            print('w_sample_sim_diff','{:.1f}'.format(w_sample_sim_diff))
        
        plot_file = generate_plot_file(m, temp=T[0], w_importance=w_importance,  
                                       w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
                                       specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

        no_gate_models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

        n_run_models_1 = []
    
        for i, model in enumerate(no_gate_models): 
            print('Model', i)
            # Initialise the new expert weights to the weights of the experts of the trained attentive gate model.
            # Fix all the weights of the new experts so they are not trained.
            new_expert_models = experts(total_experts, num_classes, expert_layers_type=expert_layers).to(device)
            old_expert_models = model[model_name]['experts'][total_experts]['model'].experts
            for i, expert in enumerate(new_expert_models):
                old_expert = old_expert_models[i]
                expert.load_state_dict(old_expert.state_dict())
                if expert_no_grad:
                    for param in expert.parameters():
                        param.requires_grad = False

                
            gate_model = gate_layers(total_experts).to(device)

            add_model_name = ''
            
            models = {out_model_name:{'model':moe_model_types[out_model_name][0],'loss':moe_model_types[out_model_name][1],
                                           'experts':{}},}
            add_model_name = moe_model_types[out_model_name][2]

            for key, val in models.items():

                if k > 0:
                    moe_model = val['model'](k=k, num_experts=total_experts, num_classes=num_classes,
                                             experts=new_expert_models, gate=gate_model, device=device).to(device)
                else:
                    moe_model = val['model'](num_experts=total_experts, num_classes=num_classes,
                                             experts=new_expert_models, gate=gate_model, device=device).to(device)

                optimizer_moe = optim.Adam(gate_model.parameters(), lr=0.001, amsgrad=False)

                optimizer = default_optimizer(optimizer_moe=optimizer_moe)
                
                if split_training:
                    train_num_epochs = int(num_epochs/2)
                else:
                    train_num_epochs = num_epochs
                print('training epochs', train_num_epochs)
                
                hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer = optimizer,
                                       T = T, accuracy=accuracy, epochs=train_num_epochs)
                
                if split_training:

                    for expert in moe_model.experts:
                        for param in expert.parameters():
                            param.requires_grad = True

                    optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False)

                    optimizer = default_optimizer(optimizer_moe=optimizer_moe)

                    hist_1 = moe_model.train(trainloader, testloader,  val['loss'], optimizer = optimizer,
                                           T = T, accuracy=accuracy, epochs=train_num_epochs)

                    for key, value in hist_1.items():
                        hist[key] = hist[key]+value
                    
                val['experts'][total_experts] = {'model':moe_model, 'history':hist}

            plot_file = generate_plot_file('new_'+m+add_model_name, temp=T[0], w_importance=w_importance, 
                                            w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
                                            specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')
        
            if os.path.exists(os.path.join(model_path, plot_file)):
                n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))
                
            n_run_models_1.append(models)                                
            torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
            n_run_models_1 = []
            print(plot_file)
            


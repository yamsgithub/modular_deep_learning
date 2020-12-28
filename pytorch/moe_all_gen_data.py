#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps
import seaborn as sns
                                        
#get_ipython().run_line_magic('matplotlib', 'inline')

import pickle

from sklearn import datasets
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from statistics import mean
from math import ceil, floor, modf, sin, cos


import torch
import torchvision

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

from visualise_results import *

from moe_models import moe_stochastic_model, moe_stochastic_loss, moe_expectation_model, moe_pre_softmax_expectation_model
from experts_gates import *

from single_model import *

# ### Generate dataset for training

def generate_data(dataset, size):
    num_classes = 2
    X = y = None
    if 'checker_board_rotated' in dataset:
        num_classes = 2
        x_a = [-1.0,-1.0,0.0,0.0]
        x_b = [0.0,0.0,1.0,1.0]
        y_a = [-1.0,0.0,-1.0,0.0]
        y_b = [0.0,1.0,0.0,1.0]
        X = 2 * np.random.random((size,2)) - 1
        def classifier4(X): # a 4x4 checkerboard pattern -- you can use the same method to make up your own checkerboard patterns
            return (np.sum( np.ceil( 2 * X).astype(int), axis=1 ) % 2).astype(float)
        y = classifier4(X)

        deg = [30,45,180,270]
        X_new = None
        y_new = None
        for i in range(0,4):
            rm = np.asarray([[cos(deg[i]),-1*sin(deg[i])],
                             [sin(deg[i]),cos(deg[i])]])
            index = (X[:,0]>=x_a[i])&(X[:,0]<=x_b[i])&(X[:,1]>=y_a[i])&(X[:,1]<=y_b[i])
            X_sub = X[index]
            X_tmp = np.transpose(np.dot(rm, np.transpose(X_sub)))
            r_min = X_tmp[:,0].min()
            r_max = X_tmp[:,0].max()
            X_tmp[:,0] = ((x_b[i]-x_a[i])*(X_tmp[:,0]-r_min)/(r_max-r_min))+x_a[i]
            r_min = X_tmp[:,1].min()
            r_max = X_tmp[:,1].max()
            X_tmp[:,1] = ((y_b[i]-y_a[i])*(X_tmp[:,1]-r_min)/(r_max-r_min))+y_a[i]
            
            if not X_new is None:
                X_new = np.vstack((X_new,X_tmp))
            else:
                X_new = X_tmp
            if not y_new is None:
                y_new = np.concatenate((y_new, y[index]))
            else:
                y_new = y[index]
    
        X = X_new
        y = y_new

    elif 'checker_board' in dataset:
        clf = int(dataset.split('-')[-1])
        
        X = 2 * np.random.random((size,2)) - 1
        def classifier0(X):
            return (np.sum( X * X, axis=1) < 0.66 ).astype(float)
        def classifier1(X): # a 3x2 checkerboard pattern
            return (( np.ceil((3/2)*(X[:,0]+1)).astype(int) + np.ceil( X[:,1]+1).astype(int) ) %2).astype(float)
        def classifier2(X): # a 4x4 checkerboard pattern -- you can use the same method to make up your own checkerboard patterns
            return (np.sum( np.ceil( 2 * X).astype(int), axis=1 ) % 2).astype(float)
        classifiers = [classifier0, classifier1, classifier2]

        y = classifiers[clf]( X )


    plot_data(X, y, num_classes, 'figures/all/'+dataset+'_'+str(num_classes)+'_.png')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(len(y_train))
    print(sum(y_train))
    print(len(y_test))
    print(sum(y_test))

    # Create trainloader
    batchsize = 128
    trainset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), 
                                              torch.tensor(y_train, dtype=torch.long))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=1, pin_memory=True)
    testset = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                             torch.tensor(y_test, dtype=torch.long))
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=True, num_workers=1, pin_memory=True)


    return X, y, trainset, trainloader, testset, testloader, num_classes




        
# ### Mixture of experts model

# compute
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def run_experiment(dataset, trainset, trainloader, testset, testloader, num_classes, total_experts = 3, epochs = 10):

    # experiment with models with different number of experts
    models = {'moe_stochastic_model':{'model':moe_stochastic_model, 'loss':moe_stochastic_loss,'experts':{}}, 
              'moe_expectation_model':{'model':moe_expectation_model,'loss':nn.CrossEntropyLoss(),'experts':{}}, 
              'moe_pre_softmax_expectation_model':{'model':moe_pre_softmax_expectation_model,'loss':nn.CrossEntropyLoss(),'experts':{}},
              'single_model': {'model':single_model, 'loss':nn.CrossEntropyLoss(), 'experts':{}}
    }
    for key, val in models.items():
        print('Model:', key)
        for num_experts in range(1, total_experts+1):
            print('Number of experts ', num_experts)
            if not key == 'single_model':
                expert_models = experts(expert_layers, num_experts, num_classes)
                gate_model = gate_layers(num_experts)
                model = val['model'](num_experts, expert_models, gate_model)
            else:
                moe_model_params = models['moe_stochastic_model']['experts'][num_experts]['parameters']
                model = val['model'](moe_model_params, num_experts, num_classes)

            model_params = sum([p.numel() for p in model.parameters()])
            optimizer = optim.RMSprop(model.parameters(),
                                      lr=0.001, momentum=0.9)
            hist = model.train(trainloader, testloader, optimizer, val['loss'], accuracy, epochs=epochs)
            val['experts'][num_experts] = {'model':model, 'history':hist, 'parameters':model_params}

    return  models

def run_experiment_1(dataset,  single_model, trainset, trainloader, testset, testloader, num_classes, total_experts = 3, epochs = 10):
    
    # experiment with models with different number of experts
    models = {'moe_stochastic_model':{'model':moe_stochastic_model, 'loss':moe_stochastic_loss,'experts':{}}, 
              'moe_expectation_model':{'model':moe_expectation_model,'loss':nn.CrossEntropyLoss(),'experts':{}}, 
              'moe_pre_softmax_expectation_model':{'model':moe_pre_softmax_expectation_model,'loss':nn.CrossEntropyLoss(),'experts':{}},
              'single_model': {'model':single_model, 'loss':nn.CrossEntropyLoss(), 'experts':{}}
    }
    for key, val in models.items():
                                           
        print('Model:', key)
        for num_experts in range(1, total_experts+1):
            print('Number of experts ', num_experts)
            if not key == 'single_model':
                expert_models = experts(expert_layers_1, num_experts, num_classes)
                gate_model = gate_layers_1(num_experts)
                model = val['model'](num_experts, expert_models, gate_model)

            else:
                moe_model_params = models['moe_stochastic_model']['experts'][num_experts]['parameters']
                model = val['model'](moe_model_params, num_experts, num_classes)

            model_params = sum([p.numel() for p in model.parameters()])
            optimizer = optim.RMSprop(model.parameters(),
                                      lr=0.001, momentum=0.9)
            hist = model.train(trainloader, testloader, optimizer, val['loss'], accuracy, epochs=epochs)
            val['experts'][num_experts] = {'model':model, 'history':hist, 'parameters':model_params}

    return  models

def aggregate_results(runs, total_experts):
    results = runs[0]
    for models in runs[1:]:
        for m_key, m_val in models.items():
            for expert in range(1, total_experts+1):
                results[m_key]['experts'][expert]['history']['loss'] = list(np.asarray(results[m_key]['experts'][expert]['history']['loss']) +np.asarray(models[m_key]['experts'][expert]['history']['loss']))
                results[m_key]['experts'][expert]['history']['accuracy'] = list(np.asarray(results[m_key]['experts'][expert]['history']['accuracy']) +np.asarray(models[m_key]['experts'][expert]['history']['accuracy']))
                results[m_key]['experts'][expert]['history']['val_accuracy'] = list(np.asarray(results[m_key]['experts'][expert]['history']['val_accuracy']) +np.asarray(models[m_key]['experts'][expert]['history']['val_accuracy']))

    for m_key, m_val in models.items():
        for expert in range(1, total_experts+1):
            results[m_key]['experts'][expert]['history']['loss'] = list(np.asarray(results[m_key]['experts'][expert]['history']['loss'])/len(runs))
            results[m_key]['experts'][expert]['history']['accuracy'] = list(np.asarray(results[m_key]['experts'][expert]['history']['accuracy'])/len(runs))
            results[m_key]['experts'][expert]['history']['val_accuracy'] = list(np.asarray(results[m_key]['experts'][expert]['history']['val_accuracy'])/len(runs))

    return results

def log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp):
    for m_key, m_val in results.items():
        for i in range(1, total_experts+1):
            fp.write(','.join([dataset, str(num_classes), str(num_runs), str(epochs)])+',')
            fp.write(str(m_val['experts'][i]['parameters'])+',')
            fp.write(','.join([m_key, str(i), str(m_val['experts'][i]['history']['loss'][-1]),
                               str(m_val['experts'][i]['history']['accuracy'][-1]),
                               str(m_val['experts'][i]['history']['val_accuracy'][-1])])+'\n')


def main():

    fp = open('results.csv', 'w')
    col_names = ['dataset', 'number of classes', 'number of runs', 'epochs', 'number of parameters-total', 'model', 'number of experts','loss','training accuracy','validation accuracy']
    fp.write(','.join(col_names)+'\n')

    
    num_runs = 2

    #dataset =  'expert_0_gate_0_checker_board-1'
    
    # X, y, trainset, trainloader, testset, testloader, num_classes = generate_data(dataset)

    # total_experts = 2
    # epochs = 2

    # runs = []
    # for r in range(0, num_runs):
    #     models = run_experiment(dataset, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
    #     runs.append(models)
    
    # results = runs[0]
    # if num_runs > 1:
    #     results = aggregate_results(runs, total_experts)

    # pickle.dump(results,open('../results/'+dataset+'_results.pkl','wb'))

    # log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)
    
    # plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
    
    # plot_accuracy(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

    # dataset =  'expert_0_gate_0_checker_board-2'

    #X, y, trainset, trainloader, testset, testloader, num_classes = generate_data(dataset)
    
    # total_experts = 2
    # epochs = 2
    
    # runs = []
    # for r in range(0, num_runs):
    #     models = run_experiment(dataset, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
    #     runs.append(models)
    
    # results = runs[0]
    # if num_runs > 1:
    #     results = aggregate_results(runs, total_experts)

    # pickle.dump(results,open('../results/'+dataset+'_results.pkl','wb'))

    # log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)

    # plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
    
    # plot_accuracy(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

    # for size in [3000, 5000, 8000]:
    #     dataset =  'expert_1_gate_1_single_shallow_checker_board_'+str(size)+'-2'
    #     X, y, trainset, trainloader, testset, testloader, num_classes = generate_data(dataset, size)
        
    #     total_experts = 2
    #     epochs = 2
        
    #     runs = []
    #     for r in range(0, num_runs):
    #         models = run_experiment_1(dataset, single_model_shallow, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
    #         runs.append(models)
    
    #     results = runs[0]
    #     if num_runs > 1:
    #         results = aggregate_results(runs, total_experts)
            
    #     pickle.dump(results,open('../results/'+dataset+'_results.pkl','wb'))
        
    #     log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)
        
    #     plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
        
    #     plot_accuracy(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

    for size in [3000, 5000, 8000]:
        dataset =  'expert_1_gate_1_single_deep_checker_board_rotated_'+str(size)
        X, y, trainset, trainloader, testset, testloader, num_classes = generate_data(dataset, size)
        
        total_experts = 20
        epochs = 40
        
        runs = []
        for r in range(0, num_runs):
            models = run_experiment_1(dataset, single_model_deep, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
            runs.append(models)
    
        results = runs[0]
        if num_runs > 1:
            results = aggregate_results(runs, total_experts)

        pickle.dump(results,open('../results/'+dataset+'_results.pkl','wb'))

        log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)
        
        plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
        
        plot_accuracy(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')
        
    fp.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()

#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps
import seaborn as sns
                                        
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import datasets
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from statistics import mean
from math import ceil, floor, modf


import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from visualise_results import *

from moe_models import moe_stochastic_model, moe_stochastic_loss, moe_expectation_model, moe_pre_softmax_expectation_model

# ### Generate dataset for training

def generate_data(dataset):
    num_classes = 2
    X = y = None
    if 'checker_board' in dataset:
        clf = int(dataset.split('-')[-1])
        
        X = 2 * np.random.random((3000,2)) - 1
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
    batchsize = 32
    trainset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), 
                                              torch.tensor(y_train, dtype=torch.long))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=2)
    testset = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                             torch.tensor(y_test, dtype=torch.long))
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=True, num_workers=2)


    return X, y, trainset, trainloader, testset, testloader, num_classes

# ### Networks and callbacks

#Expert network
class expert_layers(nn.Module):
    def __init__(self, output):
        super(expert_layers, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.ReLU(),
                    nn.Linear(4,output),
                    nn.Softmax(dim=1)
                )        
        
    def forward(self, input):
        return self.model(input)

class expert_layers_1(nn.Module):
    def __init__(self, output):
        super(expert_layers_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4,output),
            nn.Softmax(dim=1)
        )        
        
    def forward(self, input):
        return self.model(input)

# create a set of experts
def experts(expert_layers, num_experts, num_classes):
    models = []
    for i in range(num_experts):
        models.append(expert_layers(num_classes))
    return nn.ModuleList(models)


#Gate network (Similar to the expert layer)
class gate_layers(nn.Module):
    def __init__(self, num_experts):
        super(gate_layers, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.ReLU(),
                    nn.Linear(4,num_experts),
                    nn.Softmax(dim=1)
                )
        
    def forward(self, input):
        return self.model(input)

class gate_layers_1(nn.Module):
    def __init__(self, num_experts):
        super(gate_layers_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,num_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, input):
        return self.model(input)

class single_model(nn.Module):
    def __init__(self, parameters, num_experts, num_classes):
        super(single_model, self).__init__()
        output = parameters/(2*(num_experts+1)*2)
        if modf(output)[0] < 0.5:
            output = floor(output)
        else:
            output = ceil(output)
        print('parameters', parameters, 'num_experts', num_experts+1, 'output', output)
        self.model = nn.Sequential(
            nn.Linear(2, output*4),
            nn.ReLU(),
            nn.Linear(output*4, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, input):
        return self.model(input)

    def train(self, trainloader, testloader, optimizer, loss_criterion, accuracy, epochs):    

        history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}
        for epoch in range(0, epochs):
            running_loss = 0.0
            training_accuracy = 0.0
            test_accuracy = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = self(inputs)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
                training_accuracy += accuracy(outputs, labels)
                
            for j, test_data in enumerate(testloader, 0):
                test_input, test_labels = test_data
                test_outputs = self(test_input)
                test_accuracy += accuracy(test_outputs, test_labels)
            history['loss'].append(running_loss/(i+1))
            history['accuracy'].append(training_accuracy/(i+1))
            history['val_accuracy'].append(test_accuracy/(j+1))
            print('epoch: %d loss: %.3f training accuracy: %.3f val accuracy: %.3f' %
                  (epoch + 1, running_loss / (i+1), training_accuracy/(i+1), test_accuracy/(j+1)))
        return history



        
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
                model = single_model(moe_model_params, total_experts, num_classes)

            model_params = sum([p.numel() for p in model.parameters()])
            optimizer = optim.RMSprop(model.parameters(),
                                      lr=0.001, momentum=0.9)
            hist = model.train(trainloader, testloader, optimizer, val['loss'], accuracy, epochs=epochs)
            val['experts'][num_experts] = {'model':model, 'history':hist, 'parameters':model_params}

    return  models

def run_experiment_1(dataset,  trainset, trainloader, testset, testloader, num_classes, total_experts = 3, epochs = 10):
    
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
                model = single_model(moe_model_params, total_experts, num_classes)

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

    dataset =  'expert_0_gate_0_checker_board-1'

    X, y, trainset, trainloader, testset, testloader, num_classes = generate_data(dataset)

    num_runs = 10
    total_experts = 10
    epochs = 20

    runs = []
    for r in range(0, num_runs):
        models = run_experiment(dataset, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
        runs.append(models)
    
    results = runs[0]
    if num_runs > 1:
        results = aggregate_results(runs, total_experts)

    log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)
    
    plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
    
    plot_accuracy(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

    dataset =  'expert_0_gate_0_checker_board-2'
    total_experts = 20
    epochs = 40
    
    runs = []
    for r in range(0, num_runs):
        models = run_experiment(dataset, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
        runs.append(models)
    
    results = runs[0]
    if num_runs > 1:
        results = aggregate_results(runs, total_experts)

    log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)

    plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
    
    plot_accuracy(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

    dataset =  'expert_1_gate_1_checker_board-2'
    total_experts = 20
    epochs = 40

    runs = []
    for r in range(0, num_runs):
        models = run_experiment_1(dataset, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
        runs.append(models)
    
    results = runs[0]
    if num_runs > 1:
        results = aggregate_results(runs, total_experts)

    log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)

    plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
    
    plot_accuracy(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')
    
    fp.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()

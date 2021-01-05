#!/usr/bin/env python
# coding: utf-8

import numpy as np
from statistics import mean
from math import ceil, floor, modf

import torch
import torchvision

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import data_generator
from moe_models import *
from experts_gates import *
from single_model import *
from visualise_results import *

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
                if key == 'moe_pre_softmax_expectation_model':
                    expert_models = experts(expert_layers_shallow_presoftmax, num_experts, num_classes)
                else:
                    expert_models = experts(expert_layers_shallow, num_experts, num_classes)
                gate_model = gate_layers_shallow(num_experts)
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
                if key == 'moe_pre_softmax_expectation_model':
                    expert_models = experts(expert_layers_deep_presoftmax, num_experts, num_classes)
                else:
                    expert_models = experts(expert_layers_deep, num_experts, num_classes)
                gate_model = gate_layers_deep(num_experts)
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

def run_experiment_2(dataset,  single_model, trainset, trainloader, testset, testloader, num_classes, total_experts = 3, epochs = 10):
    
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
                if key == 'moe_pre_softmax_expectation_model':
                    expert_models = experts(expert_layers_deep_presoftmax, num_experts, num_classes)
                else:
                    expert_models = experts(expert_layers_deep_escort, num_experts, num_classes)
                gate_model = gate_layers_deep_escort(num_experts)
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
    
    # plot_error_rate(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

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
    
    # plot_error_rate(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

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
        
    #     plot_error_rate(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

    # for size in [3000]:#, 5000, 8000]:
    #     dataset =  'expert_1_gate_1_single_deep_checker_board_rotated_'+str(size)
    #     X, y, trainset, trainloader, testset, testloader, num_classes = generate_data(dataset, size)
        
    #     total_experts = 2
    #     epochs = 2
        
    #     runs = []
    #     for r in range(0, num_runs):
    #         models = run_experiment_1(dataset, single_model_deep, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
    #         runs.append(models)
    
    #     results = runs[0]
    #     if num_runs > 1:
    #         results = aggregate_results(runs, total_experts)

    #     torch.save(results,open('../results/'+dataset+'_results.pt','wb'))

    #     log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)
        
    #     plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
        
    #     plot_error_rate(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

    for size in [3000, 5000, 8000]:
        dataset =  'expert_1_gate_1_single_deep_escort_checker_board_rotated_'+str(size)
        X, y, trainset, trainloader, testset, testloader, num_classes = data_generator.generate_data(dataset, size)
        
        total_experts = 20
        epochs = 40
        
        runs = []
        for r in range(0, num_runs):
            models = run_experiment_2(dataset, single_model_deep, trainset, trainloader, testset, testloader, num_classes, total_experts, epochs)
            runs.append(models)
    
        results = runs[0]
        if num_runs > 1:
            results = aggregate_results(runs, total_experts)

        torch.save(results,open('../results/'+dataset+'_results.pt','wb'))

        log_results(results, total_experts, num_classes, num_runs, epochs, dataset, fp)

        generated_data = data_generator.create_meshgrid(X)
        plot_results(X, y, generated_data, num_classes, trainset, trainloader, testset, testloader, runs[0], dataset, total_experts)
        
        plot_error_rate(results, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

        
    fp.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()

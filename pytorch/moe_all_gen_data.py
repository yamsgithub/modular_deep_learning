#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps 
                                        
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import datasets
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns

import numpy as np
from statistics import mean
from math import ceil


import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from moe_models import moe_stochastic_model, moe_stochastic_loss, moe_expectation_model, moe_pre_softmax_expectation_model

def plot_data(X, y, num_classes, save_as):
    f, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple'][0:num_classes]
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y,palette=colors, ax=ax)
    ax.set_title("2D 3 classes Generated Data")
    plt.ylabel('Dim 2')
    plt.xlabel('Dim 1')
    plt.savefig(save_as)
    plt.show()
    plt.clf()


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

    return X, y, num_classes

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

# create a set of experts
def experts(num_experts, num_classes):
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


# ### Mixture of experts model

# compute
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def run_experiment(dataset, total_experts = 3, epochs = 10):

    X, y, num_classes = generate_data(dataset)
    
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



    # experiment with models with different number of experts
    models = {'moe_stochastic_model':{'model':moe_stochastic_model, 'loss':moe_stochastic_loss,'experts':{}}, 
              'moe_expectation_model':{'model':moe_expectation_model,'loss':nn.CrossEntropyLoss(),'experts':{}}, 
              'moe_pre_softmax_expectation_model':{'model':moe_pre_softmax_expectation_model,'loss':nn.CrossEntropyLoss(),'experts':{}}}
    for key, val in models.items():
        print('Model:', key)
        for num_experts in range(1, total_experts+1):
            print('Number of experts ', num_experts)
            expert_models = experts(num_experts, num_classes)
            gate_model = gate_layers(num_experts)
            moe_model = val['model'](num_experts, expert_models, gate_model)
            optimizer = optim.RMSprop(moe_model.parameters(),
                                      lr=0.001, momentum=0.9)
            hist = moe_model.train(trainloader, testloader, optimizer, val['loss'], accuracy, epochs=epochs)
            val['experts'][num_experts] = {'model':moe_model, 'history':hist}

    return X, y, num_classes, trainset, trainloader, testset, testloader, models


# ### Visualise decision boundaries of mixture of expert model, expert model and gate model

def create_meshgrid(X):
    #create meshgrid
    resolution = 100 # 100x100 background pixels
    a2d_min, a2d_max = np.min(X[:,0]), np.max(X[:,0])
    b2d_min, b2d_max = np.min(X[:,1]), np.max(X[:,1])
    a, b = np.meshgrid(np.linspace(a2d_min, a2d_max, resolution), 
                       np.linspace(b2d_min, b2d_max, resolution))
    generated_data = torch.tensor(np.c_[a.ravel(), b.ravel()], dtype=torch.float32)

    return generated_data

def labels(p, palette=['r','c','y','g']):
    pred_labels = torch.argmax(p, dim=1)+1
    uniq_y = np.unique(pred_labels)
    pred_color = [palette[i-1] for i in uniq_y]
    return pred_color, pred_labels


def predict(dataloader, model):
        
        pred_labels = []
        true_labels = []
        for i, data in enumerate(dataloader):
            inputs, labels = data
            true_labels.append(labels)
            pred_labels.append(torch.argmax(model(inputs), dim=1))
            
        return torch.stack(true_labels), torch.stack(pred_labels)

def plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, models, dataset, total_experts):

    generated_data = create_meshgrid(X)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    
    for e in range(1, total_experts+1):
        nrows = (e*1)+3
        ncols = 3
        thefigsize = (ncols*5,nrows*5)
        
        print('Number of Experts:', e)
        
        fig,ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=thefigsize)
        ax = ax.flatten()
    
        keys = models.keys()
        print(keys)

        index = 0
        for m_key, m_val in models.items():
        
            moe_model = m_val['experts'][e]['model']
            
            pred = moe_model(generated_data)
            pred_color,pred_labels = labels(pred)
            sns.scatterplot(x=generated_data[:,0],y=generated_data[:,1],
                            hue=pred_labels,palette=pred_color, legend=False, ax=ax[index])
            sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette=colors[0:num_classes], ax=ax[index])
            ax[index].set_title('Mixture of Experts')
            ax[index].set_ylabel('Dim 2')
            ax[index].set_xlabel('Dim 1')
            
        
            experts = moe_model.experts

            for i in range(0, e):
                pred = experts[i](generated_data)
                pred_color,pred_labels = labels(pred)
                sns.scatterplot(x=generated_data[:,0],y=generated_data[:,1],
                                hue=pred_labels,palette=pred_color, legend=False, ax=ax[((i+1)*3)+index])
                sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette=colors[0:num_classes],  ax=ax[((i+1)*3)+index])
                
                ax[((i+1)*3)+index].set_title('Expert '+str(i+1)+' Model')
                ax[((i+1)*3)+index].set_ylabel('Dim 2')
                ax[((i+1)*3)+index].set_xlabel('Dim 1')

            palette = sns.husl_palette(total_experts)
            pred_gate = moe_model.gate(generated_data)
            pred_gate_color, pred_gate_labels = labels(pred_gate, palette)
            
            sns.scatterplot(x=generated_data[:,0],y=generated_data[:,1],
                            hue=pred_gate_labels,palette=pred_gate_color, legend=False, ax=ax[((e+1)*3)+index])
            sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette=colors[0:num_classes], ax=ax[((e+1)*3)+index])
            ax[((e+1)*3)+index].set_title('Gate Model')
            ax[((e+1)*3)+index].set_ylabel('Dim 2')
            ax[((e+1)*3)+index].set_xlabel('Dim 1')
        
        
            pred_gate = moe_model.gate(trainset[:][0])
            pred_gate_color, pred_gate_labels = labels(pred_gate, palette)
            
            sns.scatterplot(x=trainset[:][0][:,0],y=trainset[:][0][:,1],
                            hue=pred_gate_labels,palette=pred_gate_color, ax=ax[((e+2)*3)+index])       
            
            
            index += 1
        plt.savefig('figures/all/'+dataset+'_'+str(num_classes)+'_'+str(e)+'_experts.png')
        plt.show()
        plt.clf()

def plot_accuracy(models, total_experts, save_as):
    labels = []
    for m_key, m_val in models.items():
        labels.append(m_key)
        accuracies = []
        for i in range(1, total_experts+1):                
            history = m_val['experts'][i]['history']
            accuracies.append(history['accuracy'][-1])
        print(range(1,len(accuracies)+1), accuracies)
        plt.plot(range(1,len(accuracies)+1), accuracies)
    plt.legend(labels)
    plt.ylim(0, 1)
    plt.xticks(range(1, total_experts+1), [str(i) for i in range(1, total_experts+1)])
    plt.xlabel('Number of Experts')
    plt.ylabel('Accuracy')
    plt.savefig(save_as)
    plt.show()
    plt.clf()
    

def main():
    dataset =  'checker_board-1'
    total_experts = 2
    epochs = 1
    X, y, num_classes, trainset, trainloader, testset, testloader,  models = run_experiment(dataset, total_experts, epochs)
    
    plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, models, dataset, total_experts)
    
    plot_accuracy(models, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

    # dataset =  'checker_board-2'
    # total_experts = 20
    # epochs = 40
    # X, y, num_classes, trainset, trainloader, testset, testloader,  models = run_experiment(dataset, total_experts, epochs)
    
    # plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, models, dataset, total_experts)
    
    # plot_accuracy(models, total_experts, 'figures/all/accuracy_'+dataset+'_'+ str(num_classes)+'_experts.png')

if __name__ == "__main__":
    # execute only if run as a script
    main()

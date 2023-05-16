import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

from sklearn.metrics import confusion_matrix

from helper import moe_models
from moe_models.moe_models_base import moe_models_base

def entropy(x, y):
    p = x[torch.arange(x.shape[0]).type_as(y),y]
    eps=1e-15
    return -1*torch.log2(p+eps)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device= torch.device("cpu")):
        self.std = std
        self.mean = mean
        self.device = device
        
    def __call__(self, tensor):
        rand_noise = torch.randn(tensor.size()).to(self.device) 
        return  tensor + rand_noise * self.std + self.mean

def mae(x,y):
    return torch.mean(torch.abs(y-x))
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class moe_no_gate_self_information_model(moe_models_base):

    def __init__(self, output_type='argmax', num_experts=5, num_classes=10, experts=None, gate=None, 
                 task='classification', device = torch.device("cpu")):
        super(moe_no_gate_self_information_model,self).__init__(num_experts=num_experts, num_classes=num_classes, experts=experts, 
                                                gate=gate, task=task, device=device)
        self.argmax = False
        self.stochastic = False
        self.expectation = False

        if output_type == 'argmax':
            self.argmax = True
        elif output_type == 'stochastic':
            self.stochastic = True
        elif output_type == 'expectation':
            self.expectation = True
            
    def forward(self, inputs, targets=None, T=1.0):
        
        y = []
        h = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            y.append(expert_output.view(1,-1,self.num_classes))
            if self.task == 'classification':
                measure = entropy(expert_output, targets)
            elif self.task == 'regression':
                measure = mae(expert_output, targets)
            h.append(measure)

        y = torch.vstack(y).transpose_(0,1).to(self.device)
        self.expert_outputs = y
        
        h = torch.vstack(h).transpose_(0,1).to(self.device)
        self.per_sample_entropy = h
        add_noise = AddGaussianNoise(device=self.device)
        p = F.softmin(add_noise(h)/T, dim=1).detach()
                
        self.gate_outputs = p

        self.samples = torch.argmax(p, dim=1).to(self.device)
            
        output = y[torch.arange(y.shape[0]).type_as(self.samples), self.samples, :].squeeze()

        return output


class moe_no_gate_entropy_model(moe_models_base):

    def __init__(self, output_type='argmax', num_experts=5, num_classes=10, experts=None, gate=None, task='classification',device = torch.device("cpu")):
        super( moe_no_gate_entropy_model,self).__init__(num_experts=num_experts, num_classes=num_classes, experts=experts, gate=gate, task=task, device=device)
        self.argmax = False
        self.stochastic = False
        self.expectation = False

        if output_type == 'argmax':
            self.argmax = True
        elif output_type == 'stochastic':
            self.stochastic = True
        elif output_type == 'expectation':
            self.expectation = True

    def forward(self, inputs, targets=None, T=1.0):

        batch_size = inputs.shape[0]
        
        y = []
        h = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            y.append(expert_output.view(1,-1,self.num_classes))
            expert_entropy = moe_models.entropy(expert_output, reduction='none')
            h.append(expert_entropy)

        y = torch.vstack(y).transpose_(0,1).to(self.device)
        self.expert_outputs = y
        
        h = torch.vstack(h).transpose_(0,1).to(self.device)

        self.per_sample_entropy = h
        
        p = F.softmin(h/T, dim=1).detach()

        self.gate_outputs = p
        
        if self.expectation:
            # reshape gate output so probabilities correspond 
            # to each expert
            p = p.reshape(p.shape[0],p.shape[1], 1)

            # repeat probabilities number of classes times so
            # dimensions correspond
            p = p.repeat(1,1,y.shape[2])
            
            # expected sum of expert outputs
            output = torch.sum(p*y, 1)
        else:
            try:
                if self.stochastic:
                    m  = Categorical(p)
                    self.samples = m.sample().reshape(len(p), 1).to(self.device)
                elif self.argmax:
                    self.samples = torch.argmax(p, dim=1).to(self.device)
            except:
                raise
                        
            output = y[torch.arange(y.shape[0]).type_as(self.samples), self.samples, :].squeeze()

        return output

                
            

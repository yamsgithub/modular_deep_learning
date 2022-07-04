import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

from sklearn.metrics import confusion_matrix

from helper import moe_models
from moe_models.moe_models_base import moe_models_base

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

class moe_no_gate_model(moe_models_base):

    def __init__(self, output_type='argmax', num_experts=5, num_classes=10, experts=None, gate=None, task='classification'):
        super( moe_no_gate_model,self).__init__(num_experts=num_experts, num_classes=num_classes, experts=experts, gate=gate, task=task)
        self.argmax = False
        self.stochastic = False
        self.expectation = False

        if output_type == 'argmax':
            self.argmax = True
        elif output_type == 'stochastic':
            self.stochastic = True
        elif output_type == 'expectation':
            self.expectation = True

    def forward(self, inputs, T=1.0):

        batch_size = inputs.shape[0]
        
        y = []
        h = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            y.append(expert_output.view(1,-1,self.num_classes))
            expert_entropy = moe_models.entropy(expert_output, reduction='none')
            h.append(expert_entropy)

        y = torch.vstack(y).transpose_(0,1).to(device)
        self.expert_outputs = y
        
        h = torch.vstack(h).transpose_(0,1).to(device)

        # TESTING: using the scalar weights. 
        p = F.softmin(h/T, dim=1).detach()
        #print('gate', p.shape)

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
                    self.samples = m.sample().reshape(len(p), 1).to(device)
                elif self.argmax:
                    self.samples = torch.argmax(p, dim=1).to(device)
            except:
                raise
            
            output = y[torch.arange(0,batch_size).reshape(batch_size,1).to(device), self.samples.reshape(batch_size,1).to(device), :].squeeze()

        return output

                
            

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# The moe architecture that outputs an expected output of the experts
# based on the gate probabilities
class moe_expectation_model(moe_models_base):
    
    def __init__(self, num_experts=5, num_classes=10, augment=0, attention_flag=0, hidden=None, experts=None, gate=None, task='classification'):
        super(moe_expectation_model,self).__init__(num_experts, num_classes, augment, attention_flag, hidden, experts, gate, task)
        
    def forward(self,inputs, T=1.0):

        y = []
        h = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            y.append(expert_output.view(1,-1,self.num_classes))
            if self.attention:
                hidden_output = expert.hidden
                h.append(hidden_output.view(1,-1,hidden_output.shape[1]))

                
        y = torch.vstack(y).transpose_(0,1).to(device)
        
        self.expert_outputs = y

        if self.augment:
            output_aug = torch.flatten(y, start_dim=1)
            input_aug = torch.cat((inputs, output_aug), dim=1)
            p = self.gate(input_aug)
        elif self.attention:
            h = torch.vstack(h).transpose_(0,1).to(device)
            h_gate = self.gate(inputs)

            attention = self.attn(h, h_gate)

            # attention scores are the gate output
            p = attention 
        else:
            p = self.gate(inputs, T)
        
        self.gate_outputs = p
               
        # reshape gate output so probabilities correspond 
        # to each expert
        p = p.reshape(p.shape[0],p.shape[1], 1)

        # repeat probabilities number of classes times so
        # dimensions correspond
        p = p.repeat(1,1,y.shape[2])

        # expected sum of expert outputs
        output = torch.sum(p*y, 1)

        return output

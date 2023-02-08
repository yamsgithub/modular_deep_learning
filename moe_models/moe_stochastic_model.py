import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

from moe_models.moe_models_base import moe_models_base

# The moe architecture that outputs a stochastic selection of an expert
# based on the gate probabilities. 
class moe_stochastic_model(moe_models_base):
    
    def __init__(self, num_experts=5, num_classes=10, augment=0, attention_flag=0, hidden=None, softmax=True, experts=None, gate=None, task='classification',device = torch.device("cpu")):
        super(moe_stochastic_model,self).__init__(num_experts, num_classes, augment, attention_flag, hidden, softmax, experts, gate, task, device)
        
    def forward(self,inputs, T=1.0):
        y = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            y.append(expert_output.view(1,-1,self.num_classes))
        y = torch.vstack(y).transpose_(0,1).to(self.device)

        self.expert_outputs = y
        try:
            if self.augment:
                context = self.attn(inputs, y)
                input_aug = torch.cat((inputs, context.squeeze(1)), dim=1)
                p = self.gate(input_aug, T)
            else:
                p = self.gate(inputs, T)

            self.gate_outputs = p
            
            if self.training:
                m  = Categorical(p)
                # self.samples = m.sample().reshape(len(p), 1).to(self.device)
                self.samples = m.sample().to(self.device)
            else:
                self.samples = torch.argmax(p, dim=1).to(self.device)
        except:
            raise
        
        output = y[torch.arange(y.shape[0]).type_as(self.samples), self.samples, :].squeeze()
        
        return output
    

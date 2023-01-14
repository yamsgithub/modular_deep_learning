import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

from moe_models.moe_models_base import moe_models_base

class moe_expert_loss_model(moe_models_base):

    def __init__(self, num_experts=5, num_classes=10, augment=0, attention_flag=0, hidden=None, softmax=False, experts=None, gate=None, task='classification',device = torch.device("cpu")):
        super(moe_expert_loss_model,self).__init__(num_experts, num_classes, augment, attention_flag, hidden, softmax, experts, gate, task, device)
        self.device = device
    
    def forward(self,inputs, T=1.0):

        p = self.gate(inputs, T)        
        
        y = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            y.append(expert_output)
            
        y = torch.stack(y).transpose_(0,1).to(self.device)

        self.expert_outputs = y
        self.gate_outputs = p
        
        selected_experts = torch.argmax(p, dim=1)
        
        output = y[torch.arange(y.shape[0]).type_as(selected_experts), selected_experts]

        return output

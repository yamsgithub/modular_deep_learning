import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

from moe_models.moe_models_base import moe_models_base

# The moe architecture that outputs a stochastic selection of an expert
# based on the gate probabilities. 
class moe_top_k_model(moe_models_base):
    
    def __init__(self, k=1, num_experts=5, num_classes=10, augment=0, attention_flag=0, hidden=None, softmax=False, experts=None, gate=None, task='classification', device=torch.device("cpu")):
        super(moe_top_k_model,self).__init__(num_experts, num_classes, augment, attention_flag, hidden, softmax, experts, gate, task, device)
        self.k = k
        
    def forward(self,inputs, T=1.0):
        batch_size = inputs.shape[0]
        y = []
        h = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            y.append(expert_output.view(1,-1,self.num_classes))
            if self.attention:
                hidden_output = expert.hidden
                h.append(hidden_output.view(1,-1,hidden_output.shape[1]))

        y = torch.vstack(y).transpose_(0,1).to(self.device)

        if self.attention:
            h = torch.vstack(h).transpose_(0,1).to(self.device)
            h_gate = self.gate(inputs)

            attention = self.attn(h, h_gate)

            # attention scores are the gate output
            del h_gate
            del h
            p = attention 
        else:
            p = self.gate(inputs, T)
        
        self.expert_outputs = y
        mask = torch.full(p.shape, True,dtype=torch.bool).to(self.device)
      
        if self.k==1:
            p = F.softmax(p, dim=1)
            selected_experts = torch.topk(p,self.k,dim=1).indices
            self.gate_outputs = p
        else:
            selected_experts = torch.topk(p,self.k,dim=1).indices
            values = torch.empty(selected_experts.shape,dtype=torch.bool).fill_(False).to(self.device)

            mask = torch.scatter(mask, 1, selected_experts,values)    
            self.gate_outputs = torch.masked_fill(p, mask, float("-inf"))
            self.gate_outputs = F.softmax(self.gate_outputs, dim=1)
            p = self.gate_outputs
            
        p = p.reshape(p.shape[0],p.shape[1], 1)        
        p = p.repeat(1,1,y.shape[2])

        output = p[torch.arange(p.shape[0]).unsqueeze(-1).type_as(selected_experts), selected_experts]*y[torch.arange(y.shape[0]).unsqueeze(-1).type_as(selected_experts), selected_experts]

        if self.k > 1:
            output = torch.sum(output, 1)
        else:
            output = F.softmax(output.squeeze(), dim=1)
        return output
    

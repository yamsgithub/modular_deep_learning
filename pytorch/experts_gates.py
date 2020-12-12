import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

